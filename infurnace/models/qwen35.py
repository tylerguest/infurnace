"""Thin local runner for tinygrad's existing Qwen3.5 GGUF implementation.

Tinygrad owns loading, transformer math, recurrence, caches, RNG and JIT. This
adapter exposes Infurnace's bounded generation/lifecycle API and, on NV, swaps
quantized projection weights for Infurnace's packed tinygrad UOp kernels (the
same `_NVLinear` adapter the Qwen3 path uses). Block math is never replaced.
"""
from pathlib import Path
from tinygrad import Context, Tensor
from tinygrad.llm.model import Transformer
from tinygrad.nn.state import get_state_dict
from infurnace.gguf import parse_gguf, read_gguf_metadata
from infurnace.kernels.quant_gemm import nv_prefill_supported
from infurnace.models.qwen3 import Qwen3Config, Qwen3Model, _NVLinear


class _PackedLinear(_NVLinear):
  """Packed projection that keeps no dense decode graph alive.

  Every live lazy graph is walked on each realize, so retained per-weight
  dequantization graphs cost tens of milliseconds per call. Rows go through
  the packed GEMM when available, otherwise the multirow packed GEMV.
  """
  def __init__(self, original, entry, *, prefill: bool, keep_weight: bool = False):
    super().__init__(original, entry, prefill=prefill, q6_aligned=True)
    if not keep_weight: self.weight = original.weight = None

  def __call__(self, x):
    if self.prefill: return super().__call__(x)
    return self.decode(x)


class Qwen35Model(Qwen3Model):
  def __init__(self, transformer: Transformer, metadata: dict, device: str | None = None):
    if metadata.get('general.architecture') != 'qwen35': raise ValueError('expected dense Qwen3.5 GGUF')
    if not transformer.blk: raise ValueError('Qwen3.5 must contain transformer blocks')
    c = transformer.blk[0].config
    self.config = Qwen3Config(c.num_blocks, c.dim, c.hidden_dim, c.n_heads, c.n_kv_heads,
      c.head_dim, c.v_head_dim, c.rope_dim, c.rope_theta, c.norm_eps, c.vocab_size, transformer.max_context)
    self.transformer, self.metadata = transformer, metadata
    self.device = device or transformer.token_embd.weight.device
    self._generating = False
    self.nv_quant_linears = 0
    self._path, self._host_embedding = None, None

  def embed(self, tokens: list[int]) -> Tensor:
    """Embedding rows as a (1, len(tokens), dim) float tensor on the model device.

    Multirow one-hot gathers from the packed tied table scan the whole
    vocabulary on GPU; a host FP16 copy turns lookups into a small upload.
    """
    if self._path is None:
      from tinygrad import dtypes
      with Context(DEV=self.device):
        ids = Tensor([tokens], dtype=dtypes.int32, device=self.device)
        return Tensor.cat(*(self.transformer.token_embd(ids[:, r:r+1]) for r in range(len(tokens))), dim=1).float()
    from tinygrad import dtypes
    dim = self.config.embedding_length
    if self._host_embedding is None:
      _, entries = parse_gguf(self._path)
      with Context(DEV='CPU'):
        table = entries['token_embd.weight'].to('CPU').decode().cast(dtypes.float16).contiguous().realize()
      self._host_embedding = (table, table.data().cast('B'))
    view, width = self._host_embedding[1], dim * 2
    rows = b''.join(view[t * width:(t + 1) * width] for t in tokens)
    return Tensor(rows, device=self.device).bitcast(dtypes.float16).reshape(1, len(tokens), dim).float()

  @classmethod
  def from_gguf(cls, path: str | Path, max_context: int | None = None, device: str = 'CPU', *,
                nv_quant: bool = False, nv_prefill: bool | None = None) -> 'Qwen35Model':
    if max_context is not None and (type(max_context) is not int or max_context <= 0): raise ValueError('invalid max_context')
    # Architecture dispatch is a cheap metadata-only check. Tensor loading,
    # configuration and block math remain upstream.
    if read_gguf_metadata(path).get('general.architecture') != 'qwen35': raise ValueError('expected dense Qwen3.5 GGUF')
    if not nv_quant:
      with Context(DEV=device):
        transformer, metadata = Transformer.from_gguf(Path(path), max_context=max_context, realize=False)
      return cls(transformer, metadata)
    if device.split(':')[0] != 'NV': raise ValueError('Qwen3.5 packed projections require NV')
    # Stage upstream's raw file in host memory; only packed projections and the
    # small remaining parameters are copied to the GPU. Upstream otherwise
    # re-unpacks every quantized weight inside each step's kernels.
    with Context(DEV='CPU'):
      transformer, metadata = Transformer.from_gguf(Path(path), max_context=max_context, realize=False)
    result = cls(transformer, metadata, device)
    result._path = Path(path)
    result._enable_packed(path, nv_prefill_supported(device) if nv_prefill is None else nv_prefill)
    # Placeholder initialization leaves an unrealized host RNG update chain that
    # every later realize would walk. Swap in a fresh counter rather than realize
    # it (that would materialize the placeholders); host sampling is unused.
    if (counter := Tensor._device_rng_counters.get('CPU')) is not None:
      counter.replace(Tensor.zeros(*counter.shape, dtype=counter.dtype, device='CPU').contiguous().realize())
    return result

  def _enable_packed(self, path, prefill: bool):
    _, entries = parse_gguf(path)
    # Converted Qwen3.5 GGUFs tie the output head to the token embedding.
    entries.setdefault('output.weight', entries['token_embd.weight'])
    packed = set()
    with Context(DEV=self.device):
      for name, entry in entries.items():
        if name == 'token_embd.weight' or len(entry.shape) != 2 or entry.shape[1] % 256 or entry.shape[0] % 16: continue
        if entry.ggml_type not in (12, 13, 14): continue
        if name == 'output.weight': owner, attribute = self.transformer, 'output'
        elif name.startswith('blk.'):
          _, layer, attribute, _ = name.split('.')
          owner = self.transformer.blk[int(layer)]
        else: continue
        tied = name == 'output.weight'
        adapter = _PackedLinear(getattr(owner, attribute), entry.to(self.device), prefill=prefill,
                                keep_weight=tied)
        setattr(owner, attribute, adapter)
        packed.add(name)
        if tied:
          # Embedding lookups read rows from the same packed allocation.
          self.transformer.token_embd.weight = adapter.weight
          adapter.weight = adapter.original.weight = None
        self.nv_quant_linears += 1
      # Move what remains (norms, conv, decay, small Q8_0 gates) as realized FP tensors.
      for name, tensor in get_state_dict(self.transformer).items():
        if tensor.device == self.device: continue
        tensor.replace(tensor.to(self.device).contiguous().realize())
