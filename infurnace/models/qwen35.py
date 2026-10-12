"""Thin local runner for tinygrad's existing Qwen3.5 GGUF implementation.

Tinygrad owns loading, transformer math, recurrence, caches, RNG and JIT. This
adapter exposes Infurnace's bounded generation/lifecycle API and, on NV, swaps
quantized projection weights for Infurnace's packed tinygrad UOp kernels (the
same `_NVLinear` adapter the Qwen3 path uses). Block math is never replaced.
"""
import contextlib
import functools
import hashlib
import math
import os
from pathlib import Path
from tinygrad import Context, Tensor, UOp
import tinygrad.llm.model as upstream_model
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.model import GatedDeltaNetBlock, Transformer
from tinygrad.nn.state import get_state_dict
from infurnace import disk_cache
from infurnace.gguf import parse_gguf, read_gguf_metadata
from infurnace.kernels.gdn import gdn_step
from infurnace.kernels.quant_gemm import nv_prefill_supported
from infurnace.models.qwen3 import Qwen3Config, Qwen3Model, _NVLinear


class _PackedLinear(_NVLinear):
  """Packed projection that keeps no dense decode graph alive.

  Every live lazy graph is walked on each realize, so retained per-weight
  dequantization graphs cost tens of milliseconds per call. Rows go through
  the packed GEMM when available, otherwise the multirow packed GEMV.
  """
  def __init__(self, original, entry, *, prefill: bool, keep_weight: bool = False, realize: bool = True):
    super().__init__(original, entry, prefill=prefill, q6_aligned=True, realize=realize)
    if not keep_weight: self.weight = original.weight = None

  def __call__(self, x):
    if self.prefill: return super().__call__(x)
    return self.decode(x)


def _fused_gdn_decode(block: GatedDeltaNetBlock):
  """Upstream GatedDeltaNetBlock._attention for one-row decoding, with the recurrence,
  RMSNorm and gate in one NV kernel (upstream fuses this only on AMD; elsewhere the
  per-step scan is dozens of small kernels). Other shapes use upstream's path."""
  upstream_attention = block._attention
  H, K, V = block.num_v_heads, block.head_k_dim, block.head_v_dim
  def attention(x: Tensor, start_pos):
    B, T, _ = x.shape
    if B != 1 or T != 1: return upstream_attention(x, start_pos)
    if not isinstance(start_pos, UOp): start_pos = UOp.variable('start_pos', 0, block.config.max_context - 1).bind(start_pos)
    initial = Tensor(start_pos).eq(0)
    x = x.half()
    out_gate = block.attn_gate(x).reshape(H, V)
    beta = block.ssm_beta(x).sigmoid().reshape(H)
    log_alpha = ((block.ssm_alpha(x).float() + block.ssm_dt['bias']).softplus().reshape(H, -1) * block.ssm_a.reshape(H, -1)).reshape(H)
    # conv over [conv_state | this row]; the last conv_kernel-1 rows become the next conv state
    conv_state = initial.where(0, block.conv_state)
    window = conv_state.cat(block.attn_qkv(x).cast(conv_state.dtype), dim=1).contiguous()
    conv_store = block.conv_state.uop.store(window[:, 1:].cast(block.conv_state.dtype).uop)
    conv_out = functools.reduce(lambda a, b: a + b, (window[:, i:i+1] * block.ssm_conv1d['weight'][:, i]
                                                     for i in range(block.ssm_conv_kernel))).silu()
    q, k, v = conv_out.split([block.q_dim, block.q_dim, block.conv_channels - 2*block.q_dim], dim=-1)
    q, k = (z.reshape(block.num_k_heads, K).normalize(dim=-1, eps=1e-6).repeat(H // block.num_k_heads, 1) for z in (q, k))
    # Like upstream: the conv write rides on the recurrent state, which the step reads and updates.
    state = Tensor(block.recurrent_state.uop.buf_uop.after(conv_store)).reshape(H, V, K)
    z, _ = gdn_step(q * K**-0.5, k, v.reshape(H, V), beta, log_alpha.exp(), state, out_gate, block.ssm_norm.weight,
                    (1 - initial.float()).reshape(1), block.ssm_norm.eps)
    return block.ssm_out(z.reshape(1, 1, H * V))
  return attention


@contextlib.contextmanager
def _lazy_upstream_weights(path):
  """Build upstream's Transformer without reading its weights into host memory.

  Upstream's gguf_shard realizes every tensor of the file (2.7 GB copied to host
  RAM, one realize each), but the packed load replaces nearly all of them with
  packed NV copies read straight from disk. Inside this block the weights stay
  lazy, so only the ones that are kept get read (when moved to the device), and
  upstream reuses the header we already parsed (same metadata and entries)."""
  parse, shard = upstream_model.gguf_parse, upstream_model.gguf_shard
  def cached_parse(fn):
    metadata, entries = parse_gguf(path)
    return metadata, {name: (entry.data, entry.shape, entry.ggml_type) for name, entry in entries.items()}
  def lazy_shard(entries, devices, shard_map=None):
    if shard_map or len(devices) != 1: return shard(entries, devices, shard_map)
    return {name: ggml_data_to_tensor(data.to(devices[0]).flatten(), math.prod(shape), typ).reshape(shape)
            for name, (data, shape, typ) in entries.items()}
  upstream_model.gguf_parse, upstream_model.gguf_shard = cached_parse, lazy_shard
  try: yield
  finally: upstream_model.gguf_parse, upstream_model.gguf_shard = parse, shard


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
    self._tied_head = False  # the packed output head is the token embedding table

  def embed(self, tokens: list[int]) -> Tensor:
    """Embedding rows as a (1, len(tokens), dim) float tensor on the model device.

    Multirow one-hot gathers from the packed table scan the whole vocabulary on
    GPU. With a tied head, only the requested rows are decoded from its packed
    weights on device; otherwise a host FP16 copy turns lookups into an upload.
    Both round through FP16 like upstream's token embedding.
    """
    if self._path is None:
      from tinygrad import dtypes
      with Context(DEV=self.device):
        ids = Tensor([tokens], dtype=dtypes.int32, device=self.device)
        return Tensor.cat(*(self.transformer.token_embd(ids[:, r:r+1]) for r in range(len(tokens))), dim=1).float()
    from tinygrad import dtypes
    dim = self.config.embedding_length
    if self._tied_head:
      from infurnace.kernels.embedding import packed_gather
      head = self.transformer.output
      with Context(DEV=self.device):
        ids = Tensor(tokens, dtype=dtypes.int32, device=self.device)
        rows = packed_gather(ids, head.packed, width=head.in_features, ggml_type=head.ggml_type, q6_aligned=head.q6_aligned,
                             q6_swizzled=head.q6_swizzled, q4_swizzled=head.q4_swizzled)
        return rows.half().float().reshape(1, len(tokens), dim)
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
    with Context(DEV='CPU'), _lazy_upstream_weights(path):
      transformer, metadata = Transformer.from_gguf(Path(path), max_context=max_context, realize=False)
    result = cls(transformer, metadata, device)
    result._path = Path(path)
    result._enable_packed(path, nv_prefill_supported(device) if nv_prefill is None else nv_prefill)
    if os.environ.get('INFURNACE_FUSED_GDN', '1') != '0':
      for block in transformer.blk:
        if (isinstance(block, GatedDeltaNetBlock) and getattr(block, 'ssm_g_a', None) is None
            and block.head_k_dim == 128 and block.head_v_dim == 128):
          block._attention = _fused_gdn_decode(block)
          # The fused step updates the recurrent state in place, so it must be a real buffer.
          with Context(DEV=device):
            block._init_state(Tensor.empty(1, 1, result.config.embedding_length))
            Tensor.realize(block.conv_state, block.recurrent_state)
    # Placeholder initialization leaves an unrealized host RNG update chain that
    # every later realize would walk. Swap in a fresh counter rather than realize
    # it (that would materialize the placeholders); host sampling is unused.
    if (counter := Tensor._device_rng_counters.get('CPU')) is not None:
      counter.replace(Tensor.zeros(*counter.shape, dtype=counter.dtype, device='CPU').contiguous().realize())
    return result

  def _param_cache_path(self, path):
    if os.environ.get('INFURNACE_PARAM_CACHE', '1') == '0': return None
    key = hashlib.sha256(f'{disk_cache.sources()} {disk_cache.file_identity(path)} {self.device}'.encode()).hexdigest()[:32]
    return disk_cache.root() / 'params' / f'{key}.bin'

  def _enable_packed(self, path, prefill: bool):
    _, entries = parse_gguf(path)
    # Converted Qwen3.5 GGUFs tie the output head to the token embedding.
    self._tied_head = 'output.weight' not in entries
    entries.setdefault('output.weight', entries['token_embd.weight'])
    packed, layers = set(), {}
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
        # Lazy copy + repack; each layer is realized in one schedule below
        # (one realize per weight costs ~20 ms of graph bookkeeping each).
        adapter = _PackedLinear(getattr(owner, attribute), entry.to(self.device, realize=False), prefill=prefill,
                                keep_weight=tied, realize=False)
        layers.setdefault(name.split('.')[1] if name.startswith('blk.') else name, []).append(adapter.packed)
        setattr(owner, attribute, adapter)
        packed.add(name)
        if tied:
          # Embedding lookups read rows from the same packed allocation.
          self.transformer.token_embd.weight = adapter.weight
          adapter.weight = adapter.original.weight = None
        self.nv_quant_linears += 1
      # Batches of ~512 MB of raw weights: few schedules, and the staged raw
      # copies of the whole file never coexist on the GPU.
      group, size = [], 0
      for packed_layer in layers.values():
        group += packed_layer
        size += sum(t.nbytes() for t in packed_layer)
        if size >= 512 << 20:
          Tensor.realize(*group)
          group, size = [], 0
      if group: Tensor.realize(*group)
      # Move what remains (norms, conv, decay, small Q8_0 gates) as realized FP16 tensors. Preparing
      # them schedules ~700 tiny kernels (~2 s) for ~10 MB, so the result is cached on disk.
      remaining = {name: tensor for name, tensor in get_state_dict(self.transformer).items() if tensor.device != self.device}
      cache = self._param_cache_path(path) if remaining else None
      cached = disk_cache.load_tensors(cache, self.device) if cache is not None and cache.is_file() else None
      if cached is not None and set(cached) == set(remaining) and all(
          cached[name].shape == tensor.shape and cached[name].dtype == tensor.dtype for name, tensor in remaining.items()):
        for name, tensor in remaining.items(): tensor.replace(cached[name])
      elif remaining:
        moved = [(tensor, tensor.to(self.device).contiguous()) for tensor in remaining.values()]
        Tensor.realize(*[copy for _, copy in moved])
        for tensor, copy in moved: tensor.replace(copy)
        if cache is not None:
          disk_cache.save_tensors(cache, remaining)
          disk_cache.prune(cache.parent, keep=4)
