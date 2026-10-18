"""Tinygrad DFlash GGUF block-diffusion drafter (no external model runtime)."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path

from tinygrad import Context, Tensor, UOp, nn, dtypes
from tinygrad.llm.model import apply_rope, precompute_freqs_cis
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict
from infurnace.gguf import parse_gguf
from infurnace.linear import _NVLinear
from infurnace.kernels.quant_gemv import QUANT_SIZES
from infurnace.kernels.quant_gemm import nv_prefill_supported
from infurnace.engine import Model
from infurnace.dflash.jit import captured
from infurnace.kernels.attention import attention, store_rows
from infurnace.device import nvidia

@dataclass(frozen=True)
class DFlashConfig:
  layers: int
  dim: int
  hidden: int
  heads: int
  kv_heads: int
  head_dim: int
  epsilon: float
  theta: float
  block_size: int
  mask_token: int
  feature_layers: tuple[int, ...]  # before target block N == HF hidden_states[N]
  sliding: tuple[bool, ...]
  window: int
  causal: bool | None = None

  def layer_causal(self, index: int) -> bool:
    # Current z-lab/dflash defaults: sliding layers causal, full layers
    # noncausal. An explicit config flag overrides every layer.
    return self.sliding[index] if self.causal is None else self.causal

  @classmethod
  def from_gguf(cls, metadata: dict, target: Model):
    if metadata.get('general.architecture') != 'dflash': raise ValueError('expected DFlash GGUF')
    def integer(key):
      value = metadata.get('dflash.' + key)
      if type(value) is not int or value <= 0: raise ValueError(f'invalid dflash.{key}')
      return value
    layers, dim = integer('block_count'), integer('embedding_length')
    heads, kv_heads, hd = integer('attention.head_count'), integer('attention.head_count_kv'), integer('attention.key_length')
    if heads % kv_heads or hd % 2: raise ValueError('invalid DFlash grouped heads/RoPE dimensions')
    if metadata.get('dflash.attention.value_length', hd) != hd: raise ValueError('unequal DFlash K/V head sizes')
    if dim != target.config.embedding_length: raise ValueError('DFlash/target hidden dimensions differ')
    features = metadata.get('dflash.target_layers')
    if (not isinstance(features, list) or not features or any(type(i) is not int or not 0 < i < target.config.block_count for i in features)
        or features != sorted(set(features))): raise ValueError('invalid DFlash feature layers')
    if metadata.get('tokenizer.ggml.tokens') != target.metadata.get('tokenizer.ggml.tokens'):
      raise ValueError('DFlash and target must have identical token vocabularies')
    mask = metadata.get('tokenizer.ggml.mask_token_id')
    if type(mask) is not int or not 0 <= mask < target.config.vocab_size: raise ValueError('missing/invalid DFlash mask token')
    pattern = metadata.get('dflash.attention.sliding_window_pattern', [False] * layers)
    if not isinstance(pattern, list) or len(pattern) != layers or any(type(v) is not bool for v in pattern):
      raise ValueError('invalid DFlash sliding attention pattern')
    window = metadata.get('dflash.attention.sliding_window', 0)
    if type(window) is not int or window < 0 or (any(pattern) and window <= 0): raise ValueError('invalid DFlash sliding window')
    epsilon, theta = (metadata.get('dflash.' + key) for key in ('attention.layer_norm_rms_epsilon', 'rope.freq_base'))
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for v in (epsilon, theta)):
      raise ValueError('invalid DFlash norm/RoPE parameters')
    causal = metadata.get('dflash.attention.causal')
    if causal is not None and type(causal) is not bool: raise ValueError('invalid DFlash causal flag')
    for key in ('expert_count', 'conv_kernel_size', 'selector_rank', 'sample_from_anchor'):
      if metadata.get('dflash.' + key, 0): raise ValueError(f'unsupported DFlash feature: {key}')
    if metadata.get('dflash.rope.scaling.type', 'none') not in ('none', 'default'):
      raise ValueError('unsupported DFlash RoPE scaling')
    return cls(layers, dim, integer('feed_forward_length'), heads, kv_heads, hd, epsilon, theta,
               integer('block_size'), mask, tuple(features), tuple(pattern), window, causal)

def projection(layer, x):
  # Draft rows are simultaneous. Multirow packed GEMM reads weights once;
  # otherwise use the packed GEMV rather than materializing a dense head.
  if not isinstance(layer, _NVLinear): return layer(x)
  return layer(x) if layer.prefill and x.numel() != layer.in_features else layer.decode(x)

class DraftBlock:
  def __init__(self, config: DFlashConfig):
    c = config
    self.attn_norm, self.ffn_norm = nn.RMSNorm(c.dim, c.epsilon), nn.RMSNorm(c.dim, c.epsilon)
    self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(c.head_dim, c.epsilon), nn.RMSNorm(c.head_dim, c.epsilon)
    self.attn_q, self.attn_k, self.attn_v = (nn.Linear(c.dim, n * c.head_dim, bias=False) for n in (c.heads, c.kv_heads, c.kv_heads))
    self.attn_output = nn.Linear(c.heads * c.head_dim, c.dim, bias=False)
    self.ffn_gate, self.ffn_up = (nn.Linear(c.dim, c.hidden, bias=False) for _ in range(2))
    self.ffn_down = nn.Linear(c.hidden, c.dim, bias=False)

class DraftWeights:
  def __init__(self, config):
    self.fc = nn.Linear(config.dim * len(config.feature_layers), config.dim, bias=False)
    self.enc = {'output_norm': nn.RMSNorm(config.dim, config.epsilon)}
    self.output_norm = nn.RMSNorm(config.dim, config.epsilon)
    self.blk = [DraftBlock(config) for _ in range(config.layers)]

class DFlashDraft:
  def __init__(self, config: DFlashConfig, target: Model, weights: dict[str, Tensor]):
    self.config, self.target, self.device = config, target, target.device
    self.capacity = target.config.context_length
    self.length = 0
    # Target feature rows observed so far (positions 0..observed-1). The last
    # `_staged` of them sit in the 16-row staging buffer; earlier rows are in
    # the draft K/V cache.
    self.observed = self._staged = 0
    self._jits = {}
    # Full-block proposals can run on the fused round's draft graph instead of capturing their own (~4 s of
    # warmup): a callable returning (graph, its noise buffer), set by the parallel verifier.
    self.full_block_graph = None
    with Context(DEV=self.device):
      self.weights = DraftWeights(config)
      expected = get_state_dict(self.weights)
      if set(weights) != set(expected): raise ValueError('DFlash weight keys mismatch')
      for name, value in weights.items():
        if value.shape != expected[name].shape: raise ValueError(f'invalid DFlash shape: {name}, {value.shape}')
      state = {name: value.to(self.device).cast(dtypes.float32 if name.endswith('norm.weight') else dtypes.float16)
               for name, value in weights.items()}
      load_state_dict(self.weights, state, verbose=False, realize=False)
      # One extra chunk of rows lets a padded context chunk end past capacity.
      self.cache_length = self.capacity + config.block_size
      # Sliding layers keep only what their window can see, in a ring: position x at slot x % ring_length.
      # Two blocks of slack cover a padded context chunk plus the noise block written past the window.
      self.ring_length = config.window + 2 * config.block_size
      self.ring = [config.sliding[i] and config.window > 0 and self.ring_length < self.cache_length for i in range(config.layers)]
      self.cache = [Tensor.zeros(2, 1, config.kv_heads, self.ring_length if ring else self.cache_length, config.head_dim,
                                 dtype=dtypes.float16, device=self.device).contiguous().realize() for ring in self.ring]
      # Ring slots below the floor hold positions from a later pass (see load_state); attention skips them.
      self._floor = Tensor.zeros(1, dtype=dtypes.int32, device=self.device).contiguous().realize()
      self._floor_value, self._high = 0, 0  # the floor, and an upper bound on every position written
      self.frequencies = precompute_freqs_cis(config.head_dim, self.cache_length, config.theta, device=self.device).realize()
      self._start = UOp.variable('dflash_context_start', 0, self.capacity - 1)
      self._absorb_jit = captured(self._absorb)
      self._noise_inputs: dict[int, Tensor] = {}
      # Fixed-shape staging for observed rows; stores use a variable offset, so
      # only the row count (1..block) selects a kernel, never the history shape.
      self._context = Tensor.empty(1, config.block_size, config.dim * len(config.feature_layers), device=self.device).realize()
      self._offset = UOp.variable('dflash_stage_offset', 0, config.block_size - 1)

  @classmethod
  def from_gguf(cls, path: str | Path, target: Model, *, packed: bool = True, legacy_noncausal: bool = False):
    if not isinstance(target, Model): raise ValueError('DFlash requires an infurnace Model target')
    metadata, entries = parse_gguf(path)
    config = DFlashConfig.from_gguf(metadata, target)
    if type(legacy_noncausal) is not bool: raise ValueError('invalid legacy attention policy')
    if legacy_noncausal: config = replace(config, causal=False)
    # Validate names/shapes before copying bytes to GPU.
    with Context(DEV=target.device): expected = get_state_dict(DraftWeights(config))
    if set(entries) != set(expected): raise ValueError('DFlash GGUF weight keys mismatch')
    for name, entry in entries.items():
      if entry.shape != expected[name].shape: raise ValueError(f'invalid DFlash shape: {name}, {entry.shape}')
    # Lazy copies: everything is realized in one schedule at the end.
    resident = {name: entry.to(target.device, realize=False) for name, entry in entries.items()}
    result = cls(config, target, {name: entry.decode() for name, entry in resident.items()})
    result.metadata, result.path = metadata, Path(path)
    result.packed_linears = 0
    if packed and nvidia(target.device):
      for name, entry in resident.items():
        if entry.ggml_type not in QUANT_SIZES or len(entry.shape) != 2 or entry.shape[0] % 16 or entry.shape[1] % 256: continue
        if name.startswith('blk.'):
          _, layer, attr, _ = name.split('.')
          owner = result.weights.blk[int(layer)]
        elif name == 'fc.weight': owner, attr = result.weights, 'fc'
        else: continue
        adapter = _NVLinear(getattr(owner, attr), entry, q6_aligned=True,
                            prefill=entry.ggml_type in (12, 13, 14) and nv_prefill_supported(target.device), realize=False)
        # projection() always uses the packed GEMV; keep no dense decode graph alive.
        adapter.weight = adapter.original.weight = None
        setattr(owner, attr, adapter)
        result.packed_linears += 1
    # Realize the packed projections and what stays dense (norms, small projections)
    # together: captured graphs then read real buffers instead of recomputing casts.
    with Context(DEV=result.device):
      if lazy := [t for t in get_parameters(result.weights) if not t.uop.is_realized]: Tensor.realize(*lazy)
    return result

  def reset(self):
    # Old noise/cache rows are masked. Each reused prefix is overwritten by
    # actual target context before it is authorized again.
    self.length = self.observed = self._staged = 0
    self._set_floor(0)

  def _set_floor(self, value: int):
    if value == self._floor_value: return
    with Context(DEV=self.device):
      Tensor(self._floor.uop.after(self._floor.uop.store(Tensor([value], dtype=dtypes.int32, device=self.device).uop))).realize()
    self._floor_value = value

  def _wrote(self, position: int):
    self._high = max(self._high, position)

  def save_state(self):
    """What restores this draft to its current prompt position: the position and the staged target
    rows not yet absorbed. Its K/V cache needs no copy: rows before the position stay valid in place,
    except ring slots a later pass wrote over (load_state hides those)."""
    with Context(DEV=self.device): context = self._context.clone().realize()
    return self.observed, self._staged, context

  def load_state(self, state):
    observed, staged, context = state
    with Context(DEV=self.device): Tensor(self._context.uop.after(self._context.uop.store(context.uop))).realize()
    self.length, self.observed, self._staged = observed, observed, staged
    # A ring slot for position k was overwritten if anything at k + ring_length or later was written:
    # hide positions up to the highest write less a lap (never past the restored position itself).
    if any(self.ring): self._set_floor(min(max(0, self._high - self.ring_length), observed))

  def synced(self, position: int):
    """Rows up to `position` were absorbed outside `observe` (fused rounds)."""
    self.length = self.observed = position
    self._staged = 0
    self._wrote(position + 2 * self.config.block_size)  # that round's padded chunk and noise block

  def observe(self, features: Tensor):
    """Absorb target feature rows (1, rows, width) for the next positions.

    Rows are copied into the staging buffer before returning, so the caller may
    overwrite `features` afterwards (e.g. the next captured target step)."""
    c = self.config
    width = c.dim * len(c.feature_layers)
    if len(features.shape) != 3 or features.shape[0] != 1 or features.shape[2] != width or features.device != self.device:
      raise ValueError('invalid DFlash target feature shape')
    rows, chunk = features.shape[1], c.block_size
    if not 0 < rows or self.observed + rows > self.capacity: raise ValueError('DFlash observation exceeds the context')
    with Context(DEV=self.device):
      done = 0
      while done < rows:
        take = min(chunk - self._staged, rows - done)
        offset = self._offset.bind(self._staged)
        part = features[:, done:done + take].float()
        Tensor(self._context.uop.after(self._context[:, offset:offset + take].uop.store(part.uop))).realize()
        self._staged, self.observed, done = self._staged + take, self.observed + take, done + take
        if self._staged == chunk:
          self._absorb_jit(self._context, self._start.bind(self.observed - chunk))
          self._staged = 0
      self._wrote(self.observed + chunk)

  def _absorb(self, features: Tensor, start: UOp):
    # Fixed-width context chunk: write draft K/V for observed target rows.
    # Padding rows land at positions that are overwritten (later context or the
    # noise block) or masked before any query can see them.
    c, weights, width = self.config, self.weights, features.shape[1]
    context = weights.enc['output_norm'](projection(weights.fc, features))
    stored = []
    for i, block in enumerate(weights.blk):
      k = block.attn_k_norm(projection(block.attn_k, context).reshape(1, width, c.kv_heads, c.head_dim)).transpose(1, 2)
      v = projection(block.attn_v, context).reshape(1, width, c.kv_heads, c.head_dim).transpose(1, 2)
      k = apply_rope(k, self.frequencies[start:start + width])
      stored.append(store_rows(self.cache[i], Tensor.stack(k, v).half(), start, ring=self.ring[i]))
    Tensor.realize(*stored)

  def _denoise(self, noise: Tensor, position: UOp):
    c, weights = self.config, self.weights
    rows, x = noise.shape[1], noise
    for i, block in enumerate(weights.blk):
      residual, norm = x, block.attn_norm(x)
      q = block.attn_q_norm(projection(block.attn_q, norm).reshape(1, rows, c.heads, c.head_dim)).transpose(1, 2)
      k = block.attn_k_norm(projection(block.attn_k, norm).reshape(1, rows, c.kv_heads, c.head_dim)).transpose(1, 2)
      v = projection(block.attn_v, norm).reshape(1, rows, c.kv_heads, c.head_dim).transpose(1, 2)
      q, k = apply_rope(q, self.frequencies[position:position + rows]), apply_rope(k, self.frequencies[position:position + rows])
      cache = store_rows(self.cache[i], Tensor.stack(k, v).half(), position, ring=self.ring[i])
      # Over the active length only: rows see keys up to position+rows (causal: up to their own),
      # within the window on sliding layers.
      attended = attention(q, cache, position, causal=c.layer_causal(i), window=c.window if c.sliding[i] else 0,
                           ring=self.ring[i], floor=self._floor if self.ring[i] else None).cast(q.dtype)
      x = residual + projection(block.attn_output, attended.transpose(1, 2).reshape(1, rows, -1))
      norm = block.ffn_norm(x)
      x = x + projection(block.ffn_down, projection(block.ffn_gate, norm).silu() * projection(block.ffn_up, norm))
    return weights.output_norm(x[:, 1:])

  def _observe(self, features: Tensor, noise: Tensor, position: UOp | int, *, tokens: bool = False) -> Tensor:
    """Eager absorb-then-denoise over one observation (reference/test path)."""
    chunk, rows = self.config.block_size, features.shape[1]
    for offset in range(0, rows, chunk):
      part = features[:, offset:offset + chunk].float()
      self._absorb(part.pad(((0, 0), (0, chunk - part.shape[1]), (0, 0))), position - rows + offset)
    return self._forward(noise, position) if tokens else self._denoise(noise, position)

  def _forward(self, noise: Tensor, position: UOp):
    logits = projection(self.target.transformer.output, self._denoise(noise, position))
    return logits.argmax(-1).contiguous()

  def _denoise_graph(self, count: int):
    """The captured denoise graph for `count` proposals and its persistent noise
    input (one buffer per shape: captured graphs always see the same address)."""
    if count == self.config.block_size - 1 and self.full_block_graph is not None: return self.full_block_graph()
    if count not in self._jits:
      with Context(DEV=self.device):
        self._noise_inputs[count] = Tensor.empty(1, count + 1, self.config.dim, device=self.device).realize()
      self._jits[count] = captured(self._forward)
    return self._jits[count], self._noise_inputs[count]

  def propose(self, anchor: int, position: int, count: int) -> list[int]:
    """Draft `count` tokens after `anchor` at `position`; every target row
    before `position` must have been observed."""
    c = self.config
    if type(count) is not int or not 0 <= count < c.block_size: raise ValueError('invalid DFlash proposal count')
    if type(anchor) is not int or not 0 <= anchor < self.target.config.vocab_size: raise ValueError('invalid DFlash anchor')
    if type(position) is not int or position != self.observed or position <= 0 or position + count + 1 > self.capacity:
      raise ValueError('DFlash proposal must follow the observed prefix and leave space for the noise block')
    if count == 0: return []
    self._wrote(position + c.block_size + 1)
    with Context(DEV=self.device):
      # A partial chunk is absorbed padded; its padding rows land where later
      # context or this noise block overwrites them, or stay masked.
      if self._staged: self._absorb_jit(self._context, self._start.bind(self.observed - self._staged))
      # Always the full block, as fused rounds draft, and keep the first `count`: drafting fewer rows costs
      # about the same, and each new row count would capture its own graph (seconds) on first use.
      # Only the last positions of the context need a smaller block.
      rows = c.block_size - 1 if position + c.block_size <= self.capacity else count
      graph, noise = self._denoise_graph(rows)
      embedded = self.target.embed([anchor] + [c.mask_token] * rows)
      Tensor(noise.uop.after(noise.uop.store(embedded.uop))).realize()
      pos = UOp.variable('dflash_position', 1, self.capacity - rows - 1).bind(position)
      decisions = graph(noise, pos).realize()
      output = [int(v) for v in Tensor(decisions.uop.buf_uop).reshape(rows).tolist()][:count]
    self.length = position  # Transient noise K/V never authorizes future prefix reuse.
    return output
