"""Tinygrad DFlash GGUF block-diffusion drafter (no external model runtime)."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path

from tinygrad import Context, Tensor, TinyJit, UOp, nn, dtypes
from tinygrad.llm.model import apply_rope, precompute_freqs_cis
from tinygrad.nn.state import get_state_dict, load_state_dict
from infurnace.gguf import parse_gguf
from infurnace.model import _NVLinear
from infurnace.kernels.quant_gemv import QUANT_SIZES
from infurnace.qwen35 import Qwen35Model


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
  def from_gguf(cls, metadata: dict, target: Qwen35Model):
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
  # Draft rows are simultaneous. Reuse our packed tinygrad Q8 projection for
  # concrete multi-row shapes rather than materializing a dense target head.
  return layer.decode(x) if isinstance(layer, _NVLinear) else layer(x)


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
  def __init__(self, config: DFlashConfig, target: Qwen35Model, weights: dict[str, Tensor]):
    self.config, self.target, self.device = config, target, target.device
    self.capacity = target.config.context_length
    self.length = 0
    self._jits = {}
    with Context(DEV=self.device):
      self.weights = DraftWeights(config)
      expected = get_state_dict(self.weights)
      if set(weights) != set(expected): raise ValueError('DFlash weight keys mismatch')
      for name, value in weights.items():
        if value.shape != expected[name].shape: raise ValueError(f'invalid DFlash shape: {name}, {value.shape}')
      state = {name: value.to(self.device).cast(dtypes.float32 if name.endswith('norm.weight') else dtypes.float16)
               for name, value in weights.items()}
      load_state_dict(self.weights, state, verbose=False, realize=False)
      self.cache = [Tensor.zeros(2, 1, config.kv_heads, self.capacity, config.head_dim, dtype=dtypes.float16,
                                device=self.device).contiguous().realize() for _ in range(config.layers)]
      self.frequencies = precompute_freqs_cis(config.head_dim, self.capacity, config.theta, device=self.device).realize()

  @classmethod
  def from_gguf(cls, path: str | Path, target: Qwen35Model, *, packed: bool = True, legacy_noncausal: bool = False):
    if not isinstance(target, Qwen35Model): raise ValueError('DFlash currently requires a Qwen3.5 target')
    metadata, entries = parse_gguf(path)
    config = DFlashConfig.from_gguf(metadata, target)
    if type(legacy_noncausal) is not bool: raise ValueError('invalid legacy attention policy')
    if legacy_noncausal: config = replace(config, causal=False)
    # Validate names/shapes before copying bytes to GPU.
    with Context(DEV=target.device): expected = get_state_dict(DraftWeights(config))
    if set(entries) != set(expected): raise ValueError('DFlash GGUF weight keys mismatch')
    for name, entry in entries.items():
      if entry.shape != expected[name].shape: raise ValueError(f'invalid DFlash shape: {name}, {entry.shape}')
    resident = {name: entry.to(target.device) for name, entry in entries.items()}
    result = cls(config, target, {name: entry.decode() for name, entry in resident.items()})
    result.metadata = metadata
    result.packed_linears = 0
    if packed and target.device.split(':')[0] == 'NV':
      for name, entry in resident.items():
        if entry.ggml_type not in QUANT_SIZES or len(entry.shape) != 2 or entry.shape[0] % 16 or entry.shape[1] % 256: continue
        if name.startswith('blk.'):
          _, layer, attr, _ = name.split('.')
          owner = result.weights.blk[int(layer)]
        elif name == 'fc.weight': owner, attr = result.weights, 'fc'
        else: continue
        setattr(owner, attr, _NVLinear(getattr(owner, attr), entry, q6_aligned=True))
        result.packed_linears += 1
    return result

  def reset(self):
    # Old noise/cache rows are masked. Each reused prefix is overwritten by
    # actual target context before it is authorized again.
    self.length = 0

  def _denoise(self, features: Tensor, noise: Tensor, position: UOp):
    c, weights = self.config, self.weights
    ctx_rows, rows = features.shape[1], noise.shape[1]
    context = weights.enc['output_norm'](projection(weights.fc, features))
    x = noise
    query_pos = Tensor.arange(rows).reshape(rows, 1) + Tensor(position)
    key_pos = Tensor.arange(self.capacity).reshape(1, self.capacity)
    visible = (key_pos < Tensor(position) + rows).expand(rows, self.capacity)
    for i, block in enumerate(weights.blk):
      residual, norm = x, block.attn_norm(x)
      q = block.attn_q_norm(projection(block.attn_q, norm).reshape(1, rows, c.heads, c.head_dim)).transpose(1, 2)
      k = projection(block.attn_k, context).cat(projection(block.attn_k, norm), dim=1)
      v = projection(block.attn_v, context).cat(projection(block.attn_v, norm), dim=1)
      k = block.attn_k_norm(k.reshape(1, ctx_rows + rows, c.kv_heads, c.head_dim)).transpose(1, 2)
      v = v.reshape(1, ctx_rows + rows, c.kv_heads, c.head_dim).transpose(1, 2)
      q = apply_rope(q, self.frequencies[position:position + rows])
      k = apply_rope(k, self.frequencies[position - ctx_rows:position + rows])
      store = self.cache[i][:, :, :, position - ctx_rows:position + rows].uop.store(Tensor.stack(k, v).half().uop)
      cache = Tensor(self.cache[i].uop.after(store))
      layer_causal = c.layer_causal(i)
      layer_visible = visible & (key_pos <= query_pos) if layer_causal else visible
      if c.sliding[i]:
        layer_visible = layer_visible & (query_pos - key_pos < c.window)
        if not layer_causal: layer_visible = layer_visible & (key_pos - query_pos < c.window)
      mask = layer_visible.where(0.0, float('-inf')).reshape(1, 1, rows, self.capacity)
      attention = q.scaled_dot_product_attention(cache[0], cache[1], attn_mask=mask, enable_gqa=True)
      x = residual + projection(block.attn_output, attention.transpose(1, 2).reshape(1, rows, -1))
      norm = block.ffn_norm(x)
      x = x + projection(block.ffn_down, projection(block.ffn_gate, norm).silu() * projection(block.ffn_up, norm))
    return weights.output_norm(x[:, 1:])

  def _forward(self, features: Tensor, noise: Tensor, position: UOp):
    logits = projection(self.target.transformer.output, self._denoise(features, noise, position))
    return logits.argmax(-1).contiguous()

  def propose(self, features: Tensor, anchor: int, position: int, count: int) -> list[int]:
    c = self.config
    if type(count) is not int or not 0 <= count < c.block_size: raise ValueError('invalid DFlash proposal count')
    if type(anchor) is not int or not 0 <= anchor < self.target.config.vocab_size: raise ValueError('invalid DFlash anchor')
    width = c.dim * len(c.feature_layers)
    if features.shape[:1] != (1,) or len(features.shape) != 3 or features.shape[2] != width:
      raise ValueError('invalid DFlash target feature shape')
    ctx_rows = features.shape[1]
    if type(position) is not int or position - ctx_rows != self.length or position + count + 1 > self.capacity:
      raise ValueError('DFlash observation must extend the committed prefix and leave space for the noise block')
    if ctx_rows <= 0 or features.device != self.device or features.dtype not in dtypes.floats:
      raise ValueError('invalid DFlash feature dtype/device/rows')
    if count == 0: return []
    with Context(DEV=self.device):
      # Capture inputs are concrete realized buffers, never executed target graphs.
      features = features.contiguous().realize()
      features = Tensor(UOp.from_buffer(features.uop.buffer, self.device)).reshape(features.shape)
      ids = Tensor([[anchor] + [c.mask_token] * count], dtype=dtypes.int32, device=self.device)
      noise = self.target.transformer.token_embd(ids).float().contiguous().realize()
      noise = Tensor(UOp.from_buffer(noise.uop.buffer, self.device)).reshape(noise.shape)
      key = (ctx_rows, count)
      if key not in self._jits: self._jits[key] = TinyJit(self._forward)
      pos = UOp.variable('dflash_position', ctx_rows, self.capacity - count - 1).bind(position)
      decisions = self._jits[key](features, noise, pos).realize()
      output = [int(v) for v in Tensor(decisions.uop.buf_uop).reshape(count).tolist()]
    self.length = position  # Transient noise K/V never authorizes future prefix reuse.
    return output
