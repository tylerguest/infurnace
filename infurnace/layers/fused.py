"""Fused layers: upstream's attention and feed-forward rewired onto infurnace's kernels, and the shared-input
projections they and the families' extra passes (Qwen3.5's GDN, DFlash's verifier) use. A norm that feeds
Hadamard-folded projections is folded into the kernel that rotates and quantizes their input."""
import os
from tinygrad import Tensor, UOp, dtypes
from tinygrad.llm.model import TransformerBlock, apply_rope, precompute_freqs_cis
from tinygrad.uop.ops import Ops
from infurnace.kernels.attention import attention, attention_prologue, store_kv
from infurnace.kernels.hadamard import norm_fwht_q8
from infurnace.kernels.quant_gemv import q8_quantize
from infurnace.quant.linear import _NVLinear

PREFILL_ROWS = 16  # prompt chunks are padded to a multiple of this (the fused GDN kernel scans 16 rows at a time)
_SHARED_Q8 = os.environ.get('INFURNACE_SHARED_Q8', '1') != '0'

class FusibleNorm:
  """An RMSNorm that remembers the input of its last call, so the consumer it feeds can take that input (take)
  and norm, rotate and quantize it for Hadamard-folded projections in one kernel (_one_row's and _rows' norm): the
  lazy output, then never read, is never computed. Every consumer of a fusible norm takes, so no input outlives its
  call (tensors can't be weakly referenced, and live lazy graphs are walked on every realize)."""
  def __init__(self, norm):
    # the fused kernels' weight in FP32 once (casting the FP16 weight in each captured step is a kernel a norm)
    self.norm, self.weight, self.eps, self._last = norm, norm.weight.float().contiguous().realize(), norm.eps, None

  def __call__(self, x: Tensor) -> Tensor:
    out = self.norm(x)
    self._last = (out, x)
    return out

  def take(self, h: Tensor) -> Tensor | None:
    """The input that produced `h` (its last output), else None; forgets it either way."""
    last, self._last = self._last, None
    return last[1] if last is not None and last[0] is h else None

def _one_row(x: Tensor, *layers, norm: FusibleNorm | None = None, half: bool = False, joined: bool = False) -> tuple[Tensor, ...]:
  """One-row outputs of projections that read the same input, quantizing it to Q8 once: each packed
  GEMV would otherwise quantize it again. Other inputs or layers go through their own call. With `norm`, `x` may be
  that norm's output: Hadamard-folded projections then take its input and norm, rotate and quantize it in one kernel
  (`half`: x is the norm's output rounded to FP16, as the GDN pass feeds it)."""
  width = getattr(layers[0], 'in_features', None)
  source = norm.take(x) if norm is not None else None
  if _SHARED_Q8 and all(isinstance(layer, _NVLinear) and layer.rotation is layers[0].rotation for layer in layers) and \
     isinstance(x.numel(), int) and x.numel() == width:
    rotation = layers[0].rotation
    if rotation is not None and source is not None: q8 = _norm_q8(norm, source, rotation, half=half)
    # Hadamard-folded weights: one rotation for all of them, in the same kernel as the quantization
    else: q8 = rotation.quantized(x, 32) if rotation is not None else q8_quantize(x, 1, width)
    return _project(layers, x, q8, one_row=True, joined=joined)
  return tuple(layer(x) for layer in layers)

def _project(layers: tuple, x: Tensor, q8, *, one_row: bool, joined: bool = False) -> tuple[Tensor, ...]:
  """Projections of one quantized input: one launch of their merged linear (Model._merge_projections) when they
  were merged in this order, its output split (or with `joined` returned whole, a 1-tuple, for a consumer that reads
  its parts itself: a custom kernel's slice of it would be copied first), else one launch each."""
  group = getattr(layers[0], 'merged_group', None)
  if group is not None and group[1] == tuple(layers):
    out = group[0].decode(x, q8) if one_row else group[0](x, q8)
    return (out,) if joined else tuple(out.split([layer.out_features for layer in layers], dim=-1))
  return tuple(layer.decode(x, q8) if one_row else layer(x, q8) for layer in layers)

def _materialized(*tensors: Tensor) -> bool:
  """Whether each tensor is a view of a kernel's output (so a custom kernel can read it without another kernel)."""
  return all(t.uop.base.op in (Ops.AFTER, Ops.BUFFER) for t in tensors)

def _addends(source: Tensor) -> tuple[Tensor, Tensor] | None:
  """The two FP32 operands of a lazy residual sum (a block's x + attention, h + feed-forward), else None."""
  u = source.uop
  if u.op is not Ops.ADD or len(u.src) != 2 or source.dtype != dtypes.float32 or not isinstance(source.numel(), int): return None
  return (Tensor(u.src[0]), Tensor(u.src[1])) if all(s.shape == u.shape for s in u.src) else None

def _norm_q8(norm: FusibleNorm, source: Tensor, rotation, *, half: bool = False, normed: bool = False):
  """norm(source), rotated and quantized in one kernel (norm_fwht_q8): the Q8 tuple for the projections, and with
  `normed` also the normalized row in FP16. A residual sum's two operands go in separately: the kernel adds them,
  stores the sum, and `source` becomes that output, so the block's later residual add reads it instead of adding
  them again (one kernel fewer each residual add)."""
  signs = rotation.signs if rotation.signs is not None else Tensor.ones(source.shape[-1], device=source.device)
  parts = _addends(source)
  out = norm_fwht_q8(parts[0] if parts else source, norm.weight, norm.eps, signs, block=rotation.block, group=32, normed=normed,
                     half=half, addend=parts[1] if parts else None)
  if parts:
    source.replace(out[-1].reshape(source.shape))
    out = out[:-1]
  return (out[:3], out[3]) if normed else out

def _norm_of(block, name: str) -> FusibleNorm | None:
  """The block's norm `name` when it is fusible (Model._fuse_norms), else None."""
  return norm if isinstance(norm := getattr(block, name, None), FusibleNorm) else None

def _norm_rows_q8(layer: _NVLinear, norm: FusibleNorm, source: Tensor, *, half: bool = False, normed: bool = False):
  """layer.rows_q8 of norm(source) in one kernel (norm_fwht_q8): rows padded to a multiple of 16 (padding rows
  normalize to zero), rotated and quantized in 64-value groups; with `normed`, also the FP16 normalized rows in
  source's shape."""
  symbolic = not isinstance(source.numel(), int)
  activation = source.pad_to(source.max_shape) if symbolic else source
  tokens = activation.numel() // layer.in_features
  def padded(t: Tensor) -> Tensor: return t.reshape(tokens, layer.in_features).pad(((0, -tokens % 16), (0, 0)))
  parts = _addends(source)  # a residual sum: added in the kernel, as _norm_q8 does
  rotation = layer.rotation
  signs = rotation.signs if rotation.signs is not None else Tensor.ones(layer.in_features, device=source.device)
  out = norm_fwht_q8(padded(parts[0] if parts else activation), norm.weight, norm.eps, signs, block=rotation.block, group=64,
                     normed=normed, half=half, addend=padded(parts[1]) if parts else None)
  if parts:
    source.replace(out[-1][:tokens].reshape(source.shape))
    out = out[:-1]
  if not normed: return out
  rows_normed = out[2][:tokens].reshape(activation.shape)
  if symbolic: rows_normed = rows_normed.shrink(tuple((0, size) for size in source.shape))
  return out[:2], rows_normed

def _rows(x: Tensor, *layers, norm: FusibleNorm | None = None, source: Tensor | None = None, joined: bool = False) -> tuple[Tensor, ...]:
  """Multirow outputs of projections that read the same input: Hadamard-folded Q2_0 weights share one rotated
  int8 input (_NVLinear.rows_q8) instead of each rotating and quantizing it. Other layers go through their own call.
  With `norm`, `x` may be that norm's output (or `source` its input): the shared input is then normalized, rotated
  and quantized in one kernel."""
  if source is None and norm is not None: source = norm.take(x)
  if all(isinstance(layer, _NVLinear) and layer.rows_quantized and layer.rotation is layers[0].rotation for layer in layers):
    q8 = _norm_rows_q8(layers[0], norm, source) if source is not None else layers[0].rows_q8(x)
    return _project(layers, x, q8, one_row=False, joined=joined)
  return tuple(layer(x) for layer in layers)

def fused_feed_forward(block):
  """Upstream's dense feed-forward, with gate and up sharing one Q8 activation for one row."""
  upstream_feed_forward = block._feed_forward
  down = block.ffn_down
  def feed_forward(x: Tensor) -> Tensor:
    if hasattr(block, 'ffn_gate_exps'): return upstream_feed_forward(x)
    width = getattr(down, 'in_features', None)
    if not isinstance(x.numel(), int) or x.numel() != getattr(block.ffn_gate, 'in_features', None):
      outs = _rows(x, block.ffn_gate, block.ffn_up, norm=_norm_of(block, 'ffn_norm'), joined=True)
      # merged gate and up: the down projection's rotate-and-quantize kernel reads both halves of the one output
      if len(outs) == 1 and down.rows_quantized: return down(outs[0][..., :width], down.rows_q8(outs[0], pair=True))
      gate, up = outs if len(outs) == 2 else outs[0].split(width, dim=-1)
      # silu(gate) * up made by the same kernel, when both are kernel outputs already (a 16-row verification's are
      # lazy sums of split partials, which the multiply's kernel adds)
      if isinstance(down, _NVLinear) and down.rows_quantized and _materialized(gate, up): return down(gate, down.rows_q8(gate, up=up))
      return down(gate.silu().contiguous() * up)
    outs = _one_row(x, block.ffn_gate, block.ffn_up, norm=_norm_of(block, 'ffn_norm'), joined=True)
    if len(outs) == 1 and _SHARED_Q8 and down.rotation is not None:
      return down.decode(outs[0][..., :width], down.rotation.quantized(outs[0], 32, pair=True))
    gate, up = outs if len(outs) == 2 else outs[0].split(width, dim=-1)
    if _SHARED_Q8 and isinstance(down, _NVLinear) and down.rotation is not None and _materialized(gate, up):
      return down.decode(gate, down.rotation.quantized(gate, 32, up=up))
    return down(gate.silu().contiguous() * up)
  feed_forward.takes_norm = True  # its one-row path takes a FusibleNorm's input (_one_row)
  return feed_forward

def cached_attention(block: TransformerBlock, q: Tensor, k: Tensor, v: Tensor, start_pos) -> Tensor:
  """Write k, v (1, kv_heads, rows, dim) as cache rows start_pos.. of the block's KV cache, then attend over it
  (kernels/attention.py): an int8 cache (int8_kv_cache) when the block has one."""
  if (scales := getattr(block, 'cache_scale', None)) is not None:
    cache, scales = store_kv(block.cache_kv, scales, k, v, start_pos)
    return attention(q, cache, start_pos, scales=scales)
  store = block.cache_kv[:, :, :, start_pos:start_pos + k.shape[2], :].uop.store(Tensor.stack(k, v).cast(block.cache_kv.dtype).uop)
  return attention(q, Tensor(block.cache_kv.uop.after(store)), start_pos)

def int8_kv_cache(block: TransformerBlock):
  """Create the block's KV cache as int8 keys and FP8 values with one scale a vector (kernels/attention.store_kv) in
  place of upstream's FP16 one: half the bytes a decoding step reads, which at long contexts is most of a step
  (Bonsai-4B at 64k keys: ~6 of ~12 ms). Only cached_attention reads it. Its capacity is rounded up to whole
  32-key tiles (the multirow kernel reads values a tile at a time)."""
  init, c = block._init_state, block.config
  def init_state(x: Tensor):
    if not hasattr(block, 'cache_kv'):  # upstream creates the RoPE table with its cache: so does this
      batch, device, capacity = x.shape[0], x.device, -(-c.max_context // 32) * 32
      block.cache_kv = Tensor.zeros(2, batch, c.n_kv_heads, capacity, c.head_dim // 4, dtype=dtypes.uint32, device=device).contiguous()
      block.cache_scale = Tensor.zeros(2, batch, c.n_kv_heads, capacity, dtype=dtypes.float16, device=device).contiguous()
      block.freqs_cis = precompute_freqs_cis(c.rope_dim, c.max_context, c.rope_theta, device=device, yarn=c.yarn)
    init(x)
  block._init_state = init_state

def fused_attention(block: TransformerBlock):
  """Upstream TransformerBlock._attention for one-row decoding and multirow prompt chunks, with the
  attention itself in kernels/attention.py: upstream's attention over a runtime length slows as the
  conversation grows (it swaps in its own kernel only on AMD, at this same spot). Other shapes use
  upstream's path."""
  upstream_attention, c = block._attention, block.config
  def attention_step(x: Tensor, start_pos, idx_mask=None):
    # idx_mask: an indexer's token selection (DeepSeek-style sparse attention), which these models don't have
    if idx_mask is not None: return upstream_attention(x, start_pos, idx_mask)
    B, T, _ = x.shape
    if B == 1 and (isinstance(T, UOp) or T > 1) and x.max_shape[1] % PREFILL_ROWS == 0:
      # A prompt chunk (1..16 rows, or up to a prompt chunk's size), padded to its maximum like the GDN prefill. The padding rows come
      # from zero inputs, so their cached keys and values are finite (the kernel reads every key below the
      # padded end), and the prompt's next rows overwrite them. The prompt reader keeps the padded end in the cache.
      norm = _norm_of(block, 'attn_norm')
      source = norm.take(x) if norm is not None else None
      x = x.pad_to((B, x.max_shape[1], x.shape[2]))
      if source is not None: source = source.pad_to(x.shape)
      return rows_attention(x, start_pos, *_rows(x, block.attn_q, block.attn_k, block.attn_v, norm=norm, source=source))[:, :T]
    if B != 1 or isinstance(T, UOp) or T != 1:
      if hasattr(block, 'cache_scale'): raise NotImplementedError("upstream attention can't read an int8 KV cache")
      return upstream_attention(x, start_pos)
    outs = _one_row(x, block.attn_q, block.attn_k, block.attn_v, norm=_norm_of(block, 'attn_norm'), joined=True)
    if len(outs) == 1:  # merged: the fused prologue reads the one output, or it is split as three
      if (weights := getattr(block, 'qk_norm_f32', None)) is not None and hasattr(block, 'cache_scale'):
        return prologue_row(outs[0], start_pos, weights)
      outs = outs[0].split([block.attn_q.out_features, block.attn_k.out_features, block.attn_v.out_features], dim=-1)
    return rows_attention(x, start_pos, *outs)
  def prologue_row(joined: Tensor, start_pos, weights: tuple[Tensor, Tensor]) -> Tensor:
    # q/k norms, RoPE and the int8 K/V write in one kernel (kernels/attention.attention_prologue), then attention
    H, D = c.n_heads, c.head_dim
    joined = joined.reshape(-1)
    q, cache, scales = attention_prologue(joined, *weights, block.freqs_cis, block.cache_kv, block.cache_scale, start_pos, heads=H,
                                          kv_heads=c.n_kv_heads, dim=D, rope_dim=c.rope_dim, eps=block.attn_q_norm.eps)
    attn = attention(q.reshape(1, H, 1, D), cache, start_pos, scales=scales).reshape(1, 1, H * D)
    gate = joined[:H * 2 * D].reshape(H, 2, D)[:, 1].reshape(1, 1, H * D)
    return block.attn_output(attn * gate.sigmoid())
  def rows_attention(x: Tensor, start_pos, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    B, T, _ = x.shape
    if c.qk_norm and c.qk_norm != c.head_dim: q, k = block.attn_q_norm(q), block.attn_k_norm(k)
    if c.attn_output_gate:
      qg = q.reshape(B, T, c.n_heads, 2, c.head_dim)
      q, gate = qg[:, :, :, 0, :], qg[:, :, :, 1, :].reshape(B, T, c.n_heads * c.head_dim)
    q = q.reshape(B, T, c.n_heads, c.head_dim).transpose(1, 2)
    k = k.reshape(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
    v = v.reshape(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
    if c.qk_norm == c.head_dim: q, k = block.attn_q_norm(q), block.attn_k_norm(k)
    q = apply_rope(q[..., :c.rope_dim], block.freqs_cis[start_pos:start_pos + T]).cat(q[..., c.rope_dim:], dim=-1)
    k = apply_rope(k[..., :c.rope_dim], block.freqs_cis[start_pos:start_pos + T]).cat(k[..., c.rope_dim:], dim=-1)
    attn = cached_attention(block, q, k, v, start_pos).cast(q.dtype).transpose(1, 2).reshape(B, T, -1)
    return block.attn_output(attn * gate.sigmoid() if c.attn_output_gate else attn)
  attention_step.takes_norm = True  # its one-row path takes a FusibleNorm's input (_one_row)
  return attention_step
