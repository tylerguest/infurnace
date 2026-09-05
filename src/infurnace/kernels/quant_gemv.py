from __future__ import annotations
"""NV decode GEMV kernels for GGML Q4_K/Q6_K packed weights (C1).

Port of tinygrad's AMD quant-linear decode kernels
(``tinygrad/llm/kernels/amd.py`` ``q8_linear`` / ``_quant_decode_kernel``) to
the NV backend. The decode GEMV is M==1 (one token per row), so it runs as a
warp-per-output-row kernel: each lane owns one 32-wide group of the in-axis,
dequantizes the Q4_K/Q6_K block in registers, and the full wave reduces the
partial dots. Weights stay packed on device (read once per token) and
activations are Q8-quantized per 32-wide group so the dot product uses integer
``__dp4a`` hardware, matching the llama.cpp decode design that sustains
~312 GB/s on this card.

The kernels are built from UOps rendered by tinygrad's CUDA C renderer
(``DEV=NV`` targets ``CUDARenderer`` + NVRTC on ``sm_120``). CUDA-specific
intrinsics are injected through ``Ops.CUSTOM``/``Ops.CUSTOMI`` format strings:
``__dp4a`` and ``__shfl_xor_sync``. Everything else uses renderer-agnostic UOps,
so the module is the Infurnace-owned custom-UOp kernel surface.

Only the decode path is ported (the AMD WMMA token-batched path is prefill
scope, Phase 10). The kernels are TinyJit-capturable: static shapes, no Python
control flow inside the captured graph.
"""
import functools
import math
from typing import Callable, cast

from tinygrad import Tensor, UOp, dtypes
from tinygrad.device import Buffer
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import prod
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

# GGML block geometry (bytes per 256-weight block): Q4_K 144, Q6_K 210.
Q4_K, Q6_K, GGML_BLOCK_SIZE, Q8_GROUP_SIZE, Q4_WORDS, Q6_BYTES = 12, 14, 256, 32, 36, 210
QUANT_SIZES = {Q4_K: Q4_WORDS * 4, Q6_K: Q6_BYTES}  # bytes per 256-weight block

class QuantGemvError(ValueError):
  """Quant-GEMV inputs do not satisfy the kernel contract."""

def warp_reduce(val: UOp, maximum: bool = False, full_wave: bool = False) -> UOp:
  """Full-wave (32-lane) xor-shuffle reduction through CUDA ``__shfl_xor_sync``."""
  for offset in ((16, 8, 4, 2, 1) if full_wave else (8, 4, 2, 1)):
    if val.op is Ops.INDEX and val.addrspace == AddrSpace.REG:
      val = val.load()
    other = UOp(Ops.CUSTOM, dtypes.float, (val,), arg=f"__shfl_xor_sync(0xffffffff, {{0}}, {offset})")
    val = val.maximum(other) if maximum else val + other
  return val

def _dp4a(a: UOp, b: UOp, c: UOp) -> UOp:
  return UOp(Ops.CUSTOMI, dtypes.int32, (a.int(), b.int(), c), arg="__dp4a({}, {}, {})")

def _load_lanes(ptr: UOp, lanes: int) -> UOp:
  """Load ``lanes`` consecutive elements at a buffer index as a vector value."""
  assert ptr.op is Ops.INDEX
  buf, coords = ptr.src[0], ptr.src[1:]
  idx = sum((coord * math.prod(buf.shape[i + 1:]) for i, coord in enumerate(coords)), UOp.const(0))
  return UOp(Ops.SHRINK, src=(buf.flatten(), idx, UOp.const(lanes))).load(dtype=ptr.dtype)

def _load_byte(raw: UOp, base: UOp, offset: UOp) -> UOp:
  return (raw[base + offset // 4] >> ((offset & 3) * 8).cast(dtypes.uint32)) & 255

def _half(value: UOp) -> UOp:
  return value.cast(dtypes.uint16).bitcast(dtypes.float16).float()

def _q4_scales(raw: UOp, base: UOp, subgroup: UOp) -> tuple[UOp, UOp, UOp, UOp]:
  """Q4_K/Q5_K shared scale layout: d, dmin, scale[4], minimum[4] in 12 bytes."""
  scale = (subgroup < 4).where(_load_byte(raw, base, 4 + subgroup) & 63,
    (_load_byte(raw, base, 8 + subgroup) & 15) | ((_load_byte(raw, base, subgroup) >> 6) << 4))
  minimum = (subgroup < 4).where(_load_byte(raw, base, 8 + subgroup) & 63,
    (_load_byte(raw, base, 8 + subgroup) >> 4) | ((_load_byte(raw, base, 4 + subgroup) >> 6) << 4))
  d, dmin = (raw[base] & 0xffff).cast(dtypes.uint16), (raw[base] >> 16).cast(dtypes.uint16)
  return _half(d), _half(dmin), scale.float(), minimum.float()

def _q8_quantize_kernel(q: UOp, scale: UOp, x: UOp, tokens: int, in_features: int) -> UOp:
  """Per-32-group Q8_0-style activation quantize: 8 int8 per 4 lanes, one fp32 scale."""
  groups = in_features // Q8_GROUP_SIZE
  token_group, lane = UOp.range(tokens * groups, 0), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  token, group = token_group // groups, token_group % groups
  x = x.reshape(tokens, groups, 32)
  group_scale = (warp_reduce(x[token, group, lane].float().abs(), maximum=True, full_wave=True) / 127).maximum(1e-8)
  word_lane = lane.minimum(7)
  xs = tuple(x[token, group, word_lane * 4 + i].float() for i in range(4))
  word = sum(((v / group_scale).round().clip(-127, 127).cast(dtypes.int8).cast(dtypes.uint8).cast(dtypes.uint32) << (i * 8)
              for i, v in enumerate(xs)), UOp.const(0, dtypes.uint32))
  stores = (q[token, group, lane.valid(lane < 8)].store(word), scale[token, group.valid(lane.eq(0))].store(group_scale))
  return UOp.group(*stores).end(token_group, lane).sink(arg=KernelInfo(name="q8_quantize", opts_to_apply=()))

def q8_quantize(x: Tensor, tokens: int, in_features: int) -> tuple[Tensor, Tensor]:
  groups = in_features // Q8_GROUP_SIZE
  q = Tensor.empty(tokens, groups, 8, dtype=dtypes.uint32, device=x.device)
  scale = Tensor.empty(tokens, groups, dtype=dtypes.float32, device=x.device)
  q, scale = Tensor.custom_kernel(q, scale, x, fxn=functools.partial(_q8_quantize_kernel, tokens=tokens, in_features=in_features))[:2]
  return q, scale

def _decode_linear(out: UOp, out_features: int, group_count: int, group_dot: Callable[[UOp, UOp, UOp], UOp], name: str) -> UOp:
  """Warp-per-output-row decode GEMV: each 32-lane warp covers 32 in-groups."""
  chunks = (group_count + 31) // 32
  token_output_chunk, lane = UOp.range(out.shape[0] * out_features * chunks, 0), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  token, output, chunk = token_output_chunk // (out_features * chunks), (token_output_chunk // chunks) % out_features, token_output_chunk % chunks
  group = lane + chunk * 32
  value = group_dot(token, output, group) if group_count % 32 == 0 else \
    (group < group_count).where(group_dot(token, output, group.minimum(group_count - 1)), UOp.const(0, dtypes.float32))
  total = warp_reduce(value, full_wave=True)
  return out[token, output, chunk.valid(lane.eq(0))].store(total.cast(out.dtype)).end(token_output_chunk, lane).sink(
    arg=KernelInfo(name=name, opts_to_apply=()))

def _quant_decode_kernel(out: UOp, raw: UOp, xq: UOp, xd: UOp, out_features: int, in_features: int, ggml_type: int) -> UOp:
  group_count = in_features // Q8_GROUP_SIZE
  def group_dot(token: UOp, output: UOp, group: UOp) -> UOp:
    block, subgroup = group // 8, group % 8
    xwords = _load_lanes(xq[token, group, 0], 8)
    if ggml_type == Q4_K:
      base = (output * in_features // GGML_BLOCK_SIZE + block) * Q4_WORDS
      qs_base, dot, qsum = base + 4 + (subgroup // 2) * 8, UOp.const(0, dtypes.int32), UOp.const(0, dtypes.int32)
      for word_idx in range(8):
        word = (raw[qs_base + word_idx] >> ((subgroup & 1) * 4).cast(dtypes.uint32)) & 0x0f0f0f0f
        dot, qsum = _dp4a(word, xwords[word_idx], dot), _dp4a(UOp.const(0x01010101, dtypes.uint32), xwords[word_idx], qsum)
      d, dmin, scale, minimum = _q4_scales(raw, base, subgroup)
      return (dot.float() * d * scale - qsum.float() * dmin * minimum) * xd[token, group]
    if ggml_type == Q6_K:
      base = (output * in_features // GGML_BLOCK_SIZE + block) * Q6_BYTES
      dots = [UOp.const(0, dtypes.int32)] * 2
      for word_idx in range(8):
        pos, within = subgroup * 32 + word_idx * 4, (subgroup * 32 + word_idx * 4) % 128
        low = _load_lanes(raw[base + (pos // 128) * 64 + within % 64], 4) >> ((within // 64) * 4).cast(dtypes.uint8)
        high = _load_lanes(raw[base + 128 + (pos // 128) * 32 + within % 32], 4) >> ((within // 32) * 2).cast(dtypes.uint8)
        quant = ((low & 15) | ((high & 3) << 4)).bitcast(dtypes.int8) - 32
        word = sum((quant[i].cast(dtypes.uint8).cast(dtypes.uint32) << (i * 8) for i in range(4)), UOp.const(0, dtypes.uint32))
        dots[word_idx // 4] = _dp4a(word, xwords[word_idx], dots[word_idx // 4])
      scales = [raw[base + 192 + subgroup * 2 + i].cast(dtypes.uint8).bitcast(dtypes.int8).float() for i in range(2)]
      dbits = raw[base + 208].cast(dtypes.uint16) | (raw[base + 209].cast(dtypes.uint16) << 8)
      return (dots[0].float() * scales[0] + dots[1].float() * scales[1]) * xd[token, group] * _half(dbits)
    raise QuantGemvError(f"unsupported quant type {ggml_type}")
  names = {Q4_K: "linear_q4_k", Q6_K: "linear_q6"}
  return _decode_linear(out, out_features, group_count, group_dot, names[ggml_type])

def quant_gemv(x: Tensor, packed: Tensor, *, out_features: int, in_features: int, ggml_type: int) -> Tensor:
  """Q4_K/Q6_K decode GEMV: ``x`` (fp32, [..., in_features]) @ packed^T -> [..., out_features].

  ``packed`` is the packed GGML weight buffer viewed as uint32 (Q4_K) or uint8
  (Q6_K) words. The kernel quantizes ``x`` to Q8_0-style groups, runs the
  warp-per-output decode kernel, and sums the per-chunk partials.
  """
  if x.dtype not in dtypes.floats: raise QuantGemvError(f"x must be float, got {x.dtype}")
  if in_features % Q8_GROUP_SIZE != 0: raise QuantGemvError(f"in_features {in_features} not a multiple of 32")
  if out_features % 16 != 0: raise QuantGemvError(f"out_features {out_features} not a multiple of 16")
  tokens = int(x.numel()) // in_features
  raw = packed.uop.buf_uop
  xq, xd = q8_quantize(x, tokens, in_features)
  decode = functools.partial(_quant_decode_kernel, ggml_type=ggml_type)
  chunks = (in_features + 1023) // 1024
  out = Tensor.empty(tokens, out_features, chunks, dtype=dtypes.float32, device=x.device).uop
  params = tuple(UOp.placeholder_like(s, slot=i) for i, s in enumerate((out, raw, xq.uop, xd.uop)))
  kernel = decode(*params, out_features=out_features, in_features=in_features).call(out, raw, xq.uop, xd.uop)
  result = Tensor(out.after(kernel))
  result = result.sum(-1)
  return result.reshape(*x.shape[:-1], out_features)

def packed_weight(decoded: Tensor) -> tuple[Tensor, int] | None:
  """Extract the packed GGML weight view from a lazy fp16-cast quantized tensor.

  Mirrors ``tinygrad.llm.kernels.amd.Linear.set_quantized``: finds the uint8
  packed-block SHRINK inside the dequant expression and views the resident
  buffer as packed words. Returns ``None`` for non-quantized weights so callers
  fall back to the generic matmul path.
  """
  if decoded.dtype != dtypes.float16: return None
  packed_sizes = {decoded.numel() // GGML_BLOCK_SIZE * type_size: typ for typ, type_size in QUANT_SIZES.items()}
  raw = next((u for u in decoded.uop.toposort()
              if u.op is Ops.SHRINK and u.dtype == dtypes.uint8 and prod(u.shape) in packed_sizes), None)
  if raw is None: return None
  raw_offset = raw.contiguous_view_offset()
  if raw_offset is None or raw_offset % 4 != 0 or raw.buf_uop.dtype != dtypes.uint8: return None
  ggml_type = packed_sizes[prod(raw.shape)]
  packed_dtype = dtypes.uint8 if ggml_type == Q6_K else dtypes.uint32
  packed = Tensor(UOp.from_buffer(cast(Buffer, raw.buf_uop.buffer)
    .view(raw.max_numel() * raw.dtype.itemsize // packed_dtype.itemsize, packed_dtype, raw_offset)))
  return packed, ggml_type