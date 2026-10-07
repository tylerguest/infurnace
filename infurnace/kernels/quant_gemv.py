from __future__ import annotations
import functools
import math
from typing import Callable
from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

Q4_K = 12
Q6_K = 14
GGML_BLOCK_SIZE = 256
Q8_GROUP_SIZE = 32
Q4_WORDS = 36
Q6_BYTES = 210
QUANT_SIZES = {Q4_K:Q4_WORDS*4, Q6_K:Q6_BYTES}

class QuantGemvError(ValueError):
  """Quant-GEMV inputs do not satisfy the kernel contract."""

def warp_reduce(val:UOp, maximum:bool = False, full_wave:bool = False) -> UOp:
  """Full-wave (32-lane) xor-shuffle reduction through CUDA ``__shfl_xor_sync``."""
  for offset in ((16, 8, 4, 2, 1) if full_wave else (8, 4, 2, 1)):
    if val.op is Ops.INDEX and val.addrspace == AddrSpace.REG: val = val.load()
    other = UOp(Ops.CUSTOM, src=(val,), arg=(f"__shfl_xor_sync(0xffffffff, {{0}}, {offset})", dtypes.float))
    val = val.maximum(other) if maximum else val + other
  return val

def _dp4a(a:UOp, b:UOp, c:UOp) -> UOp:
  return UOp(Ops.CUSTOMI, src=(a.int(), b.int(), c), arg=("__dp4a({}, {}, {})", dtypes.int32))

def _load_lanes(ptr: UOp, lanes: int) -> UOp:
  """Load ``lanes`` consecutive elements at a buffer index as a vector value."""
  assert ptr.op is Ops.INDEX
  buf, coords = ptr.src[0], ptr.src[1:]
  idx = sum((coord * math.prod(buf.shape[i + 1:]) for i, coord in enumerate(coords)), UOp.const(0))
  return UOp(Ops.SHRINK, src=(buf.flatten(), idx, UOp.const(lanes))).load()

def _load_byte(raw:UOp, base:UOp, offset:UOp) -> UOp:
  return (raw[base + offset // 4] >> ((offset & 3) * 8).cast(dtypes.uint32)) & 255

def _half(value:UOp) -> UOp:
  return value.cast(dtypes.uint16).bitcast(dtypes.float16).float()

def _q6_word(raw:UOp, base:UOp, byte_offset:UOp) -> UOp:
  # Q6 blocks are 210 bytes: every other block is only halfword-aligned.
  # Combine two aligned u16 loads, never reinterpret the address as an aligned u32.
  index = base + byte_offset // 2
  return raw[index].load().cast(dtypes.uint32) | (raw[index + 1].load().cast(dtypes.uint32) << 16)

def _q4_scales(raw:UOp, base:UOp, subgroup:UOp) -> tuple[UOp, UOp, UOp, UOp]:
  """Q4_K/Q5_K shared scale layout: d, dmin, scale[4], minimum[4] in 12 bytes."""
  scale = (subgroup < 4).where(_load_byte(raw, base, 4 + subgroup) & 63, (_load_byte(raw, base, 8 + subgroup) & 15) | ((_load_byte(raw, base, subgroup) >> 6) << 4))
  minimum = (subgroup < 4).where(_load_byte(raw, base, 8 + subgroup) & 63, (_load_byte(raw, base, 8 + subgroup) >> 4) | ((_load_byte(raw, base, 4 + subgroup) >> 6) << 4))
  d, dmin = (raw[base] & 0xffff).cast(dtypes.uint16), (raw[base] >> 16).cast(dtypes.uint16)
  return _half(d), _half(dmin), scale.float(), minimum.float()

def _q8_quantize_kernel(q:UOp, scale:UOp, xsum:UOp, x:UOp, tokens:int, in_features:int) -> UOp:
  """Per-32-group Q8_0-style activation quantize: 8 int8 per 4 lanes, one fp32 scale."""
  groups = in_features // Q8_GROUP_SIZE
  token_group, lane = UOp.range(tokens * groups, 0), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  token, group = token_group // groups, token_group % groups
  x = x.reshape(tokens, groups, 32)
  group_scale = (warp_reduce(x[token, group, lane].float().abs(), maximum=True, full_wave=True) / 127).maximum(1e-8)
  word_lane = lane.minimum(7)
  xs = tuple(x[token, group, word_lane * 4 + i].float() for i in range(4))
  word = sum(((v / group_scale).round().clip(-127, 127).cast(dtypes.int8).cast(dtypes.uint8).cast(dtypes.uint32) << (i * 8) for i, v in enumerate(xs)), UOp.const(0, dtypes.uint32))
  # Sum signed quantized bytes once per activation group, not per output row.
  word_sum = _dp4a(UOp.const(0x01010101, dtypes.uint32), word, UOp.const(0, dtypes.int32))
  # Each half has its own Q6 scale. Q4 uses the sum of both halves.
  # Integer sums in [-2032, 2032] are exact throughout these FP32 reductions.
  half_sums = tuple(warp_reduce(((lane >= half*4) & (lane < (half+1)*4)).where(word_sum.float(), UOp.const(0, dtypes.float32)), full_wave=True) for half in range(2))
  stores = (q[token, group, lane.valid(lane < 8)].store(word),
            scale[token, group.valid(lane.eq(0))].store(group_scale),
            *(xsum[token, group.valid(lane.eq(0)), half].store(value.cast(dtypes.int32)) for half, value in enumerate(half_sums)))
  return UOp.group(*stores).end(token_group, lane).sink(arg=KernelInfo(name="q8_quantize", opts_to_apply=()))

def q8_quantize(x:Tensor, tokens:int, in_features:int) -> tuple[Tensor, Tensor, Tensor]:
  groups = in_features // Q8_GROUP_SIZE
  q = Tensor.empty(tokens, groups, 8, dtype=dtypes.uint32, device=x.device)
  scale = Tensor.empty(tokens, groups, dtype=dtypes.float32, device=x.device)
  xsum = Tensor.empty(tokens, groups, 2, dtype=dtypes.int32, device=x.device)
  q, scale, xsum = Tensor.custom_kernel(q, scale, xsum, x, fxn=functools.partial(_q8_quantize_kernel, tokens=tokens, in_features=in_features))[:3]
  return q, scale, xsum

def _decode_warps(ggml_type: int, in_features: int, out_features: int) -> int:
  if ggml_type == Q4_K: return 4
  # RTX 5060 screen: larger blocks helped this Q6 output head, not FFN-down.
  if ggml_type == Q6_K and (out_features, in_features) == (151936, 2560): return 8
  return 1

def _decode_linear(out:UOp, out_features:int, group_count:int, group_dot:Callable[[UOp, UOp, UOp], UOp], name:str, warps_per_block:int=1) -> UOp:
  assert warps_per_block in (1, 2, 4, 8)
  assert out_features % warps_per_block == 0
  chunks = (group_count + 31) // 32
  row_blocks = out_features // warps_per_block
  block = UOp.range(out.shape[0]*row_blocks*chunks, 0)
  lane = UOp.range(32, -1, axis_type=AxisType.WARP)
  warp = UOp.range(warps_per_block, 1, axis_type=AxisType.LOCAL)
  chunk = block%chunks
  token = block // (row_blocks*chunks)
  output = ((block//chunks)%row_blocks)*warps_per_block+warp
  group = lane+chunk*32
  if group_count%32 == 0:
    value = group_dot(token, output, group)
  else:
    value = (group < group_count).where(group_dot(token, output, group.minimum(group_count-1),), UOp.const(0, dtypes.float32))
  total = warp_reduce(value, full_wave=True)
  return out[token, output, chunk.valid(lane.eq(0))].store(total.cast(out.dtype)).end(block, warp, lane).sink(arg=KernelInfo(name=name, opts_to_apply=()))

def _quant_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, out_features:int, in_features:int, ggml_type:int) -> UOp:
  group_count = in_features // Q8_GROUP_SIZE
  def group_dot(token: UOp, output: UOp, group: UOp) -> UOp:
    block, subgroup = group // 8, group % 8
    xwords = _load_lanes(xq[token, group, 0], 8)
    if ggml_type == Q4_K:
      base = (output * in_features // GGML_BLOCK_SIZE + block) * Q4_WORDS
      qs_base, dot = base + 4 + (subgroup // 2) * 8, UOp.const(0, dtypes.int32)
      for word_idx in range(8):
        word = (raw[qs_base + word_idx] >> ((subgroup & 1) * 4).cast(dtypes.uint32)) & 0x0f0f0f0f
        dot = _dp4a(word, xwords[word_idx], dot)
      d, dmin, scale, minimum = _q4_scales(raw, base, subgroup)
      qsum = (xs[token, group, 0].load() + xs[token, group, 1].load()).float()
      return (dot.float() * d * scale - qsum * dmin * minimum) * xd[token, group]
    if ggml_type == Q6_K:
      base = (output*in_features//GGML_BLOCK_SIZE+block) * (Q6_BYTES//2)
      dots = [UOp.const(0, dtypes.int32)] * 2
      for word_idx in range(8):
        low_offset = (subgroup//4)*64 + (subgroup%2)*32 + word_idx*4
        high_offset = 128 + (subgroup//4)*32 + word_idx*4
        low = _q6_word(raw, base, low_offset) >> ((subgroup%4//2)*4).cast(dtypes.uint32)
        high = _q6_word(raw, base, high_offset) >> ((subgroup%4)*2).cast(dtypes.uint32)
        # Four independent values in [0, 63]; no byte unpack/repack or signed offset here.
        word = (low & 0x0f0f0f0f) | ((high & 0x03030303) << 4)
        dots[word_idx // 4] = _dp4a(word, xwords[word_idx], dots[word_idx // 4])
      scale_word = raw[base + 96 + subgroup].load().cast(dtypes.uint32)
      scales = [(scale_word >> (half*8)).cast(dtypes.uint8).bitcast(dtypes.int8).float() for half in range(2)]
      # dot((q-32), x) == dot(q, x) - 32*sum(x), independently for each 16-value scale group.
      corrected = [(dots[half] - 32*xs[token, group, half].load()).float() for half in range(2)]
      return (corrected[0]*scales[0] + corrected[1]*scales[1]) * xd[token, group] * _half(raw[base + 104].load())
    raise QuantGemvError(f"unsupported quant type {ggml_type}")
  names = {Q4_K: "linear_q4_k", Q6_K: "linear_q6"}
  warps = _decode_warps(ggml_type, in_features, out_features)
  return _decode_linear(out, out_features, group_count, group_dot, names[ggml_type], warps_per_block=warps)

def quant_gemv(x:Tensor, packed:Tensor, *, out_features:int, in_features:int, ggml_type:int) -> Tensor:
  if x.dtype not in dtypes.floats: raise QuantGemvError(f"x must be float, got {x.dtype}")
  if ggml_type not in QUANT_SIZES: raise QuantGemvError(f"unsupported quant type {ggml_type}")
  if in_features <= 0 or in_features % GGML_BLOCK_SIZE: raise QuantGemvError("in_features must be a positive multiple of 256")
  if out_features <= 0 or out_features % 16: raise QuantGemvError("out_features must be a positive multiple of 16")
  if not x.shape or x.shape[-1] != in_features: raise QuantGemvError("activation shape does not match in_features")
  if x.device != packed.device: raise QuantGemvError("activations and packed weights must share a device")

  expected_dtype = dtypes.uint32 if ggml_type == Q4_K else dtypes.uint16
  expected_bytes = out_features * in_features // GGML_BLOCK_SIZE * QUANT_SIZES[ggml_type]
  if packed.dtype != expected_dtype or packed.numel() * packed.dtype.itemsize != expected_bytes:
    raise QuantGemvError("packed weight dtype/size does not match the quantization contract")

  tokens = int(x.numel()) // in_features
  raw = packed.uop
  xq, xd, xs = q8_quantize(x, tokens, in_features)
  decode = functools.partial(_quant_decode_kernel, ggml_type=ggml_type)
  chunks = (in_features + 1023) // 1024
  out = Tensor.empty(tokens, out_features, chunks, dtype=dtypes.float32, device=x.device).uop
  sources = (out, raw, xq.uop, xd.uop, xs.uop)
  params = tuple(UOp.placeholder_like(s, slot=i) for i, s in enumerate(sources))
  kernel = decode(*params, out_features=out_features, in_features=in_features).call(*sources)
  result = Tensor(out.after(kernel))
  result = result.sum(-1)

  return result.reshape(*x.shape[:-1], out_features)

def packed_weight(decoded: Tensor) -> tuple[Tensor, int] | None:
  """Extract the packed GGML weight view from a lazy fp16-cast quantized tensor."""