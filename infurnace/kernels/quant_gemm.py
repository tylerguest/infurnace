"""First NV packed prefill GEMM: m16n8k16 FP16 tensor cores, FP32 accumulation.

Weights are decoded into warp fragments, never a persistent dense matrix.
Activations and decoded weights round to FP16; this differs from FP32-activation
fallback math and Q8-activation decode. No shared-memory staging/autotuning yet.
"""
from __future__ import annotations
import functools
import math

from tinygrad import Context, Device, Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.uop.ops import AxisType, KernelInfo

from infurnace.kernels.quant_gemv import Q4_K, Q5_K, Q6_K, Q6_PREPARED_BYTES, QUANT_SIZES, _half, _q4_scales, _q6_swizzled_index, _q4_word


class QuantGemmError(ValueError):
  """Inputs/backend do not satisfy the first prefill kernel contract."""


def _q6_byte(raw: UOp, base: UOp, offset: UOp, blocks_per_row: int = 0) -> UOp:
  size = raw.dtype.itemsize
  index = _q6_swizzled_index(base, offset//size, blocks_per_row) if blocks_per_row else base+offset//size
  return (raw[index].load().cast(dtypes.uint32) >> ((offset%size)*8).cast(dtypes.uint32)) & 255


def _weight(raw: UOp, row: UOp, col: UOp, width: int, typ: int, q6_aligned: bool = False,
            q6_swizzled: bool = False, q4_swizzled: bool = False) -> UOp:
  block, within = col//256, col%256
  if typ == Q4_K:
    logical = row*(width//256)+block
    base = logical if q4_swizzled else logical*36
    swizzle_width = width//256 if q4_swizzled else 0
    subgroup = within//32
    word = _q4_word(raw, base, 4+(subgroup//2)*8+(within%32)//4, swizzle_width)
    quant = (word >> ((subgroup%2)*4+(within%4)*8).cast(dtypes.uint32)) & 15
    d, dmin, scale, minimum = _q4_scales(raw, base, subgroup, swizzle_width)
    return (d*scale*quant.float()-dmin*minimum).cast(dtypes.float16)
  if typ == Q5_K:
    # 44 words: d/dmin, 12 scale bytes, 32 high-bit bytes (bit `subgroup` of
    # byte j), then Q4_K-ordered low nibbles.
    base, subgroup, j = (row*(width//256)+block)*44, within//32, within%32
    low = (raw[base+12+(subgroup//2)*8+j//4].load() >> ((subgroup%2)*4+(j%4)*8).cast(dtypes.uint32)) & 15
    high = (raw[base+4+j//4].load() >> ((j%4)*8+subgroup).cast(dtypes.uint32)) & 1
    d, dmin, scale, minimum = _q4_scales(raw, base, subgroup)
    return (d*scale*(low | (high << 4)).float()-dmin*minimum).cast(dtypes.float16)
  logical = row*(width//256)+block
  base = logical if q6_swizzled else logical*(Q6_PREPARED_BYTES//4 if q6_aligned else 105)
  swizzle_width = width//256 if q6_swizzled else 0
  half, pos = within//128, within%128
  low = _q6_byte(raw, base, half*64+pos%64, swizzle_width) >> ((pos//64)*4).cast(dtypes.uint32)
  high = _q6_byte(raw, base, 128+half*32+pos%32, swizzle_width) >> ((pos//32)*2).cast(dtypes.uint32)
  quant = ((low & 15) | ((high & 3) << 4)).cast(dtypes.int32)-32
  scale = _q6_byte(raw, base, 192+within//16, swizzle_width).cast(dtypes.uint8).bitcast(dtypes.int8).float()
  d_index = _q6_swizzled_index(logical, 52, width//256) if q6_swizzled else base+(52 if q6_aligned else 104)
  return (_half(raw[d_index].load())*quant.float()*scale).cast(dtypes.float16)


@functools.cache
def _prefill_kernel(out: UOp, raw: UOp, x: UOp, *, tokens: int, width: int, rows: int, typ: int,
                    q6_aligned: bool = False, q6_swizzled: bool = False, q4_swizzled: bool = False) -> UOp:
  warps = math.gcd(rows//8, 4)
  output_blocks = rows//(warps*8)
  block = UOp.range((tokens//16)*output_blocks, 0, axis_type=AxisType.GLOBAL)
  lane = UOp.range(32, -1, axis_type=AxisType.WARP)
  warp = UOp.range(warps, 1, axis_type=AxisType.LOCAL)
  token_base = (block//output_blocks)*16
  output_base = (block%output_blocks)*warps*8+warp*8
  group, thread = lane//4, lane%4
  acc = UOp.placeholder((4,), dtypes.float32, slot=3, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.const_like(0)))
  k = UOp.range(width//16, 2, axis_type=getattr(AxisType, 'REDUCE', AxisType.LOOP))
  # NVIDIA mma.sync m16n8k16.row.col fragment order (not AMD WMMA order).
  a = UOp.stack(*(x[token_base+group+row, k*16+thread*2+col].load()
                  for col_high in (0, 8) for row in (0, 8) for col in (col_high, col_high+1)))
  b = UOp.stack(*(_weight(raw, output_base+group, k*16+thread*2+col, width, typ, q6_aligned, q6_swizzled, q4_swizzled) for col in (0, 1, 8, 9)))
  update = acc.store(UOp.wmma(a, b, acc.after(k), (8, 16, 16), 32)).end(k)
  result = acc.after(update)
  stores = [out[token_base+group+row, output_base+thread*2+col].store(result[i].load())
            for i, (row, col) in enumerate(((0, 0), (0, 1), (8, 0), (8, 1)))]
  return UOp.group(*stores).end(block, warp, lane).sink(
    arg=KernelInfo(name=f"prefill_q{ {Q4_K: 4, Q5_K: 5}.get(typ, 6)}_k_mma", opts_to_apply=()))


@functools.cache
def nv_prefill_supported(device: str) -> bool:
  if not isinstance(device, str) or device.split(":")[0] != "NV": return False
  with Context(ALLOW_DEVICE_USAGE=1): renderer = Device[device].renderer
  return isinstance(renderer, CUDARenderer) and any(
    tc.dims == (8, 16, 16) and tc.dtype_in == dtypes.float16 and tc.dtype_out == dtypes.float32 for tc in renderer.tensor_cores)


def quant_gemm(x: Tensor, packed: Tensor, *, out_features: int, in_features: int, ggml_type: int,
               q6_aligned: bool = False, q6_swizzled: bool = False, q4_swizzled: bool = False) -> Tensor:
  """Concrete multi-token packed linear; symbolic padding belongs to the adapter."""
  if ggml_type not in (Q4_K, Q5_K, Q6_K): raise QuantGemmError("only Q4_K/Q5_K/Q6_K prefill is supported")
  if in_features <= 0 or in_features%256 or out_features <= 0 or out_features%8:
    raise QuantGemmError("prefill requires input width multiple of 256 and output rows multiple of 8")
  if x.dtype not in dtypes.floats or not x.shape or x.shape[-1] != in_features:
    raise QuantGemmError("activation dtype/shape does not match prefill contract")
  if not isinstance(x.numel(), int) or x.numel()//in_features < 2:
    raise QuantGemmError("prefill requires a concrete multi-token shape")
  if q4_swizzled and ggml_type != Q4_K: raise QuantGemmError("swizzled Q4 requires Q4_K")
  if q6_swizzled and not q6_aligned: raise QuantGemmError("swizzled Q6 requires prepared storage")
  if q6_aligned and ggml_type != Q6_K: raise QuantGemmError("aligned Q6 layout requires Q6_K")
  dtype = dtypes.uint32 if ggml_type in (Q4_K, Q5_K) or q6_aligned else dtypes.uint16
  expected = out_features*in_features//256*(Q6_PREPARED_BYTES if q6_aligned else QUANT_SIZES[ggml_type])
  if packed.dtype != dtype or packed.numel()*dtype.itemsize != expected:
    raise QuantGemmError("packed dtype/size does not match prefill contract")
  if x.device != packed.device or not isinstance(x.device, str) or x.device.split(":")[0] != "NV":
    raise QuantGemmError("prefill requires activations/weights on the same NV device")
  if not nv_prefill_supported(x.device):
    raise QuantGemmError("prefill requires CUDA-C FP16 m16n8k16 tensor cores")
  tokens = x.numel()//in_features
  padded_tokens = (tokens+15)//16*16
  activation = x.reshape(tokens, in_features).cast(dtypes.float16).pad(((0, padded_tokens-tokens), (0, 0))).contiguous()
  out = Tensor.empty(padded_tokens, out_features, dtype=dtypes.float32, device=x.device)
  result = Tensor.custom_kernel(out, packed.flatten().contiguous(), activation,
    fxn=functools.partial(_prefill_kernel, tokens=padded_tokens, width=in_features, rows=out_features, typ=ggml_type,
                         q6_aligned=q6_aligned, q6_swizzled=q6_swizzled, q4_swizzled=q4_swizzled))[0]
  return result[:tokens].reshape(*x.shape[:-1], out_features)
