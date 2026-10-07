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

from infurnace.kernels.quant_gemv import Q4_K, QUANT_SIZES, _half, _q4_scales


class QuantGemmError(ValueError):
  """Inputs/backend do not satisfy the first prefill kernel contract."""


def _q6_byte(raw: UOp, base: UOp, offset: UOp) -> UOp:
  return (raw[base+offset//2].load().cast(dtypes.uint32) >> ((offset%2)*8).cast(dtypes.uint32)) & 255


def _weight(raw: UOp, row: UOp, col: UOp, width: int, typ: int) -> UOp:
  block, within = col//256, col%256
  if typ == Q4_K:
    base = (row*(width//256)+block)*36
    subgroup = within//32
    word = raw[base+4+(subgroup//2)*8+(within%32)//4].load()
    quant = (word >> ((subgroup%2)*4+(within%4)*8).cast(dtypes.uint32)) & 15
    d, dmin, scale, minimum = _q4_scales(raw, base, subgroup)
    return (d*scale*quant.float()-dmin*minimum).cast(dtypes.float16)
  base = (row*(width//256)+block)*105
  half, pos = within//128, within%128
  low = _q6_byte(raw, base, half*64+pos%64) >> ((pos//64)*4).cast(dtypes.uint32)
  high = _q6_byte(raw, base, 128+half*32+pos%32) >> ((pos//32)*2).cast(dtypes.uint32)
  quant = ((low & 15) | ((high & 3) << 4)).cast(dtypes.int32)-32
  scale = _q6_byte(raw, base, 192+within//16).cast(dtypes.uint8).bitcast(dtypes.int8).float()
  return (_half(raw[base+104].load())*quant.float()*scale).cast(dtypes.float16)


@functools.cache
def _prefill_kernel(out: UOp, raw: UOp, x: UOp, *, tokens: int, width: int, rows: int, typ: int) -> UOp:
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
  k = UOp.range(width//16, 2, axis_type=AxisType.REDUCE)
  # NVIDIA mma.sync m16n8k16.row.col fragment order (not AMD WMMA order).
  a = UOp.stack(*(x[token_base+group+row, k*16+thread*2+col].load()
                  for col_high in (0, 8) for row in (0, 8) for col in (col_high, col_high+1)))
  b = UOp.stack(*(_weight(raw, output_base+group, k*16+thread*2+col, width, typ) for col in (0, 1, 8, 9)))
  update = acc.store(UOp.wmma(a, b, acc.after(k), (8, 16, 16), 32)).end(k)
  result = acc.after(update)
  stores = [out[token_base+group+row, output_base+thread*2+col].store(result[i].load())
            for i, (row, col) in enumerate(((0, 0), (0, 1), (8, 0), (8, 1)))]
  return UOp.group(*stores).end(block, warp, lane).sink(
    arg=KernelInfo(name=f"prefill_q{4 if typ == Q4_K else 6}_k_mma", opts_to_apply=()))


@functools.cache
def nv_prefill_supported(device: str) -> bool:
  if not isinstance(device, str) or device.split(":")[0] != "NV": return False
  with Context(ALLOW_DEVICE_USAGE=1): renderer = Device[device].renderer
  return isinstance(renderer, CUDARenderer) and any(
    tc.dims == (8, 16, 16) and tc.dtype_in == dtypes.float16 and tc.dtype_out == dtypes.float32 for tc in renderer.tensor_cores)


def quant_gemm(x: Tensor, packed: Tensor, *, out_features: int, in_features: int, ggml_type: int) -> Tensor:
  """Concrete multi-token packed linear; symbolic padding belongs to the adapter."""
  if ggml_type not in QUANT_SIZES: raise QuantGemmError("only Q4_K/Q6_K prefill is supported")
  if in_features <= 0 or in_features%256 or out_features <= 0 or out_features%8:
    raise QuantGemmError("prefill requires input width multiple of 256 and output rows multiple of 8")
  if x.dtype not in dtypes.floats or not x.shape or x.shape[-1] != in_features:
    raise QuantGemmError("activation dtype/shape does not match prefill contract")
  if not isinstance(x.numel(), int) or x.numel()//in_features < 2:
    raise QuantGemmError("prefill requires a concrete multi-token shape")
  dtype = dtypes.uint32 if ggml_type == Q4_K else dtypes.uint16
  expected = out_features*in_features//256*QUANT_SIZES[ggml_type]
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
    fxn=functools.partial(_prefill_kernel, tokens=padded_tokens, width=in_features, rows=out_features, typ=ggml_type))[0]
  return result[:tokens].reshape(*x.shape[:-1], out_features)
