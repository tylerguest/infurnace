"""Small NV BF16 draft GEMV: coalesced word-pair loads, FP32 accumulation."""
import functools

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo

from infurnace.kernels.quant_gemv import warp_reduce

def _bf16_gemv(out, raw, x, *, rows, width, tokens):
  block = UOp.range(tokens*(rows//4), 0, axis_type=AxisType.GLOBAL)
  warp = UOp.range(4, 1, axis_type=AxisType.LOCAL)
  lane = UOp.range(32, -1, axis_type=AxisType.WARP)
  token, row = block//(rows//4), (block%(rows//4))*4+warp
  step = UOp.range(width//64, 2, axis_type=AxisType.LOOP)
  acc = UOp.placeholder((1,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  acc = acc.after(acc[0].store(UOp.const(0.0, dtypes.float32)))
  # Reinterpret the native BF16 storage inside the kernel. Tensor.bitcast
  # across item sizes schedules a full packing kernel on this tinygrad pin.
  raw_words = raw.bitcast(dtypes.uint32)
  x_words = x.bitcast(dtypes.uint32)
  wword = raw_words.index(row*(width//2)+step*32+lane).load()
  xword = x_words.index(token*(width//2)+step*32+lane).load()
  def low(word): return ((word & 0xffff) << 16).bitcast(dtypes.float32)
  def high(word): return (word & 0xffff0000).bitcast(dtypes.float32)
  acc = acc.after(acc[0].store(acc[0].load()+low(wword)*low(xword)+high(wword)*high(xword)).end(step))
  value = warp_reduce(acc[0].load(), full_wave=True)
  return out[token, row.valid(lane.eq(0))].store(value.cast(out.dtype)).end(block, warp, lane).sink(
    arg=KernelInfo(name='draft_bf16_gemv', opts_to_apply=()))

def bf16_gemv(x: Tensor, weight: Tensor) -> Tensor:
  rows, width = weight.shape
  tokens = x.numel()//width
  if (x.dtype != dtypes.bfloat16 or weight.dtype != dtypes.bfloat16 or not 1 <= tokens <= 8
      or x.numel() != tokens*width
      or x.shape[-1] != width or rows%4 or width%64 or x.device != weight.device
      or x.device.split(':')[0] != 'NV'):
    raise ValueError('BF16 GEMV requires one to eight NV tokens, matching BF16 weights and aligned dimensions')
  out = Tensor.empty(tokens, rows, dtype=dtypes.bfloat16, device=x.device)
  result = UOp.custom_kernel(out.uop, weight.flatten().uop,
                            x.contiguous().flatten().uop,
                            fxn=functools.partial(_bf16_gemv, rows=rows, width=width, tokens=tokens))[0]
  return Tensor(result).reshape(*x.shape[:-1], rows)
