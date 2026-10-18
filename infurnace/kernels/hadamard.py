"""Blockwise Walsh-Hadamard transform of activations (NV), for Hadamard-folded weights (infurnace/hadamard.py).

One CTA per block of `block` values, one thread per value: the reorder and signs are folded into the load, then
log2(block) butterfly stages run in shared memory (a buffer per stage, 2 KB each for 512 values: tinygrad won't
store twice to one offset of a buffer; one barrier a stage), and the normalization into the store. A dense
block x block matmul does the same with a 1 MB matrix read per call (~17 us a call for 2,560 values on an
RTX 5060, against a few here).
"""
import functools
import math

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo

@functools.cache
def _fwht_kernel(out: UOp, x: UOp, signs: UOp, *, rows: int, width: int, block: int, perm: tuple[int, int, int] | None,
                 signs_after: bool) -> UOp:
  item = UOp.range(rows * (width // block), 0, axis_type=AxisType.GLOBAL)
  lane = UOp.range(block, 1, axis_type=AxisType.LOCAL)
  row, position = item // (width // block), (item % (width // block)) * block + lane
  source = position
  if perm is not None:  # grouped position (k, r, d) reads tiled (r, k, d)
    rep, nk, hd = perm
    source = ((position // hd) % rep) * (nk * hd) + (position // (rep * hd)) * hd + position % hd
  value = x[row, source].load().float()
  if not signs_after: value = value * signs[position].load()
  stages = int(math.log2(block))
  buffers = [UOp.placeholder((block,), dtypes.float32, slot=i, addrspace=AddrSpace.LOCAL) for i in range(stages + 1)]
  current = buffers[0].after(buffers[0][lane].store(value).barrier())
  for stage in range(stages):
    h = 1 << stage
    low = ((lane // h) % 2).eq(0)  # this lane holds the first of its pair (a, b) -> (a + b, a - b)
    mine, other = current[lane].load(), current[low.where(lane + h, lane - h)].load()
    target = buffers[stage + 1]
    current = target.after(target[lane].store(low.where(mine + other, other - mine)).barrier())
  result = current[lane].load() * (block ** -0.5)
  if signs_after: result = result * signs[position].load()
  return out[row, position].store(result).end(item, lane).sink(arg=KernelInfo(name=f"fwht_{block}", opts_to_apply=()))

def fwht(x: Tensor, signs: Tensor, *, block: int, perm: tuple[int, int, int] | None = None, signs_after: bool = False) -> Tensor:
  """x (..., width) -> FP32 (..., width): optional reorder, * signs, then the normalized transform per block (or,
  with `signs_after`, the transform then * signs: the inverse)."""
  width = x.shape[-1]
  if block > 1024 or block & (block - 1) or width % block: raise ValueError("fwht needs a power-of-two block up to 1024 dividing the width")
  rows = x.numel() // width
  out = Tensor.empty(rows, width, dtype=dtypes.float32, device=x.device)
  out = Tensor.custom_kernel(out, x.reshape(rows, width).contiguous(), signs.float().contiguous(),
                             fxn=functools.partial(_fwht_kernel, rows=rows, width=width, block=block, perm=perm, signs_after=signs_after))[0]
  return out.reshape(x.shape)
