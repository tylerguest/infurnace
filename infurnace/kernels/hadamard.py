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

from infurnace.kernels.quant_gemv import q8_64_slot, warp_reduce

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

@functools.cache
def _fwht_q8_kernel(q: UOp, scale: UOp, *sources: UOp, rows: int, width: int, block: int, perm: tuple[int, int, int] | None,
                    group: int) -> UOp:
  """The forward transform, quantized to int8 in the same pass: q8_quantize's layout (group 32, with the halves'
  sums) or q8_quantize64's (group 64). A 32-value group is one warp; a 64-value group is two, whose maxima meet in
  shared memory. Every fourth lane packs its four values from the last stage's buffer into a word."""
  xsum, x, signs = sources if group == 32 else (None, *sources)
  item = UOp.range(rows * (width // block), 0, axis_type=AxisType.GLOBAL)
  # Warps and their lanes as their own ranges: a single block-wide range may be split across thread dimensions
  # (by the lane % 4 below), and warp shuffles need 32 consecutive values in a warp
  warp, thread = UOp.range(block // 32, 1, axis_type=AxisType.LOCAL), UOp.range(32, -1, axis_type=AxisType.WARP)
  lane = warp * 32 + thread
  row, position = item // (width // block), (item % (width // block)) * block + lane
  source = position
  if perm is not None:
    rep, nk, hd = perm
    source = ((position // hd) % rep) * (nk * hd) + (position // (rep * hd)) * hd + position % hd
  value = x[row, source].load().float() * signs[position].load()
  stages = int(math.log2(block))
  buffers = [UOp.placeholder((block,), dtypes.float32, slot=i, addrspace=AddrSpace.LOCAL) for i in range(stages + 1)]
  current = buffers[0].after(buffers[0][lane].store(value).barrier())
  for stage in range(stages):
    h = 1 << stage
    low = ((lane // h) % 2).eq(0)
    mine, other = current[lane].load(), current[low.where(lane + h, lane - h)].load()
    target = buffers[stage + 1]
    current = target.after(target[lane].store(low.where(mine + other, other - mine)).barrier())
  norm = block ** -0.5
  biggest = warp_reduce(current[lane].load().abs(), maximum=True, full_wave=True) * norm
  if group == 64:  # the other warp of the pair
    maxima = UOp.placeholder((block // 32,), dtypes.float32, slot=stages + 1, addrspace=AddrSpace.LOCAL)
    shared = maxima.after(maxima[(lane // 32).valid((lane % 32).eq(0))].store(biggest).barrier())
    pair = (lane // 64) * 2
    biggest = shared[pair].load().maximum(shared[pair + 1].load())
  step = (biggest / 127).maximum(1e-8)
  def code(v: UOp) -> UOp: return (v * norm / step).round().clip(-127, 127)
  first = lane - lane % 4  # this word's first value
  word = sum((code(current[first + i].load()).cast(dtypes.int8).cast(dtypes.uint8).cast(dtypes.uint32) << (i * 8)
              for i in range(4)), UOp.const(0, dtypes.uint32))
  slot = position // 4 if group == 32 else position // 64 * 16 + q8_64_slot(position % 64 // 4)  # q8_quantize64's word order
  stores = [q[row, slot.valid((lane % 4).eq(0))].store(word),
            scale[row, (position // group).valid((lane % group).eq(0))].store(step)]
  if xsum is not None:  # each 16-value half's sum of codes, for the K-quant GEMVs' offsets (exact integers in FP32)
    mine = code(current[lane].load())
    for half in range(2):
      total = warp_reduce(((lane % 32) // 16).eq(half).where(mine, UOp.const(0.0, dtypes.float32)), full_wave=True)
      stores.append(xsum[row, (position // group * 2 + half).valid((lane % 32).eq(0))].store(total.cast(dtypes.int32)))
  return UOp.group(*stores).end(item, warp, thread).sink(arg=KernelInfo(name=f"fwht_q8_{block}_{group}", opts_to_apply=()))

def fwht_q8(x: Tensor, signs: Tensor, *, block: int, perm: tuple[int, int, int] | None = None, group: int = 32) -> tuple[Tensor, ...]:
  """fwht(x, signs, ...) quantized to int8 in one kernel, without the FP32 result: (words, scales, half sums) as
  quant_gemv.q8_quantize returns for group 32, (words, scales) as q8_quantize64 for group 64."""
  width = x.shape[-1]
  if block > 1024 or block & (block - 1) or width % block or block < 64:
    raise ValueError("fwht_q8 needs a power-of-two block of 64-1024 dividing the width")
  if group not in (32, 64): raise ValueError("fwht_q8 quantizes 32- or 64-value groups")
  rows, groups = x.numel() // width, width // group
  outputs = [Tensor.empty(rows, width // 4, dtype=dtypes.uint32, device=x.device), Tensor.empty(rows, groups, dtype=dtypes.float32, device=x.device)]
  if group == 32: outputs.append(Tensor.empty(rows, groups * 2, dtype=dtypes.int32, device=x.device))
  kernel = functools.partial(_fwht_q8_kernel, rows=rows, width=width, block=block, perm=perm, group=group)
  results = Tensor.custom_kernel(*outputs, x.reshape(rows, width).contiguous(), signs.float().contiguous(), fxn=kernel)[:len(outputs)]
  shapes = ((rows, groups, group // 4), (rows, groups), (rows, groups, 2))
  return tuple(r.reshape(s) for r, s in zip(results, shapes))

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
