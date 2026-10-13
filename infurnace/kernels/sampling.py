"""Fused sampling of many logit rows (NV): Gumbel-max with the noise generated in the kernel.

  score = logit + T * g,   g = -ln(-ln u),   u from threefry(key, step << 32 | element)

At T = 0 this is exactly argmax, the first maximum on ties. For T > 0 the argmax of the score is a
sample from softmax(logit / T): it is the Gumbel-max score logit / T + g scaled by T, which doesn't
change the argmax. Upstream computes the same sample with tensor ops (rand_like, two logs, a divide
and an argmax over the whole row); here the logits are read once and the noise never leaves
registers. A second launch combines each row's per-block winners and advances `step`, so every
call draws fresh noise from tinygrad's own counter-based generator.
"""
from __future__ import annotations
import functools
import math

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo
from infurnace.kernels.quant_gemv import warp_reduce

THREADS, PER_THREAD = 256, 8
CHUNK = THREADS * PER_THREAD  # logits per block
_LN2 = math.log(2)

def _best_of(score: UOp, index: UOp) -> tuple[UOp, UOp]:
  """Warp-wide maximum score, and the smallest index holding it (argmax's tie order)."""
  top = warp_reduce(score, maximum=True, full_wave=True)
  candidate = (score >= top).where(index, UOp.const(float(2 ** 24), dtypes.float32))
  return top, -warp_reduce(-candidate, maximum=True, full_wave=True)

@functools.cache
def _partial_kernel(best: UOp, logits: UOp, temperature: UOp, key: UOp, step: UOp, *, rows: int, vocab: int) -> UOp:
  chunks = -(-vocab // CHUNK)
  block = UOp.range(rows * chunks, 0, axis_type=AxisType.GLOBAL)
  warp = UOp.range(THREADS // 32, 1, axis_type=AxisType.LOCAL)
  lane = UOp.range(32, -1, axis_type=AxisType.WARP)
  row, chunk = block // chunks, block % chunks
  thread = warp * 32 + lane
  t = temperature[0].load().float()
  seed = (key[1].load().cast(dtypes.uint64) << 32) | key[0].load().cast(dtypes.uint64)
  counter = step[0].load().cast(dtypes.uint64) << 32
  neg = UOp.const(float('-inf'), dtypes.float32)
  score, index = None, None
  for e in range(PER_THREAD):  # columns increase with e, so a strict > keeps the first maximum
    column = chunk * CHUNK + e * THREADS + thread
    valid = column < vocab
    bits = (counter | (row * vocab + column).cast(dtypes.uint64)).threefry(seed).cast(dtypes.uint32)
    u = ((bits >> 9).cast(dtypes.float32) + 0.5) * (2.0 ** -23)               # (0, 1), 23 random bits
    gumbel = -((-(u.log2() * _LN2)).log2() * _LN2)
    # Clamped loads, masked values: a lane-gated load could let tinygrad drop threads the shuffles need.
    value = valid.where(logits[row, column.minimum(vocab - 1)].load().float() + t * gumbel, neg)
    position = column.cast(dtypes.float32)
    if score is None: score, index = value, position
    else:
      better = value > score
      score, index = better.where(value, score), better.where(position, index)
  top, at = _best_of(score, index)
  shared = UOp.placeholder((THREADS // 32, 2), dtypes.float32, slot=0, addrspace=AddrSpace.LOCAL)
  shared = shared.after(UOp.group(shared[warp.valid(lane.eq(0)), 0].store(top), shared[warp.valid(lane.eq(0)), 1].store(at)).barrier())
  # Warp 0 combines the block's warps.
  mine = lane < THREADS // 32
  slot = lane.minimum(THREADS // 32 - 1)
  top, at = _best_of(mine.where(shared[slot, 0].load(), neg), mine.where(shared[slot, 1].load(), float(2 ** 24)))
  first = warp.eq(0) & lane.eq(0)
  stores = [best[row, chunk.valid(first), 0].store(top), best[row, chunk.valid(first), 1].store(at)]
  return UOp.group(*stores).end(block, warp, lane).sink(arg=KernelInfo(name="sample_partial", opts_to_apply=()))

@functools.cache
def _merge_kernel(out: UOp, step: UOp, best: UOp, *, rows: int, vocab: int) -> UOp:
  chunks = -(-vocab // CHUNK)
  row = UOp.range(rows, 0, axis_type=AxisType.GLOBAL)
  lane = UOp.range(32, -1, axis_type=AxisType.WARP)
  neg = UOp.const(float('-inf'), dtypes.float32)
  score, index = None, None
  for i in range(-(-chunks // 32)):  # chunks increase with i: a strict > keeps the first maximum
    chunk = lane + 32 * i
    valid = chunk < chunks
    # Load at a clamped index and mask the value: gating the load on the lane would let tinygrad shrink
    # the warp to `chunks` threads, and the shuffles below need all 32.
    held = chunk.minimum(chunks - 1)
    value = valid.where(best[row, held, 0].load(), neg)
    position = valid.where(best[row, held, 1].load(), float(2 ** 24))
    if score is None: score, index = value, position
    else:
      better = value > score
      score, index = better.where(value, score), better.where(position, index)
  _, at = _best_of(score, index)
  first = lane.eq(0)
  stores = [out[row.valid(first)].store(at.cast(dtypes.int32)),
            step[(row * 0).valid(first & row.eq(0))].store(step[0].load() + 1)]  # fresh noise for the next call
  return UOp.group(*stores).end(row, lane).sink(arg=KernelInfo(name="sample_merge", opts_to_apply=()))

class Sampler:
  """Owns the noise state: a 64-bit key and a step counter on the device, advanced by every call."""
  def __init__(self, device: str, seed: int):
    self.device = device
    self.key = Tensor([seed & 0xffffffff, (seed >> 32) & 0xffffffff], dtype=dtypes.uint32, device=device).contiguous().realize()
    self.step = Tensor([0], dtype=dtypes.uint32, device=device).contiguous().realize()

  def __call__(self, logits: Tensor, temperature: Tensor) -> Tensor:
    """(rows, vocab) logits -> (rows,) int32 tokens: argmax at T = 0, else a sample from softmax(logits / T)."""
    rows, vocab = logits.shape
    if self.device.split(':')[0] != 'NV':  # tensor ops elsewhere (CPU tests): same score, tinygrad's rand
      u = Tensor.rand(rows, vocab, device=logits.device).maximum(2.0 ** -24)
      return (logits.float() + temperature.float() * -(-u.log()).log()).argmax(-1).cast(dtypes.int32)
    best = Tensor.empty(rows, -(-vocab // CHUNK), 2, dtype=dtypes.float32, device=logits.device)
    best = Tensor.custom_kernel(best, logits.float().contiguous(), temperature.float().contiguous(), self.key, self.step,
                                fxn=functools.partial(_partial_kernel, rows=rows, vocab=vocab))[0]
    out = Tensor.empty(rows, dtype=dtypes.int32, device=logits.device)
    return Tensor.custom_kernel(out, self.step, best, fxn=functools.partial(_merge_kernel, rows=rows, vocab=vocab))[0]
