"""Top-k of each row over a large width (NV): DFlash2's candidates, the 16 best of each draft position's 248,320 logits.

Two passes of one kernel. Each block takes the k best of a 2,048-value chunk in k rounds: every thread holds 8
values, the block finds the round's best (the sampling kernel's warp reduction: largest value, smallest index on
ties, as argmax), and the thread holding it drops it. The second pass does the same over the chunks' winners.
k argmax passes over the whole row read it k times; this reads it once.
"""
import functools

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo

from infurnace.kernels.sampling import _best_of

THREADS, PER_THREAD = 256, 8
CHUNK = THREADS * PER_THREAD
_NO_INDEX = float(2 ** 24)  # above any index (exact in FP32), so it never wins a tie

@functools.cache
def _topk_kernel(out_values: UOp, out_index: UOp, values: UOp, index: UOp | None = None, *, rows: int, width: int, k: int) -> UOp:
  chunks = -(-width // CHUNK)
  block = UOp.range(rows * chunks, 0, axis_type=AxisType.GLOBAL)
  warp = UOp.range(THREADS // 32, 1, axis_type=AxisType.LOCAL)
  lane = UOp.range(32, -1, axis_type=AxisType.WARP)
  row, chunk = block // chunks, block % chunks
  thread = warp * 32 + lane
  neg = UOp.const(float('-inf'), dtypes.float32)
  held, positions = [], []
  for e in range(PER_THREAD):  # columns increase with e: a strict > keeps the first maximum
    column = chunk * CHUNK + e * THREADS + thread
    valid, clamped = column < width, column.minimum(width - 1)  # clamped loads, masked values: no lane is gated off
    held.append(valid.where(values[row, clamped].load().float(), neg))
    positions.append(index[row, clamped].load() if index is not None else column.cast(dtypes.float32))
  first, stores = warp.eq(0) & lane.eq(0), []
  for r in range(k):
    score, at = held[0], positions[0]
    for e in range(1, PER_THREAD):
      better = held[e] > score
      score, at = better.where(held[e], score), better.where(positions[e], at)
    top, where = _best_of(score, at)
    # The block's best: each warp's in shared memory (a buffer pair per round: tinygrad stores once per offset),
    # then every warp combines all eight, so each thread knows the winner without another broadcast
    shared = [UOp.placeholder((THREADS // 32,), dtypes.float32, slot=2 * r + i, addrspace=AddrSpace.LOCAL) for i in range(2)]
    staged = UOp.group(shared[0][warp.valid(lane.eq(0))].store(top), shared[1][warp.valid(lane.eq(0))].store(where)).barrier()
    mine, slot = lane < THREADS // 32, lane.minimum(THREADS // 32 - 1)
    best, held_at = shared[0].after(staged)[slot].load(), shared[1].after(staged)[slot].load()
    top, where = _best_of(mine.where(best, neg), mine.where(held_at, _NO_INDEX))
    stores += [out_values[row, chunk.valid(first), r].store(top), out_index[row, chunk.valid(first), r].store(where)]
    held = [positions[e].eq(where).where(neg, held[e]) for e in range(PER_THREAD)]  # the winner's holder drops it
  return UOp.group(*stores).end(block, warp, lane).sink(arg=KernelInfo(name=f"topk_{k}", opts_to_apply=()))

def _pass(values: Tensor, index: Tensor | None, k: int) -> tuple[Tensor, Tensor]:
  rows, width = values.shape
  chunks = -(-width // CHUNK)
  out_values = Tensor.empty(rows, chunks, k, dtype=dtypes.float32, device=values.device)
  out_index = Tensor.empty(rows, chunks, k, dtype=dtypes.float32, device=values.device)
  sources = (values.float().contiguous(),) + ((index.contiguous(),) if index is not None else ())
  fxn = functools.partial(_topk_kernel, rows=rows, width=width, k=k)
  out_values, out_index = Tensor.custom_kernel(out_values, out_index, *sources, fxn=fxn)[:2]
  return out_values.reshape(rows, chunks * k), out_index.reshape(rows, chunks * k)

def topk(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
  """The k largest values of each row of x (rows, width), in descending order, and their int32 column indices;
  equal values in column order (as argmax)."""
  rows, width = x.shape
  if not 0 < k <= THREADS // 32 * 32: raise ValueError("topk supports k up to 256")
  values, index = _pass(x, None, k)
  while values.shape[1] > k:  # the chunks' winners, until one chunk's worth remains
    values, index = _pass(values, index, k)
  return values, index.cast(dtypes.int32)
