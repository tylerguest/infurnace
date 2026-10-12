"""Experimental NV two-pass, first-index argmax over materialized FP32 scores."""
import functools

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

def _reduce(values_out, ids_out, values, ids, *, count, vocabulary, tile, indexed):
  blocks = (count+tile-1)//tile
  block = UOp.range(blocks, 0, axis_type=AxisType.GLOBAL)
  lane = UOp.range(32, 1, axis_type=AxisType.WARP)
  step = UOp.range(tile//32, 2, axis_type=AxisType.LOOP)
  def pack(value, index): return value.bitcast(dtypes.uint32).cast(dtypes.uint64) | (index.cast(dtypes.uint64) << 32)
  def unpack(pair): return pair.cast(dtypes.uint32).bitcast(dtypes.float32), (pair >> 32).cast(dtypes.int32)
  accumulator = UOp.placeholder((1,), dtypes.uint64, slot=0, addrspace=AddrSpace.REG)
  initial = pack(UOp.const(float('-inf'), dtypes.float32), UOp.const(vocabulary, dtypes.int32))
  accumulator = accumulator.after(accumulator[0].store(initial))
  position = block*tile+step*32+lane
  value = values[position.valid(position < count)].load()
  index = ids[position.valid(position < count)].load() if indexed else position
  valid = (position < count) & value.eq(value)
  # Keep the mutable-register load inside the loop. Without this dependency,
  # codegen hoisted it and selected a late value instead of the maximum.
  previous = accumulator.after(step)[0].load()
  best, winner = unpack(previous)
  take = valid & ((value > best) | (value.eq(best) & (index < winner)))
  selected = UOp(Ops.CUSTOM, src=(take, pack(value, index), previous),
                 arg=('({0} ? {1} : {2})', dtypes.uint64))
  update = accumulator[0].store(selected).end(step)
  value, index = unpack(accumulator.after(update)[0].load())
  for offset in (16, 8, 4, 2, 1):
    other = UOp(Ops.CUSTOM, src=(value,), arg=(f'__shfl_xor_sync(0xffffffff, {{0}}, {offset})', dtypes.float32))
    other_index = UOp(Ops.CUSTOM, src=(index,), arg=(f'__shfl_xor_sync(0xffffffff, {{0}}, {offset})', dtypes.int32))
    take = (other > value) | (other.eq(value) & (other_index < index))
    value, index = take.where(other, value), take.where(other_index, index)
  return UOp.group(values_out[block.valid(lane.eq(0))].store(value),
                   ids_out[block.valid(lane.eq(0))].store(index)).end(block, lane).sink(
                     arg=KernelInfo(name='eagle_parallel_argmax_merge' if indexed else 'eagle_parallel_argmax', opts_to_apply=()))

def parallel_score_argmax(row, temperature, random, *, materialize_scores=False):
  if type(materialize_scores) is not bool: raise ValueError('materialize_scores must be boolean')
  scores = row/temperature.maximum(1e-12) - (random.maximum(1e-12).log().neg()).log()
  return parallel_argmax(scores.contiguous())

def parallel_argmax(scores):
  if scores.device.split(':')[0] != 'NV' or scores.dtype != dtypes.float32 or len(scores.shape) != 2 or scores.shape[0] != 1:
    raise ValueError('parallel argmax requires one NV FP32 row')
  vocabulary = scores.shape[-1]
  if vocabulary < 1: raise ValueError('empty vocabulary')
  values = scores.contiguous().flatten()
  ids = Tensor.empty(1, dtype=dtypes.int32, device=scores.device)
  for indexed, tile in ((False, 512), (True, ((vocabulary+511)//512+31)//32*32)):
    count = values.numel()
    blocks = (count+tile-1)//tile
    out_values = Tensor.empty(blocks, dtype=dtypes.float32, device=scores.device)
    out_ids = Tensor.empty(blocks, dtype=dtypes.int32, device=scores.device)
    result = UOp.custom_kernel(out_values.uop, out_ids.uop, values.uop, ids.uop,
                              fxn=functools.partial(_reduce, count=count, vocabulary=vocabulary, tile=tile, indexed=indexed))
    values, ids = Tensor(result[0]), Tensor(result[1])
  return ids.reshape(1, 1)
