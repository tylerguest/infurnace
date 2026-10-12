"""NV decode attention: parallel-softmax reference and fused split-context GQA.

The fused path reuses native FP16 KV tiles across four query warps, with FP32
online softmax/accumulation and a merge launch. Scratch is bounded by capacity;
there is no full score/probability matrix or FP16 query conversion.
"""
from __future__ import annotations
import functools
import inspect
import math

from tinygrad import Context, Device, Tensor, UOp, dtypes
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo
from infurnace.kernels.quant_gemv import warp_reduce

class AttentionError(ValueError):
  """Inputs do not satisfy the initial NV decode attention contract."""

@functools.cache
def nv_attention_supported(device: str) -> bool:
  if not isinstance(device, str) or device.split(":")[0] != "NV": return False
  with Context(ALLOW_DEVICE_USAGE=1): renderer = Device[device].renderer
  return isinstance(renderer, CUDARenderer)

def _validate_attention(q: Tensor, assigned_kv: Tensor, active_tokens: int | UOp) -> tuple[int, int, int]:
  if len(q.shape) != 4 or q.shape[0] != 1 or q.shape[2:] != (1, 128) or q.dtype != dtypes.float32:
    raise AttentionError("decode attention requires FP32 Q with shape (1, heads, 1, 128)")
  if (len(assigned_kv.shape) != 5 or assigned_kv.shape[:2] != (2, 1) or assigned_kv.shape[-1] != 128
      or assigned_kv.dtype != dtypes.float16):
    raise AttentionError("decode attention requires FP16 KV with shape (2, 1, kv_heads, capacity, 128)")
  heads, kv_heads, capacity = q.shape[1], assigned_kv.shape[2], assigned_kv.shape[3]
  if not all(isinstance(s, int) and s > 0 for s in (heads, kv_heads, capacity)) or heads%kv_heads:
    raise AttentionError("decode attention requires concrete capacity and divisible query/KV heads")
  if isinstance(active_tokens, UOp):
    if not dtypes.is_int(active_tokens.dtype) or active_tokens.vmin < 1 or active_tokens.vmax > capacity:
      raise AttentionError("symbolic active length must stay within cache capacity")
  elif type(active_tokens) is not int or not 1 <= active_tokens <= capacity:
    raise AttentionError("active length must stay within cache capacity")
  if q.device != assigned_kv.device or not nv_attention_supported(q.device):
    raise AttentionError("decode attention requires Q/KV on the same NV CUDA-C device")

  return heads, kv_heads, capacity

def decode_attention(q: Tensor, assigned_kv: Tensor, active_tokens: int | UOp) -> Tensor:
  """Parallel-softmax reference path, preserving FP32 Q and native FP16 KV."""
  heads, kv_heads, capacity = _validate_attention(q, assigned_kv, active_tokens)
  # Only QK/PV use the active prefix. Do not expand those computations to capacity.
  k = assigned_kv[0, :, :, :active_tokens, :].repeat_interleave(heads//kv_heads, dim=1)
  v = assigned_kv[1, :, :, :active_tokens, :].repeat_interleave(heads//kv_heads, dim=1)
  scores = (q.matmul(k.transpose(-2, -1), dtype=dtypes.float32)/math.sqrt(128)).cast(q.dtype)
  # A concrete index range avoids symbolic arange/cumsum. Mask padding before
  # both max and exp: zeros or stale/NaN future cache slots must not participate.
  scores = scores.pad_to((1, heads, 1, capacity))
  valid = Tensor.arange(capacity).to(q.device) < active_tokens
  probabilities = valid.where(scores, float("-inf")).softmax(-1)
  probabilities = probabilities[:, :, :, :active_tokens]
  return probabilities @ v

def fused_chain_attention(q: Tensor, assigned_kv: Tensor, start_pos: int | UOp) -> Tensor:
  """Short causal chain using exactly the existing per-query fused arithmetic.

  All candidate KV rows are already stored. Query i masks every position after
  start_pos+i, so rejected/future candidates cannot affect an accepted prefix.
  This reference-preserving scaffold does not yet reuse KV tiles across queries.
  """
  if len(q.shape) != 4 or not isinstance(q.shape[2], int) or not 1 <= q.shape[2] <= 8:
    raise AttentionError('chain attention requires one to eight concrete queries')
  size = q.shape[2]
  if size == 1: return fused_decode_attention(q, assigned_kv, start_pos+1)
  heads, kv_heads, capacity = _validate_attention(q[:, :, :1], assigned_kv, start_pos+size)
  if heads != kv_heads*4: raise AttentionError('chain attention requires four query heads per KV head')
  active = start_pos+1
  partial = Tensor.empty(size*heads, (capacity+63)//64, 130, device=q.device, dtype=dtypes.float32)
  partial = _launch_attention(partial, q.contiguous(), assigned_kv, scalar=active,
    fxn=functools.partial(_fused_partial, active=active, heads=heads, capacity=capacity, partition=64, queries=size))
  out = Tensor.empty(*q.shape, device=q.device, dtype=dtypes.float32)
  return _launch_attention(out, partial, scalar=active,
    fxn=functools.partial(_fused_merge, heads=heads, live=(active+63)//64, queries=size, active=active))

_VARIABLE_PARAM = {'param': True} if 'param' in inspect.signature(UOp.variable).parameters else {}

def _kernel_scalar(value: int | UOp) -> int | UOp:
  if not isinstance(value, UOp): return value
  value = value.unbind_all()[0]
  return value.substitute({v: UOp.variable(v.expr, v.vmin, v.vmax, dtype=v.dtype,
                          multiple_of=v.arg.multiple_of, **_VARIABLE_PARAM) for v in value.toposort() if v.is_variable})

def _launch_attention(out: Tensor, *inputs: Tensor, scalar: int | UOp, fxn) -> Tensor:
  sources = (out.uop, *(t.uop for t in inputs))
  params = tuple(UOp.placeholder_like(s, slot=i) for i, s in enumerate(sources))
  # Explicit scalar dependency retains bound values for standalone calls as well
  # as nested upstream functions. Kernels use ALU PARAMs, not tensor-graph BUFFERs.
  scalar_args = tuple(v.cast(dtypes.int32) for v in scalar.toposort() if v.is_bound_var) if isinstance(scalar, UOp) else ()
  call = fxn(*params).call(*sources, *scalar_args)
  return Tensor(out.uop.after(call))

def _fused_partial(out: UOp, q: UOp, kv: UOp, *, active: int | UOp, heads: int, capacity: int, partition: int,
                   queries: int = 1) -> UOp:
  """Four query warps reuse one shared KV tile; partition-local online softmax."""
  active = _kernel_scalar(active)
  maximum_active = active+queries-1
  live = (maximum_active+partition-1)//partition
  query_index = UOp.range(queries, 5, axis_type=AxisType.GLOBAL) if queries > 1 else UOp.const(0)
  block = UOp.range((heads//4)*live, 0, axis_type=AxisType.GLOBAL)
  lane = UOp.range(32, -1, axis_type=AxisType.WARP)
  warp = UOp.range(4, 1, axis_type=AxisType.LOCAL)
  kv_head, part = block//live, block%live
  head, start = kv_head*4+warp, part*partition
  query_active = active+query_index
  # Uniform loop bound avoids a pinned rangeifier failure for bounds depending
  # on block%parts. Only live partitions launch; each contains at least one key.
  tiles = (maximum_active+15)//16
  tiles = tiles.minimum(partition//16) if isinstance(tiles, UOp) else min(tiles, partition//16)
  tile = UOp.range(tiles, 2, axis_type=AxisType.LOOP)
  loader = UOp.range(4, 4, axis_type=AxisType.LOOP)
  row = loader*4+warp
  position = start+tile*16+row
  shared = UOp.placeholder((2, 16, 128), dtypes.float16, slot=0, addrspace=AddrSpace.LOCAL)
  # Each warp stages a complete row with four adjacent halfwords per lane;
  # the compiler can coalesce them into vector loads/stores (four loop trips).
  stage = UOp.group(*(shared[half, row, lane*4+i].store(
    kv[half, 0, kv_head, position.valid(position < query_active), lane*4+i].load())
    for half in (0, 1) for i in range(4))).end(loader).barrier()
  local = shared.after(stage)
  state = UOp.placeholder((6,), dtypes.float32, slot=1, addrspace=AddrSpace.REG)
  state = state.after(UOp.group(state[0].store(UOp.const(float("-inf"), dtypes.float32)),
                                *(state[i].store(UOp.const(0.0, dtypes.float32)) for i in range(1, 6))))
  query = tuple(q[0, head, query_index, lane+i*32].load() for i in range(4))
  key = UOp.range(16, 3, axis_type=AxisType.LOOP)
  dot = sum((query[i]*local[0, key, lane+i*32].load().float() for i in range(4)), UOp.const(0.0, dtypes.float32))
  score = warp_reduce(dot, full_wave=True)/math.sqrt(128)
  score = (start+tile*16+key < query_active).where(score, UOp.const(float("-inf"), dtypes.float32))
  previous = state.after(key)
  maximum = previous[0].load().maximum(score)
  alpha, weight = (previous[0].load()-maximum).exp(), (score-maximum).exp()
  # Freeze coefficients before overwriting any recurrent state registers.
  coefficients = UOp.placeholder((3,), dtypes.float32, slot=2, addrspace=AddrSpace.REG)
  coefficients = coefficients.after(UOp.group(*(coefficients[i].store(value) for i, value in enumerate((maximum, alpha, weight)))))
  current = state.after(coefficients)
  updates = UOp.group(current[0].store(coefficients[0].load()),
    current[1].store(current[1].load()*coefficients[1].load()+coefficients[2].load()),
    *(current[2+i].store(current[2+i].load()*coefficients[1].load()
                        +coefficients[2].load()*local[1, key, lane+i*32].load().float()) for i in range(4)))
  # All four warps must finish consuming a tile before the shared buffer is reused.
  done = state.after(updates.end(key).barrier().end(tile))
  stores = [out[query_index*heads+head, part, lane+i*32].store(done[2+i].load()) for i in range(4)]
  stores += [out[query_index*heads+head, part, UOp.const(128+i).valid(lane.eq(0))].store(done[i].load()) for i in range(2)]
  axes = (block, warp, lane, query_index) if queries > 1 else (block, warp, lane)
  return UOp.group(*stores).end(*axes).sink(arg=KernelInfo(name="attention_fused_partial", opts_to_apply=()))

def _fused_merge(out: UOp, partial: UOp, *, heads: int, live: int | UOp,
                 queries: int = 1, active: int | UOp | None = None) -> UOp:
  live = _kernel_scalar(live)
  head = UOp.range(heads, 0, axis_type=AxisType.GLOBAL)
  query_index = UOp.range(queries, 3, axis_type=AxisType.GLOBAL) if queries > 1 else UOp.const(0)
  source_head = query_index*heads+head
  if queries > 1: live = (_kernel_scalar(active)+query_index+63)//64
  lane = UOp.range(32, -1, axis_type=AxisType.WARP)
  maximum = UOp.placeholder((1,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  maximum = maximum.after(maximum[0].store(UOp.const(float("-inf"), dtypes.float32)))
  p = UOp.range(live, 1, axis_type=AxisType.LOOP)
  maximum = maximum.after(maximum.after(p)[0].store(maximum.after(p)[0].load().maximum(partial[source_head, p, 128].load())).end(p))
  total = UOp.placeholder((5,), dtypes.float32, slot=1, addrspace=AddrSpace.REG)
  total = total.after(UOp.group(*(total[i].store(UOp.const(0.0, dtypes.float32)) for i in range(5))))
  part = UOp.range(live, 2, axis_type=AxisType.LOOP)
  weight = (partial[source_head, part, 128].load()-maximum[0].load()).exp()
  updates = UOp.group(total.after(part)[0].store(total.after(part)[0].load()+weight*partial[source_head, part, 129].load()),
    *(total.after(part)[1+i].store(total.after(part)[1+i].load()+weight*partial[source_head, part, lane+i*32].load()) for i in range(4)))
  done = total.after(updates.end(part))
  axes = (head, lane, query_index) if queries > 1 else (head, lane)
  return UOp.group(*(out[0, head, query_index, lane+i*32].store(done[1+i].load()/done[0].load()) for i in range(4))).end(*axes).sink(
    arg=KernelInfo(name="attention_fused_merge", opts_to_apply=()))

def fused_decode_attention(q: Tensor, assigned_kv: Tensor, active_tokens: int | UOp, *, partition: int = 64) -> Tensor:
  """FP32 online-softmax decode with native FP16 KV, shared GQA tiles and merge.

  Initial contract: four query heads per KV head, D=128. Two GPU launches; scratch
  is bounded by capacity, never an active-length score/probability matrix.
  RTX 5060 paired decode checks favor 64-token partitions over the original 128;
  explicit 128/256 remain available for other workloads and smaller scratch.
  """
  heads, kv_heads, capacity = _validate_attention(q, assigned_kv, active_tokens)
  if heads != kv_heads*4: raise AttentionError("fused attention requires four query heads per KV head")
  if type(partition) is not int or partition not in (64, 128, 256): raise AttentionError("partition must be 64, 128 or 256")
  if assigned_kv.uop.contiguous_view() is None: raise AttentionError("fused attention requires contiguous native KV storage")
  parts = (capacity+partition-1)//partition
  partial = Tensor.empty(heads, parts, 130, device=q.device, dtype=dtypes.float32)
  fxn = functools.partial(_fused_partial, active=active_tokens, heads=heads, capacity=capacity, partition=partition)
  partial = _launch_attention(partial, q.contiguous(), assigned_kv, scalar=active_tokens, fxn=fxn)
  out = Tensor.empty(*q.shape, device=q.device, dtype=dtypes.float32)
  live = (active_tokens+partition-1)//partition
  return _launch_attention(out, partial, scalar=live, fxn=functools.partial(_fused_merge, heads=heads, live=live))
