"""Multirow GQA attention over the active length (NV): split-KV partials, then a merge.

Ported from the design of vLLM's unified attention (vllm/v1/attention/ops/triton_unified_attention.py,
Apache-2.0): every query row of every grouped head that shares a KV head is packed into one block,
so each key and value is read once per KV head; each row has its own causal limit; the context is
split into fixed segments processed in parallel and merged afterwards, as llama.cpp's
flash_attn_combine_results does (ggml/src/ggml-cuda/fattn-common.cuh, MIT). Written in tinygrad UOps
with FP32 FMA math over the FP16 cache.

Only segments that hold active keys launch, so the cost follows `start + rows`, not the allocated
cache, with one captured graph for every length. Row t (absolute position start+t) sees keys
lo <= key < hi:
  causal:      hi = start+t+1                  non-causal: hi = start+rows
  window > 0:  lo = start+t-window+1, and for non-causal also hi <= start+t+window
"""
from __future__ import annotations
import functools
import inspect
import math

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo
from infurnace.kernels.quant_gemv import warp_reduce

TILE = 32      # keys staged per step: one per lane
SEGMENT = 256  # keys per block; segments are merged afterwards
QUERIES = 8    # packed query vectors per warp, at most
_NEG = float('-inf')
_VARIABLE_PARAM = {'param': True} if 'param' in inspect.signature(UOp.variable).parameters else {}

def _kernel_scalar(value: int | UOp) -> int | UOp:
  """A bound position as the kernel's own unbound variable (its value is passed at launch)."""
  if not isinstance(value, UOp): return value
  value = value.unbind_all()[0]
  return value.substitute({v: UOp.variable(v.expr, v.vmin, v.vmax, dtype=v.dtype, multiple_of=v.arg.multiple_of, **_VARIABLE_PARAM)
                           for v in value.toposort() if v.is_variable})

def _launch(outputs: tuple[Tensor, ...], inputs: tuple[Tensor, ...], start: int | UOp, fxn) -> tuple[Tensor, ...]:
  sources = tuple(t.uop for t in outputs + inputs)
  params = tuple(UOp.placeholder_like(s, slot=i) for i, s in enumerate(sources))
  scalars = tuple(v.cast(dtypes.int32) for v in start.toposort() if v.is_bound_var) if isinstance(start, UOp) else ()
  call = fxn(*params).call(*sources, *scalars)
  return tuple(Tensor(t.uop.after(call)) for t in outputs)

def _live(start: int | UOp, rows: int, segment: int) -> int | UOp:
  return (start + rows + segment - 1) // segment

def _visible(key: UOp, start: int | UOp, row: UOp, rows: int, causal: bool, window: int) -> UOp:
  position = start + row
  hi = position + 1 if causal else start + rows
  if window and not causal: hi = hi.minimum(position + window) if isinstance(hi, UOp) else min(hi, position + window)
  visible = key < hi
  if window: visible = visible & (key >= position - window + 1)
  return visible

@functools.cache
def _partial_kernel(part: UOp, stats: UOp, q: UOp, kv: UOp, *, start: int | UOp, kv_heads: int, groups: int, rows: int,
                    dim: int, causal: bool, window: int, segment: int) -> UOp:
  queries, lanes_dim = groups * rows, dim // 32
  per = min(QUERIES, queries)  # one-row decode packs only the grouped heads
  warps = -(-queries // per)
  live = _live(start, rows, segment)
  block = UOp.range(kv_heads * live, 0, axis_type=AxisType.GLOBAL)
  warp = UOp.range(warps, 1, axis_type=AxisType.LOCAL)
  lane = UOp.range(32, -1, axis_type=AxisType.WARP)
  kv_head, part_index = block // live, block % live
  active = start + rows
  ms = [warp * per + j for j in range(per)]                 # packed query vector: group * rows + row
  valid = [m < queries for m in ms]
  heads, row_of = [kv_head * groups + m // rows for m in ms], [m % rows for m in ms]
  F32, zero, neg = dtypes.float32, UOp.const(0.0, dtypes.float32), UOp.const(_NEG, dtypes.float32)

  # Running softmax state per query (replicated across lanes) and this lane's output columns.
  state = UOp.placeholder((2 * per,), F32, slot=0, addrspace=AddrSpace.REG)    # max[j], sum[j]
  out = UOp.placeholder((per * lanes_dim,), F32, slot=1, addrspace=AddrSpace.REG)
  state = state.after(UOp.group(*(state[j].store(neg) for j in range(per)), *(state[per + j].store(zero) for j in range(per))))
  out = out.after(UOp.group(*(out[i].store(zero) for i in range(per * lanes_dim))))

  # The bound must be uniform across blocks: stop at the active length for short contexts.
  tiles = (active + TILE - 1) // TILE
  tiles = tiles.minimum(segment // TILE) if isinstance(tiles, UOp) else min(tiles, segment // TILE)
  tile = UOp.range(tiles, 2, axis_type=AxisType.LOOP)
  base = part_index * segment + tile * TILE
  # Stage the tile's keys (transposed: lanes read consecutive keys) and values (lanes read consecutive dims).
  k_tile = UOp.placeholder((dim, TILE), dtypes.float16, slot=2, addrspace=AddrSpace.LOCAL)
  v_tile = UOp.placeholder((TILE, dim), dtypes.float16, slot=3, addrspace=AddrSpace.LOCAL)
  threads, elements = warps * 32, TILE * dim
  loader = UOp.range(-(-elements // threads), 3, axis_type=AxisType.LOOP)
  element = loader * threads + warp * 32 + lane
  in_tile = element < elements
  key, column = element // dim, element % dim
  live_key = in_tile & (base + key < active)
  kv_index = (base + key).valid(live_key)
  stage = UOp.group(k_tile[column.valid(in_tile), key].store(kv[0, 0, kv_head, kv_index, column].load()),
                    v_tile[key.valid(in_tile), column].store(kv[1, 0, kv_head, kv_index, column].load())).end(loader).barrier()
  k_tile, v_tile = k_tile.after(stage), v_tile.after(stage)

  # Scores: lane = key, accumulated over the head dimension; queries broadcast from global.
  score = UOp.placeholder((per,), F32, slot=4, addrspace=AddrSpace.REG)
  score = score.after(UOp.group(*(score.after(stage)[j].store(zero) for j in range(per))))
  d = UOp.range(dim, 4, axis_type=AxisType.LOOP)
  acc, kd = score.after(d), k_tile[d, lane].load().float()
  score = score.after(UOp.group(*(acc[j].store(acc[j].load() + q[0, heads[j], row_of[j], d.valid(valid[j])].load().float() * kd)
                                  for j in range(per))).end(d))
  scale = 1 / math.sqrt(dim)
  keys = base + lane
  scores = [(valid[j] & (keys < active) & _visible(keys, start, row_of[j], rows, causal, window)).where(score[j].load() * scale, neg)
            for j in range(per)]

  # Online softmax: freeze this tile's coefficients before overwriting the running state.
  coef = UOp.placeholder((3 * per,), F32, slot=5, addrspace=AddrSpace.REG)      # safe new max, alpha, new max
  previous = state.after(score)
  frozen = []
  for j in range(per):
    new_max = previous[j].load().maximum(warp_reduce(scores[j], maximum=True, full_wave=True))
    safe = new_max.ne(neg).where(new_max, zero)        # all masked so far: keep exp() finite
    frozen += [coef[j].store(safe), coef[per + j].store((previous[j].load() - safe).exp()), coef[2 * per + j].store(new_max)]
  coef = coef.after(UOp.group(*frozen))
  probs = UOp.placeholder((per * warps, TILE), F32, slot=6, addrspace=AddrSpace.LOCAL)
  ps = [(scores[j] - coef[j].load()).exp() for j in range(per)]
  current = state.after(coef)
  updates = [probs[warp * per + j, lane].store(ps[j]) for j in range(per)]
  updates += [current[j].store(coef[2 * per + j].load()) for j in range(per)]
  updates += [current[per + j].store(current[per + j].load() * coef[per + j].load() + warp_reduce(ps[j], full_wave=True))
              for j in range(per)]
  rescaled = out.after(coef)
  updates += [rescaled[j * lanes_dim + i].store(rescaled[j * lanes_dim + i].load() * coef[per + j].load())
              for j in range(per) for i in range(lanes_dim)]
  settled = UOp.group(*updates).barrier()
  probs, out = probs.after(settled), out.after(settled)

  # Values: lane owns dims lane + 32*i; probabilities broadcast from shared memory.
  k = UOp.range(TILE, 5, axis_type=AxisType.LOOP)
  acc = out.after(k)
  vs = [v_tile[k, lane + 32 * i].load().float() for i in range(lanes_dim)]
  accumulated = UOp.group(*(acc[j * lanes_dim + i].store(acc[j * lanes_dim + i].load() + probs[warp * per + j, k].load() * vs[i])
                            for j in range(per) for i in range(lanes_dim))).end(k)
  # Every warp must finish with the tile before the next one is staged over it.
  done = accumulated.barrier().end(tile)
  out, state = out.after(done), state.after(done)

  stores = [part[kv_head, part_index, ms[j].valid(valid[j]), lane + 32 * i].store(out[j * lanes_dim + i].load())
            for j in range(per) for i in range(lanes_dim)]
  stores += [stats[kv_head, part_index, ms[j].valid(valid[j] & lane.eq(0)), s].store(state[s * per + j].load())
             for j in range(per) for s in range(2)]
  return UOp.group(*stores).end(block, warp, lane).sink(arg=KernelInfo(name=f"attention_partial_d{dim}", opts_to_apply=()))

@functools.cache
def _merge_kernel(out: UOp, part: UOp, stats: UOp, *, start: int | UOp, kv_heads: int, groups: int, rows: int, dim: int,
                  segment: int) -> UOp:
  queries, lanes_dim = groups * rows, dim // 32
  live = _live(start, rows, segment)
  block = UOp.range(kv_heads * queries, 0, axis_type=AxisType.GLOBAL)
  lane = UOp.range(32, -1, axis_type=AxisType.WARP)
  kv_head, m = block // queries, block % queries
  F32, zero = dtypes.float32, UOp.const(0.0, dtypes.float32)
  top = UOp.placeholder((1,), F32, slot=0, addrspace=AddrSpace.REG)
  top = top.after(top[0].store(UOp.const(_NEG, F32)))
  s = UOp.range(live, 1, axis_type=AxisType.LOOP)
  top = top.after(top.after(s)[0].store(top.after(s)[0].load().maximum(stats[kv_head, s, m, 0].load())).end(s))
  total = UOp.placeholder((1 + lanes_dim,), F32, slot=1, addrspace=AddrSpace.REG)
  total = total.after(UOp.group(*(total[i].store(zero) for i in range(1 + lanes_dim))))
  s = UOp.range(live, 2, axis_type=AxisType.LOOP)
  acc = total.after(s)
  weight = (stats[kv_head, s, m, 0].load() - top[0].load()).exp()   # segments past this row's limit weigh 0
  total = total.after(UOp.group(acc[0].store(acc[0].load() + weight * stats[kv_head, s, m, 1].load()),
                                *(acc[1 + i].store(acc[1 + i].load() + weight * part[kv_head, s, m, lane + 32 * i].load())
                                  for i in range(lanes_dim))).end(s))
  head, row = kv_head * groups + m // rows, m % rows
  stores = [out[0, head, row, lane + 32 * i].store(total[1 + i].load() / total[0].load()) for i in range(lanes_dim)]
  return UOp.group(*stores).end(block, lane).sink(arg=KernelInfo(name=f"attention_merge_d{dim}", opts_to_apply=()))

def _masked(q: Tensor, kv: Tensor, start: int | UOp, causal: bool, window: int) -> Tensor:
  """Tensor-op fallback for other devices: softmax over the whole cache with each row's key range."""
  rows, capacity = q.shape[2], kv.shape[3]
  keys, position = Tensor.arange(capacity).reshape(1, capacity), Tensor.arange(rows).reshape(rows, 1) + Tensor(start)
  hi = position + 1 if causal else (position - Tensor.arange(rows).reshape(rows, 1)) + rows
  if window and not causal: hi = hi.minimum(position + window)
  visible = keys < hi
  if window: visible = visible & (keys >= position - window + 1)
  mask = visible.where(0.0, float('-inf')).reshape(1, 1, rows, capacity)
  return q.float().scaled_dot_product_attention(kv[0].float(), kv[1].float(), attn_mask=mask, enable_gqa=True)

def attention(q: Tensor, kv: Tensor, start: int | UOp, *, causal: bool = True, window: int = 0, segment: int = SEGMENT) -> Tensor:
  """q: (1, heads, rows, dim) at positions start..start+rows-1; kv: (2, 1, kv_heads, capacity, dim) FP16
  cache already holding those rows. Returns (1, heads, rows, dim) FP32. dim must be a multiple of 32.
  The fused kernels need NV (CUDA warp shuffles); other devices use plain tensor ops over the whole cache."""
  _, heads, rows, dim = q.shape
  _, _, kv_heads, capacity, kv_dim = kv.shape
  if dim != kv_dim or heads % kv_heads: raise ValueError("attention needs matching head sizes and grouped heads")
  if q.device.split(':')[0] != 'NV' or dim % 32 or kv.dtype != dtypes.float16: return _masked(q, kv, start, causal, window)
  groups, segments = heads // kv_heads, -(-capacity // segment)
  part = Tensor.empty(kv_heads, segments, groups * rows, dim, dtype=dtypes.float32, device=q.device)
  stats = Tensor.empty(kv_heads, segments, groups * rows, 2, dtype=dtypes.float32, device=q.device)
  scalar = _kernel_scalar(start)
  part, stats = _launch((part, stats), (q.contiguous(), kv), start, functools.partial(
    _partial_kernel, start=scalar, kv_heads=kv_heads, groups=groups, rows=rows, dim=dim, causal=causal, window=window, segment=segment))
  out = Tensor.empty(1, heads, rows, dim, dtype=dtypes.float32, device=q.device)
  return _launch((out,), (part, stats), start, functools.partial(
    _merge_kernel, start=scalar, kv_heads=kv_heads, groups=groups, rows=rows, dim=dim, segment=segment))[0]
