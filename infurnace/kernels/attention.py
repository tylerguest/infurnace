"""Multirow GQA attention over the active length (NV): split-KV partials, then a merge.

Ported from the design of vLLM's unified attention (vllm/v1/attention/ops/triton_unified_attention.py,
Apache-2.0): every query row of every grouped head that shares a KV head is packed into one block,
so each key and value is read once per KV head; each row has its own causal limit; the context is
split into fixed segments processed in parallel and merged afterwards, as llama.cpp's
flash_attn_combine_results does (ggml/src/ggml-cuda/fattn-common.cuh, MIT). Written in tinygrad UOps.

Three partial kernels share that merge: one-row decode (flash-decoding: a lane owns dim/32 dimensions, softmax
in registers), multirow on FP16 tensor cores (FlashAttention-2's fragment layout on mma.sync m16n8k16, FP32
accumulation; INFURNACE_MMA_ATTENTION=0 turns it off), and the FP32 FMA kernel for everything else.

Only segments that hold active keys launch (with a window, only those inside it), so the cost
follows the visible length, not the allocated cache, with one captured graph for every length.
Row t (absolute position start+t) sees keys lo <= key < hi:
  causal:      hi = start+t+1                  non-causal: hi = start+rows
  window > 0:  lo = start+t-window+1, and for non-causal also hi <= start+t+window
  floor:       lo >= floor[0] (a one-element buffer, so captured graphs read its current value)
A ring cache (`ring=True`, sliding windows only) holds position x at slot x % its length, which
must be at least window + rows: it only keeps what a window can see. `store_rows` writes it.
"""
from __future__ import annotations
import functools
import inspect
import math
import os

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from infurnace.kernels.quant_gemv import _load_lanes, warp_reduce
from infurnace.device import nvidia

TILE = 32      # keys staged per step: one per lane
SEGMENT = 256  # keys per block; segments are merged afterwards
QUERIES = 8    # packed query vectors per warp, at most
# One-row decode (no window, ring or floor): each warp walks DECODE_SEGMENT keys of one KV head, DECODE_TILE
# keys per softmax update, DECODE_WARPS warps per block.
DECODE_SEGMENT, DECODE_TILE, DECODE_WARPS = 128, 8, 4
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

def _first(start: int | UOp, window: int, segment: int) -> int | UOp:
  """The first segment any row can see: the window's start for the first row."""
  if not window: return 0
  lo = start - window + 1
  return (lo.maximum(0) if isinstance(lo, UOp) else max(lo, 0)) // segment

def _visible(key: UOp, start: int | UOp, row: UOp, rows: int, causal: bool, window: int) -> UOp:
  position = start + row
  hi = position + 1 if causal else start + rows
  if window and not causal: hi = hi.minimum(position + window) if isinstance(hi, UOp) else min(hi, position + window)
  visible = key < hi
  if window: visible = visible & (key >= position - window + 1)
  return visible

@functools.cache
def _partial_kernel(part: UOp, stats: UOp, q: UOp, kv: UOp, floor: UOp | None = None, *, start: int | UOp, kv_heads: int,
                    groups: int, rows: int, dim: int, causal: bool, window: int, segment: int, ring: int) -> UOp:
  queries, lanes_dim = groups * rows, dim // 32
  per = min(QUERIES, queries)  # one-row decode packs only the grouped heads
  warps = -(-queries // per)
  first = _first(start, window, segment)
  count = _live(start, rows, segment) - first
  block = UOp.range(kv_heads * count, 0, axis_type=AxisType.GLOBAL)
  warp = UOp.range(warps, 1, axis_type=AxisType.LOCAL)
  lane = UOp.range(32, -1, axis_type=AxisType.WARP)
  kv_head, part_index = block // count, block % count
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
  base = (first + part_index) * segment + tile * TILE
  # Stage the tile's keys (transposed: lanes read consecutive keys) and values (lanes read consecutive dims).
  k_tile = UOp.placeholder((dim, TILE), dtypes.float16, slot=2, addrspace=AddrSpace.LOCAL)
  v_tile = UOp.placeholder((TILE, dim), dtypes.float16, slot=3, addrspace=AddrSpace.LOCAL)
  threads, elements = warps * 32, TILE * dim
  loader = UOp.range(-(-elements // threads), 3, axis_type=AxisType.LOOP)
  element = loader * threads + warp * 32 + lane
  in_tile = element < elements
  key, column = element // dim, element % dim
  live_key = in_tile & (base + key < active)
  kv_index = ((base + key) % ring if ring else base + key).valid(live_key)
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
  seen = (keys < active) & (keys >= floor[0].load().cast(keys.dtype)) if floor is not None else keys < active
  scores = [(valid[j] & seen & _visible(keys, start, row_of[j], rows, causal, window)).where(score[j].load() * scale, neg)
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

_MMA = os.environ.get('INFURNACE_MMA_ATTENTION', '1') != '0'

def _tensor_cores(device: str) -> bool:
  from infurnace.kernels.quant_gemm import nv_prefill_supported  # FP16 m16n8k16 MMA through the CUDA C renderer
  return nv_prefill_supported(device)

def _shuffle_rows(value: UOp, maximum: bool) -> UOp:
  """Reduce across the four lanes that hold one row of an MMA accumulator (lanes 4g..4g+3)."""
  for offset in (1, 2):
    other = UOp(Ops.CUSTOM, src=(value,), arg=(f"__shfl_xor_sync(0xffffffff, {{0}}, {offset})", dtypes.float))
    value = value.maximum(other) if maximum else value + other
  return value

@functools.cache
def _mma_kernel(part: UOp, stats: UOp, q: UOp, kv: UOp, floor: UOp | None = None, *, start: int | UOp, kv_heads: int,
                groups: int, rows: int, dim: int, causal: bool, window: int, segment: int, ring: int, capacity: int) -> UOp:
  """Multirow attention on FP16 tensor cores (FlashAttention-2's layout, mma.sync m16n8k16): one warp per 16
  packed queries, S = Q·Kᵀ per 32-key tile, online softmax on the accumulator fragments (a row's values sit
  in four neighbouring lanes), then P·V with P taken straight from the score fragments, which are already
  in the A-operand layout. Same partial/stats outputs as _partial_kernel."""
  queries, steps, tiles_n = groups * rows, dim // 16, dim // 8
  # Up to 4 warps (64 queries) a block: at dim 256 a lane takes ~254 registers, so a block can't be larger.
  # More queries (a 64-row prompt chunk) go to more blocks, each reading the same keys.
  warps = min(-(-queries // 16), 4)
  qblocks = -(-queries // (16 * warps))
  first = _first(start, window, segment)
  count = _live(start, rows, segment) - first
  block = UOp.range(kv_heads * count * qblocks, 0, axis_type=AxisType.GLOBAL)
  warp = UOp.range(warps, 1, axis_type=AxisType.LOCAL)
  lane = UOp.range(32, -1, axis_type=AxisType.WARP)
  kv_head, rest = block // (count * qblocks), block % (count * qblocks)
  part_index, qblock = rest // qblocks, rest % qblocks
  group, thread = lane // 4, lane % 4
  active = start + rows
  F32, F16, zero, neg = dtypes.float32, dtypes.float16, UOp.const(0.0, dtypes.float32), UOp.const(_NEG, dtypes.float32)
  # This lane's two query rows: packed query m = group*rows + row, as in _partial_kernel.
  ms = [(qblock * warps + warp) * 16 + group + 8 * r for r in (0, 1)]
  valid_m = [m < queries for m in ms]
  heads, row_of = [kv_head * groups + m // rows for m in ms], [m % rows for m in ms]
  # Q as A fragments, loaded once: slot order (r0,c0) (r0,c1) (r8,c0) (r8,c1) (r0,c8) (r0,c9) (r8,c8) (r8,c9).
  qa = UOp.placeholder((steps * 8,), F16, slot=0, addrspace=AddrSpace.REG)
  slots = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 8), (0, 9), (1, 8), (1, 9)]
  qa = qa.after(UOp.group(*(qa[ks * 8 + i].store(
    q[0, heads[r], row_of[r], (ks * 16 + thread * 2 + c).valid(valid_m[r])].load().cast(F16))
    for ks in range(steps) for i, (r, c) in enumerate(slots))))
  # max and sum for the two rows; the output fragments in their own 4-slot accumulators
  state = UOp.placeholder((4,), F32, slot=1, addrspace=AddrSpace.REG)
  state = state.after(UOp.group(state[0].store(neg), state[1].store(neg), state[2].store(zero), state[3].store(zero)))
  outs = []
  for c in range(tiles_n):
    o = UOp.placeholder((4,), F32, slot=2 + c, addrspace=AddrSpace.REG)
    outs.append(o.after(o.store(o.const_like(0))))
  scores = [UOp.placeholder((4,), F32, slot=2 + tiles_n + nt, addrspace=AddrSpace.REG) for nt in range(4)]
  tile = UOp.range(segment // 32, 2, axis_type=AxisType.LOOP)
  base = (first + part_index) * segment + tile * 32
  def slot_of(key: UOp) -> tuple[UOp, UOp]:
    return key < active, (key % ring if ring else key.minimum(capacity - 1))
  # A P·V B fragment needs four keys of one dimension: four separate cache rows. With dim <= 128, stage the
  # tile's values transposed (dim x keys) in shared memory first, so a lane reads adjacent keys. At dim 256 the
  # output fragments alone take 128 registers a lane (254 in all: 8 warps an SM), and the staging barriers
  # cost more than the scattered loads (RTX 5060: 352 vs 461 us for 16 rows at 16k keys).
  stage_v = dim <= 128
  if stage_v:
    threads, pad = warps * 32, 8
    vt = UOp.placeholder((dim, 32 + pad), F16, slot=3 + tiles_n + 4, addrspace=AddrSpace.LOCAL)
    chunks = 32 * dim // 8
    loader = UOp.range(-(-chunks // threads), 3, axis_type=AxisType.LOOP)
    chunk = loader * threads + warp * 32 + lane
    in_tile = chunk < chunks
    v_key, v_col = chunk % 32, (chunk // 32) * 8  # neighbouring lanes take neighbouring keys: no bank conflicts
    v_live, v_index = slot_of(base + v_key)
    row = _load_lanes(kv[1, 0, kv_head, v_index, v_col], 8)
    stage = UOp.group(*(vt[(v_col + i).valid(in_tile), v_key].store(v_live.where(row[i], UOp.const(0.0, F16))) for i in range(8)))
    vt = vt.after(stage.end(loader).barrier())
  qv = [qa.after(tile)[i].load() for i in range(steps * 8)]
  computed = []
  for nt in range(4):
    _, index = slot_of(base + nt * 8 + group)  # past the active length: masked after the MMA
    acc = UOp.stack(zero, zero, zero, zero)
    for ks in range(steps):
      pairs = [_load_lanes(kv[0, 0, kv_head, index, ks * 16 + thread * 2 + c], 2) for c in (0, 8)]  # 4-byte loads
      b = UOp.stack(pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1])
      acc = UOp.wmma(UOp.stack(*qv[ks * 8:ks * 8 + 8]), b, acc, (8, 16, 16), 32)
    computed.append(scores[nt].after(scores[nt].store(acc)))
  scale = 1 / math.sqrt(dim)
  cur = state.after(tile)
  values = [[None] * 8 for _ in range(2)]  # per row: 4 n-tiles x 2 columns
  for nt in range(4):
    for i in range(4):
      r, col = i // 2, i % 2
      key = base + nt * 8 + thread * 2 + col
      seen = key < active
      if floor is not None: seen = seen & (key >= floor[0].load().cast(key.dtype))
      visible = valid_m[r] & seen & _visible(key, start, row_of[r], rows, causal, window)
      values[r][nt * 2 + col] = visible.where(computed[nt][i].load() * scale, neg)
  updates, alphas, probs = [], [], [None, None]
  for r in range(2):
    top = values[r][0]
    for v in values[r][1:]: top = top.maximum(v)
    old = cur[r].load()
    new_max = old.maximum(_shuffle_rows(top, True))
    safe = new_max.ne(neg).where(new_max, zero)        # all masked so far: keep exp() finite
    alpha = (old - safe).exp()
    probs[r] = [(v - safe).exp() for v in values[r]]
    total = probs[r][0]
    for p in probs[r][1:]: total = total + p
    updates += [cur[r].store(new_max), cur[2 + r].store(cur[2 + r].load() * alpha + _shuffle_rows(total, False))]
    alphas.append(alpha)
  # P as A fragments: k-chunk kk covers n-tiles 2kk (columns c0,c1) and 2kk+1 (c8,c9).
  pa = [UOp.stack(*(probs[r][(2 * kk + h) * 2 + col].cast(F16) for h in (0, 1) for r in (0, 1) for col in (0, 1))) for kk in range(2)]
  for c in range(tiles_n):
    o = outs[c].after(tile)
    acc = UOp.stack(*(o[i].load() * alphas[i // 2] for i in range(4)))
    for kk in range(2):
      if stage_v: parts = [vt[c * 8 + group, kk * 16 + thread * 2 + j].load() for j in (0, 1, 8, 9)]
      else:
        parts = []
        for j in (0, 1, 8, 9):
          live, index = slot_of(base + kk * 16 + thread * 2 + j)
          parts.append(live.where(kv[1, 0, kv_head, index, c * 8 + group].load(), UOp.const(0.0, F16)))
      acc = UOp.wmma(pa[kk], UOp.stack(*parts), acc, (8, 16, 16), 32)
    updates.append(o.store(acc))
  # Every warp must finish with the tile before the next one is staged over it.
  done = (UOp.group(*updates).barrier() if stage_v else UOp.group(*updates)).end(tile)
  state, outs = state.after(done), [o.after(done) for o in outs]
  stores = [part[kv_head, part_index, ms[i // 2].valid(valid_m[i // 2]), c * 8 + thread * 2 + i % 2].store(outs[c][i].load())
            for c in range(tiles_n) for i in range(4)]
  stores += [stats[kv_head, part_index, ms[r].valid(valid_m[r] & thread.eq(0)), s].store(state[2 * s + r].load())
             for r in range(2) for s in range(2)]
  return UOp.group(*stores).end(block, warp, lane).sink(arg=KernelInfo(name=f"attention_mma_d{dim}", opts_to_apply=()))

@functools.cache
def _decode_kernel(part: UOp, stats: UOp, q: UOp, kv: UOp, *, start: int | UOp, kv_heads: int, groups: int, dim: int,
                   capacity: int, segment: int, tile: int, warps: int) -> UOp:
  """One query row (every grouped head of a KV head) after flash-decoding: lane owns dim/32 consecutive
  dimensions and reads them with one vector load per key, scores are summed across the warp by shuffles, and
  the online softmax lives in registers (no shared memory or barriers). Each warp writes one partial."""
  lanes_dim, F32 = dim // 32, dtypes.float32
  zero, neg = UOp.const(0.0, F32), UOp.const(_NEG, F32)
  active = start + 1
  blocks = (active + segment * warps - 1) // (segment * warps)
  block = UOp.range(kv_heads * blocks, 0, axis_type=AxisType.GLOBAL)
  warp = UOp.range(warps, 1, axis_type=AxisType.LOCAL)
  lane = UOp.range(32, -1, axis_type=AxisType.WARP)
  kv_head, entry = block // blocks, (block % blocks) * warps + warp
  scale = 1 / math.sqrt(dim)
  qs = UOp.placeholder((groups * lanes_dim,), F32, slot=0, addrspace=AddrSpace.REG)
  qs = qs.after(UOp.group(*(qs[j * lanes_dim + i].store(q[0, kv_head * groups + j, 0, lane * lanes_dim + i].load().float() * scale)
                            for j in range(groups) for i in range(lanes_dim))))
  # max[j], sum[j], then this lane's output columns per query
  state = UOp.placeholder((2 * groups + groups * lanes_dim,), F32, slot=1, addrspace=AddrSpace.REG)
  state = state.after(UOp.group(*(state[j].store(neg) for j in range(groups)),
                                *(state[groups + j].store(zero) for j in range(groups + groups * lanes_dim))))
  t = UOp.range(segment // tile, 2, axis_type=AxisType.LOOP)
  cur, qv = state.after(t), [qs[i].load() for i in range(groups * lanes_dim)]
  scores, values = [[] for _ in range(groups)], []
  for u in range(tile):
    key = entry * segment + t * tile + u
    live = key < active
    row = key.minimum(capacity - 1)  # past the active length: in bounds, masked below
    k = _load_lanes(kv[0, 0, kv_head, row, lane * lanes_dim], lanes_dim)
    v = _load_lanes(kv[1, 0, kv_head, row, lane * lanes_dim], lanes_dim)
    # Unwritten cache rows may hold any bits: zero their values so a 0 weight can't meet a NaN.
    values.append([live.where(v[i].float(), zero) for i in range(lanes_dim)])
    for j in range(groups):
      dot = qv[j * lanes_dim] * k[0].float()
      for i in range(1, lanes_dim): dot = dot + qv[j * lanes_dim + i] * k[i].float()
      scores[j].append(live.where(warp_reduce(dot, full_wave=True), neg))
  updates = []
  for j in range(groups):
    top = scores[j][0]
    for sc in scores[j][1:]: top = top.maximum(sc)
    old = cur[j].load()
    new_max = old.maximum(top)
    safe = new_max.ne(neg).where(new_max, zero)        # all masked so far: keep exp() finite
    alpha, ps = (old - safe).exp(), [(sc - safe).exp() for sc in scores[j]]
    total = cur[groups + j].load() * alpha
    for p in ps: total = total + p
    updates += [cur[j].store(new_max), cur[groups + j].store(total)]
    for i in range(lanes_dim):
      slot = 2 * groups + j * lanes_dim + i
      acc = cur[slot].load() * alpha
      for p, v in zip(ps, values): acc = acc + p * v[i]
      updates.append(cur[slot].store(acc))
  state = state.after(UOp.group(*updates).end(t))
  stores = [part[kv_head, entry, j, lane * lanes_dim + i].store(state[2 * groups + j * lanes_dim + i].load())
            for j in range(groups) for i in range(lanes_dim)]
  stores += [stats[kv_head, entry, UOp.const(j, lane.dtype).valid(lane.eq(0)), s].store(state[s * groups + j].load())
             for j in range(groups) for s in range(2)]  # lane 0 only
  return UOp.group(*stores).end(block, warp, lane).sink(arg=KernelInfo(name=f"attention_decode_d{dim}", opts_to_apply=()))

@functools.cache
def _merge_kernel(out: UOp, part: UOp, stats: UOp, *, start: int | UOp, kv_heads: int, groups: int, rows: int, dim: int,
                  window: int, segment: int) -> UOp:
  queries, lanes_dim = groups * rows, dim // 32
  live = _live(start, rows, segment) - _first(start, window, segment)
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

def _masked(q: Tensor, kv: Tensor, start: int | UOp, causal: bool, window: int, ring: bool, floor: Tensor | None) -> Tensor:
  """Tensor-op fallback for other devices: softmax over the whole cache with each row's key range."""
  rows, capacity = q.shape[2], kv.shape[3]
  keys, position = Tensor.arange(capacity).reshape(1, capacity), Tensor.arange(rows).reshape(rows, 1) + Tensor(start)
  if ring:  # slot j holds the newest position at or below the last row's that is j modulo the length
    top = Tensor(start) + (rows - 1)
    keys = top - (top - keys + capacity) % capacity
  hi = position + 1 if causal else (position - Tensor.arange(rows).reshape(rows, 1)) + rows
  if window and not causal: hi = hi.minimum(position + window)
  visible = (keys < hi) & (keys >= 0)
  if window: visible = visible & (keys >= position - window + 1)
  if floor is not None: visible = visible & (keys >= floor.reshape(1, 1))
  mask = visible.where(0.0, float('-inf')).reshape(1, 1, rows, capacity)
  return q.float().scaled_dot_product_attention(kv[0].float(), kv[1].float(), attn_mask=mask, enable_gqa=True)

def attention(q: Tensor, kv: Tensor, start: int | UOp, *, causal: bool = True, window: int = 0, segment: int = SEGMENT,
              ring: bool = False, floor: Tensor | None = None) -> Tensor:
  """q: (1, heads, rows, dim) at positions start..start+rows-1; kv: (2, 1, kv_heads, capacity, dim) FP16
  cache already holding those rows (a ring of length capacity with `ring`). floor: optional (1,) int32
  buffer, the lowest visible key. Returns (1, heads, rows, dim) FP32. dim must be a multiple of 32.
  The fused kernels need NV (CUDA warp shuffles); other devices use plain tensor ops over the whole cache.
  One row without a window, ring or floor (decoding) uses the flash-decoding kernel, which splits the keys
  itself (DECODE_SEGMENT per warp); `segment` applies to the other shapes."""
  _, heads, rows, dim = q.shape
  _, _, kv_heads, capacity, kv_dim = kv.shape
  if dim != kv_dim or heads % kv_heads: raise ValueError("attention needs matching head sizes and grouped heads")
  if ring and not 0 < window <= capacity - rows: raise ValueError("a ring cache needs a window and room for it plus the rows")
  if not nvidia(q.device) or dim % 32 or kv.dtype != dtypes.float16: return _masked(q, kv, start, causal, window, ring, floor)
  groups, scalar = heads // kv_heads, _kernel_scalar(start)
  if rows == 1 and not window and not ring and floor is None and dim // 32 in (2, 4, 8):
    seg = DECODE_SEGMENT
    entries = -(-capacity // (seg * DECODE_WARPS)) * DECODE_WARPS
    part = Tensor.empty(kv_heads, entries, groups, dim, dtype=dtypes.float32, device=q.device)
    stats = Tensor.empty(kv_heads, entries, groups, 2, dtype=dtypes.float32, device=q.device)
    part, stats = _launch((part, stats), (q.contiguous(), kv), start, functools.partial(
      _decode_kernel, start=scalar, kv_heads=kv_heads, groups=groups, dim=dim, capacity=capacity, segment=seg,
      tile=DECODE_TILE, warps=DECODE_WARPS))
    out = Tensor.empty(1, heads, 1, dim, dtype=dtypes.float32, device=q.device)
    return _launch((out,), (part, stats), start, functools.partial(
      _merge_kernel, start=scalar, kv_heads=kv_heads, groups=groups, rows=1, dim=dim, window=0, segment=seg))[0]
  # A ring's window can straddle one more segment boundary than its length spans.
  if rows > 16 and not (dim % 16 == 0 and segment % 32 == 0 and _MMA and _tensor_cores(q.device)):
    # The FP32 kernel stages a probability row per query: past 16 rows its shared memory runs out. Slices of
    # up to 16 rows are the same attention at later starting positions.
    return Tensor.cat(*(attention(q[:, :, i:i + 16], kv, start + i, causal=causal, window=window, segment=segment, ring=ring,
                                  floor=floor) for i in range(0, rows, 16)), dim=2)
  segments = -(-capacity // segment) + int(ring)
  part = Tensor.empty(kv_heads, segments, groups * rows, dim, dtype=dtypes.float32, device=q.device)
  stats = Tensor.empty(kv_heads, segments, groups * rows, 2, dtype=dtypes.float32, device=q.device)
  if rows > 1 and dim % 16 == 0 and segment % 32 == 0 and _MMA and _tensor_cores(q.device):
    part, stats = _launch((part, stats), (q.contiguous(), kv) + ((floor,) if floor is not None else ()), start, functools.partial(
      _mma_kernel, start=scalar, kv_heads=kv_heads, groups=groups, rows=rows, dim=dim, causal=causal, window=window,
      segment=segment, ring=capacity if ring else 0, capacity=capacity))
    out = Tensor.empty(1, heads, rows, dim, dtype=dtypes.float32, device=q.device)
    return _launch((out,), (part, stats), start, functools.partial(
      _merge_kernel, start=scalar, kv_heads=kv_heads, groups=groups, rows=rows, dim=dim, window=window, segment=segment))[0]
  part, stats = _launch((part, stats), (q.contiguous(), kv) + ((floor,) if floor is not None else ()), start, functools.partial(
    _partial_kernel, start=scalar, kv_heads=kv_heads, groups=groups, rows=rows, dim=dim, causal=causal, window=window, segment=segment,
    ring=capacity if ring else 0))
  out = Tensor.empty(1, heads, rows, dim, dtype=dtypes.float32, device=q.device)
  return _launch((out,), (part, stats), start, functools.partial(
    _merge_kernel, start=scalar, kv_heads=kv_heads, groups=groups, rows=rows, dim=dim, window=window, segment=segment))[0]

@functools.cache
def _ring_store_kernel(cache: UOp, values: UOp, *, start: int | UOp, kv_heads: int, rows: int, dim: int, length: int) -> UOp:
  i = UOp.range(2 * kv_heads * rows, 0, axis_type=AxisType.GLOBAL)
  d = UOp.range(dim, 1, axis_type=AxisType.LOCAL)
  c, h, r = i // (kv_heads * rows), (i // rows) % kv_heads, i % rows
  return cache[c, 0, h, (start + r) % length, d].store(values[c, 0, h, r, d].load()).end(i, d).sink(
    arg=KernelInfo(name=f"ring_store_d{dim}", opts_to_apply=()))

def store_rows(cache: Tensor, values: Tensor, start: int | UOp, *, ring: bool = False) -> Tensor:
  """Write values (2, 1, kv_heads, rows, dim) as positions start..start+rows-1 of cache, which then
  holds them (a ring of its length with `ring`, so the rows may wrap around its end)."""
  rows, length = values.shape[3], cache.shape[3]
  if not ring: return Tensor(cache.uop.after(cache[:, :, :, start:start + rows].uop.store(values.uop)))
  if nvidia(cache.device):
    return _launch((cache,), (values.contiguous(),), start, functools.partial(
      _ring_store_kernel, start=_kernel_scalar(start), kv_heads=cache.shape[2], rows=rows, dim=cache.shape[4], length=length))[0]
  offset = (Tensor.arange(length) - Tensor(start) % length + length) % length    # each slot's row, if < rows
  new = (offset < rows).reshape(1, 1, 1, length, 1).where(values[:, :, :, offset.clip(0, rows - 1)], cache).contiguous()
  return Tensor(cache.uop.after(cache.uop.store(new.uop)))
