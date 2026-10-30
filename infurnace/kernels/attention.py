"""Multirow GQA attention over the active length (NV): split-KV partials, then a merge.

Ported from the design of vLLM's unified attention (vllm/v1/attention/ops/triton_unified_attention.py,
Apache-2.0): every query row of every grouped head that shares a KV head is packed into one block,
so each key and value is read once per KV head; each row has its own causal limit; the context is
split into fixed segments processed in parallel and merged afterwards, as llama.cpp's
flash_attn_combine_results does (ggml/src/ggml-cuda/fattn-common.cuh, MIT). Written in tinygrad UOps.

Four partial kernels share that merge: one-row decode (flash-decoding: a lane owns dim/32 dimensions, softmax
in registers); multirow over the whole context (target verification, prompt chunks) on FP16 tensor cores with
each key tile staged in shared memory for a block of warps and the next one loaded meanwhile (_flash_kernel, a
fixed number of key slices; INFURNACE_FLASH_ATTENTION=0 turns it off); multirow with a window, ring or floor (the
DFlash draft) on tensor cores reading fragments from the cache (FlashAttention-2's fragment layout on mma.sync
m16n8k16, FP32 accumulation; INFURNACE_MMA_ATTENTION=0 turns both off); and the FP32 FMA kernel for everything else.

Only segments that hold active keys launch (with a window, only those inside it), so the cost
follows the visible length, not the allocated cache, with one captured graph for every length.
Row t (absolute position start+t) sees keys lo <= key < hi:
  causal:      hi = start+t+1                  non-causal: hi = start+rows
  window > 0:  lo = start+t-window+1, and for non-causal also hi <= start+t+window
  floor:       lo >= floor[0] (a one-element buffer, so captured graphs read its current value)
A ring cache (`ring=True`, sliding windows only) holds position x at slot x % its length, which
must be at least window + rows: it only keeps what a window can see. `store_rows` writes it.

An int8 cache (`store_kv`) holds each key and value vector as int8 with one FP16 scale, four values a uint32
word: a key's scale multiplies its scores and a value's its probabilities, so the kernels convert the integers
and apply one multiply a key. Bonsai-4B perplexity (eval.txt, 16-row path): 18.7368 -> 18.7586 at 2048 tokens,
6.5237 -> 6.5251 at 8192.
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

def _byte(word: UOp, index: int | UOp) -> UOp:
  """Signed byte `index` (0-3) of a uint32 word, as an int32."""
  shift = (3 - index) * 8
  if isinstance(shift, UOp): shift = shift.cast(dtypes.int32)
  return (word.bitcast(dtypes.int32) << shift) >> 24

_LDMATRIX = ('[&]() {{ unsigned a, b; asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{%0,%1}}, [%2];" : "=r"(a), "=r"(b) '
             ': "r"((unsigned)__cvta_generic_to_shared({0}))); return ((unsigned long long)b << 32) | a; }}()')

def _ldmatrix_words(ptr: UOp) -> list[UOp]:
  """ldmatrix.x2 as its two 32-bit registers (as _ldmatrix): on int8 rows of 16 bytes, an m16n8k32 B fragment."""
  pair = UOp(Ops.CUSTOM, src=(ptr,), arg=(_LDMATRIX, dtypes.uint64))
  return [pair.cast(dtypes.uint32), (pair >> 32).cast(dtypes.uint32)]

def _ldmatrix(ptr: UOp) -> list[UOp]:
  """ldmatrix.x2: two 8x8 FP16 tiles of shared memory, lanes 0-7 giving the rows of the first and 8-15 of the
  second; each lane gets row lane / 4, columns 2 (lane % 4) and the next of both, as four halves. Emitted once
  (a CUSTOM, not inline): one instruction for an MMA's B fragment in place of two 4-byte loads."""
  pair = UOp(Ops.CUSTOM, src=(ptr,), arg=(_LDMATRIX, dtypes.uint64))
  words = [pair.cast(dtypes.uint32), (pair >> 32).cast(dtypes.uint32)]
  return [(w >> s).cast(dtypes.uint16).bitcast(dtypes.float16) for w in words for s in (0, 16)]

# Four int8 values of a word as two FP16 pairs times a scale: bytes made unsigned (+128) become the low mantissa
# of 1024 + b (0x64XX), then 1152 comes off; two values an instruction (__hsub2, __hmul2) instead of shifts and
# a conversion each.
_INT8_PAIR = ('[&]() {{ unsigned u = __byte_perm({0} ^ 0x80808080u, 0x64646464u, {2}); __half2 h = __hmul2(__hsub2(*(__half2*)&u, '
              '__float2half2_rn(1152.f)), __float2half2_rn({1})); return *(unsigned*)&h; }}()')

def _int8_halves(word: UOp, scale: UOp) -> list[UOp]:
  """The four signed bytes of a uint32 word, times `scale`, as FP16."""
  pairs = [UOp(Ops.CUSTOM, src=(word, scale), arg=(_INT8_PAIR.replace('{2}', selector), dtypes.uint32)) for selector in ('0x4140u', '0x4342u')]
  return [(p >> s).cast(dtypes.uint16).bitcast(dtypes.float16) for p in pairs for s in (0, 16)]

# FP8 e4m3: two FP16 (as a word) or two FP32 values to a byte pair (the first value the low byte), saturating
_E4M3_F16X2 = '[&]() {{ unsigned short r; asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(r) : "r"({0})); return (unsigned)r; }}()'
_E4M3_F32X2 = '[&]() {{ unsigned short r; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"({1}), "f"({0})); return (unsigned)r; }}()'
_FMMA = ('[&]() {{ float d0, d1, d2, d3; asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, '
         '{{%8,%9}}, {{%10,%11,%12,%13}};" : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"({0}), "r"({1}), "r"({2}), "r"({3}), '
         '"r"({4}), "r"({5}), "f"({6}), "f"({7}), "f"({8}), "f"({9})); return d{index}; }}()')

def _int8_e4m3(word: UOp) -> UOp:
  """The four signed bytes of a uint32 word as FP8 e4m3 bytes (exact to 16, three mantissa bits above)."""
  one = UOp.const(1.0, dtypes.float32)
  pairs = [UOp(Ops.CUSTOM, src=(word, one), arg=(_INT8_PAIR.replace('{2}', selector), dtypes.uint32)) for selector in ('0x4140u', '0x4342u')]
  low, high = (UOp(Ops.CUSTOMI, src=(p,), arg=(_E4M3_F16X2, dtypes.uint32)) for p in pairs)
  return low | (high << 16)

def _fmma(a: list[UOp], b: list[UOp]) -> list[UOp]:
  """D = A x B on FP8 e4m3 tensor cores, FP32 out (m16n8k32; fragments as _imma's): 2x the FP16-in, FP32-accumulate
  rate on an RTX 5060. Each starts from zero, so no inline MMA expression holds another."""
  zero = UOp.const(0.0, dtypes.float32)
  return [UOp(Ops.CUSTOMI, src=(*a, *b, zero, zero, zero, zero), arg=(_FMMA.replace('{index}', str(i)), dtypes.float32)) for i in range(4)]

def _extras(extra: tuple[UOp, ...], int8: bool) -> tuple[UOp | None, UOp | None]:
  """A kernel's optional sources after kv: the int8 cache's scales, then the floor."""
  return (extra[0] if int8 else None), (extra[int(int8)] if len(extra) > int(int8) else None)

@functools.cache
def _partial_kernel(part: UOp, stats: UOp, q: UOp, kv: UOp, *extra: UOp, start: int | UOp, kv_heads: int, groups: int, rows: int,
                    dim: int, causal: bool, window: int, segment: int, ring: int, int8: bool = False) -> UOp:
  scales, floor = _extras(extra, int8)
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
  def element_of(c: int) -> UOp:
    if scales is None: return kv[c, 0, kv_head, kv_index, column].load()
    value = _byte(kv[c, 0, kv_head, kv_index, column // 4].load(), column % 4).float() * scales[c, 0, kv_head, kv_index].load().float()
    return value.cast(dtypes.float16)
  stage = UOp.group(k_tile[column.valid(in_tile), key].store(element_of(0)),
                    v_tile[key.valid(in_tile), column].store(element_of(1))).end(loader).barrier()
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
_FLASH = os.environ.get('INFURNACE_FLASH_ATTENTION', '1') != '0'
FLASH_BLOCKS = 120  # _flash_kernel blocks to aim for

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
def _mma_kernel(part: UOp, stats: UOp, q: UOp, kv: UOp, *extra: UOp, start: int | UOp, kv_heads: int, groups: int, rows: int,
                dim: int, causal: bool, window: int, segment: int, ring: int, capacity: int, int8: bool = False) -> UOp:
  """Multirow attention on FP16 tensor cores (FlashAttention-2's layout, mma.sync m16n8k16): one warp per 16
  packed queries, S = Q·Kᵀ per 32-key tile, online softmax on the accumulator fragments (a row's values sit
  in four neighbouring lanes), then P·V with P taken straight from the score fragments, which are already
  in the A-operand layout. Same partial/stats outputs as _partial_kernel. An int8 cache's integers are exact in
  FP16: the MMAs take them as they are, and the scales multiply the scores (keys) and probabilities (values)."""
  scales, floor = _extras(extra, int8)
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
    if scales is None: row = _load_lanes(kv[1, 0, kv_head, v_index, v_col], 8)
    else:
      words = _load_lanes(kv[1, 0, kv_head, v_index, v_col // 4], 2)
      row = [_byte(words[i // 4], i % 4).cast(F16) for i in range(8)]
    stage = UOp.group(*(vt[(v_col + i).valid(in_tile), v_key].store(v_live.where(row[i], UOp.const(0.0, F16))) for i in range(8)))
    vt = vt.after(stage.end(loader).barrier())
  qv = [qa.after(tile)[i].load() for i in range(steps * 8)]
  computed = []
  for nt in range(4):
    _, index = slot_of(base + nt * 8 + group)  # past the active length: masked after the MMA
    acc = UOp.stack(zero, zero, zero, zero)
    for ks in range(steps):
      if scales is None:
        pairs = [_load_lanes(kv[0, 0, kv_head, index, ks * 16 + thread * 2 + c], 2) for c in (0, 8)]  # 4-byte loads
        b = UOp.stack(pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1])
      else:  # dims thread*2, +1 (and +8, +9) of the step: two bytes of one word each
        words = [kv[0, 0, kv_head, index, ks * 4 + c // 4 + thread // 2].load() for c in (0, 8)]
        b = UOp.stack(*(_byte(w, (thread % 2) * 2 + j).cast(F16) for w in words for j in (0, 1)))
      acc = UOp.wmma(UOp.stack(*qv[ks * 8:ks * 8 + 8]), b, acc, (8, 16, 16), 32)
    computed.append(scores[nt].after(scores[nt].store(acc)))
  scale = 1 / math.sqrt(dim)
  cur = state.after(tile)
  # An int8 cache: this lane's eight score columns' key and value scales (n-tile nt, column col at nt * 2 + col)
  key_scales, value_scales = [], []
  if scales is not None:
    for nt in range(4):
      for col in range(2):
        live, index = slot_of(base + nt * 8 + thread * 2 + col)
        key_scales.append(scales[0, 0, kv_head, index].load().float())
        value_scales.append(live.where(scales[1, 0, kv_head, index].load().float(), zero))
  values = [[None] * 8 for _ in range(2)]  # per row: 4 n-tiles x 2 columns
  for nt in range(4):
    for i in range(4):
      r, col = i // 2, i % 2
      key = base + nt * 8 + thread * 2 + col
      seen = key < active
      if floor is not None: seen = seen & (key >= floor[0].load().cast(key.dtype))
      visible = valid_m[r] & seen & _visible(key, start, row_of[r], rows, causal, window)
      score = computed[nt][i].load() * (scale * key_scales[nt * 2 + col] if key_scales else scale)
      values[r][nt * 2 + col] = visible.where(score, neg)
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
  weights = [[p * value_scales[n] for n, p in enumerate(probs[r])] if value_scales else probs[r] for r in range(2)]
  pa = [UOp.stack(*(weights[r][(2 * kk + h) * 2 + col].cast(F16) for h in (0, 1) for r in (0, 1) for col in (0, 1))) for kk in range(2)]
  for c in range(tiles_n):
    o = outs[c].after(tile)
    acc = UOp.stack(*(o[i].load() * alphas[i // 2] for i in range(4)))
    for kk in range(2):
      if stage_v: parts = [vt[c * 8 + group, kk * 16 + thread * 2 + j].load() for j in (0, 1, 8, 9)]
      else:
        parts = []
        for j in (0, 1, 8, 9):
          live, index = slot_of(base + kk * 16 + thread * 2 + j)
          value = (kv[1, 0, kv_head, index, c * 8 + group].load() if scales is None else
                   _byte(kv[1, 0, kv_head, index, c * 2 + group // 4].load(), group % 4).cast(F16))
          parts.append(live.where(value, UOp.const(0.0, F16)))
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
def _flash_kernel(part: UOp, stats: UOp, q: UOp, kv: UOp, *extra: UOp, start: int | UOp, kv_heads: int, groups: int, rows: int,
                  dim: int, causal: bool, capacity: int, parts: int, warps_m: int, int8: bool = False) -> UOp:
  """Multirow attention without a window, ring or floor (target verification, prompt chunks), FlashAttention-2 style:
  a block of warps_m x 2 warps takes up to 16 * warps_m packed queries of one KV head and one of `parts` slices of
  the active keys. Each 32-key tile is staged in shared memory once (keys row-major, values transposed; an int8
  cache dequantized there) from registers loaded during the previous tile, so global loads overlap the MMAs. Warp
  (wm, wn) scores its 16 queries against keys wn * 16.. of the tile; the pair shares its row maxima and
  probabilities through shared memory, and each accumulates output dims wn * dim / 2.. against all 32 keys, so no
  score is computed twice and a lane holds half a head's output. Fragments come from shared memory by ldmatrix
  (one instruction a B fragment). With an int8 cache it follows SageAttention2: the scores run on int8 tensor
  cores (keys as stored, Q quantized a row) and P V on FP8 ones, the values' integers as e4m3 against the
  probabilities times their key's scale, scaled so the tile's largest fills e4m3's range (~3% rounding each;
  Bonsai-4B perplexity +0.04% at 8192 tokens). Same partial/stats outputs as _mma_kernel, one per slice. RTX 5060,
  Bonsai-4B's heads at 64k keys, int8 cache: 16 rows 445 us against _mma_kernel's 1656 (616 in FP16, 492 with
  int8 scores), 64 rows 1897 against 5480 (2743, 2157), and the merge reads a few dozen partials instead of one a
  256-key segment (16 us, from up to 660)."""
  scales, _ = _extras(extra, int8)
  queries, steps, tiles_n = groups * rows, dim // 16, dim // 8
  half_tiles = tiles_n // 2
  warps, threads = warps_m * 2, warps_m * 64
  qblocks = -(-queries // (16 * warps_m))
  active = start + rows
  per = ((active + 31) // 32 + parts - 1) // parts  # tiles a slice
  block = UOp.range(kv_heads * parts * qblocks, 0, axis_type=AxisType.GLOBAL)
  warp = UOp.range(warps, 1, axis_type=AxisType.LOCAL)
  lane = UOp.range(32, -1, axis_type=AxisType.WARP)
  kv_head, rest = block // (parts * qblocks), block % (parts * qblocks)
  part_index, qblock = rest // qblocks, rest % qblocks
  wm, wn, tid = warp // 2, warp % 2, warp * 32 + lane
  group, thread = lane // 4, lane % 4
  F32, F16, U32 = dtypes.float32, dtypes.float16, dtypes.uint32
  zero, neg = UOp.const(0.0, F32), UOp.const(_NEG, F32)
  ms = [(qblock * warps_m + wm) * 16 + group + 8 * r for r in (0, 1)]
  valid_m = [m < queries for m in ms]
  heads, row_of = [kv_head * groups + m // rows for m in ms], [m % rows for m in ms]
  if int8:
    # SageAttention's INT8 Q/K (TensorRT-LLM's Blackwell recipe, one scale a token): the int8 cache's keys go to
    # int8 tensor cores as they are (4x the FP16-in, FP32-accumulate rate on an RTX 5060), against Q quantized here
    # with one scale a query row, kept as the m16n8k32 A fragments: (row g, k 4t..), (g+8, 4t..), (g, 16+4t..), (g+8, 16+4t..)
    words4 = [(r, off) for off in (0, 16) for r in (0, 1)]
    def q_value(r: int, d) -> UOp: return q[0, heads[r], row_of[r], d.valid(valid_m[r])].load().float()
    rowvals = [[q_value(r, s32 * 32 + off + thread * 4 + j) for s32 in range(dim // 32) for off in (0, 16) for j in range(4)] for r in (0, 1)]
    q_scale = []
    for r in range(2):
      top = rowvals[r][0].abs()
      for v in rowvals[r][1:]: top = top.maximum(v.abs())
      q_scale.append((_shuffle_rows(top, True) / 127).maximum(1e-30))
    qw = UOp.placeholder((dim // 32 * 4,), U32, slot=0, addrspace=AddrSpace.REG)
    def word(r: int, s32: int, off: int) -> UOp:
      codes = [(q_value(r, s32 * 32 + off + thread * 4 + j) / q_scale[r]).round().clip(-127, 127) for j in range(4)]
      return sum((c.cast(dtypes.int8).cast(dtypes.uint8).cast(U32) << (8 * j) for j, c in enumerate(codes)), UOp.const(0, U32))
    qs = UOp.placeholder((2,), F32, slot=9 + half_tiles, addrspace=AddrSpace.REG)
    qw = qw.after(UOp.group(*(qw[s32 * 4 + n].store(word(r, s32, off)) for s32 in range(dim // 32) for n, (r, off) in enumerate(words4)),
                            *(qs[r].store(q_scale[r]) for r in range(2))))
    qs = qs.after(qw)
  else:
    qa = UOp.placeholder((steps * 8,), F16, slot=0, addrspace=AddrSpace.REG)
    slots = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 8), (0, 9), (1, 8), (1, 9)]
    qa = qa.after(UOp.group(*(qa[ks * 8 + i].store(
      q[0, heads[r], row_of[r], (ks * 16 + thread * 2 + c).valid(valid_m[r])].load().cast(F16))
      for ks in range(steps) for i, (r, c) in enumerate(slots))))
  # max and this warp's half of the sum for rows g, g + 8; output fragments in their own 4-slot accumulators
  state = UOp.placeholder((4,), F32, slot=1, addrspace=AddrSpace.REG)
  state = state.after(UOp.group(state[0].store(neg), state[1].store(neg), state[2].store(zero), state[3].store(zero)))
  outs = []
  for c in range(half_tiles):
    o = UOp.placeholder((4,), F32, slot=2 + c, addrspace=AddrSpace.REG)
    outs.append(o.after(o.store(o.const_like(0))))
  slot = 2 + half_tiles
  scores = [UOp.placeholder((4,), F32, slot=slot + nt, addrspace=AddrSpace.REG) for nt in range(2)]
  # This thread's 16-byte chunks of a tile: keys row-major (K) and key-minor (V, staged transposed: neighbouring
  # lanes take neighbouring keys, no bank conflicts), held in registers from one tile to the next.
  width = 16 if int8 else 8  # values a 16-byte load
  chunks = 32 * dim // width
  count = -(-chunks // threads)
  lanes = 4 if int8 else 8
  held = UOp.placeholder((2 * count * lanes,), U32 if int8 else F16, slot=slot + 2, addrspace=AddrSpace.REG)
  held_scales = UOp.placeholder((2 * count,), F32, slot=slot + 3, addrspace=AddrSpace.REG) if int8 else None
  def place(c: int, i: int) -> tuple[UOp, UOp, UOp]:
    chunk = i * threads + tid
    key, col = (chunk // (dim // width), (chunk % (dim // width)) * width) if c == 0 else (chunk % 32, (chunk // 32) * width)
    return chunk < chunks, key, col
  def fetch(t: UOp) -> list[UOp]:
    """Stores of tile t's chunks into the held registers."""
    stores = []
    for c in range(2):
      for i in range(count):
        inside, key, col = place(c, i)
        index = ((part_index * per + t) * 32 + key).minimum(capacity - 1).valid(inside)
        n = c * count + i
        if int8:
          words = _load_lanes(kv[c, 0, kv_head, index, col // 4], 4)
          stores += [held[n * 4 + w].store(words[w]) for w in range(4)]
          stores.append(held_scales[n].store(scales[c, 0, kv_head, index].load().float()))
        else:
          values = _load_lanes(kv[c, 0, kv_head, index, col], 8)
          stores += [held[n * 8 + w].store(values[w]) for w in range(8)]
    return stores
  held = held.after(UOp.group(*fetch(lane.const_like(0))))
  if int8: held_scales = held_scales.after(held)
  # Padded rows: a row of keys is 4 banks off the next (K), a dim of values 20 words off the next (V transposed),
  # a row of probabilities 20 words off the next, so the lanes of a fragment load hit 32 different banks.
  # an int8 tile's keys stay int8 (rows of dim + 16 bytes, as words), with their scales
  k_lds = (UOp.placeholder((32, (dim + 16) // 4), U32, slot=slot + 4, addrspace=AddrSpace.LOCAL) if int8 else
           UOp.placeholder((32, dim + 8), F16, slot=slot + 4, addrspace=AddrSpace.LOCAL))
  k_scales = UOp.placeholder((32,), F32, slot=slot + 8, addrspace=AddrSpace.LOCAL) if int8 else None
  if int8:  # FP8 values (transposed, 48-byte rows: 16-byte aligned, ldmatrix rows 12 banks apart) and probabilities
    v_lds = UOp.placeholder((dim, 48), dtypes.uint8, slot=slot + 5, addrspace=AddrSpace.LOCAL)
    p_lds = UOp.placeholder((warps_m, 16, 48), dtypes.uint8, slot=slot + 6, addrspace=AddrSpace.LOCAL)
    v_scales = UOp.placeholder((32,), F32, slot=slot + 9, addrspace=AddrSpace.LOCAL)
  else:
    v_lds = UOp.placeholder((dim, 32 + 8), F16, slot=slot + 5, addrspace=AddrSpace.LOCAL)
    p_lds = UOp.placeholder((warps_m, 16, 32 + 8), F16, slot=slot + 6, addrspace=AddrSpace.LOCAL)
  m_lds = UOp.placeholder((warps_m, 2, 16), F32, slot=slot + 7, addrspace=AddrSpace.LOCAL)
  tile = UOp.range(per, 2, axis_type=AxisType.LOOP)
  base = (part_index * per + tile) * 32
  cur_held = held.after(tile)
  cur_scales = held_scales.after(tile) if int8 else None
  stage = []
  for c in range(2):
    for i in range(count):
      inside, key, col = place(c, i)
      n = c * count + i
      if int8 and c == 0:  # keys as they are: past the active length they're masked out of the scores
        stage += [k_lds[key.valid(inside), col // 4 + w].store(cur_held[n * 4 + w].load()) for w in range(4)]
        stage.append(k_scales[key.valid(inside & col.eq(0))].store(cur_scales[n].load()))
        continue
      live = base + key < active
      if int8:  # values as FP8 integers, their scales apart (folded into the probabilities)
        for w in range(4):
          fp8 = _int8_e4m3(cur_held[n * 4 + w].load())
          stage += [v_lds[(col + 4 * w + j).valid(inside), key].store(((fp8 >> (8 * j)) & 255).cast(dtypes.uint8)) for j in range(4)]
        stage.append(v_scales[key.valid(inside & col.eq(0))].store(live.where(cur_scales[n].load(), zero)))
        continue
      values = [cur_held[n * 8 + w].load() for w in range(8)]
      live = base + key < active  # unwritten rows may hold any bits: zero keys and values past the active length
      values = [live.where(v, UOp.const(0.0, F16)) for v in values]
      stage += [(k_lds[key.valid(inside), col + w] if c == 0 else v_lds[(col + w).valid(inside), key]).store(v) for w, v in enumerate(values)]
  staged = UOp.group(*stage).barrier()
  k_lds, v_lds = k_lds.after(staged), v_lds.after(staged)
  if int8:
    k_scales, v_scales = k_scales.after(staged), v_scales.after(staged)
    # the tile's largest value scale: probabilities times their key's scale, over it, fill e4m3's range (448)
    v_top = warp_reduce(v_scales[lane].load(), maximum=True, full_wave=True).maximum(1e-30)
  # The next tile's loads, issued now: they land while this one's MMAs run
  prefetch = UOp.group(*fetch(tile + 1))
  computed = []
  if int8:
    from infurnace.kernels.quant_gemm import _imma
    qv, izero = [qw.after(tile)[i].load() for i in range(dim // 32 * 4)], UOp.const(0, dtypes.int32)
    for nt in range(2):
      acc = None
      for s32 in range(dim // 32):  # 32 dims a step: 16 bytes of the 8 keys' rows (+16 for the second tile)
        # each step's MMA from zero, its sums added after: chained through the accumulator, every inline MMA
        # expression would repeat the previous four (8 steps: a source of gigabytes)
        d = _imma(qv[s32 * 4:s32 * 4 + 4], _ldmatrix_words(k_lds[wn * 16 + nt * 8 + lane % 8, s32 * 8 + (lane // 8) % 2 * 4]), [izero] * 4)
        acc = d if acc is None else [a + b for a, b in zip(acc, d)]
      computed.append(scores[nt].after(scores[nt].store(UOp.stack(*(a.float() for a in acc)))))
  else:
    qv = [qa.after(tile)[i].load() for i in range(steps * 8)]
    for nt in range(2):
      acc = UOp.stack(zero, zero, zero, zero)
      for ks in range(steps):
        b = _ldmatrix(k_lds[wn * 16 + nt * 8 + lane % 8, ks * 16 + (lane // 8) % 2 * 8])
        acc = UOp.wmma(UOp.stack(*qv[ks * 8:ks * 8 + 8]), UOp.stack(*b), acc, (8, 16, 16), 32)
      computed.append(scores[nt].after(scores[nt].store(acc)))
  scale = 1 / math.sqrt(dim)
  cur = state.after(tile)
  values = [[None] * 4 for _ in range(2)]  # per row: 2 n-tiles x 2 columns
  for nt in range(2):
    for i in range(4):
      r, col = i // 2, i % 2
      key = base + wn * 16 + nt * 8 + thread * 2 + col
      visible = valid_m[r] & (key < active) & _visible(key, start, row_of[r], rows, causal, 0)
      factor = qs[r].load() * k_scales[wn * 16 + nt * 8 + thread * 2 + col].load() * scale if int8 else scale
      values[r][nt * 2 + col] = visible.where(computed[nt][i].load() * factor, neg)
  # The pair's row maxima through shared memory
  mine = []
  for r in range(2):
    top = values[r][0]
    for v in values[r][1:]: top = top.maximum(v)
    mine.append(_shuffle_rows(top, True))
  shared = UOp.group(*(m_lds[wm, wn, (group + 8 * r).valid(thread.eq(0))].store(mine[r]) for r in range(2))).barrier()
  m_lds_now = m_lds.after(shared)
  updates, alphas, probs = [], [], []
  for r in range(2):
    old = cur[r].load()
    new_max = old.maximum(mine[r].maximum(m_lds_now[wm, 1 - wn, group + 8 * r].load()))
    safe = new_max.ne(neg).where(new_max, zero)        # all masked so far: keep exp() finite
    alpha = (old - safe).exp()
    probs.append([(v - safe).exp() for v in values[r]])
    total = probs[r][0]
    for p in probs[r][1:]: total = total + p
    updates += [cur[r].store(new_max), cur[2 + r].store(cur[2 + r].load() * alpha + _shuffle_rows(total, False))]
    alphas.append(alpha)
  # This warp's probabilities into the pair's shared rows; both read all 32 keys' as A fragments
  if int8:
    written = []
    for r in range(2):
      for nt in range(2):
        key = wn * 16 + nt * 8 + thread * 2
        pair = UOp(Ops.CUSTOMI, src=tuple(probs[r][nt * 2 + col] * v_scales[key + col].load() * (448 / v_top) for col in range(2)),
                   arg=(_E4M3_F32X2, U32))
        written += [p_lds[wm, group + 8 * r, key + col].store(((pair >> (8 * col)) & 255).cast(dtypes.uint8)) for col in range(2)]
    p_now = p_lds.after(UOp.group(*written).barrier())
    # A fragment (k32): rows 0-7 and 8-15 at keys 0-15, then at 16-31
    pa = [w for c in (0, 16) for w in _ldmatrix_words(p_now[wm, lane % 8 + (lane // 8) % 2 * 8, c])]
    for c in range(half_tiles):
      o = outs[c].after(tile)
      d = (wn * half_tiles + c) * 8 + lane % 8
      pv = _fmma(pa, _ldmatrix_words(v_lds[d, (lane // 8) % 2 * 16]))
      updates.append(o.store(UOp.stack(*(o[i].load() * alphas[i // 2] + pv[i] * (v_top / 448) for i in range(4)))))
  else:
    written = UOp.group(*(p_lds[wm, group + 8 * r, wn * 16 + nt * 8 + thread * 2 + col].store(probs[r][nt * 2 + col].cast(F16))
                          for r in range(2) for nt in range(2) for col in range(2))).barrier()
    p_now = p_lds.after(written)
    # A fragments: (rows 0-7, 8-15) at keys kk * 16 + 0-7, then at + 8-15
    pa = [UOp.stack(*(v for c in (0, 8) for v in _ldmatrix(p_now[wm, lane % 8 + (lane // 8) % 2 * 8, kk * 16 + c]))) for kk in range(2)]
    for c in range(half_tiles):
      o = outs[c].after(tile)
      acc = UOp.stack(*(o[i].load() * alphas[i // 2] for i in range(4)))
      d = (wn * half_tiles + c) * 8 + lane % 8
      for kk in range(2):
        acc = UOp.wmma(pa[kk], UOp.stack(*_ldmatrix(v_lds[d, kk * 16 + (lane // 8) % 2 * 8])), acc, (8, 16, 16), 32)
      updates.append(o.store(acc))
  # Every warp must finish with the tile before the next one is staged over it.
  done = UOp.group(*updates, prefetch).barrier().end(tile)
  state, outs = state.after(done), [o.after(done) for o in outs]
  # The pair's halves of each row's sum
  sums = UOp.group(*(m_lds.after(done)[wm, wn, (group + 8 * r).valid(thread.eq(0))].store(state[2 + r].load()) for r in range(2))).barrier()
  m_lds = m_lds.after(sums)
  totals = [m_lds[wm, 0, group + 8 * r].load() + m_lds[wm, 1, group + 8 * r].load() for r in range(2)]
  stores = [part[kv_head, part_index, ms[i // 2].valid(valid_m[i // 2]), (wn * half_tiles + c) * 8 + thread * 2 + i % 2].store(outs[c][i].load())
            for c in range(half_tiles) for i in range(4)]
  stores += [stats[kv_head, part_index, ms[r].valid(valid_m[r] & thread.eq(0) & wn.eq(0)), s].store(state[r].load() if s == 0 else totals[r])
             for r in range(2) for s in range(2)]
  return UOp.group(*stores).end(block, warp, lane).sink(arg=KernelInfo(name=f"attention_flash_d{dim}", opts_to_apply=()))

@functools.cache
def _decode_kernel(part: UOp, stats: UOp, q: UOp, kv: UOp, *extra: UOp, start: int | UOp, kv_heads: int, groups: int, dim: int,
                   capacity: int, segment: int, tile: int, warps: int, int8: bool = False) -> UOp:
  """One query row (every grouped head of a KV head) after flash-decoding: lane owns dim/32 consecutive
  dimensions and reads them with one vector load per key, scores are summed across the warp by shuffles, and
  the online softmax lives in registers (no shared memory or barriers). Each warp writes one partial. An int8
  cache (dim a multiple of 128: whole words a lane) scales each key's score and each value's probability."""
  scales, _ = _extras(extra, int8)
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
  scores, values, value_scales = [[] for _ in range(groups)], [], []
  for u in range(tile):
    key = entry * segment + t * tile + u
    live = key < active
    row = key.minimum(capacity - 1)  # past the active length: in bounds, masked below
    if scales is None:
      k, v = (_load_lanes(kv[c, 0, kv_head, row, lane * lanes_dim], lanes_dim) for c in (0, 1))
      k, v = [x.float() for x in k], [x.float() for x in v]
    else:
      words = [_load_lanes(kv[c, 0, kv_head, row, lane * lanes_dim // 4], lanes_dim // 4) if lanes_dim > 4 else
               [kv[c, 0, kv_head, row, lane].load()] for c in (0, 1)]
      k, v = ([_byte(w[i // 4], i % 4).float() for i in range(lanes_dim)] for w in words)
      value_scales.append(live.where(scales[1, 0, kv_head, row].load().float(), zero))
    # Unwritten cache rows may hold any bits: zero their values so a 0 weight can't meet a NaN.
    values.append([live.where(v[i], zero) for i in range(lanes_dim)])
    for j in range(groups):
      dot = qv[j * lanes_dim] * k[0]
      for i in range(1, lanes_dim): dot = dot + qv[j * lanes_dim + i] * k[i]
      score = warp_reduce(dot, full_wave=True)
      if scales is not None: score = score * scales[0, 0, kv_head, row].load().float()
      scores[j].append(live.where(score, neg))
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
    weights = [p * m for p, m in zip(ps, value_scales)] if value_scales else ps
    for i in range(lanes_dim):
      slot = 2 * groups + j * lanes_dim + i
      acc = cur[slot].load() * alpha
      for p, v in zip(weights, values): acc = acc + p * v[i]
      updates.append(cur[slot].store(acc))
  state = state.after(UOp.group(*updates).end(t))
  stores = [part[kv_head, entry, j, lane * lanes_dim + i].store(state[2 * groups + j * lanes_dim + i].load())
            for j in range(groups) for i in range(lanes_dim)]
  stores += [stats[kv_head, entry, UOp.const(j, lane.dtype).valid(lane.eq(0)), s].store(state[s * groups + j].load())
             for j in range(groups) for s in range(2)]  # lane 0 only
  return UOp.group(*stores).end(block, warp, lane).sink(arg=KernelInfo(name=f"attention_decode_d{dim}", opts_to_apply=()))

@functools.cache
def _merge_kernel(out: UOp, part: UOp, stats: UOp, *, start: int | UOp, kv_heads: int, groups: int, rows: int, dim: int,
                  window: int, segment: int, parts: int = 0) -> UOp:
  """Adds the partials of each query; `parts`: a fixed count of them (_flash_kernel's slices) instead of one a segment."""
  queries, lanes_dim = groups * rows, dim // 32
  live = parts or _live(start, rows, segment) - _first(start, window, segment)
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
              ring: bool = False, floor: Tensor | None = None, scales: Tensor | None = None) -> Tensor:
  """q: (1, heads, rows, dim) at positions start..start+rows-1; kv: (2, 1, kv_heads, capacity, dim) FP16
  cache already holding those rows (a ring of length capacity with `ring`), or with `scales` an int8 cache
  (store_kv: (2, 1, kv_heads, capacity, dim / 4) words and (2, 1, kv_heads, capacity) scales). floor: optional
  (1,) int32 buffer, the lowest visible key. Returns (1, heads, rows, dim) FP32. dim must be a multiple of 32.
  The fused kernels need NV (CUDA warp shuffles); other devices use plain tensor ops over the whole cache.
  One row without a window, ring or floor (decoding) uses the flash-decoding kernel, which splits the keys
  itself (DECODE_SEGMENT per warp); `segment` applies to the other shapes."""
  _, heads, rows, dim = q.shape
  _, _, kv_heads, capacity, kv_dim = kv.shape
  int8 = scales is not None
  if int8 and (kv.dtype != dtypes.uint32 or scales.shape != kv.shape[:4]): raise ValueError("an int8 cache is store_kv's words and scales")
  if dim != kv_dim * (4 if int8 else 1) or heads % kv_heads: raise ValueError("attention needs matching head sizes and grouped heads")
  if ring and not 0 < window <= capacity - rows: raise ValueError("a ring cache needs a window and room for it plus the rows")
  if not nvidia(q.device) or dim % 32 or not (int8 or kv.dtype == dtypes.float16):
    return _masked(q, dequantize_kv(kv, scales) if int8 else kv, start, causal, window, ring, floor)
  groups, scalar = heads // kv_heads, _kernel_scalar(start)
  extra = ((scales,) if int8 else ()) + ((floor,) if floor is not None else ())
  if rows == 1 and not window and not ring and floor is None and dim // 32 in ((4, 8) if int8 else (2, 4, 8)):
    seg = DECODE_SEGMENT
    entries = -(-capacity // (seg * DECODE_WARPS)) * DECODE_WARPS
    part = Tensor.empty(kv_heads, entries, groups, dim, dtype=dtypes.float32, device=q.device)
    stats = Tensor.empty(kv_heads, entries, groups, 2, dtype=dtypes.float32, device=q.device)
    part, stats = _launch((part, stats), (q.contiguous(), kv, *extra), start, functools.partial(
      _decode_kernel, start=scalar, kv_heads=kv_heads, groups=groups, dim=dim, capacity=capacity, segment=seg,
      tile=DECODE_TILE, warps=DECODE_WARPS, int8=int8))
    out = Tensor.empty(1, heads, 1, dim, dtype=dtypes.float32, device=q.device)
    return _launch((out,), (part, stats), start, functools.partial(
      _merge_kernel, start=scalar, kv_heads=kv_heads, groups=groups, rows=1, dim=dim, window=0, segment=seg))[0]
  if rows > 1 and not window and not ring and floor is None and dim % 32 == 0 and _FLASH and _MMA and _tensor_cores(q.device):
    queries = groups * rows
    warps_m = min(-(-queries // 16), 4)
    qblocks = -(-queries // (16 * warps_m))
    # About four blocks an SM (an RTX 5060 has 30) between the KV heads, query blocks and key slices
    parts = max(1, round(FLASH_BLOCKS / (kv_heads * qblocks)))
    part = Tensor.empty(kv_heads, parts, queries, dim, dtype=dtypes.float32, device=q.device)
    stats = Tensor.empty(kv_heads, parts, queries, 2, dtype=dtypes.float32, device=q.device)
    part, stats = _launch((part, stats), (q.contiguous(), kv, *extra), start, functools.partial(
      _flash_kernel, start=scalar, kv_heads=kv_heads, groups=groups, rows=rows, dim=dim, causal=causal, capacity=capacity,
      parts=parts, warps_m=warps_m, int8=int8))
    out = Tensor.empty(1, heads, rows, dim, dtype=dtypes.float32, device=q.device)
    return _launch((out,), (part, stats), start, functools.partial(
      _merge_kernel, start=scalar, kv_heads=kv_heads, groups=groups, rows=rows, dim=dim, window=0, segment=segment, parts=parts))[0]
  # A ring's window can straddle one more segment boundary than its length spans.
  if rows > 16 and not (dim % 16 == 0 and segment % 32 == 0 and _MMA and _tensor_cores(q.device)):
    # The FP32 kernel stages a probability row per query: past 16 rows its shared memory runs out. Slices of
    # up to 16 rows are the same attention at later starting positions.
    return Tensor.cat(*(attention(q[:, :, i:i + 16], kv, start + i, causal=causal, window=window, segment=segment, ring=ring,
                                  floor=floor, scales=scales) for i in range(0, rows, 16)), dim=2)
  segments = -(-capacity // segment) + int(ring)
  part = Tensor.empty(kv_heads, segments, groups * rows, dim, dtype=dtypes.float32, device=q.device)
  stats = Tensor.empty(kv_heads, segments, groups * rows, 2, dtype=dtypes.float32, device=q.device)
  if rows > 1 and dim % 16 == 0 and segment % 32 == 0 and _MMA and _tensor_cores(q.device):
    part, stats = _launch((part, stats), (q.contiguous(), kv, *extra), start, functools.partial(
      _mma_kernel, start=scalar, kv_heads=kv_heads, groups=groups, rows=rows, dim=dim, causal=causal, window=window,
      segment=segment, ring=capacity if ring else 0, capacity=capacity, int8=int8))
    out = Tensor.empty(1, heads, rows, dim, dtype=dtypes.float32, device=q.device)
    return _launch((out,), (part, stats), start, functools.partial(
      _merge_kernel, start=scalar, kv_heads=kv_heads, groups=groups, rows=rows, dim=dim, window=window, segment=segment))[0]
  part, stats = _launch((part, stats), (q.contiguous(), kv, *extra), start, functools.partial(
    _partial_kernel, start=scalar, kv_heads=kv_heads, groups=groups, rows=rows, dim=dim, causal=causal, window=window, segment=segment,
    ring=capacity if ring else 0, int8=int8))
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

def quantize_kv(values: Tensor) -> tuple[Tensor, Tensor]:
  """(..., dim) keys or values as int8, one FP16 scale (max |x| / 127) a vector: (words (..., dim / 4) uint32, scales (...))."""
  x = values.float()
  scale = (x.abs().max(-1) / 127).cast(dtypes.float16)
  codes = (x / scale.float().maximum(1e-30).unsqueeze(-1)).round().clip(-127, 127).cast(dtypes.int8)
  return codes.bitcast(dtypes.uint32), scale

def dequantize_kv(kv: Tensor, scales: Tensor) -> Tensor:
  """An int8 cache (store_kv's words and scales) as FP16."""
  return (kv.bitcast(dtypes.int8).float() * scales.float().unsqueeze(-1)).cast(dtypes.float16)

def store_kv(cache: Tensor, scales: Tensor, k: Tensor, v: Tensor, start: int | UOp) -> tuple[Tensor, Tensor]:
  """Write keys and values (1, kv_heads, rows, dim) as positions start..start+rows-1 of an int8 cache: words
  (2, 1, kv_heads, capacity, dim / 4) uint32 and scales (2, 1, kv_heads, capacity) FP16. Returns both after the write."""
  words, scale = quantize_kv(Tensor.stack(k, v))
  rows = k.shape[2]
  return (Tensor(cache.uop.after(cache[:, :, :, start:start + rows].uop.store(words.uop))),
          Tensor(scales.uop.after(scales[:, :, :, start:start + rows].uop.store(scale.uop))))

@functools.cache
def _prologue_kernel(q_out: UOp, cache: UOp, scales: UOp, joined: UOp, q_w: UOp, k_w: UOp, freqs: UOp, *, start: int | UOp,
                     heads: int, kv_heads: int, dim: int, rope_dim: int, eps: float) -> UOp:
  """A decoding step's attention inputs in one kernel (TensorRT-LLM's fused QK-norm/RoPE/KV-insert): a block per
  head of the merged q (+ output gate) / k / v projection's row, `dim` threads. Q and K heads RMS-normalize (their
  own weights) and rotate their first rope_dim dims (rotate-half, freqs[start]); Q heads write FP32 q, K and V heads
  quantize as store_kv (int8, one FP16 scale a vector) into the cache at `start`. In place of ~8 small kernels."""
  warps, half = dim // 32, rope_dim // 2
  block = UOp.range(heads + 2 * kv_heads, 0, axis_type=AxisType.GLOBAL)
  warp = UOp.range(warps, 1, axis_type=AxisType.LOCAL)
  lane = UOp.range(32, -1, axis_type=AxisType.WARP)
  t = warp * 32 + lane
  F32 = dtypes.float32
  is_q, is_v = block < heads, block >= heads + kv_heads
  k_head, v_head = block - heads, block - heads - kv_heads
  base = heads * 2 * dim
  source = is_q.where(block * 2 * dim, is_v.where(base + kv_heads * dim + v_head * dim, base + k_head * dim)) + t
  x = joined[source].load().float()
  def block_sum(value: UOp, slot: int, maximum: bool = False) -> UOp:
    partial = warp_reduce(value, maximum=maximum, full_wave=True)
    buf = UOp.placeholder((warps,), F32, slot=slot, addrspace=AddrSpace.LOCAL)
    buf = buf.after(buf[warp.valid(lane.eq(0))].store(partial).barrier())
    return functools.reduce(lambda a, b: a.maximum(b) if maximum else a + b, (buf[i].load() for i in range(warps)))
  inverse = (block_sum(x * x, 0) * (1 / dim) + eps).rsqrt()
  y = is_v.where(x, x * inverse * is_q.where(q_w[t].load().float(), k_w[t].load().float()))
  shared = UOp.placeholder((dim,), F32, slot=1, addrspace=AddrSpace.LOCAL)
  shared = shared.after(shared[t].store(y).barrier())
  low = t < half
  partner = shared[low.where(t + half, t - half).minimum(dim - 1)].load()
  cos, sin = freqs[start, t % half].load(), freqs[start, half + t % half].load()
  z = (~is_v & (t < rope_dim)).where(low.where(y * cos - partner * sin, y * cos + partner * sin), y)
  stores = [q_out[block.valid(is_q), t].store(z)]
  # K and V: int8 with one FP16 scale, rounded as store_kv rounds
  scale = (block_sum(z.abs(), 2, maximum=True) / 127).cast(dtypes.float16)
  code = (z / scale.float().maximum(1e-30)).round().clip(-127, 127).cast(dtypes.int32)
  def down(value: UOp, n: int) -> UOp: return UOp(Ops.CUSTOM, src=(value,), arg=(f"__shfl_down_sync(0xffffffff, {{0}}, {n})", dtypes.int32))
  word = functools.reduce(lambda a, b: a | b, [((down(code, n) if n else code).cast(dtypes.uint32) & 255) << (8 * n) for n in range(4)])
  which, head = is_v.cast(dtypes.int32), is_v.where(v_head, k_head)
  stores.append(cache[which, 0, head.valid(~is_q & (lane % 4).eq(0)), start, t // 4].store(word))
  stores.append(scales[which, 0, head.valid(~is_q & t.eq(0)), start].store(scale))
  return UOp.group(*stores).end(block, warp, lane).sink(arg=KernelInfo(name=f"attention_prologue_d{dim}", opts_to_apply=()))

def attention_prologue(joined: Tensor, q_weight: Tensor, k_weight: Tensor, freqs: Tensor, cache: Tensor, scales: Tensor, start: int | UOp, *,
                       heads: int, kv_heads: int, dim: int, rope_dim: int, eps: float) -> tuple[Tensor, Tensor, Tensor]:
  """One decoding row's q (heads, dim) FP32, normalized and rotated, and its K/V written into the int8 cache at
  `start`, from the merged projection's output `joined` (heads x [q, gate], then K, then V; _prologue_kernel).
  Returns (q, cache, scales) after the write."""
  if dim % 32 or rope_dim % 2 or rope_dim > dim or joined.numel() != (heads * 2 + kv_heads * 2) * dim:
    raise ValueError("attention_prologue needs gated q, 32-aligned heads and an even rotary width")
  q = Tensor.empty(heads, dim, dtype=dtypes.float32, device=joined.device)
  return _launch((q, cache, scales), (joined.reshape(-1).contiguous(), q_weight, k_weight, freqs), start, functools.partial(
    _prologue_kernel, start=_kernel_scalar(start), heads=heads, kv_heads=kv_heads, dim=dim, rope_dim=rope_dim, eps=float(eps)))
