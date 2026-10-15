"""First NV packed prefill GEMM: m16n8k16 FP16 tensor cores, FP32 accumulation.

Weights are decoded into warp fragments, never a persistent dense matrix.
Activations and decoded weights round to FP16; this differs from FP32-activation
fallback math and Q8-activation decode. No shared-memory staging/autotuning yet.
"""
from __future__ import annotations
import functools
import math
import os

from tinygrad import Context, Device, Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

from infurnace.kernels.quant_gemv import Q4_K, Q5_K, Q6_K, Q6_PREPARED_BYTES, QUANT_SIZES, _half, _load_lanes, _q4_scales, _q4_word
from infurnace.device import nvidia

class QuantGemmError(ValueError):
  """Inputs/backend do not satisfy the first prefill kernel contract."""

def _q6_byte(raw: UOp, base: UOp, offset: UOp) -> UOp:
  size = raw.dtype.itemsize
  index = base+offset//size
  return (raw[index].load().cast(dtypes.uint32) >> ((offset%size)*8).cast(dtypes.uint32)) & 255

def _weight(raw: UOp, row: UOp, col: UOp, width: int, typ: int, q6_aligned: bool = False,
            q6_tiled: bool = False, q4_tiled: bool = False) -> UOp:
  block, within = col//256, col%256
  if typ == Q4_K:
    base = q4_block_base(row, block, width) if q4_tiled else (row*(width//256)+block)*36
    subgroup = within//32
    word = _q4_word(raw, base, 4+(subgroup//2)*8+(within%32)//4)
    quant = (word >> ((subgroup%2)*4+(within%4)*8).cast(dtypes.uint32)) & 15
    d, dmin, scale, minimum = _q4_scales(raw, base, subgroup)
    return (d*scale*quant.float()-dmin*minimum).cast(dtypes.float16)
  if typ == Q5_K:
    # 44 words: d/dmin, 12 scale bytes, 32 high-bit bytes (bit `subgroup` of
    # byte j), then Q4_K-ordered low nibbles.
    base = q4_block_base(row, block, width, 44) if q4_tiled else (row*(width//256)+block)*44
    subgroup, j = within//32, within%32
    low = (raw[base+12+(subgroup//2)*8+j//4].load() >> ((subgroup%2)*4+(j%4)*8).cast(dtypes.uint32)) & 15
    high = (raw[base+4+j//4].load() >> ((j%4)*8+subgroup).cast(dtypes.uint32)) & 1
    d, dmin, scale, minimum = _q4_scales(raw, base, subgroup)
    return (d*scale*(low | (high << 4)).float()-dmin*minimum).cast(dtypes.float16)
  logical = row*(width//256)+block
  base = q6_block_base(row, block, width) if q6_tiled else logical*(Q6_PREPARED_BYTES//4 if q6_aligned else 105)
  half, pos = within//128, within%128
  low = _q6_byte(raw, base, half*64+pos%64) >> ((pos//64)*4).cast(dtypes.uint32)
  high = _q6_byte(raw, base, 128+half*32+pos%32) >> ((pos//32)*2).cast(dtypes.uint32)
  quant = ((low & 15) | ((high & 3) << 4)).cast(dtypes.int32)-32
  scale = _q6_byte(raw, base, 192+within//16).cast(dtypes.uint8).bitcast(dtypes.int8).float()
  d_index = base+(52 if q6_aligned or q6_tiled else 104)
  return (_half(raw[d_index].load())*quant.float()*scale).cast(dtypes.float16)

@functools.cache
def _prefill_q4_paired(out: UOp, raw: UOp, x: UOp, *, tokens: int, width: int, rows: int,
                       tiles: int = 4, splits: int = 1, typ: int = Q4_K) -> UOp:
  """Q4_K/Q5_K MMA over a permuted k order that matches the packed nibble layout.

  A 64-column subgroup pair is 32 qs bytes: low nibbles are subgroup 2p, high
  nibbles subgroup 2p+1. Lane t of a row owns bytes 8t..8t+7 (one 8-byte load,
  each nibble used once), which fill its fragment slots {2t,2t+1,2t+8,2t+9} in
  four k-steps. The reduction is order-independent, so activations load the same
  permuted columns: c = 64p + 32*(s>=2) + 8t + 4*(s%2) + i for slot i.
  Q5_K adds bit (subgroup) of qh byte j as the fifth bit; lane t's qh bytes are
  the same 8t..8t+7 for every pair, so they load once per block.
  """
  q5 = typ == Q5_K
  warps = math.gcd(rows//(8*tiles), 4)
  output_blocks = rows//(warps*8*tiles)
  blocks = width//256//splits
  block = UOp.range((tokens//16)*output_blocks*splits, 0, axis_type=AxisType.GLOBAL)
  lane = UOp.range(32, -1, axis_type=AxisType.WARP)
  warp = UOp.range(warps, 1, axis_type=AxisType.LOCAL)
  split, rest = block%splits, block//splits
  token_base = (rest//output_blocks)*16
  output_base = (rest%output_blocks)*warps*8*tiles+warp*8*tiles
  group, thread = lane//4, lane%4
  accs = []
  for t in range(tiles):
    acc = UOp.placeholder((4,), dtypes.float32, slot=3+t, addrspace=AddrSpace.REG)
    accs.append(acc.after(acc.store(acc.const_like(0))))
  kb = UOp.range(blocks, 2, axis_type=getattr(AxisType, 'REDUCE', AxisType.LOOP))
  column_block = split*blocks+kb
  values = [acc.after(kb) for acc in accs]
  for pair in range(4):
    # Per tile: scale/min for both subgroups of the pair and this lane's two qs words.
    decoded = []
    for t in range(tiles):
      logical = (output_base+t*8+group)*(width//256)+column_block
      base = logical*(44 if q5 else 36)
      low = _q4_scales(raw, base, UOp.const(2*pair))
      high = _q4_scales(raw, base, UOp.const(2*pair+1))
      words = [_q4_word(raw, base, (12 if q5 else 4)+8*pair+2*thread+w) for w in (0, 1)]
      highs = [raw[base+4+2*thread+w].load() for w in (0, 1)] if q5 else None
      decoded.append((low, high, words, highs))
    for step in range(4):
      col0 = column_block*256+64*pair+32*(step//2)+8*thread+4*(step%2)
      cols = [col0+i for i in range(4)]
      a = UOp.stack(*(x[token_base+group+row, c].load() for (row, c) in
                      ((0, cols[0]), (0, cols[1]), (8, cols[0]), (8, cols[1]), (0, cols[2]), (0, cols[3]), (8, cols[2]), (8, cols[3]))))
      for t, (low, high, words, highs) in enumerate(decoded):
        d, dmin, scale, minimum = high if step >= 2 else low
        word, subgroup = words[step%2], 2*pair+step//2
        def quant(i):
          q = (word >> (8*i+4*(step//2))) & 15
          return q | (((highs[step%2] >> (8*i+subgroup)) & 1) << 4) if highs is not None else q
        b = UOp.stack(*((d*scale*quant(i).float()-dmin*minimum).cast(dtypes.float16) for i in range(4)))
        values[t] = UOp.wmma(a, b, values[t], (8, 16, 16), 32)
  done = UOp.group(*(acc.store(value) for acc, value in zip(accs, values))).end(kb)
  stores = [out[split, token_base+group+row, output_base+t*8+thread*2+col].store(acc.after(done)[i].load())
            for t, acc in enumerate(accs) for i, (row, col) in enumerate(((0, 0), (0, 1), (8, 0), (8, 1)))]
  return UOp.group(*stores).end(block, warp, lane).sink(arg=KernelInfo(name=f"prefill_q{5 if q5 else 4}_k_paired", opts_to_apply=()))

def q4_block_base(row: UOp, block: UOp, width: int, block_words: int = 36) -> UOp:
  """Word offset of (row, block) in the tiled Q4_K or Q5_K (44-word) layout (prepare_q4): each 16-row tile of each
  256-value block is contiguous, blocks in native word order."""
  return ((row//16*(width//256)+block)*16+row%16)*block_words

# Packed FP16 pairs: two halves in one 32-bit register, operated on with half2 instructions through inline
# CUDA (as Marlin does). Cutting a register into its two halves only where an MMA reads them lets the
# compiler hand the register straight to the tensor core; scalar FP16 arithmetic would make it re-pack.
_HALF2 = "tg_bitcast<half2>({})"
_FP16_1024 = 0x64006400  # 1024.0 in both halves: OR-ing a nibble into the mantissa gives 1024 + q exactly

def _h2(op: str, *args: UOp) -> UOp:
  return UOp(Ops.CUSTOMI, src=args, arg=(f"tg_bitcast<unsigned int>({op}({', '.join([_HALF2] * len(args))}))", dtypes.uint32))

def _halves(pair: UOp) -> list[UOp]:
  return [(pair >> s).cast(dtypes.uint16).bitcast(dtypes.float16) for s in (0, 16)]

def _q4_scale_pair(head: list[UOp], subgroup: int) -> tuple[UOp, UOp]:
  """Packed (d*scale, d*scale) and (dmin*minimum, dmin*minimum) for a static Q4_K subgroup, from the block's
  first four words. The header word is already the FP16 pair (d, dmin), and the two 6-bit values go through the
  FP16 trick as a pair: one HMUL2, exactly as rounded as the FP32 product (an FP16 times a 6-bit integer is exact)."""
  def byte(word: int, k: int) -> UOp: return (head[word] >> (8*k)) & 255
  if subgroup < 4: scale, minimum = byte(1, subgroup) & 63, byte(2, subgroup) & 63
  else:
    k = subgroup-4
    a, b, c = byte(2, k), byte(3, k), byte(1, k)
    scale, minimum = (b & 15) | ((c >> 6) << 4), (b >> 4) | ((a >> 6) << 4)
  integers = _h2("__hsub2", scale | (minimum << 16) | _FP16_1024, UOp.const(_FP16_1024, dtypes.uint32))
  products = _h2("__hmul2", integers, head[0])  # (d*scale, dmin*minimum)
  def broadcast(op: str) -> UOp:
    return UOp(Ops.CUSTOMI, src=(products,), arg=(f"tg_bitcast<unsigned int>({op}({_HALF2}))", dtypes.uint32))
  return broadcast("__low2half2"), broadcast("__high2half2")

def _q4_fragment(word: UOp, nibble: int, scale: UOp, offset: UOp, high: UOp | None = None, subgroup: int = 0) -> UOp:
  """The B fragment for nibble `nibble` of word's four bytes, as k slots (byte 0, byte 2, byte 1, byte 3):
  the pairs the FP16 trick yields directly. A Q4_K high nibble lands as 16q, so its scale comes pre-divided by 16.
  Q5_K passes `high`, the qh word whose bit `subgroup` in each byte is the value's fifth bit; its nibbles are
  shifted down so the fifth bit can join them. `offset` is dmin*minimum, subtracted."""
  if high is None:
    mask = 0x000f000f << (4*nibble)
    pairs = [((w & mask) | _FP16_1024) for w in (word, word >> 8)]
  else:
    pairs = [(((w >> (4*nibble)) & 0x000f000f) | (((h >> subgroup) & 0x00010001) << 4) | _FP16_1024)
             for w, h in ((word, high), (word >> 8, high >> 8))]
  def dequant(p: UOp) -> UOp:  # (p - 1024) * scale - offset
    q = _h2("__hsub2", p, UOp.const(_FP16_1024, dtypes.uint32))
    return UOp(Ops.CUSTOMI, src=(q, scale, offset), arg=(f"tg_bitcast<unsigned int>(__hfma2({_HALF2}, {_HALF2}, __hneg2({_HALF2})))", dtypes.uint32))
  pairs = [dequant(p) for p in pairs]
  return UOp.stack(*_halves(pairs[0]), *_halves(pairs[1]))

@functools.cache
def _fragment_kernel(out: UOp, x: UOp, *, chunks: int, width: int, typ: int = Q4_K) -> UOp:
  steps = width//16
  item = UOp.range(chunks*steps, 0, axis_type=AxisType.GLOBAL)
  lane = UOp.range(32, 1, axis_type=AxisType.LOCAL)
  chunk, step = item//steps, item%steps
  group, thread = lane//4, lane%4
  if typ == Q4_K:  # step = block*16 + pair*4 + nibble*2 + word: column = block*256 + pair*64 + nibble*32 + t*8 + word*4 + i
    column = (step//16)*256+((step//4)%4)*64+((step//2)%2)*32+thread*8+(step%2)*4
  else:  # Q6_K, step = block*16 + half*8 + nibble*4 + word: column = block*256 + half*128 + nibble*64 + t*16 + word*4 + i
    column = (step//16)*256+((step//8)%2)*128+((step//4)%2)*64+thread*16+(step%4)*4
  rows = [_load_lanes(x[chunk*16+group+8*r8, column], 4) for r8 in (0, 1)]
  # slots: (r0 i0, r0 i2, r8 i0, r8 i2, r0 i1, r0 i3, r8 i1, r8 i3)
  values = [rows[r8][i].cast(dtypes.float16) for i0 in (0, 1) for r8 in (0, 1) for i in (i0, i0+2)]
  return UOp.group(*(out[chunk, step, lane, j].store(v) for j, v in enumerate(values))).end(item, lane).sink(
    arg=KernelInfo(name="fragment_activations", opts_to_apply=()))

def fragment_activations(x: Tensor, width: int, typ: int = Q4_K) -> Tensor:
  """16-row activation chunks (rows, width) as FP16 (rows/16, width/16, 32 lanes, 8 halves): each lane's A
  fragment for one k step of _prefill_q4_mma or _prefill_q6_mma, contiguous, in that kernel's permuted k
  order, with the slots of every four taking columns (0, 2, 1, 3) to match the FP16 trick's pairs."""
  chunks = x.shape[0]//16
  out = Tensor.empty(chunks, width//16, 32, 8, dtype=dtypes.float16, device=x.device)
  return Tensor.custom_kernel(out, x.float().contiguous(), fxn=functools.partial(_fragment_kernel, chunks=chunks, width=width, typ=typ))[0]

@functools.cache
def _prefill_q4_mma(out: UOp, raw: UOp, x: UOp, *, tokens: int, width: int, rows: int, tiles: int = 2, warps: int = 4,
                    splits: int = 1, typ: int = Q4_K) -> UOp:
  """Tiled Q4_K (prepare_q4) x 16-row chunks on FP16 tensor cores, after Marlin:
  - B fragments are dequantized straight into fragment registers: per 32-bit word, two bit operations and two
    packed FP16 operations per register, with no repacking;
  - each A fragment is one 16-byte load of pre-permuted activations (fragment_activations);
  - a warp's 16 rows of one block are one contiguous 2,304-byte run of the tiled layout;
  - the next block's weights and activations are loaded while the current block's MMAs run.
  Q5_K works the same way, with the lane's two qh words (the same for every subgroup pair) adding fifth bits.
  Split K writes partial sums to out[split]."""
  q5 = typ == Q5_K
  output_blocks = rows//(warps*8*tiles)
  blocks_per_row = width//256
  blocks = blocks_per_row//splits
  block = UOp.range((tokens//16)*output_blocks*splits, 0, axis_type=AxisType.GLOBAL)
  lane = UOp.range(32, -1, axis_type=AxisType.WARP)
  warp = UOp.range(warps, 1, axis_type=AxisType.LOCAL)
  split, rest = block%splits, block//splits
  chunk = rest//output_blocks
  output_base = (rest%output_blocks)*warps*8*tiles+warp*8*tiles
  group, thread = lane//4, lane%4
  accs = []
  for t in range(tiles):
    acc = UOp.placeholder((4,), dtypes.float32, slot=3+t, addrspace=AddrSpace.REG)
    accs.append(acc.after(acc.store(acc.const_like(0))))
  # Per row: four header/scale words, Q5_K's two qh words, then two words for each of four subgroup pairs.
  extra = 2 if q5 else 0
  per_tile = 12+extra

  def fetch(column_block) -> list[UOp]:
    loaded = []
    for t in range(tiles):
      start = q4_block_base(output_base+t*8+group, column_block, width, 44 if q5 else 36)
      head = _load_lanes(raw[start], 4)
      loaded += [head[i] for i in range(4)]
      if q5:
        high = _load_lanes(raw[start+4+2*thread], 2)
        loaded += [high[0], high[1]]
      for pair in range(4):
        words = _load_lanes(raw[start+(12 if q5 else 4)+8*pair+2*thread], 2)
        loaded += [words[0], words[1]]
    return loaded

  kb = UOp.range(blocks, 2, axis_type=getattr(AxisType, 'REDUCE', AxisType.LOOP))
  staged = UOp.placeholder((tiles*per_tile,), dtypes.uint32, slot=3+tiles, addrspace=AddrSpace.REG)
  staged = staged.after(UOp.group(*(staged[i].store(v) for i, v in enumerate(fetch(split*blocks)))))
  current = [staged.after(kb)[i].load() for i in range(tiles*per_tile)]
  upcoming = fetch(split*blocks+(kb+1).minimum(blocks-1))
  def fragments(column_block) -> list[UOp]:
    """The block's 16 A fragments (one per k step), 8 halves each, flattened."""
    loaded = []
    for step in range(16):
      fragment = _load_lanes(x[chunk, column_block*16+step, lane, 0], 8)
      loaded += [fragment[i] for i in range(8)]
    return loaded

  xstaged = UOp.placeholder((128,), dtypes.float16, slot=4+tiles, addrspace=AddrSpace.REG)
  xstaged = xstaged.after(UOp.group(*(xstaged[i].store(v) for i, v in enumerate(fragments(split*blocks)))))
  xcurrent = [xstaged.after(kb)[i].load() for i in range(128)]
  xupcoming = fragments(split*blocks+(kb+1).minimum(blocks-1))
  values = [acc.after(kb) for acc in accs]
  for pair in range(4):
    factors = []
    for t in range(tiles):
      head = current[t*per_tile:t*per_tile+4]
      (scale0, min0), (scale1, min1) = _q4_scale_pair(head, 2*pair), _q4_scale_pair(head, 2*pair+1)
      sixteenth = UOp.const(0x2c002c00, dtypes.uint32)  # a Q4_K high nibble lands as 16q; dividing by 16 is exact
      factors.append(((scale0, min0), (scale1 if q5 else _h2("__hmul2", scale1, sixteenth), min1)))
    for step in range(4):
      a = UOp.stack(*xcurrent[(pair*4+step)*8:(pair*4+step+1)*8])
      for t in range(tiles):
        scale, offset = factors[t][step//2]
        word = current[t*per_tile+4+extra+2*pair+step%2]
        high = current[t*per_tile+4+step%2] if q5 else None
        frag = _q4_fragment(word, step//2, scale, offset, high, 2*pair+step//2)
        values[t] = UOp.wmma(a, frag, values[t], (8, 16, 16), 32)
  updates = [acc.store(value) for acc, value in zip(accs, values)]
  after = staged.after(*values)  # overwrite the staged words only after this iteration's MMAs have read them
  updates += [after[i].store(v) for i, v in enumerate(upcoming)]
  xafter = xstaged.after(*values)
  updates += [xafter[i].store(v) for i, v in enumerate(xupcoming)]
  done = UOp.group(*updates).end(kb)
  stores = [out[split, chunk*16+group+row, output_base+t*8+thread*2+col].store(acc.after(done)[i].load())
            for t, acc in enumerate(accs) for i, (row, col) in enumerate(((0, 0), (0, 1), (8, 0), (8, 1)))]
  return UOp.group(*stores).end(block, warp, lane).sink(arg=KernelInfo(name=f"prefill_q{5 if q5 else 4}_k_mma", opts_to_apply=()))

def q6_block_base(row: UOp, block: UOp, width: int) -> UOp:
  """Word offset of (row, block) in the tiled Q6_K layout (prepare_q6_tiled): 54-word blocks, each 16-row tile
  of each block contiguous. Words 0-31 are ql, 32-47 qh, 48-51 the 16 int8 scales, 52 holds d."""
  return ((row//16*(width//256)+block)*16+row%16)*(Q6_PREPARED_BYTES//4)

_FP16_1056 = 0x64206420  # 1024 + 32: subtracting it from 1024 + q leaves q - 32
_FP16_1152 = 0x64806480  # 1024 + 128: the offset of a signed byte XOR 0x80

def _q6_scale_pair(scale_lo: UOp, scale_hi: UOp, d2: UOp) -> UOp:
  """Packed (d*scale_lo, d*scale_hi) from two int8 scales (in the low bytes of their registers)."""
  bits = ((scale_lo & 255) | ((scale_hi & 255) << 16)) ^ 0x00800080
  return _h2("__hmul2", _h2("__hsub2", bits | _FP16_1024, UOp.const(_FP16_1152, dtypes.uint32)), d2)

def _q6_fragment(ql: UOp, qh: UOp, nibble: int, qh_shift: UOp, scale: UOp) -> UOp:
  """The B fragment for one ql word's four bytes (nibble `nibble`) with their 2 high bits from qh (at
  qh_shift), as k slots (byte 0, byte 2, byte 1, byte 3): (1024 + q) - 1056 = q - 32 exactly, times d*scale."""
  def pair(low: UOp, high: UOp) -> UOp:
    bits = ((low >> (4*nibble)) & 0x000f000f) | (((high >> qh_shift) & 0x00030003) << 4) | _FP16_1024
    return _h2("__hmul2", _h2("__hsub2", bits, UOp.const(_FP16_1056, dtypes.uint32)), scale)
  return UOp.stack(*_halves(pair(ql, qh)), *_halves(pair(ql >> 8, qh >> 8)))

@functools.cache
def _prefill_q6_mma(out: UOp, raw: UOp, x: UOp, *, tokens: int, width: int, rows: int, tiles: int = 2, warps: int = 4,
                    splits: int = 1) -> UOp:
  """Tiled Q6_K (prepare_q6_tiled) x 16-row chunks on FP16 tensor cores, the Q6_K counterpart of _prefill_q4_mma.

  In half h of a block, lane t owns ql bytes 16t..16t+15 (low nibbles: values 128h + 16t + j, scale group 8h + t;
  high nibbles: values 128h + 64 + 16t + j, group 8h + 4 + t) and qh bytes 16t..16t+15 mod 32 of the half's 32,
  whose 2-bit fields sit at shift 2*(t//2), plus 4 for the high-nibble values. Step (h, nibble, word) feeds the
  lane's word `word` of that run; each lane applies its own group's scale before the MMA."""
  output_blocks = rows//(warps*8*tiles)
  blocks_per_row = width//256
  blocks = blocks_per_row//splits
  block = UOp.range((tokens//16)*output_blocks*splits, 0, axis_type=AxisType.GLOBAL)
  lane = UOp.range(32, -1, axis_type=AxisType.WARP)
  warp = UOp.range(warps, 1, axis_type=AxisType.LOCAL)
  split, rest = block%splits, block//splits
  chunk = rest//output_blocks
  output_base = (rest%output_blocks)*warps*8*tiles+warp*8*tiles
  group, thread = lane//4, lane%4
  accs = []
  for t in range(tiles):
    acc = UOp.placeholder((4,), dtypes.float32, slot=3+t, addrspace=AddrSpace.REG)
    accs.append(acc.after(acc.store(acc.const_like(0))))
  per_tile = 21  # ql: 4 words per half; qh: 4 words per half; the 4 scale words; d

  def fetch(column_block) -> list[UOp]:
    loaded = []
    for t in range(tiles):
      start = q6_block_base(output_base+t*8+group, column_block, width)
      for h in range(2):
        ql = _load_lanes(raw[start+16*h+4*thread], 4)
        loaded += [ql[i] for i in range(4)]
      for h in range(2):
        qh = _load_lanes(raw[start+32+8*h+4*(thread%2)], 4)
        loaded += [qh[i] for i in range(4)]
      scales = _load_lanes(raw[start+48], 4)
      loaded += [scales[i] for i in range(4)] + [raw[start+52].load()]
    return loaded

  def fragments(column_block) -> list[UOp]:
    """The block's 16 A fragments (one per k step), 8 halves each, flattened."""
    loaded = []
    for step in range(16):
      fragment = _load_lanes(x[chunk, column_block*16+step, lane, 0], 8)
      loaded += [fragment[i] for i in range(8)]
    return loaded

  kb = UOp.range(blocks, 2, axis_type=getattr(AxisType, 'REDUCE', AxisType.LOOP))
  staged = UOp.placeholder((tiles*per_tile,), dtypes.uint32, slot=3+tiles, addrspace=AddrSpace.REG)
  staged = staged.after(UOp.group(*(staged[i].store(v) for i, v in enumerate(fetch(split*blocks)))))
  current = [staged.after(kb)[i].load() for i in range(tiles*per_tile)]
  upcoming = fetch(split*blocks+(kb+1).minimum(blocks-1))
  xstaged = UOp.placeholder((128,), dtypes.float16, slot=4+tiles, addrspace=AddrSpace.REG)
  xstaged = xstaged.after(UOp.group(*(xstaged[i].store(v) for i, v in enumerate(fragments(split*blocks)))))
  xcurrent = [xstaged.after(kb)[i].load() for i in range(128)]
  xupcoming = fragments(split*blocks+(kb+1).minimum(blocks-1))
  values = [acc.after(kb) for acc in accs]
  shift = (thread//2)*2
  for h in range(2):
    factors = []
    for t in range(tiles):
      words = current[t*per_tile:(t+1)*per_tile]
      d = words[20] & 0xffff
      d2 = d | (d << 16)
      # Lane t's scale groups in half h: 8h + t (low nibbles) and 8h + 4 + t (high nibbles), bytes t of words 2h, 2h + 1.
      pair = _q6_scale_pair(words[16+2*h] >> (8*thread), words[17+2*h] >> (8*thread), d2)
      low, high = (UOp(Ops.CUSTOMI, src=(pair,), arg=(f"tg_bitcast<unsigned int>({op}({_HALF2}))", dtypes.uint32))
                   for op in ("__low2half2", "__high2half2"))
      factors.append((low, high))
    for nibble in range(2):
      for word in range(4):
        a = UOp.stack(*xcurrent[(h*8+nibble*4+word)*8:(h*8+nibble*4+word+1)*8])
        for t in range(tiles):
          words = current[t*per_tile:(t+1)*per_tile]
          frag = _q6_fragment(words[4*h+word], words[8+4*h+word], nibble, shift+4*nibble, factors[t][nibble])
          values[t] = UOp.wmma(a, frag, values[t], (8, 16, 16), 32)
  updates = [acc.store(value) for acc, value in zip(accs, values)]
  after = staged.after(*values)  # overwrite the staged words only after this iteration's MMAs have read them
  updates += [after[i].store(v) for i, v in enumerate(upcoming)]
  xafter = xstaged.after(*values)
  updates += [xafter[i].store(v) for i, v in enumerate(xupcoming)]
  done = UOp.group(*updates).end(kb)
  stores = [out[split, chunk*16+group+row, output_base+t*8+thread*2+col].store(acc.after(done)[i].load())
            for t, acc in enumerate(accs) for i, (row, col) in enumerate(((0, 0), (0, 1), (8, 0), (8, 1)))]
  return UOp.group(*stores).end(block, warp, lane).sink(arg=KernelInfo(name="prefill_q6_k_mma", opts_to_apply=()))

@functools.cache
def _prefill_q6_paired(out: UOp, raw: UOp, x: UOp, *, tokens: int, width: int, rows: int,
                       tiles: int = 4, splits: int = 1) -> UOp:
  """Prepared (216-byte) Q6_K MMA over a permuted k order, per 128-value half.

  For half h, lane t owns ql bytes 64h+16t..+15 (low nibbles: values 128h+16t+j,
  high nibbles: 128h+64+16t+j) and qh bytes 128+32h+16(t%2)..+15, whose 2-bit
  fields sit at shift 2*(t//2) (+4 for the high-nibble values). Each set of 16
  values shares one int8 scale. Step s (0..7) takes word s%4, nibble s//4; its
  permuted columns are c = 128h + 64*(s//4) + 16t + 4*(s%4) + i.
  """
  warps = math.gcd(rows//(8*tiles), 4)
  output_blocks = rows//(warps*8*tiles)
  blocks = width//256//splits
  block = UOp.range((tokens//16)*output_blocks*splits, 0, axis_type=AxisType.GLOBAL)
  lane = UOp.range(32, -1, axis_type=AxisType.WARP)
  warp = UOp.range(warps, 1, axis_type=AxisType.LOCAL)
  split, rest = block%splits, block//splits
  token_base = (rest//output_blocks)*16
  output_base = (rest%output_blocks)*warps*8*tiles+warp*8*tiles
  group, thread = lane//4, lane%4
  def word(logical, index):
    return raw[logical*(Q6_PREPARED_BYTES//4)+index].load()
  accs = []
  for t in range(tiles):
    acc = UOp.placeholder((4,), dtypes.float32, slot=3+t, addrspace=AddrSpace.REG)
    accs.append(acc.after(acc.store(acc.const_like(0))))
  kb = UOp.range(blocks, 2, axis_type=getattr(AxisType, 'REDUCE', AxisType.LOOP))
  column_block = split*blocks+kb
  values = [acc.after(kb) for acc in accs]
  for half in range(2):
    decoded = []
    for t in range(tiles):
      logical = (output_base+t*8+group)*(width//256)+column_block
      ql = [word(logical, 16*half+4*thread+w) for w in range(4)]
      qh = [word(logical, 32+8*half+4*(thread%2)+w) for w in range(4)]
      d = _half(word(logical, 52) & 0xffff)
      scales = [d*((word(logical, 48+(8*half+4*n+thread)//4) >> (((8*half+4*n+thread)%4)*8).cast(dtypes.uint32)) & 255)
                .cast(dtypes.uint8).bitcast(dtypes.int8).float() for n in (0, 1)]
      decoded.append((ql, qh, scales))
    for step in range(8):
      col0 = column_block*256+128*half+64*(step//4)+16*thread+4*(step%4)
      cols = [col0+i for i in range(4)]
      a = UOp.stack(*(x[token_base+group+row, c].load() for (row, c) in
                      ((0, cols[0]), (0, cols[1]), (8, cols[0]), (8, cols[1]), (0, cols[2]), (0, cols[3]), (8, cols[2]), (8, cols[3]))))
      for t, (ql, qh, scales) in enumerate(decoded):
        low_word, high_word, scale = ql[step%4], qh[step%4], scales[step//4]
        def quant(i):
          low = (low_word >> (8*i+4*(step//4))) & 15
          high = (high_word >> (UOp.const(8*i+4*(step//4), dtypes.uint32)+(thread//2*2).cast(dtypes.uint32))) & 3
          return (low | (high << 4)).cast(dtypes.int32)-32
        b = UOp.stack(*((scale*quant(i).float()).cast(dtypes.float16) for i in range(4)))
        values[t] = UOp.wmma(a, b, values[t], (8, 16, 16), 32)
  done = UOp.group(*(acc.store(value) for acc, value in zip(accs, values))).end(kb)
  stores = [out[split, token_base+group+row, output_base+t*8+thread*2+col].store(acc.after(done)[i].load())
            for t, acc in enumerate(accs) for i, (row, col) in enumerate(((0, 0), (0, 1), (8, 0), (8, 1)))]
  return UOp.group(*stores).end(block, warp, lane).sink(arg=KernelInfo(name="prefill_q6_k_paired", opts_to_apply=()))

@functools.cache
def _prefill_kernel(out: UOp, raw: UOp, x: UOp, *, tokens: int, width: int, rows: int, typ: int,
                    q6_aligned: bool = False, q6_tiled: bool = False, q4_tiled: bool = False,
                    tiles: int = 1, splits: int = 1, unroll: int = 1) -> UOp:
  # Each warp owns 16 tokens x (8*tiles) outputs: one A fragment load feeds
  # `tiles` independent MMAs. Split-K partitions the reduction across blocks;
  # partial sums land in out[split] and are reduced by the caller.
  warps = math.gcd(rows//(8*tiles), 4)
  output_blocks = rows//(warps*8*tiles)
  steps = width//16//splits
  block = UOp.range((tokens//16)*output_blocks*splits, 0, axis_type=AxisType.GLOBAL)
  lane = UOp.range(32, -1, axis_type=AxisType.WARP)
  warp = UOp.range(warps, 1, axis_type=AxisType.LOCAL)
  split, rest = block%splits, block//splits
  token_base = (rest//output_blocks)*16
  output_base = (rest%output_blocks)*warps*8*tiles+warp*8*tiles
  group, thread = lane//4, lane%4
  accs = []
  for t in range(tiles):
    acc = UOp.placeholder((4,), dtypes.float32, slot=3+t, addrspace=AddrSpace.REG)
    accs.append(acc.after(acc.store(acc.const_like(0))))
  k = UOp.range(steps//unroll, 2, axis_type=getattr(AxisType, 'REDUCE', AxisType.LOOP))
  # Unrolled k-steps issue all their loads before the dependent MMAs, keeping
  # several global loads in flight instead of one latency per step.
  frags = []
  for u in range(unroll):
    kk = split*steps+k*unroll+u
    # NVIDIA mma.sync m16n8k16.row.col fragment order (not AMD WMMA order).
    a = UOp.stack(*(x[token_base+group+row, kk*16+thread*2+col].load()
                    for col_high in (0, 8) for row in (0, 8) for col in (col_high, col_high+1)))
    bs = [UOp.stack(*(_weight(raw, output_base+t*8+group, kk*16+thread*2+col, width, typ, q6_aligned, q6_tiled, q4_tiled)
                      for col in (0, 1, 8, 9))) for t in range(tiles)]
    frags.append((a, bs))
  updates = []
  for t, acc in enumerate(accs):
    value = acc.after(k)
    for a, bs in frags: value = UOp.wmma(a, bs[t], value, (8, 16, 16), 32)
    updates.append(acc.store(value))
  done = UOp.group(*updates).end(k)
  stores = [out[split, token_base+group+row, output_base+t*8+thread*2+col].store(acc.after(done)[i].load())
            for t, acc in enumerate(accs) for i, (row, col) in enumerate(((0, 0), (0, 1), (8, 0), (8, 1)))]
  return UOp.group(*stores).end(block, warp, lane).sink(
    arg=KernelInfo(name=f"prefill_q{ {Q4_K: 4, Q5_K: 5}.get(typ, 6)}_k_mma", opts_to_apply=()))

def _schedule(rows: int, width: int, tokens: int) -> tuple[int, int, int]:
  """(tiles per warp, K splits, k unroll), tuned on an RTX 5060 from profiler GPU
  times: four tiles always win; split K only until there are ~64 blocks (each
  split adds a reduction pass); unroll k by four where it divides."""
  tiles = next(t for t in (4, 2, 1) if rows % (8*t) == 0)
  warps = math.gcd(rows//(8*tiles), 4)
  blocks, chunks = (tokens//16)*(rows//(warps*8*tiles)), width//256
  splits = next(s for s in range(1, chunks+1) if chunks % s == 0 and (blocks*s >= 64 or s == chunks))
  steps = width//16//splits
  return tiles, splits, next(u for u in (4, 2, 1) if steps % u == 0)

@functools.cache
def nv_prefill_supported(device: str) -> bool:
  if not nvidia(device): return False
  with Context(ALLOW_DEVICE_USAGE=1): renderer = Device[device].renderer
  return isinstance(renderer, CUDARenderer) and any(
    tc.dims == (8, 16, 16) and tc.dtype_in == dtypes.float16 and tc.dtype_out == dtypes.float32 for tc in renderer.tensor_cores)

def quant_gemm(x: Tensor, packed: Tensor, *, out_features: int, in_features: int, ggml_type: int,
               q6_aligned: bool = False, q6_tiled: bool = False, q4_tiled: bool = False) -> Tensor:
  """Concrete multi-token packed linear; symbolic padding belongs to the adapter."""
  if ggml_type not in (Q4_K, Q5_K, Q6_K): raise QuantGemmError("only Q4_K/Q5_K/Q6_K prefill is supported")
  if in_features <= 0 or in_features%256 or out_features <= 0 or out_features%8:
    raise QuantGemmError("prefill requires input width multiple of 256 and output rows multiple of 8")
  if x.dtype not in dtypes.floats or not x.shape or x.shape[-1] != in_features:
    raise QuantGemmError("activation dtype/shape does not match prefill contract")
  if not isinstance(x.numel(), int) or x.numel()//in_features < 2:
    raise QuantGemmError("prefill requires a concrete multi-token shape")
  if q4_tiled and (ggml_type not in (Q4_K, Q5_K) or out_features%16): raise QuantGemmError("tiled Q4/Q5 requires Q4_K/Q5_K and 16-row tiles")
  if q6_tiled and (ggml_type != Q6_K or out_features%16): raise QuantGemmError("tiled Q6 requires Q6_K and 16-row tiles")
  if q6_aligned and ggml_type != Q6_K: raise QuantGemmError("aligned Q6 layout requires Q6_K")
  dtype = dtypes.uint32 if ggml_type in (Q4_K, Q5_K) or q6_aligned or q6_tiled else dtypes.uint16
  expected = out_features*in_features//256*(Q6_PREPARED_BYTES if q6_aligned else QUANT_SIZES[ggml_type])
  if packed.dtype != dtype or packed.numel()*dtype.itemsize != expected:
    raise QuantGemmError("packed dtype/size does not match prefill contract")
  if x.device != packed.device or not nvidia(x.device):
    raise QuantGemmError("prefill requires activations/weights on the same NVIDIA device")
  if not nv_prefill_supported(x.device):
    raise QuantGemmError("prefill requires CUDA-C FP16 m16n8k16 tensor cores")
  tokens = x.numel()//in_features
  padded_tokens = (tokens+15)//16*16
  activation = x.reshape(tokens, in_features).cast(dtypes.float16).pad(((0, padded_tokens-tokens), (0, 0))).contiguous()
  tiles, splits, unroll = _schedule(out_features, in_features, padded_tokens)
  out = Tensor.empty(splits, padded_tokens, out_features, dtype=dtypes.float32, device=x.device)
  paired = os.environ.get('INFURNACE_Q4_PAIRED', '1') != '0'
  if q4_tiled or q6_tiled:
    tiles, warps = 2, math.gcd(out_features//16, 4)
    blocks, output_blocks = in_features//256, out_features//(warps*8*tiles)
    # Split K only for narrow outputs: about two blocks per SM on a 30-SM card
    splits = next(s for s in range(1, blocks+1) if blocks % s == 0 and ((padded_tokens//16)*output_blocks*s >= 60 or s == blocks))
    out = Tensor.empty(splits, padded_tokens, out_features, dtype=dtypes.float32, device=x.device)
    # Q5_K reads its k order like Q4_K
    fragments = fragment_activations(x.reshape(tokens, in_features).pad(((0, padded_tokens-tokens), (0, 0))), in_features,
                                     Q6_K if ggml_type == Q6_K else Q4_K)
    kernel = _prefill_q6_mma if ggml_type == Q6_K else functools.partial(_prefill_q4_mma, typ=ggml_type)
    result = Tensor.custom_kernel(out, packed.flatten().contiguous(), fragments, fxn=functools.partial(
      kernel, tokens=padded_tokens, width=in_features, rows=out_features, tiles=tiles, warps=warps, splits=splits))[0]
    result = result[0] if splits == 1 else result.sum(0)
    return result[:tokens].reshape(*x.shape[:-1], out_features)
  if paired and ggml_type == Q6_K and q6_aligned:
    blocks = in_features//256
    splits = next(s for s in range(1, blocks+1) if blocks % s == 0 and (splits <= s or s == blocks))
    out = Tensor.empty(splits, padded_tokens, out_features, dtype=dtypes.float32, device=x.device)
    result = Tensor.custom_kernel(out, packed.flatten().contiguous(), activation, fxn=functools.partial(
      _prefill_q6_paired, tokens=padded_tokens, width=in_features, rows=out_features, tiles=tiles,
      splits=splits))[0]
    result = result[0] if splits == 1 else result.sum(0)
    return result[:tokens].reshape(*x.shape[:-1], out_features)
  if paired and ggml_type in (Q4_K, Q5_K):
    blocks = in_features//256
    splits = next(s for s in range(1, blocks+1) if blocks % s == 0 and (splits <= s or s == blocks))
    out = Tensor.empty(splits, padded_tokens, out_features, dtype=dtypes.float32, device=x.device)
    result = Tensor.custom_kernel(out, packed.flatten().contiguous(), activation, fxn=functools.partial(
      _prefill_q4_paired, tokens=padded_tokens, width=in_features, rows=out_features, tiles=tiles,
      splits=splits, typ=ggml_type))[0]
    result = result[0] if splits == 1 else result.sum(0)
    return result[:tokens].reshape(*x.shape[:-1], out_features)
  result = Tensor.custom_kernel(out, packed.flatten().contiguous(), activation,
    fxn=functools.partial(_prefill_kernel, tokens=padded_tokens, width=in_features, rows=out_features, typ=ggml_type,
                         q6_aligned=q6_aligned, q6_tiled=q6_tiled, q4_tiled=q4_tiled, tiles=tiles, splits=splits, unroll=unroll))[0]
  result = result[0] if splits == 1 else result.sum(0)
  return result[:tokens].reshape(*x.shape[:-1], out_features)
