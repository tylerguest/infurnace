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
from tinygrad.uop.ops import AxisType, KernelInfo

from infurnace.kernels.quant_gemv import Q4_K, Q5_K, Q6_K, Q6_PREPARED_BYTES, QUANT_SIZES, _half, _q4_scales, _q6_swizzled_index, _q4_word
from infurnace.device import nvidia

class QuantGemmError(ValueError):
  """Inputs/backend do not satisfy the first prefill kernel contract."""

def _q6_byte(raw: UOp, base: UOp, offset: UOp, blocks_per_row: int = 0) -> UOp:
  size = raw.dtype.itemsize
  index = _q6_swizzled_index(base, offset//size, blocks_per_row) if blocks_per_row else base+offset//size
  return (raw[index].load().cast(dtypes.uint32) >> ((offset%size)*8).cast(dtypes.uint32)) & 255

def _weight(raw: UOp, row: UOp, col: UOp, width: int, typ: int, q6_aligned: bool = False,
            q6_swizzled: bool = False, q4_swizzled: bool = False) -> UOp:
  block, within = col//256, col%256
  if typ == Q4_K:
    logical = row*(width//256)+block
    base = logical if q4_swizzled else logical*36
    swizzle_width = width//256 if q4_swizzled else 0
    subgroup = within//32
    word = _q4_word(raw, base, 4+(subgroup//2)*8+(within%32)//4, swizzle_width)
    quant = (word >> ((subgroup%2)*4+(within%4)*8).cast(dtypes.uint32)) & 15
    d, dmin, scale, minimum = _q4_scales(raw, base, subgroup, swizzle_width)
    return (d*scale*quant.float()-dmin*minimum).cast(dtypes.float16)
  if typ == Q5_K:
    # 44 words: d/dmin, 12 scale bytes, 32 high-bit bytes (bit `subgroup` of
    # byte j), then Q4_K-ordered low nibbles.
    base, subgroup, j = (row*(width//256)+block)*44, within//32, within%32
    low = (raw[base+12+(subgroup//2)*8+j//4].load() >> ((subgroup%2)*4+(j%4)*8).cast(dtypes.uint32)) & 15
    high = (raw[base+4+j//4].load() >> ((j%4)*8+subgroup).cast(dtypes.uint32)) & 1
    d, dmin, scale, minimum = _q4_scales(raw, base, subgroup)
    return (d*scale*(low | (high << 4)).float()-dmin*minimum).cast(dtypes.float16)
  logical = row*(width//256)+block
  base = logical if q6_swizzled else logical*(Q6_PREPARED_BYTES//4 if q6_aligned else 105)
  swizzle_width = width//256 if q6_swizzled else 0
  half, pos = within//128, within%128
  low = _q6_byte(raw, base, half*64+pos%64, swizzle_width) >> ((pos//64)*4).cast(dtypes.uint32)
  high = _q6_byte(raw, base, 128+half*32+pos%32, swizzle_width) >> ((pos//32)*2).cast(dtypes.uint32)
  quant = ((low & 15) | ((high & 3) << 4)).cast(dtypes.int32)-32
  scale = _q6_byte(raw, base, 192+within//16, swizzle_width).cast(dtypes.uint8).bitcast(dtypes.int8).float()
  d_index = _q6_swizzled_index(logical, 52, width//256) if q6_swizzled else base+(52 if q6_aligned else 104)
  return (_half(raw[d_index].load())*quant.float()*scale).cast(dtypes.float16)

@functools.cache
def _prefill_q4_paired(out: UOp, raw: UOp, x: UOp, *, tokens: int, width: int, rows: int, q4_swizzled: bool = False,
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
  if q5: q4_swizzled = False
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
  swizzle_width = width//256 if q4_swizzled else 0
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
      base = logical if q4_swizzled else logical*(44 if q5 else 36)
      low = _q4_scales(raw, base, UOp.const(2*pair), swizzle_width)
      high = _q4_scales(raw, base, UOp.const(2*pair+1), swizzle_width)
      words = [_q4_word(raw, base, (12 if q5 else 4)+8*pair+2*thread+w, swizzle_width) for w in (0, 1)]
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

@functools.cache
def _prefill_q6_paired(out: UOp, raw: UOp, x: UOp, *, tokens: int, width: int, rows: int, q6_swizzled: bool = False,
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
    return raw[_q6_swizzled_index(logical, index, width//256) if q6_swizzled else logical*(Q6_PREPARED_BYTES//4)+index].load()
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
                    q6_aligned: bool = False, q6_swizzled: bool = False, q4_swizzled: bool = False,
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
    bs = [UOp.stack(*(_weight(raw, output_base+t*8+group, kk*16+thread*2+col, width, typ, q6_aligned, q6_swizzled, q4_swizzled)
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
               q6_aligned: bool = False, q6_swizzled: bool = False, q4_swizzled: bool = False) -> Tensor:
  """Concrete multi-token packed linear; symbolic padding belongs to the adapter."""
  if ggml_type not in (Q4_K, Q5_K, Q6_K): raise QuantGemmError("only Q4_K/Q5_K/Q6_K prefill is supported")
  if in_features <= 0 or in_features%256 or out_features <= 0 or out_features%8:
    raise QuantGemmError("prefill requires input width multiple of 256 and output rows multiple of 8")
  if x.dtype not in dtypes.floats or not x.shape or x.shape[-1] != in_features:
    raise QuantGemmError("activation dtype/shape does not match prefill contract")
  if not isinstance(x.numel(), int) or x.numel()//in_features < 2:
    raise QuantGemmError("prefill requires a concrete multi-token shape")
  if q4_swizzled and ggml_type != Q4_K: raise QuantGemmError("swizzled Q4 requires Q4_K")
  if q6_swizzled and not q6_aligned: raise QuantGemmError("swizzled Q6 requires prepared storage")
  if q6_aligned and ggml_type != Q6_K: raise QuantGemmError("aligned Q6 layout requires Q6_K")
  dtype = dtypes.uint32 if ggml_type in (Q4_K, Q5_K) or q6_aligned else dtypes.uint16
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
  if paired and ggml_type == Q6_K and q6_aligned:
    blocks = in_features//256
    splits = next(s for s in range(1, blocks+1) if blocks % s == 0 and (splits <= s or s == blocks))
    out = Tensor.empty(splits, padded_tokens, out_features, dtype=dtypes.float32, device=x.device)
    result = Tensor.custom_kernel(out, packed.flatten().contiguous(), activation, fxn=functools.partial(
      _prefill_q6_paired, tokens=padded_tokens, width=in_features, rows=out_features, q6_swizzled=q6_swizzled, tiles=tiles,
      splits=splits))[0]
    result = result[0] if splits == 1 else result.sum(0)
    return result[:tokens].reshape(*x.shape[:-1], out_features)
  if paired and ggml_type in (Q4_K, Q5_K):
    blocks = in_features//256
    splits = next(s for s in range(1, blocks+1) if blocks % s == 0 and (splits <= s or s == blocks))
    out = Tensor.empty(splits, padded_tokens, out_features, dtype=dtypes.float32, device=x.device)
    result = Tensor.custom_kernel(out, packed.flatten().contiguous(), activation, fxn=functools.partial(
      _prefill_q4_paired, tokens=padded_tokens, width=in_features, rows=out_features, q4_swizzled=q4_swizzled, tiles=tiles,
      splits=splits, typ=ggml_type))[0]
    result = result[0] if splits == 1 else result.sum(0)
    return result[:tokens].reshape(*x.shape[:-1], out_features)
  result = Tensor.custom_kernel(out, packed.flatten().contiguous(), activation,
    fxn=functools.partial(_prefill_kernel, tokens=padded_tokens, width=in_features, rows=out_features, typ=ggml_type,
                         q6_aligned=q6_aligned, q6_swizzled=q6_swizzled, q4_swizzled=q4_swizzled, tiles=tiles, splits=splits, unroll=unroll))[0]
  result = result[0] if splits == 1 else result.sum(0)
  return result[:tokens].reshape(*x.shape[:-1], out_features)
