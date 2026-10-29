from __future__ import annotations
import functools
import math
from typing import Callable
from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from infurnace.device import nvidia

Q4_K = 12
Q5_K = 13
Q6_K = 14
Q2_0 = 42  # ternary: 64 two-bit codes (0, 1, 2 mean -1, 0, +1) and an FP16 scale per block
GGML_BLOCK_SIZE = 256
Q8_GROUP_SIZE = 32
Q4_WORDS = 36
Q6_BYTES = 210
QUANT_SIZES = {Q4_K:Q4_WORDS*4, Q5_K:176, Q6_K:Q6_BYTES, Q2_0:72}  # bytes per 256 values

class QuantGemvError(ValueError):
  """Quant-GEMV inputs do not satisfy the kernel contract."""

def warp_reduce(val:UOp, maximum:bool = False, full_wave:bool = False) -> UOp:
  """Full-wave (32-lane) xor-shuffle reduction through CUDA ``__shfl_xor_sync``."""
  for offset in ((16, 8, 4, 2, 1) if full_wave else (8, 4, 2, 1)):
    if val.op is Ops.INDEX and val.addrspace == AddrSpace.REG: val = val.load()
    other = UOp(Ops.CUSTOM, src=(val,), arg=(f"__shfl_xor_sync(0xffffffff, {{0}}, {offset})", dtypes.float))
    val = val.maximum(other) if maximum else val + other
  return val

def _dp4a(a:UOp, b:UOp, c:UOp) -> UOp:
  return UOp(Ops.CUSTOMI, src=(a.int(), b.int(), c), arg=("__dp4a({}, {}, {})", dtypes.int32))

def _load_lanes(ptr: UOp, lanes: int) -> UOp:
  """Load ``lanes`` consecutive elements at a buffer index as a vector value."""
  assert ptr.op is Ops.INDEX
  buf, coords = ptr.src[0], ptr.src[1:]
  idx = sum((coord * math.prod(buf.shape[i + 1:]) for i, coord in enumerate(coords)), UOp.const(0))
  return UOp(Ops.SHRINK, src=(buf.flatten(), idx, UOp.const(lanes))).load()

def _q4_word(raw: UOp, base: UOp, word: UOp | int) -> UOp:
  return raw[base+word].load()

def _load_byte(raw:UOp, base:UOp, offset:UOp) -> UOp:
  return (_q4_word(raw, base, offset//4) >> ((offset & 3)*8).cast(dtypes.uint32)) & 255

def _half(value:UOp) -> UOp:
  return value.cast(dtypes.uint16).bitcast(dtypes.float16).float()

Q6_PREPARED_BYTES = 216

def prepare_q4(raw: Tensor, *, in_features: int, realize: bool = True, ggml_type: int = Q4_K) -> Tensor:
  """Q4_K (or Q5_K) rows in the tiled layout every kernel reads: each 16-row tile of each 256-value block is one
  contiguous run (2,304 bytes for Q4_K, 2,816 for Q5_K), blocks in native word order. A 16-row tensor-core warp
  then reads one contiguous run per block (after Marlin's weight layout); a one-row warp reads the same layout."""
  blocks, size = in_features//256, QUANT_SIZES[ggml_type]
  if (raw.dtype != dtypes.uint8 or raw.numel() <= 0 or in_features <= 0 or in_features%256
      or raw.numel()%(blocks*size) or (raw.numel()//(blocks*size))%16):
    raise QuantGemvError("tiled Q4/Q5 preparation requires complete uint8 rows, width a multiple of 256 and rows of 16")
  rows = raw.numel()//(blocks*size)
  result = raw.flatten().reshape(rows//16, 16, blocks, size).permute(0, 2, 1, 3).contiguous().flatten().bitcast(dtypes.uint32)
  return result.realize() if realize else result

def untile(packed: Tensor, *, in_features: int, block_bytes: int) -> Tensor:
  """Tiled words back to native GGUF block bytes in row order."""
  words = packed.bitcast(dtypes.uint32).reshape(-1, in_features//256, 16, block_bytes//4)
  return words.permute(0, 2, 1, 3).reshape(-1, block_bytes//4).bitcast(dtypes.uint8)

def _prepare_q6_kernel(out: UOp, raw: UOp, *, gpu: bool = True) -> UOp:
  block = UOp.range((out.shape[0]+127)//128, 0, axis_type=AxisType.GLOBAL if gpu else AxisType.LOOP)
  lane = UOp.range(128, 1, axis_type=AxisType.LOCAL if gpu else AxisType.LOOP)
  index = block*128+lane
  logical_block, word = index//54, index%54
  start = logical_block*Q6_BYTES
  value = sum((raw[(start+word*4+i).valid((index < out.shape[0]) & (word*4+i < Q6_BYTES))].load().cast(dtypes.uint32)
               << (i*8) for i in range(4)), UOp.const(0, dtypes.uint32))
  return out[index.valid(index < out.shape[0])].store(value).end(block, lane).sink(
    arg=KernelInfo(name="prepare_q6_aligned", opts_to_apply=()))

def prepare_q6(raw: Tensor, *, in_features: int | None = None, tiled: bool = False, realize: bool = True) -> Tensor:
  """Value-preserving 210->216 byte blocks, in row order or tiled (prepare_q6_tiled)."""
  if tiled: return prepare_q6_tiled(raw, in_features=in_features or 0, realize=realize)
  if raw.dtype != dtypes.uint8 or raw.numel() <= 0 or raw.numel()%Q6_BYTES:
    raise QuantGemvError("Q6 preparation requires whole uint8 GGUF blocks")
  out = Tensor.empty(raw.numel()//Q6_BYTES*(Q6_PREPARED_BYTES//4), dtype=dtypes.uint32, device=raw.device)
  fxn = functools.partial(_prepare_q6_kernel, gpu=nvidia(raw.device))
  result = Tensor(UOp.custom_kernel(out.uop, raw.uop.flatten(), fxn=fxn)[0])
  return result.realize() if realize else result

def prepare_q6_tiled(raw: Tensor, *, in_features: int, realize: bool = True) -> Tensor:
  """Q6_K rows as 216-byte blocks (210 bytes, padded) in the tiled layout: each 16-row tile of each block is one
  contiguous 3,456-byte run, like prepare_q4."""
  blocks = in_features//256
  if (raw.dtype != dtypes.uint8 or raw.numel() <= 0 or in_features <= 0 or in_features%256
      or raw.numel()%(blocks*Q6_BYTES) or (raw.numel()//(blocks*Q6_BYTES))%16):
    raise QuantGemvError("tiled Q6 preparation requires complete uint8 rows, width a multiple of 256 and rows of 16")
  rows = raw.numel()//(blocks*Q6_BYTES)
  padded = raw.flatten().reshape(rows*blocks, Q6_BYTES).pad(((0, 0), (0, Q6_PREPARED_BYTES-Q6_BYTES)))
  result = padded.reshape(rows//16, 16, blocks, Q6_PREPARED_BYTES).permute(0, 2, 1, 3).contiguous().flatten().bitcast(dtypes.uint32)
  return result.realize() if realize else result

# prepare_q2's code words: a Q2_0 block's 64 codes of a row are 4 words, word t holding the 16 codes thread t of an
# m16n8k32 int8 MMA needs for the block's two 32-value steps: register r = 2*step + half (half: k 0-15 or 16-31 of the
# step) has code j (k = 32*step + 16*half + 4t + j) at bits 8j + 2r, so a register is (word >> 2r) & 0x03030303.
def q2_code_position(k: int) -> tuple[int, int]:
  """(word within the block, bit) of code k (0-63) of a Q2_0 block in prepare_q2's layout."""
  step, half, t, j = k // 32, (k % 32) // 16, (k % 16) // 4, k % 4
  return t, 8 * j + 2 * (2 * step + half)

# prepare_q2 tiles rows by 8 and blocks by 2: for each 8-row tile and pair of 64-value blocks, each row's 8 code words
# (block 2p's 4, then 2p+1's: 32 contiguous bytes, the sector the one-row GEMV reads), then the tile's 16 FP16 scales
# (8 words: block 2p's rows 0-7, then 2p+1's). An MMA warp's 8 rows of one block are 16 bytes of each row's 32;
# the pair's two loads cover 256 contiguous bytes.
Q2_TILE_ROWS, Q2_TILE_WORDS = 8, 72

def q2_word(row, block, word, blocks: int):
  """Index of code word `word` (0-3) of `row`'s Q2_0 block `block` (of `blocks` a row) in prepare_q2's words."""
  return ((row // Q2_TILE_ROWS) * (blocks // 2) + block // 2) * Q2_TILE_WORDS + (row % Q2_TILE_ROWS) * 8 + (block % 2) * 4 + word

def q2_scale_word(row, block, blocks: int):
  """Index of the word holding `row`'s scale of block `block`, in its half (row % 2)."""
  return ((row // Q2_TILE_ROWS) * (blocks // 2) + block // 2) * Q2_TILE_WORDS + 64 + (block % 2) * 4 + (row % Q2_TILE_ROWS) // 2

def q2_scale(raw: UOp, row: UOp, block, blocks: int) -> UOp:
  """`row`'s FP32 scale of Q2_0 block `block` from prepare_q2's words."""
  return _half((raw[q2_scale_word(row, block, blocks)].load() >> ((row % 2) * 16).cast(dtypes.uint32)) & 65535)

def prepare_q2(raw: Tensor, *, in_features: int, realize: bool = True) -> Tensor:
  """Q2_0 blocks (FP16 scale, 16 code bytes) as words in 8-row tiles of block pairs (q2_word, q2_scale_word): each
  word holds 16 codes ready to mask into dp4a/MMA registers in two operations; nothing is padded (2.25 bits a weight)."""
  blocks = in_features // 64
  if raw.dtype != dtypes.uint8 or in_features % 128 or raw.numel() <= 0 or raw.numel() % (18 * blocks * Q2_TILE_ROWS):
    raise QuantGemvError("Q2_0 preparation requires whole uint8 blocks, rows of a multiple of 128 values and a multiple of 8 rows")
  rows, pairs = raw.numel() // (18 * blocks), blocks // 2
  data = raw.flatten().reshape(-1, 18)
  quarters = Tensor([1, 4, 16, 64], dtype=dtypes.uint32, device=raw.device)
  codes = ((data[:, 2:].reshape(-1, 16, 1).cast(dtypes.uint32) // quarters) % 4).reshape(-1, 64)
  # (block, step, half, t, j) -> word t gets code << (8j + 2(2 step + half))
  codes = codes.reshape(-1, 2, 2, 4, 4).permute(0, 3, 1, 2, 4).reshape(-1, 4, 16)
  words = (codes * _q2_shifts(raw.device)).sum(-1, dtype=dtypes.uint32)  # the codes' bits never overlap: the sum is a bitwise or
  words = words.reshape(rows // 8, 8, pairs, 8).permute(0, 2, 1, 3).reshape(rows // 8, pairs, 64)
  scales = data[:, :2].reshape(rows // 8, 8, pairs, 2, 2).permute(0, 2, 3, 1, 4).contiguous().reshape(rows // 8, pairs, 32).bitcast(dtypes.uint32)
  result = words.cat(scales, dim=-1).flatten()
  return result.realize() if realize else result

def _q2_shifts(device) -> Tensor:
  """Each code's multiplier in its word, in (step, half, j) order (q2_code_position)."""
  shifts = [2 ** (8 * j + 2 * (2 * step + half)) for step in range(2) for half in range(2) for j in range(4)]
  return Tensor(shifts, dtype=dtypes.uint32, device=device)

def dequant_prepared_q2(packed: Tensor, *, in_features: int) -> Tensor:
  """FP32 weights straight from prepare_q2's words."""
  pairs = in_features // 128
  tiles = packed.numel() // (pairs * Q2_TILE_WORDS)
  data = packed.reshape(tiles, pairs, Q2_TILE_WORDS)
  words = data[..., :64].reshape(tiles, pairs, 8, 2, 4, 1)                              # (tile, pair, row, block, t)
  codes = (words // _q2_shifts(packed.device).reshape(1, 1, 1, 1, 1, 16)) % 4
  codes = codes.reshape(tiles, pairs, 8, 2, 4, 2, 2, 4).permute(0, 2, 1, 3, 5, 6, 4, 7)  # (tile, row, pair, block, step, half, t, j)
  scales = data[..., 64:].contiguous().bitcast(dtypes.float16).float().reshape(tiles, pairs, 2, 8).permute(0, 3, 1, 2)  # (tile, row, pair, block)
  values = (codes.reshape(tiles, 8, pairs, 2, 64).cast(dtypes.int32) - 1).float() * scales.unsqueeze(-1)
  return values.flatten()

def dequant_prepared_q4(packed: Tensor, *, in_features: int) -> Tensor:
  """FP32 weights straight from prepare_q4's tiled words (GGUF Q4_K field math)."""
  blocks = untile(packed, in_features=in_features, block_bytes=Q4_WORDS*4)
  d, dmin = (blocks[:, i:i+2].bitcast(dtypes.float16).float().unsqueeze(-1) for i in (0, 2))
  s = blocks[:, 4:16]
  scales = s[:, 0:4].bitwise_and(63).cat(s[:, 8:12].bitwise_and(15).bitwise_or(s[:, 0:4].rshift(6).lshift(4)), dim=-1)
  minimums = s[:, 4:8].bitwise_and(63).cat(s[:, 8:12].rshift(4).bitwise_or(s[:, 4:8].rshift(6).lshift(4)), dim=-1)
  qs = blocks[:, 16:144].reshape(-1, 4, 32)  # each 32-byte chunk: a low-nibble group, then a high-nibble group
  q = Tensor.stack(qs.bitwise_and(15), qs.rshift(4), dim=2).reshape(-1, 8, 32)
  return (d * scales.unsqueeze(-1) * q - dmin * minimums.unsqueeze(-1)).flatten()

def _nibbles(data: Tensor, bits: int) -> Tensor:
  shifts = Tensor.const(tuple(2**(i*bits) for i in range(8//bits)), dtype=data.dtype)
  return data.unsqueeze(-1).div(shifts, rounding_mode="trunc").bitwise_and(0xFF >> (8-bits)).transpose(-1, -2).flatten(-2)

def dequant_prepared_q6(packed: Tensor, *, in_features: int = 0, tiled: bool = False) -> Tensor:
  """FP32 weights straight from prepare_q6's 216-byte blocks (GGUF Q6_K field math, read in place)."""
  if tiled:
    blocks = untile(packed, in_features=in_features, block_bytes=Q6_PREPARED_BYTES)
  else: blocks = packed.bitcast(dtypes.uint8).reshape(-1, Q6_PREPARED_BYTES).contiguous()
  low = _nibbles(blocks[:, :128].reshape(-1, 2, 64), 4)
  high = _nibbles(blocks[:, 128:192].reshape(-1, 2, 32), 2).lshift(4)
  q = (low.bitwise_or(high).bitcast(dtypes.int8) - 32).reshape(-1, 256)
  scales = blocks[:, 192:208].bitcast(dtypes.int8).unsqueeze(-1).expand(-1, 16, 16).reshape(-1, 256)
  return (blocks[:, 208:210].bitcast(dtypes.float16).float() * q * scales).flatten()

def _q6_word(raw:UOp, base:UOp, byte_offset:UOp) -> UOp:
  # Q6 blocks are 210 bytes: every other block is only halfword-aligned.
  # Combine two aligned u16 loads, never reinterpret the address as an aligned u32.
  index = base + byte_offset // 2
  return raw[index].load().cast(dtypes.uint32) | (raw[index + 1].load().cast(dtypes.uint32) << 16)

def _q4_scales(raw:UOp, base:UOp, subgroup:UOp) -> tuple[UOp, UOp, UOp, UOp]:
  """Q4_K/Q5_K shared scale layout: d, dmin, scale[4], minimum[4] in 12 bytes."""
  a = _load_byte(raw, base, 4+subgroup)
  b = _load_byte(raw, base, 8+subgroup)
  c = _load_byte(raw, base, subgroup)
  scale = (subgroup < 4).where(a & 63, (b & 15) | ((c >> 6) << 4))
  minimum = (subgroup < 4).where(b & 63, (b >> 4) | ((a >> 6) << 4))
  header = _q4_word(raw, base, 0)
  d, dmin = (header & 0xffff).cast(dtypes.uint16), (header >> 16).cast(dtypes.uint16)
  return _half(d), _half(dmin), scale.float(), minimum.float()

def _q8_quantize_kernel(q:UOp, scale:UOp, xsum:UOp, x:UOp, tokens:int, in_features:int) -> UOp:
  """Per-32-group Q8_0-style activation quantize: 8 int8 per 4 lanes, one fp32 scale."""
  groups = in_features // Q8_GROUP_SIZE
  token_group, lane = UOp.range(tokens * groups, 0), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  token, group = token_group // groups, token_group % groups
  x = x.reshape(tokens, groups, 32)
  group_scale = (warp_reduce(x[token, group, lane].float().abs(), maximum=True, full_wave=True) / 127).maximum(1e-8)
  word_lane = lane.minimum(7)
  xs = tuple(x[token, group, word_lane * 4 + i].float() for i in range(4))
  word = sum(((v / group_scale).round().clip(-127, 127).cast(dtypes.int8).cast(dtypes.uint8).cast(dtypes.uint32)
              << (i * 8) for i, v in enumerate(xs)), UOp.const(0, dtypes.uint32))
  # Sum signed quantized bytes once per activation group, not per output row.
  word_sum = _dp4a(UOp.const(0x01010101, dtypes.uint32), word, UOp.const(0, dtypes.int32))
  # Each half has its own Q6 scale. Q4 uses the sum of both halves.
  # Integer sums in [-2032, 2032] are exact throughout these FP32 reductions.
  half_sums = tuple(warp_reduce(((lane >= half*4) & (lane < (half+1)*4)).where(word_sum.float(), UOp.const(0, dtypes.float32)),
                                full_wave=True) for half in range(2))
  stores = (q[token, group, lane.valid(lane < 8)].store(word),
            scale[token, group.valid(lane.eq(0))].store(group_scale),
            *(xsum[token, group.valid(lane.eq(0)), half].store(value.cast(dtypes.int32)) for half, value in enumerate(half_sums)))
  return UOp.group(*stores).end(token_group, lane).sink(arg=KernelInfo(name="q8_quantize", opts_to_apply=()))

def q8_quantize(x:Tensor, tokens:int, in_features:int) -> tuple[Tensor, Tensor, Tensor]:
  groups = in_features // Q8_GROUP_SIZE
  q = Tensor.empty(tokens, groups, 8, dtype=dtypes.uint32, device=x.device)
  scale = Tensor.empty(tokens, groups, dtype=dtypes.float32, device=x.device)
  xsum = Tensor.empty(tokens, groups, 2, dtype=dtypes.int32, device=x.device)
  q, scale, xsum = Tensor.custom_kernel(q, scale, xsum, x, fxn=functools.partial(_q8_quantize_kernel, tokens=tokens, in_features=in_features))[:3]
  return q, scale, xsum

def q8_64_slot(word):
  """Where q8_quantize64 stores word `word` (values 4 word .. 4 word + 3) of a 64-value group: an int8 MMA thread t
  needs words t, t + 4 (k 0-31) and 8 + t, 12 + t (k 32-63), so they sit together at slots 4t .. 4t + 3 for one
  16-byte load: slot 4t + 2 step + w for word 8 step + 4 w + t. Works on ints and UOps."""
  return 4 * (word % 4) + 2 * (word // 8) + (word % 8) // 4

def _q8_quantize64_kernel(q: UOp, scale: UOp, x: UOp, tokens: int, in_features: int) -> UOp:
  """Int8 activations with one FP32 scale per 64 values (a Q2_0 block): a warp per group, two values a lane; lanes
  0-15 each pack four into a word, stored in q8_64_slot order."""
  groups = in_features // 64
  token_group, lane = UOp.range(tokens * groups, 0), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  token, group = token_group // groups, token_group % groups
  x = x.reshape(tokens, groups, 64)
  biggest = x[token, group, lane].float().abs().maximum(x[token, group, lane + 32].float().abs())
  group_scale = (warp_reduce(biggest, maximum=True, full_wave=True) / 127).maximum(1e-8)
  word_lane = lane.minimum(15)
  xs = tuple(x[token, group, word_lane * 4 + i].float() for i in range(4))
  word = sum(((v / group_scale).round().clip(-127, 127).cast(dtypes.int8).cast(dtypes.uint8).cast(dtypes.uint32)
              << (i * 8) for i, v in enumerate(xs)), UOp.const(0, dtypes.uint32))
  stores = (q[token, group, q8_64_slot(word_lane).valid(lane < 16)].store(word), scale[token, group.valid(lane.eq(0))].store(group_scale))
  return UOp.group(*stores).end(token_group, lane).sink(arg=KernelInfo(name="q8_quantize64", opts_to_apply=()))

def q8_quantize64(x: Tensor, tokens: int, in_features: int) -> tuple[Tensor, Tensor]:
  """(tokens, in_features / 64, 16) words of int8 values (q8_64_slot order) and (tokens, in_features / 64) scales."""
  groups = in_features // 64
  q = Tensor.empty(tokens, groups, 16, dtype=dtypes.uint32, device=x.device)
  scale = Tensor.empty(tokens, groups, dtype=dtypes.float32, device=x.device)
  q, scale = Tensor.custom_kernel(q, scale, x, fxn=functools.partial(_q8_quantize64_kernel, tokens=tokens, in_features=in_features))[:2]
  return q, scale

def _decode_warps(ggml_type: int, in_features: int, out_features: int) -> int:
  # RTX 5060, tiled Q4_K: 8 warps (eight adjacent rows, 1,152 contiguous bytes per block) 3.72 ms for all
  # projections against 3.98-4.23 ms with 4.
  if ggml_type == Q4_K: return 8
  # RTX 5060: the Qwen3.5-4B Q6 output head (tied to the embedding) takes 1.31 ms with 4 or 8 warps
  # against 1.45-1.48 ms with 1; FFN-down was screened earlier and stays at 1.
  if ggml_type == Q6_K and (out_features, in_features) == (248320, 2560): return 8
  # RTX 5060, Bonsai-4B Q2_0: the 248320-row head reads at 165 GB/s with 1 warp, 315 with 2, 363 with 4; decode
  # 73 -> 94.6 tok/s with 4 (93.7 with 2).
  if ggml_type == Q2_0: return 4
  return 1

def _decode_linear(out:UOp, out_features:int, group_count:int, group_dot:Callable[[UOp, UOp, UOp], UOp], name:str,
                   warps_per_block:int=1, interleave_tokens:bool=False, reuse_tokens:bool=False, local_tokens:bool=False,
                   loop_chunks:bool=False, fuse_chunks:bool=False, residual:UOp|None=None) -> UOp:
  assert warps_per_block in (1, 2, 4, 8)
  assert out_features % warps_per_block == 0
  chunks = (group_count + 31) // 32
  row_blocks = out_features // warps_per_block
  grid_chunks = 1 if loop_chunks else chunks
  block = UOp.range((1 if reuse_tokens or local_tokens else out.shape[0])*row_blocks*grid_chunks, 0)
  lane = UOp.range(32, -1, axis_type=AxisType.WARP)
  warp = UOp.range(warps_per_block, 1, axis_type=AxisType.LOCAL)
  chunk = UOp.range(chunks, 3, axis_type=AxisType.LOOP) if loop_chunks else block%chunks
  token = (UOp.range(out.shape[0], 2, axis_type=AxisType.LOCAL) if local_tokens else
           (block//grid_chunks)%out.shape[0] if interleave_tokens else block // (row_blocks*grid_chunks))
  output = ((block//grid_chunks)//out.shape[0] if interleave_tokens and not (reuse_tokens or local_tokens)
            else (block//grid_chunks)%row_blocks)*warps_per_block+warp
  group = lane+chunk*32
  stores = []
  # Static token unrolling gives identical weight indices to every row. UOp
  # hash-consing shares the packed loads/decode expressions across their dots;
  # each token retains its original integer dot and FP32 reduction order.
  for current in (range(out.shape[0]) if reuse_tokens else (token,)):
    current = UOp.const(current, dtypes.int32) if isinstance(current, int) else current
    if group_count%32 == 0:
      value = group_dot(current, output, group)
    else:
      value = (group < group_count).where(group_dot(current, output, group.minimum(group_count-1)), UOp.const(0, dtypes.float32))
    total = warp_reduce(value, full_wave=True)
    if fuse_chunks:
      acc = UOp.placeholder((1,), dtypes.float32, slot=len(stores), addrspace=AddrSpace.REG)
      initial = UOp.const(0.0, dtypes.float32) if residual is None else residual[current, output].load()
      acc = acc.after(acc[0].store(initial))
      # A normal ADD is flattened with the last shuffle ADD by tinygrad.
      # The opaque rounded intrinsic preserves each completed warp sum before
      # adding it to the residual/previous chunks, as a stored partial does.
      added = UOp(Ops.CUSTOMI, src=(acc[0].load(), total), arg=('__fadd_rn({}, {})', dtypes.float32))
      value = chunk.eq(0).where(total, added) if residual is None else added
      update = acc[0].store(value).end(chunk)
      summed = acc.after(update)[0].load()
      stores.append(out[current, output, UOp.const(0).valid(lane.eq(0))].store(summed.cast(out.dtype)))
    else:
      stores.append(out[current, output, chunk.valid(lane.eq(0))].store(total.cast(out.dtype)))
  stored = UOp.group(*stores).end(*((chunk,) if loop_chunks and not fuse_chunks else ()))
  return stored.end(block, warp, lane, *((token,) if local_tokens else ())).sink(arg=KernelInfo(
    name=name+('_reuse' if reuse_tokens else '_local' if local_tokens else '')+('_loop' if loop_chunks else '')
         +('_sum' if fuse_chunks else ''), opts_to_apply=()))

def _q2_slices(out_features: int, in_features: int) -> int:
  """K slices for _q2_decode_kernel: the fewest that give ~960 warps (32 per SM on an RTX 5060), each a whole
  number of 4-pair steps."""
  tiles, pairs = out_features // Q2_TILE_ROWS, in_features // 128
  fits = [s for s in range(1, pairs//4 + 1) if pairs % (4*s) == 0]
  return next((s for s in fits if tiles*s >= 960), fits[-1])

@functools.cache
def _q2_decode_kernel(out: UOp, raw: UOp, xq: UOp, xd: UOp, xs: UOp, *, tokens: int, out_features: int, in_features: int,
                      slices: int) -> UOp:
  """One-row Q2_0 GEMV over prepare_q2's tiles: a warp per 8-row tile and K slice, lane = (block pair % 4, row), so
  each step reads four consecutive pair units, ~1 KB contiguous; the dp4a dots are the generic path's. A row's sum
  meets over two shuffles, and a tile's slices (warps of one CTA) add theirs through shared memory, so the output is
  final: no separate kernel adds partial sums."""
  tiles, pairs = out_features // Q2_TILE_ROWS, in_features // 128
  per = pairs // slices
  group_tiles = math.gcd(tiles, max(1, 4 // slices))  # tiles a CTA covers: about four warps
  block = UOp.range(tokens * tiles // group_tiles, 0, axis_type=AxisType.GLOBAL)
  warp = UOp.range(group_tiles * slices, 1, axis_type=AxisType.LOCAL)
  lane = UOp.range(32, -1, axis_type=AxisType.WARP)
  token, tile = block // (tiles // group_tiles), (block % (tiles // group_tiles)) * group_tiles + warp // slices
  piece, local_tile = warp % slices, warp // slices
  row, quarter = lane % Q2_TILE_ROWS, lane // Q2_TILE_ROWS
  acc = UOp.placeholder((1,), dtypes.float32, slot=5, addrspace=AddrSpace.REG)
  acc = acc.after(acc[0].store(acc.const_like(0)))
  step = UOp.range(per // 4, 2, axis_type=AxisType.LOOP)
  pair = piece * per + step * 4 + quarter
  unit = (tile * pairs + pair) * Q2_TILE_WORDS
  total = acc.after(step)[0].load()
  for b in range(2):  # blocks 2 pair and 2 pair + 1: this row's 32 contiguous bytes, and its two scales
    words = _load_lanes(raw[unit + row * 8 + b * 4], 4)
    d = _half((raw[unit + 64 + b * 4 + row // 2].load() >> ((row % 2) * 16).cast(dtypes.uint32)) & 65535)
    for half32 in range(2):  # the block's two 32-value groups
      group = pair * 4 + b * 2 + half32
      xwords = _load_lanes(xq[token, group, 0], 8)
      dot = UOp.const(0, dtypes.int32)
      for t in range(4):
        for half in range(2):  # codes k = 16*half + 4t + j of the group, against activation word 4*half + t
          dot = _dp4a((words[t] >> (half32 * 4 + half * 2)) & 0x03030303, xwords[half * 4 + t], dot)
      qsum = xs[token, group, 0].load() + xs[token, group, 1].load()  # dot(q - 1, x) == dot(q, x) - sum(x)
      total = total + (dot - qsum).float() * d * xd[token, group]
  done = acc[0].store(total).end(step)
  value = acc.after(done)[0].load()
  for offset in (16, 8):  # the row's four pair quarters
    value = value + UOp(Ops.CUSTOM, src=(value,), arg=(f"__shfl_xor_sync(0xffffffff, {{0}}, {offset})", dtypes.float))
  if slices > 1:  # each slice's row sums in shared memory; slice 0 adds the others in order
    shared = UOp.placeholder((group_tiles, slices, Q2_TILE_ROWS), dtypes.float32, slot=6, addrspace=AddrSpace.LOCAL)
    written = shared[local_tile, piece, row.valid(lane < Q2_TILE_ROWS)].store(value).barrier()
    shared = shared.after(written)
    value = functools.reduce(lambda a, b: a + b, (shared[local_tile, i, row].load() for i in range(slices)))
  store = out[token, (tile * Q2_TILE_ROWS + row).valid((lane < Q2_TILE_ROWS) & piece.eq(0))].store(value)
  return store.end(block, warp, lane).sink(arg=KernelInfo(name="q2_0_decode_tiled", opts_to_apply=()))

def _quant_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, out_features:int, in_features:int, ggml_type:int,
                         q6_aligned: bool = False, q6_tiled: bool = False, q4_tiled: bool = False,
                         interleave_tokens:bool=False, reuse_tokens:bool=False, local_tokens:bool=False,
                         loop_chunks:bool=False, fuse_chunks:bool=False, residual:UOp|None=None) -> UOp:
  group_count = in_features // Q8_GROUP_SIZE
  def group_dot(token: UOp, output: UOp, group: UOp) -> UOp:
    block, subgroup = group // 8, group % 8
    xwords = _load_lanes(xq[token, group, 0], 8)
    if ggml_type in (Q4_K, Q5_K):
      logical = output*in_features//GGML_BLOCK_SIZE+block
      block_words = Q4_WORDS if ggml_type == Q4_K else 44
      # The tiled layout (prepare_q4, for Q4_K and Q5_K): each 16-row tile of each block contiguous, native word order.
      base = ((output//16*(in_features//GGML_BLOCK_SIZE)+block)*16+output%16)*block_words if q4_tiled else logical*block_words
      payload_start = (4 if ggml_type == Q4_K else 12)+(subgroup//2)*8
      dot = UOp.const(0, dtypes.int32)
      for word_idx in range(8):
        word = (_q4_word(raw, base, payload_start+word_idx)
                >> ((subgroup & 1)*4).cast(dtypes.uint32)) & 0x0f0f0f0f
        if ggml_type == Q5_K:
          high = (_q4_word(raw, base, 4+word_idx) >> subgroup.cast(dtypes.uint32)) & 0x01010101
          word = word | (high << 4)
        dot = _dp4a(word, xwords[word_idx], dot)
      d, dmin, scale, minimum = _q4_scales(raw, base, subgroup)
      qsum = (xs[token, group, 0].load() + xs[token, group, 1].load()).float()
      return (dot.float() * d * scale - qsum * dmin * minimum) * xd[token, group]
    if ggml_type == Q6_K:
      base = (output*in_features//GGML_BLOCK_SIZE+block) * (Q6_PREPARED_BYTES//4 if q6_aligned else Q6_BYTES//2)
      if q6_tiled: base = ((output//16*(in_features//GGML_BLOCK_SIZE)+block)*16+output%16)*(Q6_PREPARED_BYTES//4)
      dots = [UOp.const(0, dtypes.int32)] * 2
      if q6_aligned:
        # 216-byte blocks prove 8-byte alignment to the pinned coalescer.
        low_words = _load_lanes(raw[base+(subgroup//4)*16+(subgroup%2)*8], 8)
        high_words = _load_lanes(raw[base+32+(subgroup//4)*8], 8)
      for word_idx in range(8):
        low_offset = (subgroup//4)*64 + (subgroup%2)*32 + word_idx*4
        high_offset = 128 + (subgroup//4)*32 + word_idx*4
        low_word = low_words[word_idx] if q6_aligned else _q6_word(raw, base, low_offset)
        high_word = high_words[word_idx] if q6_aligned else _q6_word(raw, base, high_offset)
        low = low_word >> ((subgroup%4//2)*4).cast(dtypes.uint32)
        high = high_word >> ((subgroup%4)*2).cast(dtypes.uint32)
        # Four independent values in [0, 63]; no byte unpack/repack or signed offset here.
        word = (low & 0x0f0f0f0f) | ((high & 0x03030303) << 4)
        dots[word_idx // 4] = _dp4a(word, xwords[word_idx], dots[word_idx // 4])
      d_index = base+(52 if q6_aligned else 104)
      scale_word = ((raw[base+48+subgroup//2].load() >> ((subgroup%2)*16).cast(dtypes.uint32)) & 65535
                    if q6_aligned else raw[base + 96 + subgroup].load().cast(dtypes.uint32))
      scales = [(scale_word >> (half*8)).cast(dtypes.uint8).bitcast(dtypes.int8).float() for half in range(2)]
      # dot((q-32), x) == dot(q, x) - 32*sum(x), independently for each 16-value scale group.
      corrected = [(dots[half] - 32*xs[token, group, half].load()).float() for half in range(2)]
      return (corrected[0]*scales[0] + corrected[1]*scales[1]) * xd[token, group] * _half(raw[d_index].load())
    if ggml_type == Q2_0:
      # prepare_q2: this lane's group is one 32-value step of its block, whose 4 code words hold its 8 registers
      words = _load_lanes(raw[q2_word(output, group//2, 0, in_features//64)], 4)
      step = (group % 2).cast(dtypes.uint32)
      dot = UOp.const(0, dtypes.int32)
      for t in range(4):
        for half in range(2):  # codes k = 16*half + 4t + j of the step, against activation word 4*half + t
          codes = (words[t] >> (step*4 + half*2)) & 0x03030303
          dot = _dp4a(codes, xwords[half*4+t], dot)
      d = q2_scale(raw, output, group//2, in_features//64)
      # dot(q - 1, x) == dot(q, x) - sum(x)
      qsum = xs[token, group, 0].load() + xs[token, group, 1].load()
      return (dot - qsum).float() * d * xd[token, group]
    raise QuantGemvError(f"unsupported quant type {ggml_type}")
  names = {Q4_K: "linear_q4_k", Q5_K: "linear_q5_k", Q6_K: "linear_q6", Q2_0: "linear_q2_0"}
  # Keep CTA-local batches small: one output warp per token, at most 256
  # threads. Tokens share the SM's weight cache without unrolled registers.
  warps = 1 if local_tokens else _decode_warps(ggml_type, in_features, out_features)
  return _decode_linear(out, out_features, group_count, group_dot, names[ggml_type], warps_per_block=warps,
                        interleave_tokens=interleave_tokens, reuse_tokens=reuse_tokens, local_tokens=local_tokens,
                        loop_chunks=loop_chunks, fuse_chunks=fuse_chunks, residual=residual)

def quant_gemv(x:Tensor, packed:Tensor, *, out_features:int, in_features:int, ggml_type:int, q6_aligned: bool = False,
               q6_tiled: bool = False, q4_tiled: bool = False,
               q8: tuple[Tensor, Tensor, Tensor] | None = None, interleave_tokens:bool=False,
               exact_rows:bool=False, return_parts:bool=False, reuse_tokens:bool=False, local_tokens:bool=False,
               loop_chunks:bool=False, fuse_chunks:bool=False, residual:Tensor|None=None) -> Tensor:
  if x.dtype not in dtypes.floats: raise QuantGemvError(f"x must be float, got {x.dtype}")
  if ggml_type not in QUANT_SIZES: raise QuantGemvError(f"unsupported quant type {ggml_type}")
  if in_features <= 0 or in_features % GGML_BLOCK_SIZE: raise QuantGemvError("in_features must be a positive multiple of 256")
  if out_features <= 0 or out_features % 16: raise QuantGemvError("out_features must be a positive multiple of 16")
  if not x.shape or x.shape[-1] != in_features: raise QuantGemvError("activation shape does not match in_features")
  if x.device != packed.device: raise QuantGemvError("activations and packed weights must share a device")

  if q4_tiled and (ggml_type not in (Q4_K, Q5_K) or out_features%16): raise QuantGemvError("tiled Q4/Q5 requires Q4_K/Q5_K and 16-row tiles")
  if q6_tiled and (not q6_aligned or out_features%16): raise QuantGemvError("tiled Q6 requires prepared storage and 16-row tiles")
  if q6_aligned and ggml_type != Q6_K: raise QuantGemvError("aligned Q6 layout requires Q6_K")
  expected_dtype = dtypes.uint32 if ggml_type in (Q4_K, Q5_K, Q2_0) or q6_aligned else dtypes.uint16
  expected_bytes = out_features * in_features // GGML_BLOCK_SIZE * (Q6_PREPARED_BYTES if q6_aligned else QUANT_SIZES[ggml_type])
  if packed.dtype != expected_dtype or packed.numel() * packed.dtype.itemsize != expected_bytes:
    raise QuantGemvError("packed weight dtype/size does not match the quantization contract")

  tokens = int(x.numel()) // in_features
  if (reuse_tokens or local_tokens) and not 1 <= tokens <= 8: raise QuantGemvError('weight reuse requires one to eight tokens')
  if reuse_tokens and local_tokens: raise QuantGemvError('select either unrolled or CTA-local token reuse')
  if type(fuse_chunks) is not bool: raise QuantGemvError('fuse_chunks must be boolean')
  # These small reductions are left-to-right on the pinned CUDA scheduler.
  # Do not silently substitute sequential accumulation for larger/tree sums.
  if fuse_chunks and (in_features+1023)//1024 not in (1, 2, 3, 4, 10):
    raise QuantGemvError('fused projection supports validated chunk counts 1/2/3/4/10')
  if residual is not None and (not fuse_chunks or residual.shape != (*x.shape[:-1], out_features)
                               or residual.dtype != dtypes.float32 or residual.device != x.device):
    raise QuantGemvError('fused residual requires matching FP32 output rows and fused chunks')
  raw = packed.uop
  xq, xd, xs = q8_quantize(x, tokens, in_features) if q8 is None else q8
  groups = in_features//Q8_GROUP_SIZE
  for tensor, shape, dtype in ((xq, (tokens, groups, 8), dtypes.uint32),
                               (xd, (tokens, groups), dtypes.float32), (xs, (tokens, groups, 2), dtypes.int32)):
    if tensor.shape != shape or tensor.dtype != dtype or tensor.device != x.device:
      raise QuantGemvError("shared Q8 activation dtype/shape/device does not match the projection")
  if ggml_type == Q2_0 and in_features % 512 == 0 and not (interleave_tokens or reuse_tokens or local_tokens or loop_chunks or fuse_chunks
                                                           or exact_rows):
    # prepare_q2's tiles, read a tile at a time, four pairs a step (the generic path below reads one row's blocks)
    out = Tensor.empty(tokens, out_features, dtype=dtypes.float32, device=x.device)
    result = Tensor.custom_kernel(out, packed, xq, xd, xs, fxn=functools.partial(_q2_decode_kernel, tokens=tokens,
      out_features=out_features, in_features=in_features, slices=_q2_slices(out_features, in_features)))[0]
    return result.unsqueeze(-1) if return_parts else result.reshape(*x.shape[:-1], out_features)  # parts: one, already summed
  decode = functools.partial(_quant_decode_kernel, ggml_type=ggml_type, q6_aligned=q6_aligned, q6_tiled=q6_tiled,
                             q4_tiled=q4_tiled, interleave_tokens=interleave_tokens, reuse_tokens=reuse_tokens,
                             local_tokens=local_tokens, loop_chunks=loop_chunks or fuse_chunks, fuse_chunks=fuse_chunks)
  chunks = 1 if fuse_chunks else (in_features + 1023) // 1024
  out = Tensor.empty(tokens, out_features, chunks, dtype=dtypes.float32, device=x.device).uop
  sources = (out, raw, xq.uop, xd.uop, xs.uop)
  if residual is not None: sources += (residual.reshape(tokens, out_features).contiguous().uop,)
  params = tuple(UOp.placeholder_like(s, slot=i) for i, s in enumerate(sources))
  kernel = decode(*params[:5], out_features=out_features, in_features=in_features,
                  residual=params[5] if residual is not None else None).call(*sources)
  result = Tensor(out.after(kernel))
  if return_parts: return result
  result = (Tensor.cat(*(result[i:i+1].contiguous().sum(-1) for i in range(tokens)), dim=0)
            if exact_rows and tokens > 1 else result.sum(-1))

  return result.reshape(*x.shape[:-1], out_features)

def packed_weight(decoded: Tensor) -> tuple[Tensor, int] | None:
  """Extract the packed GGML weight view from a lazy fp16-cast quantized tensor."""