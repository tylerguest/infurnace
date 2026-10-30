"""Blockwise Walsh-Hadamard transform of activations (NV), for Hadamard-folded weights (infurnace/hadamard.py).

One CTA per block of `block` values, one thread per value: the reorder and signs are folded into the load, then
log2(block) butterfly stages run in shared memory (a buffer per stage, 2 KB each for 512 values: tinygrad won't
store twice to one offset of a buffer; one barrier a stage), and the normalization into the store. A dense
block x block matmul does the same with a 1 MB matrix read per call (~17 us a call for 2,560 values on an
RTX 5060, against a few here).
"""
import functools
import math

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

from infurnace.kernels.quant_gemv import _load_lanes, q8_64_slot, warp_reduce

VALUES = 8    # values a thread holds in _fwht_q8_kernel: 16 bytes of FP16 input
SPAN = 256    # values a warp holds there (32 lanes x VALUES)

def _shuffle(value: UOp, mask: int) -> UOp:
  return UOp(Ops.CUSTOM, src=(value,), arg=(f"__shfl_xor_sync(0xffffffff, {{0}}, {mask})", dtypes.float))

def _vector(ptr: UOp, dtype) -> list[UOp]:
  """VALUES consecutive elements at a buffer index, as FP32, in 16-byte loads."""
  lanes = 16 // dtype.itemsize
  buf, coords = ptr.src[0], ptr.src[1:]
  return [v.float() for k in range(VALUES // lanes) for v in _load_lanes(buf[(*coords[:-1], coords[-1] + k * lanes)], lanes)]

@functools.cache
def _fwht_kernel(out: UOp, x: UOp, signs: UOp, *, rows: int, width: int, block: int, perm: tuple[int, int, int] | None,
                 signs_after: bool) -> UOp:
  item = UOp.range(rows * (width // block), 0, axis_type=AxisType.GLOBAL)
  lane = UOp.range(block, 1, axis_type=AxisType.LOCAL)
  row, position = item // (width // block), (item % (width // block)) * block + lane
  source = position
  if perm is not None:  # grouped position (k, r, d) reads tiled (r, k, d)
    rep, nk, hd = perm
    source = ((position // hd) % rep) * (nk * hd) + (position // (rep * hd)) * hd + position % hd
  value = x[row, source].load().float()
  if not signs_after: value = value * signs[position].load()
  stages = int(math.log2(block))
  buffers = [UOp.placeholder((block,), dtypes.float32, slot=i, addrspace=AddrSpace.LOCAL) for i in range(stages + 1)]
  current = buffers[0].after(buffers[0][lane].store(value).barrier())
  for stage in range(stages):
    h = 1 << stage
    low = ((lane // h) % 2).eq(0)  # this lane holds the first of its pair (a, b) -> (a + b, a - b)
    mine, other = current[lane].load(), current[low.where(lane + h, lane - h)].load()
    target = buffers[stage + 1]
    current = target.after(target[lane].store(low.where(mine + other, other - mine)).barrier())
  result = current[lane].load() * (block ** -0.5)
  if signs_after: result = result * signs[position].load()
  return out[row, position].store(result).end(item, lane).sink(arg=KernelInfo(name=f"fwht_{block}", opts_to_apply=()))

@functools.cache
def _fwht_q8_kernel(q: UOp, scale: UOp, *sources: UOp, rows: int, width: int, block: int, perm: tuple[int, int, int] | None,
                    group: int, eps: float | None = None, normed: bool = False, half: bool = False, other: str = '') -> UOp:
  """The forward transform, quantized to int8 in the same pass: q8_quantize's layout (group 32, with the halves'
  sums) or q8_quantize64's (group 64). A thread holds VALUES consecutive values and a CTA max(block, SPAN): the
  butterflies within a thread's values run in registers, those across a warp's lanes by shuffles, and only those
  across warps (block 512: the last stage) through shared memory, one barrier each. A group's scale is a max over
  its 4 or 8 threads by shuffles; each thread packs its codes into two words.

  With `eps`, the input is first RMS-normalized (x / sqrt(mean(x^2) + eps) * weight, the next source): each CTA
  sums the whole row's squares itself (the row is a few KB), so the norm needs no other kernel. `normed` also
  stores the normalized row in FP16 (before the transform), `half` rounds it to FP16 before the transform too.

  `other` names a second input y (the last source) folded into the first as it loads: 'add' transforms x + y and
  stores the sum (FP32, an output after the others: the residual stream), 'swiglu' transforms silu(x) * y (a
  feed-forward's gate and up), so neither needs a kernel of its own.

  RTX 5060, 64 rows of 9216: 53 us with the earlier kernel (one value a thread, every stage through shared memory)."""
  sources = list(sources)
  xsum = sources.pop(0) if group == 32 else None
  normed_out = sources.pop(0) if normed else None
  summed_out = sources.pop(0) if other == 'add' else None
  y = sources.pop() if other in ('add', 'swiglu') else None
  x, signs = sources[0], sources[1]
  weight = sources[2] if eps is not None else None
  span = max(block, SPAN)
  warps = span // SPAN
  item = UOp.range(rows * (width // span), 0, axis_type=AxisType.GLOBAL)
  warp, lane = UOp.range(warps, 1, axis_type=AxisType.LOCAL), UOp.range(32, -1, axis_type=AxisType.WARP)
  row = item // (width // span)
  first = (item % (width // span)) * span + (warp * 32 + lane) * VALUES  # this thread's first position
  def source_of(position: UOp) -> UOp:
    if perm is None: return position
    rep, nk, hd = perm  # grouped position (k, r, d) reads tiled (r, k, d); VALUES positions stay contiguous
    return ((position // hd) % rep) * (nk * hd) + (position // (rep * hd)) * hd + position % hd
  def inputs(index: UOp) -> list[UOp]:
    values = _vector(x[row, index], x.dtype)
    if other == 'add': return [a + b for a, b in zip(values, _vector(y[row, index], y.dtype))]
    if other == 'swiglu': return [a / (1 + (-a).exp()) * b for a, b in zip(values, _vector(y[row, index], y.dtype))]
    if other == 'pair': return [a / (1 + (-a).exp()) * b for a, b in zip(values, _vector(x[row, width + index], x.dtype))]
    return values
  source = source_of(first)
  values = inputs(source)
  extra = [summed_out[row, source + i].store(v) for i, v in enumerate(values)] if summed_out is not None else []
  if eps is not None:
    # this thread's share of the row's squares, the warp's, then the CTA's
    squares = functools.reduce(lambda a, b: a + b, (v * v for j in range(width // span) for v in
                                                    inputs(j * span + (warp * 32 + lane) * VALUES)))
    total = warp_reduce(squares, full_wave=True)
    if warps > 1:
      sums = UOp.placeholder((warps,), dtypes.float32, slot=0, addrspace=AddrSpace.LOCAL)
      sums = sums.after(sums[warp.valid(lane.eq(0))].store(total).barrier())
      total = functools.reduce(lambda a, b: a + b, (sums[i].load() for i in range(warps)))
    inverse = (total * (1 / width) + eps).rsqrt()
    values = [v * inverse * w for v, w in zip(values, _vector(weight[source], dtypes.float32))]
    if half: values = [v.cast(dtypes.float16).float() for v in values]
    if normed_out is not None: extra += [normed_out[row, source + i].store(v.cast(dtypes.float16)) for i, v in enumerate(values)]
  values = [v * s for v, s in zip(values, _vector(signs[first], dtypes.float32))]
  h = 1
  while h < min(block, VALUES):  # within a thread's values
    values = [values[i] + values[i + h] if i & h == 0 else values[i - h] - values[i] for i in range(VALUES)]
    h *= 2
  while h < min(block, SPAN):  # across lanes: partner lane differs by h / VALUES
    low = (lane & (h // VALUES)).eq(0)
    others = [_shuffle(v, h // VALUES) for v in values]
    values = [low.where(v + o, o - v) for v, o in zip(values, others)]
    h *= 2
  slot = 1
  while h < block:  # across warps, through shared memory (a buffer a stage: one store per offset)
    buf = UOp.placeholder((span,), dtypes.float32, slot=slot, addrspace=AddrSpace.LOCAL)
    local = (warp * 32 + lane) * VALUES
    buf = buf.after(UOp.group(*(buf[local + i].store(v) for i, v in enumerate(values))).barrier())
    low = (local & h).eq(0)
    values = [low.where(v + buf[(local + i) ^ h].load(), buf[(local + i) ^ h].load() - v) for i, v in enumerate(values)]
    slot, h = slot + 1, h * 2
  norm = block ** -0.5
  biggest = functools.reduce(lambda a, b: a.maximum(b), (v.abs() for v in values))
  mask = 1
  while mask < group // VALUES:  # the group's threads: lanes that differ in their low bits
    biggest = biggest.maximum(_shuffle(biggest, mask))
    mask *= 2
  step = (biggest * norm / 127).maximum(1e-8)
  codes = [(v * norm / step).round().clip(-127, 127) for v in values]
  words = [sum((codes[w * 4 + i].cast(dtypes.int8).cast(dtypes.uint8).cast(dtypes.uint32) << (i * 8) for i in range(4)),
               UOp.const(0, dtypes.uint32)) for w in range(2)]
  def slot_of(position: UOp) -> UOp:  # q8_quantize64's word order for group 64
    return position // 4 if group == 32 else position // 64 * 16 + q8_64_slot(position % 64 // 4)
  in_group = lane % (group // VALUES)
  stores = [q[row, slot_of(first + w * 4)].store(word) for w, word in enumerate(words)]
  stores.append(scale[row, (first // group).valid(in_group.eq(0))].store(step))
  if xsum is not None:  # each 16-value half's sum of codes, for the K-quant GEMVs' offsets (exact integers in FP32)
    total = functools.reduce(lambda a, b: a + b, codes)
    total = total + _shuffle(total, 1)
    stores.append(xsum[row, (first // 32 * 2 + in_group // 2).valid((lane % 2).eq(0))].store(total.cast(dtypes.int32)))
  return UOp.group(*stores, *extra).end(item, warp, lane).sink(
    arg=KernelInfo(name=f"{'norm_' if eps is not None else ''}fwht_q8_{block}_{group}", opts_to_apply=()))

def _fits(width: int, block: int, perm: tuple[int, int, int] | None) -> bool:
  """Whether _fwht_q8_kernel takes this shape (else _fwht_q8_staged_kernel)."""
  return width % max(block, SPAN) == 0 and block <= 1024 and (perm is None or perm[2] % VALUES == 0)

@functools.cache
def _fwht_q8_staged_kernel(q: UOp, scale: UOp, *sources: UOp, rows: int, width: int, block: int, perm: tuple[int, int, int] | None,
                    group: int, eps: float | None = None, normed: bool = False, half: bool = False) -> UOp:
  """_fwht_q8_kernel for the shapes it doesn't take, one value a thread and a CTA a block: every butterfly stage
  through shared memory. A 32-value group is one warp; a 64-value group is two, whose maxima meet in shared
  memory. Every fourth lane packs its four values from the last stage's buffer into a word."""
  sources = list(sources)
  xsum = sources.pop(0) if group == 32 else None
  normed_out = sources.pop(0) if normed else None
  x, signs = sources[0], sources[1]
  weight = sources[2] if eps is not None else None
  item = UOp.range(rows * (width // block), 0, axis_type=AxisType.GLOBAL)
  # Warps and their lanes as their own ranges: a single block-wide range may be split across thread dimensions
  # (by the lane % 4 below), and warp shuffles need 32 consecutive values in a warp
  warp, thread = UOp.range(block // 32, 1, axis_type=AxisType.LOCAL), UOp.range(32, -1, axis_type=AxisType.WARP)
  lane = warp * 32 + thread
  row, position = item // (width // block), (item % (width // block)) * block + lane
  source = position
  if perm is not None:
    rep, nk, hd = perm
    source = ((position // hd) % rep) * (nk * hd) + (position // (rep * hd)) * hd + position % hd
  value = x[row, source].load().float()
  stages = int(math.log2(block))
  extra = []
  if eps is not None:
    # this thread's share of the row's squares, the warp's, then the CTA's (a buffer slot after the stage buffers)
    squares = functools.reduce(lambda a, b: a + b, ((v := x[row, j * block + lane].load().float()) * v for j in range(width // block)))
    partial = warp_reduce(squares, full_wave=True)
    sums = UOp.placeholder((block // 32,), dtypes.float32, slot=stages + 3, addrspace=AddrSpace.LOCAL)
    sums = sums.after(sums[warp.valid(thread.eq(0))].store(partial).barrier())
    total = functools.reduce(lambda a, b: a + b, (sums[i].load() for i in range(block // 32)))
    value = value * (total * (1 / width) + eps).rsqrt() * weight[source].load().float()
    if half: value = value.cast(dtypes.float16).float()
    if normed_out is not None: extra.append(normed_out[row, source].store(value.cast(dtypes.float16)))
  value = value * signs[position].load()
  buffers = [UOp.placeholder((block,), dtypes.float32, slot=i, addrspace=AddrSpace.LOCAL) for i in range(stages + 1)]
  current = buffers[0].after(buffers[0][lane].store(value).barrier())
  for stage in range(stages):
    h = 1 << stage
    low = ((lane // h) % 2).eq(0)
    mine, other = current[lane].load(), current[low.where(lane + h, lane - h)].load()
    target = buffers[stage + 1]
    current = target.after(target[lane].store(low.where(mine + other, other - mine)).barrier())
  norm = block ** -0.5
  biggest = warp_reduce(current[lane].load().abs(), maximum=True, full_wave=True) * norm
  if group == 64:  # the other warp of the pair
    maxima = UOp.placeholder((block // 32,), dtypes.float32, slot=stages + 1, addrspace=AddrSpace.LOCAL)
    shared = maxima.after(maxima[(lane // 32).valid((lane % 32).eq(0))].store(biggest).barrier())
    pair = (lane // 64) * 2
    biggest = shared[pair].load().maximum(shared[pair + 1].load())
  step = (biggest / 127).maximum(1e-8)
  def code(v: UOp) -> UOp: return (v * norm / step).round().clip(-127, 127)
  first = lane - lane % 4  # this word's first value
  word = sum((code(current[first + i].load()).cast(dtypes.int8).cast(dtypes.uint8).cast(dtypes.uint32) << (i * 8)
              for i in range(4)), UOp.const(0, dtypes.uint32))
  slot = position // 4 if group == 32 else position // 64 * 16 + q8_64_slot(position % 64 // 4)  # q8_quantize64's word order
  stores = [q[row, slot.valid((lane % 4).eq(0))].store(word),
            scale[row, (position // group).valid((lane % group).eq(0))].store(step)]
  if xsum is not None:  # each 16-value half's sum of codes, for the K-quant GEMVs' offsets (exact integers in FP32)
    mine = code(current[lane].load())
    for half in range(2):
      total = warp_reduce(((lane % 32) // 16).eq(half).where(mine, UOp.const(0.0, dtypes.float32)), full_wave=True)
      stores.append(xsum[row, (position // group * 2 + half).valid((lane % 32).eq(0))].store(total.cast(dtypes.int32)))
  return UOp.group(*stores, *extra).end(item, warp, thread).sink(
    arg=KernelInfo(name=f"{'norm_' if eps is not None else ''}fwht_q8_{block}_{group}", opts_to_apply=()))

def fwht_q8(x: Tensor, signs: Tensor, *, block: int, perm: tuple[int, int, int] | None = None, group: int = 32,
            up: Tensor | None = None, pair: bool = False) -> tuple[Tensor, ...]:
  """fwht(x, signs, ...) quantized to int8 in one kernel, without the FP32 result: (words, scales, half sums) as
  quant_gemv.q8_quantize returns for group 32, (words, scales) as q8_quantize64 for group 64. With `up`, of
  silu(x) * up instead (a feed-forward's down projection input, made in the same kernel); with `pair`, x's rows are
  gate then up (a merged gate-up projection's output) and the transform is of silu(gate) * up."""
  width = x.shape[-1] // (2 if pair else 1)
  if block > 1024 or block & (block - 1) or width % block or block < 64:
    raise ValueError("fwht_q8 needs a power-of-two block of 64-1024 dividing the width")
  if group not in (32, 64): raise ValueError("fwht_q8 quantizes 32- or 64-value groups")
  rows, groups = x.numel() // x.shape[-1], width // group
  outputs = [Tensor.empty(rows, width // 4, dtype=dtypes.uint32, device=x.device), Tensor.empty(rows, groups, dtype=dtypes.float32, device=x.device)]
  if group == 32: outputs.append(Tensor.empty(rows, groups * 2, dtype=dtypes.int32, device=x.device))
  fits = _fits(width, block, perm)
  if pair and not fits: x, up, pair = x[..., :width], x[..., width:], False  # the staged kernel takes neither: tensor ops
  if up is not None and not fits: x, up = x.silu() * up, None
  other = {'other': 'pair'} if pair else {'other': 'swiglu'} if up is not None else {}
  kernel = functools.partial(_fwht_q8_kernel if fits else _fwht_q8_staged_kernel, rows=rows, width=width, block=block, perm=perm,
                             group=group, **other)
  extra = (up.reshape(rows, width).contiguous(),) if up is not None else ()
  results = Tensor.custom_kernel(*outputs, x.reshape(rows, x.shape[-1]).contiguous(), signs.float().contiguous(), *extra,
                                 fxn=kernel)[:len(outputs)]
  shapes = ((rows, groups, group // 4), (rows, groups), (rows, groups, 2))
  return tuple(r.reshape(s) for r, s in zip(results, shapes))

def norm_fwht_q8(x: Tensor, weight: Tensor, eps: float, signs: Tensor, *, block: int, group: int = 32, normed: bool = False,
                 half: bool = False, addend: Tensor | None = None) -> tuple[Tensor, ...]:
  """fwht_q8 of RMSNorm(x) (x / sqrt(mean(x^2) + eps) * weight, rounded to FP16 first with `half`) in one kernel;
  with `normed`, the FP16 normalized rows (as x's shape) follow the quantized tuple. With `addend`, of x + addend,
  whose FP32 rows (as x's shape) come last."""
  width = x.shape[-1]
  if block > 1024 or block & (block - 1) or width % block or block < 64:
    raise ValueError("norm_fwht_q8 needs a power-of-two block of 64-1024 dividing the width")
  if group not in (32, 64): raise ValueError("norm_fwht_q8 quantizes 32- or 64-value groups")
  rows, groups = x.numel() // width, width // group
  outputs = [Tensor.empty(rows, width // 4, dtype=dtypes.uint32, device=x.device), Tensor.empty(rows, groups, dtype=dtypes.float32, device=x.device)]
  if group == 32: outputs.append(Tensor.empty(rows, groups * 2, dtype=dtypes.int32, device=x.device))
  if normed: outputs.append(Tensor.empty(rows, width, dtype=dtypes.float16, device=x.device))
  folded = None
  if addend is not None and not _fits(width, block, None): x, folded, addend = x + addend, x + addend, None
  if addend is not None: outputs.append(Tensor.empty(rows, width, dtype=dtypes.float32, device=x.device))
  kernel = functools.partial(_fwht_q8_kernel if _fits(width, block, None) else _fwht_q8_staged_kernel, rows=rows, width=width,
                             block=block, perm=None, group=group, eps=float(eps), normed=normed, half=half,
                             **({'other': 'add'} if addend is not None else {}))
  extra = (addend.reshape(rows, width).contiguous(),) if addend is not None else ()
  results = Tensor.custom_kernel(*outputs, x.reshape(rows, width).contiguous(), signs.float().contiguous(), weight.float().contiguous(),
                                 *extra, fxn=kernel)[:len(outputs)]
  shapes = ((rows, groups, group // 4), (rows, groups), (rows, groups, 2))[:3 if group == 32 else 2]
  quantized = tuple(r.reshape(shape) for r, shape in zip(results, shapes))
  return quantized + tuple(r.reshape(x.shape) for r in results[len(quantized):]) + ((folded,) if folded is not None else ())

def fwht(x: Tensor, signs: Tensor, *, block: int, perm: tuple[int, int, int] | None = None, signs_after: bool = False) -> Tensor:
  """x (..., width) -> FP32 (..., width): optional reorder, * signs, then the normalized transform per block (or,
  with `signs_after`, the transform then * signs: the inverse)."""
  width = x.shape[-1]
  if block > 1024 or block & (block - 1) or width % block: raise ValueError("fwht needs a power-of-two block up to 1024 dividing the width")
  rows = x.numel() // width
  out = Tensor.empty(rows, width, dtype=dtypes.float32, device=x.device)
  out = Tensor.custom_kernel(out, x.reshape(rows, width).contiguous(), signs.float().contiguous(),
                             fxn=functools.partial(_fwht_kernel, rows=rows, width=width, block=block, perm=perm, signs_after=signs_after))[0]
  return out.reshape(x.shape)
