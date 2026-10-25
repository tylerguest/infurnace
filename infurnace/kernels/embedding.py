"""Row gather from a packed GGUF table (NV): decode only the requested rows.

A one-hot gather over a 248k-row packed table scans every row, and a dense FP16
copy costs ~1.3 GB. This kernel reads token ids on device and decodes their rows
with the same element decoder as the packed GEMMs (Q2_0: prepare_q2's layout).
"""
import functools

from tinygrad import Tensor, UOp, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo

from infurnace.kernels.quant_gemm import _weight
from infurnace.kernels.quant_gemv import Q2_0, Q4_K, Q5_K, Q6_K, _half

GATHER_TYPES = (Q4_K, Q5_K, Q6_K, Q2_0)

def _q2_weight(raw: UOp, row: UOp, col: UOp, width: int, rows: int) -> UOp:
  """A Q2_0 value from prepare_q2's words: every code word (16 codes each, rows in order), then FP16 scales two a word."""
  code = (raw[(row*width+col)//16].load() >> ((col%16)*2).cast(dtypes.uint32)) & 3
  index = row*(width//64)+col//64
  scale = _half((raw[rows*width//16+index//2].load() >> ((index%2)*16).cast(dtypes.uint32)) & 65535)
  return (code.cast(dtypes.int32)-1).float()*scale

@functools.cache
def _gather_kernel(out: UOp, raw: UOp, ids: UOp, *, count: int, width: int, typ: int, q6_aligned: bool, q6_tiled: bool,
                   q4_tiled: bool, rows: int) -> UOp:
  block = UOp.range(count*(width//128), 0, axis_type=AxisType.GLOBAL)
  lane = UOp.range(128, 1, axis_type=AxisType.LOCAL)
  index, column = block//(width//128), (block%(width//128))*128+lane
  row = ids[index].load().cast(dtypes.int32)
  value = _q2_weight(raw, row, column, width, rows) if typ == Q2_0 else _weight(raw, row, column, width, typ, q6_aligned, q6_tiled, q4_tiled)
  return out[index, column].store(value.float()).end(block, lane).sink(arg=KernelInfo(name="packed_gather", opts_to_apply=()))

def packed_gather(ids: Tensor, packed: Tensor, *, width: int, ggml_type: int, q6_aligned: bool = False,
                  q6_tiled: bool = False, q4_tiled: bool = False, rows: int = 0) -> Tensor:
  """Rows `ids` (int32, any shape) of a packed (rows, width) table as FP32. Q2_0 needs `rows` (its scales follow
  every row's codes)."""
  if width % 128: raise ValueError("gather width must be a multiple of 128")
  # Another type's bytes would decode as if they were Q6_K, silently wrong
  if ggml_type not in GATHER_TYPES: raise ValueError(f"packed_gather cannot decode GGML type {ggml_type}")
  if ggml_type == Q2_0 and rows <= 0: raise ValueError("a Q2_0 gather needs the table's row count")
  flat = ids.flatten().cast(dtypes.int32).contiguous()
  out = Tensor.empty(flat.numel(), width, dtype=dtypes.float32, device=packed.device)
  out = Tensor.custom_kernel(out, packed.flatten().contiguous(), flat, fxn=functools.partial(
    _gather_kernel, count=flat.numel(), width=width, typ=ggml_type, q6_aligned=q6_aligned, q6_tiled=q6_tiled,
    q4_tiled=q4_tiled, rows=rows))[0]
  return out.reshape(*ids.shape, width)

@functools.cache
def _dense_gather_kernel(out: UOp, table: UOp, ids: UOp, *, count: int, width: int) -> UOp:
  block = UOp.range(count*(width//128), 0, axis_type=AxisType.GLOBAL)
  lane = UOp.range(128, 1, axis_type=AxisType.LOCAL)
  index, column = block//(width//128), (block%(width//128))*128+lane
  row = ids[index].load().cast(dtypes.int32)
  return out[index, column].store(table[row, column].load().float()).end(block, lane).sink(
    arg=KernelInfo(name="dense_gather", opts_to_apply=()))

def dense_gather(ids: Tensor, table: Tensor) -> Tensor:
  """Rows `ids` (int32, any shape) of a dense (rows, width) table as FP32: one read per value, where tensor indexing
  would compare each id against every row (248k for a vocabulary table)."""
  width = table.shape[-1]
  if width % 128: raise ValueError("gather width must be a multiple of 128")
  flat = ids.flatten().cast(dtypes.int32).contiguous()
  out = Tensor.empty(flat.numel(), width, dtype=dtypes.float32, device=table.device)
  out = Tensor.custom_kernel(out, table.contiguous(), flat, fxn=functools.partial(_dense_gather_kernel, count=flat.numel(), width=width))[0]
  return out.reshape(*ids.shape, width)
