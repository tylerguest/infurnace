"""Row gather from a packed GGUF table (NV): decode only the requested rows.

A one-hot gather over a 248k-row packed table scans every row, and a dense FP16
copy costs ~1.3 GB. This kernel reads token ids on device and decodes their rows
with the same element decoder as the packed GEMMs.
"""
import functools

from tinygrad import Tensor, UOp, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo

from infurnace.kernels.quant_gemm import _weight

@functools.cache
def _gather_kernel(out: UOp, raw: UOp, ids: UOp, *, count: int, width: int, typ: int, q6_aligned: bool, q6_swizzled: bool,
                   q4_tiled: bool) -> UOp:
  block = UOp.range(count*(width//128), 0, axis_type=AxisType.GLOBAL)
  lane = UOp.range(128, 1, axis_type=AxisType.LOCAL)
  index, column = block//(width//128), (block%(width//128))*128+lane
  row = ids[index].load().cast(dtypes.int32)
  value = _weight(raw, row, column, width, typ, q6_aligned, q6_swizzled, q4_tiled)
  return out[index, column].store(value.float()).end(block, lane).sink(arg=KernelInfo(name="packed_gather", opts_to_apply=()))

def packed_gather(ids: Tensor, packed: Tensor, *, width: int, ggml_type: int, q6_aligned: bool = False,
                  q6_swizzled: bool = False, q4_tiled: bool = False) -> Tensor:
  """Rows `ids` (int32, any shape) of a packed (rows, width) table as FP32."""
  if width % 128: raise ValueError("gather width must be a multiple of 128")
  flat = ids.flatten().cast(dtypes.int32).contiguous()
  out = Tensor.empty(flat.numel(), width, dtype=dtypes.float32, device=packed.device)
  out = Tensor.custom_kernel(out, packed.flatten().contiguous(), flat, fxn=functools.partial(
    _gather_kernel, count=flat.numel(), width=width, typ=ggml_type, q6_aligned=q6_aligned, q6_swizzled=q6_swizzled,
    q4_tiled=q4_tiled))[0]
  return out.reshape(*ids.shape, width)
