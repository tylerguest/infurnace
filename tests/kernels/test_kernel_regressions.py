"""CPU-only checks on the custom UOp graph; these do not prove GPU throughput."""
import importlib

import pytest
from tinygrad import Tensor, UOp, dtypes
from tinygrad.uop.ops import AxisType, Ops

kernels = importlib.import_module("infurnace.kernels.quant_gemv")


def decode_graph(typ, rows, width):
  chunks = (width+1023)//1024
  dtype = dtypes.uint32 if typ == 12 else dtypes.uint16
  tensors = (
    Tensor.empty(1, rows, chunks, dtype=dtypes.float32, device="CPU"),
    Tensor.empty(rows*width//256*kernels.QUANT_SIZES[typ]//dtype.itemsize, dtype=dtype, device="CPU"),
    Tensor.empty(1, width//32, 8, dtype=dtypes.uint32, device="CPU"),
    Tensor.empty(1, width//32, dtype=dtypes.float32, device="CPU"),
    Tensor.empty(1, width//32, 2, dtype=dtypes.int32, device="CPU"),
  )
  params = [UOp.placeholder_like(t.uop, slot=i) for i, t in enumerate(tensors)]
  return kernels._quant_decode_kernel(*params, out_features=rows, in_features=width, ggml_type=typ)


@pytest.mark.parametrize("typ,rows,width,warps", [(12, 2560, 9728, 4), (14, 151936, 2560, 8), (14, 2560, 9728, 1)])
def test_decode_graph_keeps_warps_separate_and_eight_integer_dots(typ, rows, width, warps):
  graph = decode_graph(typ, rows, width)
  nodes = list(graph.toposort())
  dots = [u for u in nodes if u.op is Ops.CUSTOMI and "__dp4a" in str(u.arg)]
  assert len(dots) == 8  # Q4 sum dots and Q6 signed-byte rebuilding must not return.
  ranges = [u for u in nodes if u.op is Ops.RANGE]
  assert any(getattr(u, 'axis_type', u.arg[-1]) is AxisType.WARP and u.src[0].arg == 32 for u in ranges)
  if warps > 1:
    assert any(getattr(u, 'axis_type', u.arg[-1]) is AxisType.LOCAL and u.src[0].arg == warps for u in ranges)
    assert not any(getattr(u, 'axis_type', u.arg[-1]) is AxisType.LOCAL and u.src[0].arg == warps*32 for u in ranges)
  if typ == 14:
    # Weight parameter uses native halfwords, not padded blocks or aligned-u32 reinterpretation.
    assert any(u.op is Ops.PARAM and u.dtype == dtypes.uint16 for u in nodes)
