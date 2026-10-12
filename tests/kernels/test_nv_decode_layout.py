import importlib
import os
import random
import struct

import pytest
from tinygrad import Tensor, TinyJit, dtypes

from tinygrad.llm.gguf import ggml_data_to_tensor as _dequant
kernels = importlib.import_module("infurnace.kernels.quant_gemv")

@pytest.mark.parametrize("typ,width,rows,expected", [
  (12, 2560, 9728, 4), (14, 2560, 151936, 8), (14, 9728, 2560, 1),
  (14, 2560, 1024, 1), (14, 2560, 65536, 1),
])
def test_shape_specific_warp_selection(typ, width, rows, expected):
  assert kernels._decode_warps(typ, width, rows) == expected

@pytest.mark.skipif(not os.environ.get("INFURNACE_TEST_NV"), reason="set INFURNACE_TEST_NV=1 on an NV GPU")
@pytest.mark.parametrize("typ,size,dtype,warps", [(12, 144, dtypes.uint32, 4), (14, 210, dtypes.uint16, 8)])
def test_grouped_warps_multiple_tokens_partial_chunk(monkeypatch, typ, size, dtype, warps):
  # Two tokens, two row blocks for Q6, and a partial last chunk; catch indexing/warp mixing.
  rows, width = 16, 1280
  rng = random.Random(31)
  data = bytearray(rng.getrandbits(8) for _ in range(rows*width//256*size))
  for offset in range(0, len(data), size):
    if typ == 12: data[offset:offset+4] = struct.pack("<ee", 0.001, 0.0002)
    else: data[offset+208:offset+210] = struct.pack("<e", 0.001)
  packed = Tensor(bytes(data), dtype=dtypes.uint8, device="NV").bitcast(dtype).flatten().realize()
  weights = _dequant(Tensor(bytes(data), dtype=dtypes.uint8, device="CPU"), rows*width, typ).reshape(rows, width).tolist()
  monkeypatch.setattr(kernels, "_decode_warps", lambda *args: warps)
  def run(x): return kernels.quant_gemv(x, packed, out_features=rows, in_features=width, ggml_type=typ)
  jit = TinyJit(run)
  for _ in range(4):
    values = [[rng.uniform(-1, 1) for _ in range(width)] for _ in range(2)]
    expected = []
    for token in values:
      rounded = []
      for offset in range(0, width, 32):
        group = token[offset:offset+32]
        scale = max(max(map(abs, group))/127, 1e-8)
        rounded.extend(round(v/scale)*scale for v in group)
      expected.append([sum(v*w for v, w in zip(rounded, weight)) for weight in weights])
    x = Tensor(values, dtype=dtypes.float32, device="NV").realize()
    for result in (run(x), jit(x)):
      for actual, reference in zip(result.tolist(), expected):
        assert actual == pytest.approx(reference, rel=3e-3, abs=2e-3)
