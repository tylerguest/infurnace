import os
import random
import struct

import pytest
from tinygrad import Tensor, TinyJit, dtypes

from infurnace.gguf import _dequant
from infurnace.kernels.quant_gemv import packed_weight, quant_gemv


def test_llm_imports():
  # This failed in the old wheel because tinygrad.llm.kernels was omitted.
  from tinygrad.llm.cli import SimpleTokenizer
  from tinygrad.llm.model import Transformer
  assert callable(SimpleTokenizer.from_gguf_kv)
  assert callable(Transformer.from_gguf)


@pytest.mark.skipif(not os.environ.get("INFURNACE_TEST_NV"), reason="set INFURNACE_TEST_NV=1 on an NV GPU")
@pytest.mark.parametrize("typ,size", [(12, 144), (14, 210)])
@pytest.mark.parametrize("width", [256, 1280])
def test_nv_quant_eager_and_jit(typ, size, width):
  rng = random.Random(7)
  rows = 16
  block_count = rows * width // 256
  data = bytearray(rng.getrandbits(8) for _ in range(block_count*size))
  for b in range(block_count):
    if typ == 12: data[b*size:b*size+4] = struct.pack("<ee", 0.001, 0.0002)
    else: data[(b+1)*size-2:(b+1)*size] = struct.pack("<e", 0.001)
  # Exercise the nonzero-offset extraction path that previously hung on NV.
  raw = Tensor(bytes(4)+bytes(data), dtype=dtypes.uint8, device="NV").realize()[4:]
  extracted = packed_weight(_dequant(raw, rows*width, typ).reshape(rows, width).half())
  assert extracted is not None
  packed, extracted_type = extracted
  assert extracted_type == typ
  weights = _dequant(Tensor(bytes(data), dtype=dtypes.uint8, device="CPU"), rows*width, typ).reshape(rows, width).tolist()
  def run(x): return quant_gemv(x, packed, out_features=rows, in_features=width, ggml_type=typ)
  jit = TinyJit(run)
  for step in range(4):
    values = [rng.uniform(-1, 1) for _ in range(width)] if step != 3 else [0.0]*width
    rounded = []
    for i in range(0, width, 32):
      group = values[i:i+32]
      scale = max(max(map(abs, group))/127, 1e-8)
      rounded.extend(round(v/scale)*scale for v in group)
    expected = [sum(x*w for x, w in zip(rounded, row)) for row in weights]
    x = Tensor([values], dtype=dtypes.float32, device="NV").realize()
    assert run(x).tolist()[0] == pytest.approx(expected, rel=3e-3, abs=2e-3)
    assert jit(x).tolist()[0] == pytest.approx(expected, rel=3e-3, abs=2e-3)
