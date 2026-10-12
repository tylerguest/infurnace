import random
import struct

import pytest
from tinygrad import Tensor, TinyJit, dtypes

from infurnace.gguf import _dequant_q4_k
from infurnace.kernels.quant_gemv import prepare_q4, quant_gemv
from infurnace.kernels.quant_gemm import quant_gemm

def data(rows, width):
  rng = random.Random(71)
  raw = bytearray(rng.getrandbits(8) for _ in range(rows*width//256*144))
  for offset in range(0, len(raw), 144): raw[offset:offset+4] = struct.pack('<ee', 0.001, 0.0002)
  return bytes(raw)

@pytest.mark.parametrize('width', [256, 768, 1280, 2560])
def test_q4_swizzle_bytes_and_generic_decode(width):
  rows = 2
  raw = data(rows, width)
  native = Tensor(raw, dtype=dtypes.uint8, device='CPU')
  packed = prepare_q4(native, in_features=width)
  actual = packed.tolist()
  expected = native.bitcast(dtypes.uint32).tolist()
  assert len(actual) == len(expected)
  for row in range(rows):
    for block in range(width//256):
      tile, within = divmod(block, 4)
      live = min(4, width//256-tile*4)
      for word in range(36):
        physical = row*(width//256)*36+tile*144+(word//4)*live*4+within*4+word%4
        assert actual[physical] == expected[(row*(width//256)+block)*36+word]
  decoded = _dequant_q4_k(packed.bitcast(dtypes.uint8), swizzled_width=width//256)
  assert (decoded-_dequant_q4_k(native)).abs().max().item() == 0

@pytest.mark.nv
@pytest.mark.parametrize('width', [1280, 2560, 9728])
def test_q4_swizzle_decode_jit_and_prefill(width):
  rows = 16
  raw = Tensor(data(rows, width), dtype=dtypes.uint8, device='NV').realize()
  native = raw.bitcast(dtypes.uint32).flatten().realize()
  packed = prepare_q4(raw, in_features=width)
  def run(x):
    return quant_gemv(x, packed, out_features=rows, in_features=width, ggml_type=12, q4_swizzled=True)
  jit = TinyJit(run)
  for _ in range(4):
    x = Tensor.randn(2, width, device='NV').realize()
    expected = quant_gemv(x, native, out_features=rows, in_features=width, ggml_type=12)
    assert (jit(x)-expected).abs().max().item() < 1e-5
  x = Tensor.randn(3, width, device='NV').realize()
  expected = quant_gemm(x, native, out_features=rows, in_features=width, ggml_type=12)
  actual = quant_gemm(x, packed, out_features=rows, in_features=width, ggml_type=12, q4_swizzled=True)
  assert (actual-expected).abs().max().item() < 1e-5
