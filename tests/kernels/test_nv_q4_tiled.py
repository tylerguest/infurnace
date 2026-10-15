import random
import struct

import pytest
from tinygrad import Tensor, TinyJit, dtypes

from tinygrad.llm.gguf import ggml_data_to_tensor
from infurnace.kernels.quant_gemv import QuantGemvError, dequant_prepared_q4, prepare_q4, quant_gemv
from infurnace.kernels.quant_gemm import quant_gemm

def data(rows, width):
  rng = random.Random(71)
  raw = bytearray(rng.getrandbits(8) for _ in range(rows*width//256*144))
  for offset in range(0, len(raw), 144): raw[offset:offset+4] = struct.pack('<ee', 0.001, 0.0002)
  return bytes(raw)

@pytest.mark.parametrize('width', [256, 768, 1280, 2560])
def test_q4_tiled_bytes_and_generic_decode(width):
  rows = 32
  raw = data(rows, width)
  native = Tensor(raw, dtype=dtypes.uint8, device='CPU')
  packed = prepare_q4(native, in_features=width)
  actual = packed.tolist()
  expected = native.bitcast(dtypes.uint32).tolist()
  assert len(actual) == len(expected)
  blocks = width//256
  for row in range(rows):
    for block in range(blocks):
      for word in range(36):  # each 16-row tile of each block is contiguous, native word order
        physical = ((row//16*blocks+block)*16+row%16)*36+word
        assert actual[physical] == expected[(row*blocks+block)*36+word]
  decoded = dequant_prepared_q4(packed, in_features=width)
  assert (decoded-ggml_data_to_tensor(native, native.numel()//144*256, 12).flatten()).abs().max().item() == 0

def test_q4_tiled_needs_16_row_tiles():
  with pytest.raises(QuantGemvError, match='rows of 16'):
    prepare_q4(Tensor(data(8, 256), dtype=dtypes.uint8, device='CPU'), in_features=256)

@pytest.mark.nv
@pytest.mark.parametrize('width', [1280, 2560, 9728])
def test_q4_tiled_decode_jit_and_prefill(width):
  rows = 16
  raw = Tensor(data(rows, width), dtype=dtypes.uint8, device='NV').realize()
  native = raw.bitcast(dtypes.uint32).flatten().realize()
  packed = prepare_q4(raw, in_features=width)
  def run(x):
    return quant_gemv(x, packed, out_features=rows, in_features=width, ggml_type=12, q4_tiled=True)
  jit = TinyJit(run)
  for _ in range(4):
    x = Tensor.randn(2, width, device='NV').realize()
    expected = quant_gemv(x, native, out_features=rows, in_features=width, ggml_type=12)
    assert (jit(x)-expected).abs().max().item() < 1e-5
  x = Tensor.randn(3, width, device='NV').realize()
  expected = quant_gemm(x, native, out_features=rows, in_features=width, ggml_type=12)
  actual = quant_gemm(x, packed, out_features=rows, in_features=width, ggml_type=12, q4_tiled=True)
  # The tiled kernel rounds d*scale to FP16 before multiplying (as Marlin does); the native one rounds once in FP32.
  assert (actual-expected).abs().max().item() < 1e-3 * expected.abs().max().item()

def q5_data(rows, width):
  rng = random.Random(97)
  raw = bytearray(rng.getrandbits(8) for _ in range(rows*width//256*176))
  for offset in range(0, len(raw), 176): raw[offset:offset+4] = struct.pack('<ee', 0.001, 0.0002)
  return bytes(raw)

@pytest.mark.parametrize('width', [256, 1280])
def test_q5_tiled_round_trips(width):
  from infurnace.kernels.quant_gemv import untile
  native = Tensor(q5_data(32, width), dtype=dtypes.uint8, device='CPU')
  packed = prepare_q4(native, in_features=width, ggml_type=13)
  assert untile(packed, in_features=width, block_bytes=176).flatten().tolist() == native.tolist()

@pytest.mark.nv
@pytest.mark.parametrize('width', [1280, 4096])
def test_q5_tiled_decode_and_prefill(width):
  rows = 32
  raw = Tensor(q5_data(rows, width), dtype=dtypes.uint8, device='NV').realize()
  native = raw.bitcast(dtypes.uint32).flatten().realize()
  packed = prepare_q4(raw, in_features=width, ggml_type=13)
  x = Tensor.randn(2, width, device='NV').realize()
  expected = quant_gemv(x, native, out_features=rows, in_features=width, ggml_type=13)
  assert (quant_gemv(x, packed, out_features=rows, in_features=width, ggml_type=13, q4_tiled=True)-expected).abs().max().item() == 0
  x = Tensor.randn(16, width, device='NV').realize()
  expected = quant_gemm(x, native, out_features=rows, in_features=width, ggml_type=13)
  actual = quant_gemm(x, packed, out_features=rows, in_features=width, ggml_type=13, q4_tiled=True)
  # The tiled kernel rounds d*scale to FP16 before multiplying (as Marlin does).
  assert (actual-expected).abs().max().item() < 1e-3 * expected.abs().max().item()
