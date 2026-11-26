import os
import random
import struct

import pytest
from tinygrad import Tensor, TinyJit, UOp, dtypes, function

from infurnace.weights.gguf import PackedTensor
from tinygrad.llm.gguf import ggml_data_to_tensor as _dequant
from infurnace.quant.linear import _NVLinear
from infurnace.kernels.quant_gemm import quant_gemm, QuantGemmError
from infurnace.kernels.quant_gemv import prepare_q4, prepare_q6

@pytest.mark.parametrize("width,rows,typ", [(32, 16, 12), (256, 3, 12), (256, 16, 99)])
def test_invalid_gemm_geometry(width, rows, typ):
  with pytest.raises(QuantGemmError):
    quant_gemm(Tensor.ones(2, width, device="CPU"), Tensor.empty(1, dtype=dtypes.uint32, device="CPU"),
               in_features=width, out_features=rows, ggml_type=typ)

def test_prefill_requires_nv_quant_before_loading():
  from infurnace.engine import Model
  with pytest.raises(ValueError, match="requires NV quantized"):
    Model.from_gguf("does-not-exist.gguf", nv_prefill=True)

def test_prefill_rejects_single_token_and_bad_packed_storage():
  with pytest.raises(QuantGemmError, match="multi-token"):
    quant_gemm(Tensor.ones(1, 256, device="CPU"), Tensor.empty(16*36, dtype=dtypes.uint32, device="CPU"),
               in_features=256, out_features=16, ggml_type=12, q4_tiled=True)
  with pytest.raises(QuantGemmError, match="dtype/size"):
    quant_gemm(Tensor.ones(2, 256, device="CPU"), Tensor.empty(16*210, dtype=dtypes.uint8, device="CPU"),
               in_features=256, out_features=16, ggml_type=14, q6_aligned=True, q6_tiled=True)
  with pytest.raises(QuantGemmError, match="tiled layout"):
    quant_gemm(Tensor.ones(2, 256, device="CPU"), Tensor.empty(16*36, dtype=dtypes.uint32, device="CPU"),
               in_features=256, out_features=16, ggml_type=12)

@pytest.mark.skipif(not os.environ.get("INFURNACE_TEST_NV"), reason="set INFURNACE_TEST_NV=1 on an NV GPU")
@pytest.mark.parametrize("typ,size", [(12, 144), (14, 210)])
@pytest.mark.parametrize("tokens,width,rows", [(2, 256, 16), (16, 256, 32), (17, 1280, 32), (32, 256, 32)])
def test_packed_prefill_fp16_reference_eager_jit(typ, size, tokens, width, rows):
  rng = random.Random(43)
  data = bytearray(rng.getrandbits(8) for _ in range(rows*width//256*size))
  for base in range(0, len(data), size):
    if typ == 12: data[base:base+4] = struct.pack("<ee", 0.001, 0.0002)
    else: data[base+208:base+210] = struct.pack("<e", 0.001)
  raw = Tensor(bytes(data), dtype=dtypes.uint8, device="NV").realize()
  packed = (prepare_q4(raw, in_features=width) if typ == 12 else prepare_q6(raw, in_features=width, tiled=True)).realize()
  options = dict(q4_tiled=True) if typ == 12 else dict(q6_aligned=True, q6_tiled=True)
  weight = _dequant(Tensor(bytes(data), dtype=dtypes.uint8, device="CPU"), rows*width, typ).reshape(rows, width).half().float().tolist()
  def run(x): return quant_gemm(x, packed, in_features=width, out_features=rows, ggml_type=typ, **options)
  jit = TinyJit(run)
  for iteration in range(4):
    values = [[rng.randint(-16, 16)/17 for _ in range(width)] for _ in range(tokens)] if iteration < 3 else [[0.0]*width]*tokens
    rounded = [[struct.unpack("<e", struct.pack("<e", value))[0] for value in token] for token in values]
    expected = [[sum(a*b for a, b in zip(token, row)) for row in weight] for token in rounded]
    # The tiled kernels round d*scale to FP16 before multiplying (as Marlin does): allow that relative to the largest output.
    tolerance = max(2e-2, 2e-3*max(abs(v) for r in expected for v in r))
    x = Tensor(values, dtype=dtypes.float32, device="NV").realize()
    for result in (run(x), jit(x)):
      assert result.shape == (tokens, rows)
      assert result.dtype == dtypes.float32
      for actual, reference in zip(result.tolist(), expected):
        assert actual == pytest.approx(reference, rel=3e-3, abs=tolerance)

@pytest.mark.skipif(not os.environ.get("INFURNACE_TEST_NV"), reason="set INFURNACE_TEST_NV=1 on an NV GPU")
@pytest.mark.parametrize("typ,size", [(12, 144), (14, 210)])
@pytest.mark.parametrize("width", [256, 1280])
def test_prefill_adapter_symbolic_tails_and_storage(typ, size, width):
  from tinygrad import nn
  from infurnace.kernels.quant_gemv import Q6_PREPARED_BYTES
  rows = 16
  rng = random.Random(53)
  data = bytearray(rng.getrandbits(8) for _ in range(rows*width//256*size))
  for base in range(0, len(data), size):
    if typ == 12: data[base:base+4] = struct.pack("<ee", 0.001, 0.0002)
    else: data[base+208:base+210] = struct.pack("<e", 0.001)
  raw = Tensor(bytes(data), dtype=dtypes.uint8, device="NV").realize()
  original = nn.Linear(width, rows, bias=False)
  original.weight = _dequant(raw, rows*width, typ).reshape(rows, width).half()
  with pytest.raises(ValueError, match="16-row tiles"):
    _NVLinear(original, PackedTensor(raw, (rows, width), typ), prefill=True)  # the untiled layout
  adapter = _NVLinear(original, PackedTensor(raw, (rows, width), typ), prefill=True, q6_aligned=True)
  assert adapter.original.weight.uop is adapter.weight.uop
  expected_size = Q6_PREPARED_BYTES if typ == 14 else size
  assert adapter.packed.numel()*adapter.packed.dtype.itemsize == len(data)//size*expected_size
  values = [[rng.randint(-16, 16)/16 for _ in range(width)] for _ in range(32)]
  # Snapshot realized device weights before constructing host expectations.
  # Do not keep a lazy device-to-host/cast graph as a reference to a reused buffer.
  weights = adapter.weight.realize().tolist()
  cpu_weights = _dequant(Tensor(bytes(data), dtype=dtypes.uint8, device="CPU"), rows*width, typ).reshape(rows, width).half().tolist()
  # CPU can emulate half in float32; allow only FP16 rounding error here.
  for row, reference in zip(weights, cpu_weights):
    assert row == pytest.approx(reference, rel=5e-4, abs=2e-6)
  expected = [[sum(a*b for a, b in zip(token, row)) for row in weights] for token in values]
  # Upstream consumes symbolic intermediates inside capture and returns a static
  # sampled-token shape. Materialize a static output here to test the same contract.
  @function(allow_implicit=True)
  def run(x): return adapter(x).pad_to((32, rows))
  jit = TinyJit(run)
  nt = UOp.variable("prefill_tokens", 1, 32)
  for count in (2, 31, 1, 17, 32):
    x = Tensor(values, device="NV", dtype=dtypes.float32).realize()[:nt.bind(count)]
    for mode, result in enumerate((run(x), jit(x))):
      assert result.shape[1] == rows
      actual = result[:count].tolist()
      assert len(actual) == count
      # Tiled Q6_K rounds d*scale to FP16 before multiplying (as Marlin does): allow that relative to the largest output.
      tolerance = max(2e-2, 2e-3*max(abs(v) for r in expected for v in r))
      for row, reference in zip(actual, expected[:count]):
        assert row == pytest.approx(reference, rel=3e-3, abs=tolerance), (count, mode, actual.index(row))
