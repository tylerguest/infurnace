import functools
import os
import random
import struct

import pytest
from tinygrad import Tensor, TinyJit, UOp, dtypes, function

from infurnace.gguf import _dequant, PackedTensor
from infurnace.model import _NVLinear
from infurnace.kernels.quant_gemm import quant_gemm, QuantGemmError, _prefill_kernel
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import Ops


@pytest.mark.parametrize("width,rows,typ", [(32, 16, 12), (256, 3, 12), (256, 16, 99)])
def test_invalid_gemm_geometry(width, rows, typ):
  with pytest.raises(QuantGemmError):
    quant_gemm(Tensor.ones(2, width, device="CPU"), Tensor.empty(1, dtype=dtypes.uint32, device="CPU"),
               in_features=width, out_features=rows, ggml_type=typ)


@pytest.mark.parametrize("tiles,splits,unroll", [(1, 1, 1), (4, 2, 2)])
@pytest.mark.parametrize("typ,size,dtype", [(12, 144, dtypes.uint32), (13, 176, dtypes.uint32), (14, 210, dtypes.uint16)])
def test_prefill_graph_uses_tensor_cores_without_dense_weights(typ, size, dtype, tiles, splits, unroll):
  out = UOp.placeholder((splits, 32, 32), dtypes.float32, slot=0)
  raw = UOp.placeholder((32*2*size//dtype.itemsize,), dtype, slot=1)
  x = UOp.placeholder((32, 512), dtypes.float16, slot=2)
  graph = _prefill_kernel(out, raw, x, tokens=32, width=512, rows=32, typ=typ, tiles=tiles, splits=splits, unroll=unroll)
  ops = graph.toposort()
  mma = [u for u in ops if u.op is Ops.WMMA]
  # One A fragment feeds every tile; each unrolled k-step issues its own MMAs.
  assert len(mma) == tiles * unroll
  assert mma[0].arg[:3] == ((8, 16, 16), dtypes.float16, 32)
  assert [s.shape for s in mma[0].src] == [(8,), (4,), (4,)]
  assert len([u for u in ops if u.op is Ops.PARAM and u.addrspace == AddrSpace.GLOBAL]) == 3


@pytest.mark.parametrize("typ,words", [(12, 36), (13, 44), (14, 54)])
def test_paired_prefill_graph_issues_one_mma_per_tile_step(typ, words):
  from infurnace.kernels.quant_gemm import _prefill_q4_paired, _prefill_q6_paired
  out = UOp.placeholder((1, 16, 64), dtypes.float32, slot=0)
  raw = UOp.placeholder((64*2*words,), dtypes.uint32, slot=1)
  x = UOp.placeholder((16, 512), dtypes.float16, slot=2)
  kernel = _prefill_q6_paired if typ == 14 else functools.partial(_prefill_q4_paired, typ=typ)
  ops = kernel(out, raw, x, tokens=16, width=512, rows=64, tiles=4).toposort()
  # Each 256-column block is 16 permuted k-steps; one A fragment feeds all tiles.
  assert len([u for u in ops if u.op is Ops.WMMA]) == 4*16
  assert len([u for u in ops if u.op is Ops.PARAM and u.addrspace == AddrSpace.GLOBAL]) == 3


def test_prefill_requires_nv_quant_before_loading():
  from infurnace.model import Qwen3Model
  with pytest.raises(ValueError, match="requires NV quantized"):
    Qwen3Model.from_gguf("does-not-exist.gguf", nv_prefill=True)


def test_prefill_rejects_single_token_and_bad_packed_storage():
  with pytest.raises(QuantGemmError, match="multi-token"):
    quant_gemm(Tensor.ones(1, 256, device="CPU"), Tensor.empty(16*36, dtype=dtypes.uint32, device="CPU"),
               in_features=256, out_features=16, ggml_type=12)
  with pytest.raises(QuantGemmError, match="dtype/size"):
    quant_gemm(Tensor.ones(2, 256, device="CPU"), Tensor.empty(16*210, dtype=dtypes.uint8, device="CPU"),
               in_features=256, out_features=16, ggml_type=14)


@pytest.mark.skipif(not os.environ.get("INFURNACE_TEST_NV"), reason="set INFURNACE_TEST_NV=1 on an NV GPU")
@pytest.mark.parametrize("typ,size", [(12, 144), (14, 210)])
@pytest.mark.parametrize("tokens,width,rows", [(2, 256, 8), (2, 256, 16), (16, 256, 32), (17, 1280, 32), (32, 256, 32)])
def test_packed_prefill_fp16_reference_eager_jit(typ, size, tokens, width, rows):
  rng = random.Random(43)
  data = bytearray(rng.getrandbits(8) for _ in range(rows*width//256*size))
  for base in range(0, len(data), size):
    if typ == 12: data[base:base+4] = struct.pack("<ee", 0.001, 0.0002)
    else: data[base+208:base+210] = struct.pack("<e", 0.001)
  dtype = dtypes.uint32 if typ == 12 else dtypes.uint16
  packed = Tensor(bytes(data), dtype=dtypes.uint8, device="NV").bitcast(dtype).flatten().realize()
  weight = _dequant(Tensor(bytes(data), dtype=dtypes.uint8, device="CPU"), rows*width, typ).reshape(rows, width).half().float().tolist()
  def run(x): return quant_gemm(x, packed, in_features=width, out_features=rows, ggml_type=typ)
  jit = TinyJit(run)
  for iteration in range(4):
    values = [[rng.randint(-16, 16)/17 for _ in range(width)] for _ in range(tokens)] if iteration < 3 else [[0.0]*width]*tokens
    rounded = [[struct.unpack("<e", struct.pack("<e", value))[0] for value in token] for token in values]
    expected = [[sum(a*b for a, b in zip(token, row)) for row in weight] for token in rounded]
    x = Tensor(values, dtype=dtypes.float32, device="NV").realize()
    for result in (run(x), jit(x)):
      assert result.shape == (tokens, rows)
      assert result.dtype == dtypes.float32
      for actual, reference in zip(result.tolist(), expected):
        assert actual == pytest.approx(reference, rel=3e-3, abs=2e-2)


@pytest.mark.skipif(not os.environ.get("INFURNACE_TEST_NV"), reason="set INFURNACE_TEST_NV=1 on an NV GPU")
@pytest.mark.parametrize("typ,size", [(12, 144), (14, 210)])
@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("width", [256, 1280])
def test_prefill_adapter_symbolic_tails_and_storage(typ, size, prepared, width):
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
  adapter = _NVLinear(original, PackedTensor(raw, (rows, width), typ), prefill=True, q6_aligned=prepared)
  assert adapter.original.weight.uop is adapter.weight.uop
  expected_size = Q6_PREPARED_BYTES if prepared and typ == 14 else size
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
      for row, reference in zip(actual, expected[:count]):
        assert row == pytest.approx(reference, rel=3e-3, abs=2e-2), (count, mode, actual.index(row))
