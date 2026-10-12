import random
import struct

import pytest
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.uop.ops import Ops

from infurnace.kernels.quant_gemv import QuantGemvError, q8_quantize, quant_gemv

def packed(typ, rows, width, device):
  size = 144 if typ == 12 else 210
  rng = random.Random(53)
  raw = bytearray(rng.getrandbits(8) for _ in range(rows*width//256*size))
  for i in range(0, len(raw), size):
    if typ == 12: raw[i:i+4] = struct.pack('<ee', 0.001, 0.0002)
    else: raw[i+208:i+210] = struct.pack('<e', 0.001)
  return Tensor(bytes(raw), dtype=dtypes.uint8, device=device).bitcast(dtypes.uint32 if typ == 12 else dtypes.uint16).realize()

def test_invalid_shared_q8_contract():
  x = Tensor.zeros(1, 256, device='CPU')
  w = Tensor.zeros(16*36, dtype=dtypes.uint32, device='CPU')
  wrong = (Tensor.zeros(1, 8, 8, dtype=dtypes.int32, device='CPU'),
           Tensor.zeros(1, 8, device='CPU'), Tensor.zeros(1, 8, 2, dtype=dtypes.int32, device='CPU'))
  with pytest.raises(QuantGemvError, match='shared Q8'):
    quant_gemv(x, w, out_features=16, in_features=256, ggml_type=12, q8=wrong)

@pytest.mark.nv
@pytest.mark.parametrize('typ', [12, 14])
def test_shared_q8_projection_parity_and_graph(typ):
  rows, width = 16, 1280
  w = packed(typ, rows, width, 'NV')
  def run(x):
    shared = q8_quantize(x, 1, width)
    values = [quant_gemv(x, w, out_features=rows, in_features=width, ggml_type=typ, q8=shared) for _ in range(3)]
    return tuple(values)
  jit = TinyJit(run)
  for _ in range(4):
    x = Tensor.randn(1, 1, width, device='NV').realize()
    graph = run(x)
    calls = [u for u in Tensor.stack(*graph).uop.toposort() if u.op is Ops.CALL and u.body.op is Ops.SINK
             and u.body.arg.name == 'q8_quantize']
    assert len(calls) == 1
    reference = quant_gemv(x, w, out_features=rows, in_features=width, ggml_type=typ)
    for actual in jit(x):
      assert (actual-reference).abs().max().item() == 0
