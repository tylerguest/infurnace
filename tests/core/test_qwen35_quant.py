"""Independent GGUF Q5_K/Q8_0 decoding and packed NV projection controls."""
import os
import random
import struct
import pytest
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.llm.gguf import ggml_data_to_tensor
from infurnace.gguf import Q5_K, Q8_0, _dequant, _tensor_nbytes
from infurnace.kernels.quant_gemv import q8_quantize, quant_gemv

def blocks(count):
  rng = random.Random(891)
  raw, decoded = bytearray(), []
  for _ in range(count):
    scales, mins = [[rng.randrange(64) for _ in range(8)] for _ in range(2)]
    values = [[rng.randrange(32) for _ in range(32)] for _ in range(8)]
    packed = bytes([(scales[j] & 63) | ((scales[j+4] >> 4) << 6) for j in range(4)] +
      [(mins[j] & 63) | ((mins[j+4] >> 4) << 6) for j in range(4)] +
      [(scales[j+4] & 15) | ((mins[j+4] & 15) << 4) for j in range(4)])
    high = bytes(sum((values[g][lane] >> 4) << g for g in range(8)) for lane in range(32))
    low = bytes((values[g][lane] & 15) | ((values[g+1][lane] & 15) << 4) for g in range(0, 8, 2) for lane in range(32))
    raw.extend(struct.pack("<ee", 0.03125, 0.015625) + packed + high + low)
    decoded.extend(0.03125 * scales[g] * q - 0.015625 * mins[g] for g in range(8) for q in values[g])
  return bytes(raw), decoded

def test_q5_scalar_and_upstream():
  raw, expected = blocks(3)
  t = Tensor(raw, dtype=dtypes.uint8, device="CPU")
  assert _dequant(t, 768, Q5_K).tolist() == expected
  assert ggml_data_to_tensor(t, 768, Q5_K).flatten().tolist() == expected
  assert _tensor_nbytes(768, Q5_K) == len(raw)

def test_q8_signed_and_row_block_size():
  values = list(range(-128, -96))
  raw = struct.pack("<e32b", 0.125, *values)
  assert _dequant(Tensor(raw, dtype=dtypes.uint8, device="CPU"), 32, Q8_0).tolist() == [x * 0.125 for x in values]
  assert _tensor_nbytes(32, Q8_0) == 34
  with pytest.raises(ValueError): _tensor_nbytes(33, Q8_0)

@pytest.mark.skipif(not os.environ.get("INFURNACE_TEST_NV"), reason="set INFURNACE_TEST_NV=1 on an NV GPU")
@pytest.mark.parametrize("width", [256, 768, 2560])
def test_q5_nv_eager_and_jit(width):
  raw, weights = blocks(16 * width // 256)
  packed = Tensor(raw, dtype=dtypes.uint8, device="NV").bitcast(dtypes.uint32).realize()
  def run(x): return quant_gemv(x, packed, out_features=16, in_features=width, ggml_type=Q5_K)
  jit = TinyJit(run)
  for step in range(4):
    rng = random.Random(step)
    x = Tensor([[rng.uniform(-2, 2) for _ in range(width)]], device="NV").realize()
    q, scale, _ = q8_quantize(x, 1, width)
    q, scale = q.tolist()[0], scale.tolist()[0]
    activation = []
    for words, d in zip(q, scale):
      for word in words:
        for byte in range(4):
          v = (word >> (byte * 8)) & 255
          activation.append((v if v < 128 else v - 256) * d)
    expected = [sum(a * w for a, w in zip(activation, weights[row * width:(row+1) * width])) for row in range(16)]
    assert run(x).tolist()[0] == pytest.approx(expected, abs=0.02, rel=2e-5)
    assert jit(x).tolist()[0] == pytest.approx(expected, abs=0.02, rel=2e-5)
