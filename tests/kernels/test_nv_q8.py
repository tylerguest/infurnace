import os
import random

import pytest
from tinygrad import Tensor, TinyJit, dtypes

from infurnace.kernels.quant_gemv import q8_quantize


@pytest.mark.skipif(not os.environ.get("INFURNACE_TEST_NV"), reason="set INFURNACE_TEST_NV=1 on an NV GPU")
@pytest.mark.parametrize("width", [32, 256, 1280])
@pytest.mark.parametrize("dtype", [dtypes.float32, dtypes.float16])
def test_q8_packing_scales_sums_eager_and_jit(width, dtype):
  rng = random.Random(19)
  def run(x): return q8_quantize(x, 2, width)
  jit = TinyJit(run)
  for iteration in range(5):
    values, expected_words, expected_scales, expected_sums = [], [], [], []
    for token in range(2):
      row, words, scales, sums = [], [], [], []
      for group in range(width//32):
        gain = (0.5, 1.0, 2.0)[group % 3]
        if iteration == 4:
          # Total sum is zero, but each Q6 scale group needs its nonzero signed sum.
          sign = -1 if token else 1
          v = [sign*127*gain]*16 + [-sign*127*gain]*16
        elif iteration == 3:
          v = [0.0]*32
        elif iteration == 2:
          v = [(-127.0 if token else 127.0)*gain]*32
        else:
          # Exact scale and rounding ties; token rows have distinct signed sums.
          v = [rng.randint(-126, 126)*gain for _ in range(32)]
          v[:8] = [x*gain for x in (0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0, -127)]
          v[-1] = 127*gain
        scale = max(max(map(abs, v))/127, 1e-8)
        quant = [max(-127, min(127, round(x/scale))) for x in v]
        words.append([sum((quant[i+j] & 255) << (8*j) for j in range(4)) for i in range(0, 32, 4)])
        scales.append(scale)
        sums.append([sum(quant[:16]), sum(quant[16:])])
        row.extend(v)
      values.append(row)
      expected_words.append(words)
      expected_scales.append(scales)
      expected_sums.append(sums)
    x = Tensor(values, dtype=dtype, device="NV").realize()
    for actual in (run(x), jit(x)):
      q, scale, xsum = actual
      assert q.tolist() == expected_words
      assert xsum.dtype == dtypes.int32
      assert xsum.tolist() == expected_sums
      for actual_row, expected_row in zip(scale.tolist(), expected_scales):
        assert actual_row == pytest.approx(expected_row, rel=1e-6)
