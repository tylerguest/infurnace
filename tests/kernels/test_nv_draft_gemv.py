
import pytest
from tinygrad import Tensor, TinyJit, dtypes

from infurnace.kernels.dense_gemv import bf16_gemv


@pytest.mark.nv
@pytest.mark.parametrize('width', [128, 2560, 9728])
@pytest.mark.parametrize('count', [1, 2, 3, 8])
def test_bf16_gemv_exact_binary_reference_and_replay(width, count):
  rows = 16
  matrix = [[((r*7+k)%9-4)*0.125 for k in range(width)] for r in range(rows)]
  w = Tensor(matrix, dtype=dtypes.float32, device='NV').bfloat16().realize()
  jit = TinyJit(lambda x: bf16_gemv(x, w))
  for shift in range(4):
    values = [[((k+shift+i)%11-5)*0.125 for k in range(width)] for i in range(count)]
    x = Tensor([values], dtype=dtypes.float32, device='NV').bfloat16().realize()
    expected = Tensor([[[sum(a*b for a, b in zip(token, row)) for row in matrix]
                        for token in values]], device='NV').bfloat16().float()
    actual = jit(x).clone().realize().float()
    assert (actual-expected).abs().max().item() == 0
