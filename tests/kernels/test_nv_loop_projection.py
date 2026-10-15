
import pytest
from tinygrad import Tensor, TinyJit, dtypes

from infurnace.kernels.quant_gemv import prepare_q4, prepare_q6, quant_gemv
from test_nv_shared_q8 import packed

pytestmark = pytest.mark.nv

@pytest.mark.parametrize('typ', [12, 14])
@pytest.mark.parametrize('count', [2, 3, 4, 8])
@pytest.mark.parametrize('width', [2560, 9728])
def test_loop_projection_preserves_every_partial_on_replay(typ, count, width):
  raw = packed(typ, 16, width, 'NV').bitcast(dtypes.uint8)
  weight = (prepare_q4(raw, in_features=width) if typ == 12 else
            prepare_q6(raw, in_features=width, tiled=True))
  settings = dict(out_features=16, in_features=width, ggml_type=typ, return_parts=True,
                  q4_tiled=typ == 12, q6_aligned=typ == 14, q6_tiled=typ == 14)
  jits = [TinyJit(lambda x, local=local: quant_gemv(x, weight, **settings, loop_chunks=True, local_tokens=local))
          for local in (False, True)]
  for _ in range(4):
    x = Tensor.randn(1, count, width, device='NV').realize()
    expected = quant_gemv(x, weight, **settings).realize()
    for jit in jits:
      actual = jit(x).clone().realize()
      assert (actual-expected).abs().max().item() == 0
