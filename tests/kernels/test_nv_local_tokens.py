
import pytest
from tinygrad import Tensor, TinyJit, dtypes

from infurnace.kernels.quant_gemv import QuantGemvError, prepare_q4, prepare_q6, quant_gemv
from test_nv_shared_q8 import packed

pytestmark = pytest.mark.nv

@pytest.mark.parametrize('typ', [12, 14])
@pytest.mark.parametrize('count', [2, 3, 4, 8])
@pytest.mark.parametrize('width', [2560, 9728])
def test_cta_local_tokens_exact_prepared_projection_and_replay(typ, count, width):
  raw = packed(typ, 16, width, 'NV').bitcast(dtypes.uint8)
  weight = (prepare_q4(raw, in_features=width) if typ == 12 else
            prepare_q6(raw, in_features=width, tiled=True))
  settings = dict(out_features=16, in_features=width, ggml_type=typ,
                  q4_tiled=typ == 12, q6_aligned=typ == 14, q6_tiled=typ == 14)
  jit = TinyJit(lambda x: quant_gemv(x, weight, **settings, local_tokens=True, exact_rows=True))
  for _ in range(4):
    x = Tensor.randn(1, count, width, device='NV').realize()
    result = jit(x)
    actual = Tensor(result.uop.buf_uop).reshape(result.shape).clone().realize()
    expected = Tensor.cat(*(quant_gemv(x[:, i:i+1], weight, **settings) for i in range(count)), dim=1)
    assert (actual-expected).abs().max().item() == 0

def test_local_token_contract_rejects_oversized_or_combined_reuse():
  weight = packed(14, 16, 2560, 'NV')
  settings = dict(out_features=16, in_features=2560, ggml_type=14, local_tokens=True)
  with pytest.raises(QuantGemvError, match='one to eight'):
    quant_gemv(Tensor.empty(1, 9, 2560, device='NV'), weight, **settings)
  with pytest.raises(QuantGemvError, match='either'):
    quant_gemv(Tensor.empty(1, 2, 2560, device='NV'), weight, **settings, reuse_tokens=True)
