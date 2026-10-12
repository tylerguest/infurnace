
import pytest
from tinygrad import Tensor, TinyJit, dtypes

from infurnace.kernels.quant_gemv import prepare_q4, prepare_q6, quant_gemv
from test_nv_shared_q8 import packed

pytestmark = pytest.mark.nv

@pytest.mark.parametrize('typ', [12, 14])
@pytest.mark.parametrize('width', [2560, 4096, 9728])
def test_fused_residual_keeps_residual_first_rounding(typ, width):
  raw = packed(typ, 32, width, 'NV').bitcast(dtypes.uint8)
  weight = prepare_q4(raw, in_features=width) if typ == 12 else prepare_q6(raw, in_features=width, swizzled=True)
  settings = dict(out_features=32, in_features=width, ggml_type=typ,
                  q4_swizzled=typ == 12, q6_aligned=typ == 14, q6_swizzled=typ == 14)
  fused = TinyJit(lambda x, h: quant_gemv(x, weight, **settings, fuse_chunks=True, local_tokens=True, residual=h))
  reference = TinyJit(lambda x, h: h+quant_gemv(x, weight, **settings, return_parts=True).sum(-1).reshape(h.shape))
  for repeat in range(4):
    x = Tensor.randn(1, 2, width, device='NV').realize()
    h = (Tensor.randn(1, 2, 32, device='NV')*(10**repeat)).realize()
    actual = fused(x, h)
    actual = Tensor(actual.uop.buf_uop).reshape(actual.shape).clone().realize()
    expected = reference(x, h)
    expected = Tensor(expected.uop.buf_uop).reshape(expected.shape)
    assert (actual-expected).abs().max().item() == 0

@pytest.mark.parametrize('typ', [12, 14])
@pytest.mark.parametrize('count', [1, 2, 3, 4])
@pytest.mark.parametrize('width', [2560, 4096, 9728])
def test_fused_chunk_sum_preserves_target_projection_and_replay(typ, count, width):
  raw = packed(typ, 32, width, 'NV').bitcast(dtypes.uint8)
  weight = prepare_q4(raw, in_features=width) if typ == 12 else prepare_q6(raw, in_features=width, swizzled=True)
  settings = dict(out_features=32, in_features=width, ggml_type=typ,
                  q4_swizzled=typ == 12, q6_aligned=typ == 14, q6_swizzled=typ == 14)
  jit = TinyJit(lambda x: quant_gemv(x, weight, **settings, fuse_chunks=True,
                                    local_tokens=count > 1, return_parts=True))
  for _ in range(4):
    x = Tensor.randn(1, count, width, device='NV').realize()
    actual = jit(x)
    actual = Tensor(actual.uop.buf_uop).reshape(count, 32).clone().realize()
    expected = quant_gemv(x, weight, **settings).reshape(count, 32)
    assert (actual-expected).abs().max().item() == 0
