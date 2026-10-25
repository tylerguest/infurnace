"""Bonsai's NV kernels on synthetic weights: the Q2_0 one-row GEMV and the blockwise Walsh-Hadamard transform."""
import os

import pytest
from tinygrad import Tensor, dtypes

from infurnace.hadamard import Rotation, hadamard_matrix
from infurnace.kernels.hadamard import fwht
from infurnace.kernels.quant_gemv import Q2_0, prepare_q2, q8_quantize, quant_gemv
from test_hadamard import q2_blocks

pytestmark = pytest.mark.skipif(not os.environ.get("INFURNACE_TEST_NV"), reason="set INFURNACE_TEST_NV=1 on an NV GPU")

@pytest.mark.parametrize("out_features,in_features,tokens", [(64, 256, 1), (48, 2560, 1), (32, 9216, 3)])
def test_q2_0_gemv_matches_dense(out_features, in_features, tokens):
  raw, values = q2_blocks(out_features * in_features // 64, seed=out_features)
  data = Tensor(raw, dtype=dtypes.uint8, device="NV")
  dense = Tensor(values, device="NV").reshape(out_features, in_features)
  x = Tensor.randn(tokens, in_features, device="NV").realize()
  reference = x @ dense.T
  got = quant_gemv(x, prepare_q2(data), out_features=out_features, in_features=in_features, ggml_type=Q2_0)
  # Q8 activations: ~1% of the output scale
  assert ((got - reference).abs().max() / reference.abs().max()).item() < 0.02
  shared = quant_gemv(x, prepare_q2(data), out_features=out_features, in_features=in_features, ggml_type=Q2_0, q8=q8_quantize(x, tokens, in_features))
  assert (shared - got).abs().max().item() == 0

@pytest.mark.parametrize("rows,width,perm", [(1, 2560, None), (64, 2560, None), (1, 4096, (2, 16, 128)), (3, 9216, None)])
def test_fwht_matches_the_dense_transform(rows, width, perm):
  matrix = Tensor(hadamard_matrix(512), device="NV").reshape(512, 512)
  signs = (Tensor.rand(width, device="NV") < 0.5).where(-1.0, 1.0).realize()
  dense = Rotation(matrix, signs, perm)._blocks  # the transform as a matmul (Rotation itself uses fwht on NV)
  x = Tensor.randn(rows, width, device="NV").realize()
  reordered = x.reshape(rows, *perm).transpose(1, 2).reshape(rows, width) if perm else x  # tiled (rep, nk, hd) -> grouped
  assert (fwht(x, signs, block=512, perm=perm) - dense(reordered * signs)).abs().max().item() < 1e-4
  assert (fwht(x, signs, block=512, signs_after=True) - dense(x) * signs).abs().max().item() < 1e-4

@pytest.mark.parametrize("int8", [True, False])
@pytest.mark.parametrize("out_features,in_features,tokens", [(64, 2560, 64), (96, 9216, 17)])
def test_q2_0_prefill_gemm_matches_dense(out_features, in_features, tokens, int8, monkeypatch):
  from infurnace.kernels.quant_gemm import quant_gemm
  monkeypatch.setenv("INFURNACE_Q2_INT8", "1" if int8 else "0")
  raw, values = q2_blocks(out_features * in_features // 64, seed=tokens)
  dense = Tensor(values, device="NV").reshape(out_features, in_features)
  x = Tensor.randn(tokens, in_features, device="NV").realize()
  reference = x.half().float() @ dense.T
  got = quant_gemm(x, prepare_q2(Tensor(raw, dtype=dtypes.uint8, device="NV")), out_features=out_features, in_features=in_features,
                   ggml_type=Q2_0)
  # int8 tensor cores take int8 activations (one scale per 32 values, ~1% of the output scale, as the decode GEMV);
  # the FP16 kernel multiplies FP16 activations and accumulates in FP32
  assert ((got - reference).abs().max() / reference.abs().max()).item() < (0.02 if int8 else 1e-3)

def test_q2_0_row_gather_matches_decoding_the_rows():
  from infurnace.kernels.embedding import packed_gather
  rows, width = 40, 2560
  raw, values = q2_blocks(rows * width // 64, seed=5)
  packed = prepare_q2(Tensor(raw, dtype=dtypes.uint8, device="NV"))
  ids = Tensor([[3, 39, 0], [17, 3, 21]], dtype=dtypes.int32, device="NV")
  got = packed_gather(ids, packed, width=width, ggml_type=Q2_0, rows=rows).tolist()
  for i, row in enumerate([[3, 39, 0], [17, 3, 21]]):
    for j, r in enumerate(row): assert got[i][j] == pytest.approx(values[r * width:(r + 1) * width], abs=1e-6)
  with pytest.raises(ValueError, match="row count"): packed_gather(ids, packed, width=width, ggml_type=Q2_0)
  with pytest.raises(ValueError, match="cannot decode"): packed_gather(ids, packed, width=width, ggml_type=8)
