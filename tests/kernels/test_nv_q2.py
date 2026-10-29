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
  got = quant_gemv(x, prepare_q2(data, in_features=in_features), out_features=out_features, in_features=in_features, ggml_type=Q2_0)
  # Q8 activations: ~1% of the output scale
  assert ((got - reference).abs().max() / reference.abs().max()).item() < 0.02
  shared = quant_gemv(x, prepare_q2(data, in_features=in_features), out_features=out_features, in_features=in_features, ggml_type=Q2_0,
                      q8=q8_quantize(x, tokens, in_features))
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

@pytest.mark.parametrize("rows,width,perm", [(1, 2560, None), (16, 9216, None), (16, 4096, (2, 16, 128))])
def test_fwht_q8_matches_the_transform_then_quantizing(rows, width, perm):
  from infurnace.kernels.hadamard import fwht_q8
  from infurnace.kernels.quant_gemv import q8_quantize64
  signs = (Tensor.rand(width, device="NV") < 0.5).where(-1.0, 1.0).realize()
  x = Tensor.randn(rows, width, device="NV").realize()
  rotated = fwht(x, signs, block=512, perm=perm)
  def codes(words): return words.flatten().bitcast(dtypes.int8).float()
  for group, reference in ((32, q8_quantize(rotated, rows, width)), (64, q8_quantize64(rotated, rows, width))):
    got = fwht_q8(x, signs, block=512, perm=perm, group=group)
    assert [g.shape for g in got] == [r.shape for r in reference]
    # The same values up to FP32 rounding of the normalization: a code may round the other way at a tie
    assert (codes(got[0]) - codes(reference[0])).abs().max().item() <= 1
    assert ((got[1] - reference[1]).abs() / reference[1]).max().item() < 1e-6
    if group == 32: assert (got[2] - reference[2]).abs().max().item() <= 1

@pytest.mark.parametrize("int8", [True, False])
@pytest.mark.parametrize("out_features,in_features,tokens", [(64, 2560, 64), (96, 9216, 17), (256, 2560, 16), (96, 9216, 13)])
def test_q2_0_prefill_gemm_matches_dense(out_features, in_features, tokens, int8, monkeypatch):
  from infurnace.kernels.quant_gemm import quant_gemm
  monkeypatch.setenv("INFURNACE_Q2_INT8", "1" if int8 else "0")
  raw, values = q2_blocks(out_features * in_features // 64, seed=tokens)
  dense = Tensor(values, device="NV").reshape(out_features, in_features)
  x = Tensor.randn(tokens, in_features, device="NV").realize()
  reference = x.half().float() @ dense.T
  packed = prepare_q2(Tensor(raw, dtype=dtypes.uint8, device="NV"), in_features=in_features)
  got = quant_gemm(x, packed, out_features=out_features, in_features=in_features,
                   ggml_type=Q2_0)
  # int8 tensor cores take int8 activations (one scale per 32 values, ~1% of the output scale, as the decode GEMV);
  # the FP16 kernel multiplies FP16 activations and accumulates in FP32
  assert ((got - reference).abs().max() / reference.abs().max()).item() < (0.02 if int8 else 1e-3)

def test_q2_0_row_gather_matches_decoding_the_rows():
  from infurnace.kernels.embedding import packed_gather
  rows, width = 40, 2560
  raw, values = q2_blocks(rows * width // 64, seed=5)
  packed = prepare_q2(Tensor(raw, dtype=dtypes.uint8, device="NV"), in_features=width)
  ids = Tensor([[3, 39, 0], [17, 3, 21]], dtype=dtypes.int32, device="NV")
  got = packed_gather(ids, packed, width=width, ggml_type=Q2_0, rows=rows).tolist()
  for i, row in enumerate([[3, 39, 0], [17, 3, 21]]):
    for j, r in enumerate(row): assert got[i][j] == pytest.approx(values[r * width:(r + 1) * width], abs=1e-6)
  with pytest.raises(ValueError, match="row count"): packed_gather(ids, packed, width=width, ggml_type=Q2_0)
  with pytest.raises(ValueError, match="cannot decode"): packed_gather(ids, packed, width=width, ggml_type=8)

@pytest.mark.parametrize("kc", [2, 4])
def test_q2_0_int8_cta_k_split_matches_unsplit(kc):
  # kc warps sharing an output's K in a CTA add their sums through shared memory: the same sums as one warp
  import functools
  from infurnace.kernels.quant_gemm import _prefill_q2_imma
  from infurnace.kernels.quant_gemv import q8_quantize64
  rows, width, tokens = 256, 2560, 16
  raw, _ = q2_blocks(rows * width // 64, seed=kc)
  packed = prepare_q2(Tensor(raw, dtype=dtypes.uint8, device="NV"), in_features=width)
  xq, xd = q8_quantize64(Tensor.randn(tokens, width, device="NV").realize(), tokens, width)
  def run(kc):
    out = Tensor.empty(1, tokens, rows, dtype=dtypes.float32, device="NV")
    fxn = functools.partial(_prefill_q2_imma, tokens=tokens, width=width, rows=rows, tiles=4, mtiles=1, unroll=5, kc=kc)
    return Tensor.custom_kernel(out, packed, xq, xd, fxn=fxn)[0]
  reference = run(1)
  assert ((run(kc) - reference).abs().max() / reference.abs().max()).item() < 1e-6
