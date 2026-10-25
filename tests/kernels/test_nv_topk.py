"""The top-k and dense row-gather kernels (DFlash2's candidates and selector tables) against plain references."""
import os
import random

import pytest
from tinygrad import Tensor, dtypes

from infurnace.kernels.embedding import dense_gather
from infurnace.kernels.topk import topk

pytestmark = pytest.mark.skipif(not os.environ.get("INFURNACE_TEST_NV"), reason="set INFURNACE_TEST_NV=1 on an NV GPU")

@pytest.mark.parametrize("rows,width,k", [(3, 248320, 16), (2, 1000, 5), (1, 2048, 16)])
def test_topk_matches_a_sort(rows, width, k):
  rng = random.Random(width)
  values = [[rng.choice([0.0, 1.0, rng.gauss(0, 1)]) for _ in range(width)] for _ in range(rows)]  # with ties
  got_values, got_index = (t.tolist() for t in topk(Tensor(values, device="NV"), k))
  for row, gv, gi in zip(values, got_values, got_index):
    want = sorted(range(width), key=lambda c: (-row[c], c))[:k]  # descending, equal values in column order
    assert gi == want and gv == pytest.approx([row[c] for c in want], rel=1e-6)

def test_dense_gather_reads_rows():
  table = Tensor.randn(1000, 256, device="NV").half().realize()
  ids = Tensor([[7, 999], [0, 7]], dtype=dtypes.int32, device="NV")
  assert dense_gather(ids, table).tolist() == table.float()[ids.flatten()].reshape(2, 2, 256).tolist()
