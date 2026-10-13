import collections
import math
import pytest
from tinygrad import Context, Tensor, TinyJit
from infurnace.kernels.sampling import Sampler

pytestmark = pytest.mark.nv

def chi_square_pvalue(counts, probs, draws):
  cells = [i for i, p in enumerate(probs) if draws * p >= 5]
  x, k = sum((counts[i] - draws * probs[i]) ** 2 / (draws * probs[i]) for i in cells), len(cells) - 1
  z = ((x / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))  # Wilson-Hilferty
  return 0.5 * math.erfc(z / math.sqrt(2))

@pytest.mark.parametrize('rows,vocab', [(1, 40), (3, 2048), (3, 2049), (16, 3000), (2, 70000), (16, 248320)])
def test_zero_temperature_is_argmax(rows, vocab):
  # 2049 and 3000 split rows into uneven chunks: the case that once returned token 0.
  with Context(DEV='NV'):
    Tensor.manual_seed(1)
    logits = (Tensor.randn(rows, vocab) * 2).contiguous().realize()
    assert Sampler('NV', 7)(logits, Tensor([0.0])).tolist() == logits.argmax(-1).tolist()
    ties = Tensor.zeros(rows, vocab).contiguous().realize()
    assert Sampler('NV', 7)(ties, Tensor([0.0])).tolist() == [0] * rows  # first maximum, like argmax

@pytest.mark.parametrize('temperature,vocab', [(1.0, 40), (0.6, 3000), (0.3, 248320)])
def test_samples_follow_softmax_inside_a_captured_graph(temperature, vocab):
  draws = 3000
  with Context(DEV='NV'):
    Tensor.manual_seed(2)
    logits = (Tensor.randn(2, vocab) * 2).contiguous().realize()
    if vocab > 100000:  # a handful of likely tokens spread over chunks, so samples cross chunk boundaries
      bumps = [(5, 3.0), (2049, 2.8), (70000, 2.6), (130001, 2.4), (200000, 2.2), (vocab - 1, 2.0)]
      values = logits.tolist()
      for row in values:
        row[:] = [v * 0.25 for v in row]
        for position, value in bumps: row[position] = value
      logits = Tensor(values).contiguous().realize()
    sampler, t = Sampler('NV', 11), Tensor([temperature]).realize()
    step = TinyJit(lambda l, t: sampler(l, t).realize())  # replays must draw fresh noise
    counts = [collections.Counter() for _ in range(2)]
    for _ in range(draws):
      for row, token in enumerate(step(logits, t).tolist()): counts[row][token] += 1
    probs = (logits / temperature).softmax(-1).tolist()
    for row in range(2): assert chi_square_pvalue(counts[row], probs[row], draws) > 0.001
    assert sampler.step.tolist() == [draws]
