"""Experimental reduction must preserve target decisions, including special values."""
import os

import pytest
from tinygrad import Tensor, TinyJit, Context, dtypes

from benchmarks.experiments.eagle3.parallel_argmax import parallel_argmax

pytestmark = pytest.mark.skipif(not os.environ.get('INFURNACE_TEST_NV'), reason='set INFURNACE_TEST_NV=1')


@pytest.mark.parametrize('size', [1, 32, 511, 512, 513, 151936])
def test_parallel_argmax_random_and_replay(size):
  with Context(DEV='NV'):
    jit = TinyJit(parallel_argmax)
    for seed in range(5):
      Tensor.manual_seed(seed)
      scores = Tensor.randn(1, size, device='NV').realize()
      expected = scores.argmax(-1, keepdim=True).item()
      assert jit(scores).item() == expected


@pytest.mark.parametrize('size', [32, 513, 151936])
@pytest.mark.parametrize('kind', ['ties', 'nan', 'allnan', 'inf', 'neginf', 'zero'])
def test_parallel_argmax_special_values(size, kind):
  with Context(DEV='NV'):
    values = [0.0]*size
    if kind == 'ties':
      values[size//3] = values[-1] = 10.0
    elif kind == 'nan':
      values[0] = values[size//2] = float('nan')
      values[-1] = 2.0
    elif kind == 'allnan': values = [float('nan')]*size
    elif kind == 'inf': values[size//3] = values[-1] = float('inf')
    elif kind == 'neginf': values = [float('-inf')]*size
    else: values[0] = -0.0
    scores = Tensor([values], device='NV').realize()
    expected = scores.argmax(-1, keepdim=True).item()
    assert parallel_argmax(scores).item() == expected


@pytest.mark.parametrize('size', [1, 2, 4, 8])
@pytest.mark.parametrize('stop', ['accept', 'reject', 'eos'])
def test_parallel_chain_preserves_prefix_counter_and_replay(size, stop, monkeypatch):
  import infurnace.eagle.sampling as sampling
  from benchmarks.experiments.eagle3.parallel_argmax import parallel_score_argmax
  original = sampling.score_argmax
  with Context(DEV='NV'):
    logits = Tensor.zeros(1, size, 32, device='NV').realize()
    temperature = Tensor([0.0], device='NV').realize()
    key = Tensor([123, 456], dtype=dtypes.uint32, device='NV').realize()
    counter = Tensor([0xfffffff0, 7], dtype=dtypes.uint32, device='NV').realize()
    eos = (31,) if stop == 'eos' else ()
    jit = TinyJit(lambda logits, ids, key, counter: sampling.sample_chain(
      logits, ids, temperature, key, counter, eos_token_ids=eos, materialize_scores=True))
    for _ in range(4):
      start = counter.tolist()
      def reset_counter(): return Tensor(start, dtype=dtypes.uint32, device='NV').realize()
      monkeypatch.setattr(sampling, 'score_argmax', original)
      probe = sampling.sample_chain(logits, Tensor.zeros(1, size, dtype=dtypes.int32, device='NV').realize(),
                                    temperature, key, reset_counter(), materialize_scores=True).tolist()[0]
      candidates = [31 if stop == 'eos' else 1]+probe[:size-1]
      if stop == 'reject' and size > 1: candidates[1] = (candidates[1]+1)%32
      ids = Tensor([candidates], dtype=dtypes.int32, device='NV').realize()
      reference_counter = reset_counter()
      expected = sampling.sample_chain(logits, ids, temperature, key, reference_counter,
                                       eos_token_ids=eos, materialize_scores=True).tolist()
      monkeypatch.setattr(sampling, 'score_argmax', parallel_score_argmax)
      actual = jit(logits, ids, key, counter)
      assert Tensor(actual.uop.buf_uop).reshape(actual.shape).tolist() == expected
      assert counter.tolist() == reference_counter.tolist()
      assert Tensor._rand(key, counter, (32,), dtypes.float32).tolist() == Tensor._rand(
        key, reference_counter, (32,), dtypes.float32).tolist()
