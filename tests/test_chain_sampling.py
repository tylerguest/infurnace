import os

import pytest
from tinygrad import Context, Tensor, TinyJit, dtypes

from infurnace.eagle.verify import BatchedVerifier
from infurnace.eagle.sampling import sample_chain, score_argmax


@pytest.mark.parametrize('device', ['CPU', 'NV'])
@pytest.mark.parametrize('size', [1, 2, 4, 8])
@pytest.mark.parametrize('stop', ['accept', 'reject', 'eos'])
@pytest.mark.parametrize('materialize_scores', [False, True])
def test_prefix_sampling_preserves_decisions_and_rng_on_replay(device, size, stop, materialize_scores):
  if device == 'NV' and not os.environ.get('INFURNACE_TEST_NV'): pytest.skip('set INFURNACE_TEST_NV=1')
  with Context(DEV=device):
    # Zero/near-zero logits exercise Gumbel draws, not only easy argmax cases.
    logits = Tensor.zeros(1, size, 32, device=device).realize()
    temperature = Tensor([0.0], device=device).realize()
    key = Tensor([123, 456], dtype=dtypes.uint32, device=device).realize()
    counter = Tensor([0xfffffff0, 7], dtype=dtypes.uint32, device=device).realize()
    eos = (31,) if stop == 'eos' else ()
    jit = TinyJit(lambda logits, ids, key, counter: sample_chain(logits, ids, temperature, key, counter,
                                                               eos_token_ids=eos, materialize_scores=materialize_scores))
    for repeat in range(4):
      start = counter.tolist()
      Tensor._device_seeds[device] = key
      reference_counter = Tensor(start, dtype=dtypes.uint32, device=device).realize()
      Tensor._device_rng_counters[device] = reference_counter
      choices = [int(BatchedVerifier._sample(logits, temperature, index).item()) for index in range(size)]
      candidates = [1]+choices[:-1]
      if stop == 'reject' and size > 1: candidates[1] = (choices[0]+1)%32
      if stop == 'eos': candidates[0] = 31
      count = 1
      while count < size and candidates[count] == choices[count-1] and candidates[count-1] not in eos: count += 1
      ids = Tensor([candidates], dtype=dtypes.int32, device=device).realize()
      actual = jit(logits, ids, key, counter)
      data = Tensor(actual.uop.buf_uop).reshape(actual.shape).tolist()[0]
      assert data[:count] == choices[:count]
      assert data[-1] == count
      expected = ((start[1]<<32)|start[0])+count*32
      assert counter.tolist() == [expected & 0xffffffff, expected>>32]
      # The next draw must match a sequential run that consumed only the prefix.
      reference_counter.assign(Tensor([expected & 0xffffffff, expected>>32], dtype=dtypes.uint32, device=device)).realize()
      expected_random = Tensor.rand(32, device=device).tolist()
      Tensor._device_rng_counters[device] = counter
      assert Tensor.rand(32, device=device).tolist() == expected_random


@pytest.mark.parametrize('device', ['CPU', 'NV'])
@pytest.mark.parametrize('temperature_value', [0.0, 0.7])
@pytest.mark.parametrize('vocabulary', [32, 1024, 151936])
def test_score_materialization_preserves_decisions_and_tie_order(device, temperature_value, vocabulary):
  if device == 'NV' and not os.environ.get('INFURNACE_TEST_NV'): pytest.skip('set INFURNACE_TEST_NV=1')
  with Context(DEV=device):
    Tensor.manual_seed(123)
    random = Tensor.rand(1, vocabulary, device=device).realize()
    temperature = Tensor([temperature_value], device=device).realize()
    # Zero rows force RNG to choose; constant positive rows at T=0 create
    # rounding ties. Both variants must retain the target's first-index rule.
    for value in (0.0, 1.0):
      row = Tensor.full((1, vocabulary), value, device=device).realize()
      original = score_argmax(row, temperature, random)
      materialized = score_argmax(row, temperature, random, materialize_scores=True)
      assert materialized.item() == original.item()


@pytest.mark.parametrize('device', ['CPU', 'NV'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_score_materialization_preserves_nonfinite_behavior(device, value):
  if device == 'NV' and not os.environ.get('INFURNACE_TEST_NV'): pytest.skip('set INFURNACE_TEST_NV=1')
  with Context(DEV=device):
    temperature = Tensor([0.7], device=device).realize()
    random = Tensor.full((1, 32), 0.5, device=device).realize()
    for position in (0, 1, 15, 31):
      values = [(index%7)*0.1 for index in range(32)]
      values[position] = value
      row = Tensor([values], device=device).realize()
      assert score_argmax(row, temperature, random, materialize_scores=True).item() == score_argmax(row, temperature, random).item()


def test_score_materialization_rejects_invalid_flags_before_rng_mutation():
  row, temperature, random = Tensor([[1.0]]), Tensor([0.0]), Tensor([[0.5]])
  with pytest.raises(ValueError, match='boolean'): score_argmax(row, temperature, random, materialize_scores=1)
  key, counter = Tensor([1, 2], dtype=dtypes.uint32), Tensor([0, 0], dtype=dtypes.uint32)
  with pytest.raises(ValueError, match='boolean'):
    sample_chain(row.reshape(1, 1, 1), Tensor([[0]]), temperature, key, counter, materialize_scores=1)
  assert counter.tolist() == [0, 0]
