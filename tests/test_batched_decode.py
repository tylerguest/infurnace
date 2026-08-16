import unittest
import pytest
from tinygrad import Tensor, dtypes
from infurnace.executor.tinygrad.buffers import ContiguousKVCache
from infurnace.executor.tinygrad.model import Qwen3Model
from infurnace.executor.tinygrad.runner import Qwen3Runner, RunnerError
from test_runner import _make_config, _make_weights, _logits_close


class TestBatchedDecode(unittest.TestCase):
  """Phase 4B: fixed-shape batched decode agrees with independent single decode."""

  def setUp(self):
    self.config = _make_config()
    self.weights = _make_weights(self.config)

  def _build(self, num_slots: int = 4):
    model = Qwen3Model(self.weights)
    kv = ContiguousKVCache(self.config, max_context=16, num_slots=num_slots)
    model.kv_cache = kv
    return model, kv

  def test_eager_batch_matches_singles(self):
    model, kv = self._build()
    model.prefill(Tensor([[1]], dtype=dtypes.int32), kv, slot=0).realize()
    model.prefill(Tensor([[2]], dtype=dtypes.int32), kv, slot=1).realize()
    model.prefill(Tensor([[3]], dtype=dtypes.int32), kv, slot=2).realize()

    batch = model._decode_batch_step(
      Tensor([[10], [11], [12]], dtype=dtypes.int32), (1, 1, 1), (0, 1, 2)
    ).realize()
    self.assertEqual(batch.shape, (3, self.config.vocab_size))

    for i, tok in enumerate((10, 11, 12)):
      single = model.decode(Tensor([[tok]], dtype=dtypes.int32), position=1, kv_cache=kv, slot=i).realize().tolist()[0]
      _logits_close(single, batch.tolist()[i])

  def test_batch_replay_matches_singles(self):
    model, kv = self._build()
    runner = Qwen3Runner(model, kv)
    runner.prefill(Tensor([[1]], dtype=dtypes.int32), slot=0).realize()
    runner.prefill(Tensor([[2]], dtype=dtypes.int32), slot=1).realize()

    for step in range(3):
      pos = 1 + step
      batch = runner.decode_batch(
        Tensor([[10 + step], [20 + step]], dtype=dtypes.int32), (pos, pos), (0, 1)
      ).realize()
      for i, tok in enumerate((10 + step, 20 + step)):
        single = model.decode(Tensor([[tok]], dtype=dtypes.int32), position=pos, kv_cache=kv, slot=i).realize().tolist()[0]
        _logits_close(single, batch.tolist()[i])

  def test_padded_batch_dummy_row_does_not_corrupt(self):
    # 3 active rows -> shape 4 with one padded row writing the reserved dummy slot.
    model, kv = self._build()
    runner = Qwen3Runner(model, kv)
    runner.prefill(Tensor([[1]], dtype=dtypes.int32), slot=0).realize()
    runner.prefill(Tensor([[2]], dtype=dtypes.int32), slot=1).realize()
    runner.prefill(Tensor([[3]], dtype=dtypes.int32), slot=2).realize()

    batch = runner.decode_batch(
      Tensor([[10], [11], [12]], dtype=dtypes.int32), (1, 1, 1), (0, 1, 2)
    ).realize()
    self.assertEqual(batch.shape, (3, self.config.vocab_size))

    for i, tok in enumerate((10, 11, 12)):
      single = model.decode(Tensor([[tok]], dtype=dtypes.int32), position=1, kv_cache=kv, slot=i).realize().tolist()[0]
      _logits_close(single, batch.tolist()[i])

    # A subsequent single decode on slot 2 must remain correct (no corruption).
    after = runner.decode(Tensor([[13]], dtype=dtypes.int32), position=2, slot=2).realize().tolist()[0]
    expected = model.decode(Tensor([[13]], dtype=dtypes.int32), position=2, kv_cache=kv, slot=2).realize().tolist()[0]
    _logits_close(expected, after)

  def test_shape_one_and_two_contracts(self):
    model, kv = self._build()
    runner = Qwen3Runner(model, kv)
    runner.prefill(Tensor([[1]], dtype=dtypes.int32), slot=0).realize()
    b1 = runner.decode_batch(Tensor([[9]], dtype=dtypes.int32), (1,), (0,)).realize()
    s1 = model.decode(Tensor([[9]], dtype=dtypes.int32), position=1, kv_cache=kv, slot=0).realize().tolist()[0]
    _logits_close(s1, b1.tolist()[0])

  def test_rejects_non_compacted_slots(self):
    model, kv = self._build()
    runner = Qwen3Runner(model, kv)
    with self.assertRaises(RunnerError):
      runner.decode_batch(Tensor([[1], [2]], dtype=dtypes.int32), (1, 1), (1, 2))

  def test_rejects_unsupported_batch_size(self):
    model, kv = self._build()
    runner = Qwen3Runner(model, kv)
    with self.assertRaises(RunnerError):
      runner.decode_batch(
        Tensor([[1]] * 5, dtype=dtypes.int32), (1,) * 5, (0, 1, 2, 3, 4)
      )

  def test_rejects_position_out_of_range(self):
    model, kv = self._build()
    runner = Qwen3Runner(model, kv)
    with self.assertRaises(RunnerError):
      runner.decode_batch(Tensor([[1]], dtype=dtypes.int32), (16,), (0,))

  def test_move_slot_preserves_decode(self):
    model, kv = self._build()
    runner = Qwen3Runner(model, kv)
    runner.prefill(Tensor([[1]], dtype=dtypes.int32), slot=2).realize()
    runner.decode(Tensor([[7]], dtype=dtypes.int32), position=1, slot=2).realize()
    expected = model.decode(Tensor([[8]], dtype=dtypes.int32), position=2, kv_cache=kv, slot=2).realize().tolist()[0]

    runner.move_slot(2, 0)
    actual = runner.decode(Tensor([[8]], dtype=dtypes.int32), position=2, slot=0).realize().tolist()[0]
    _logits_close(expected, actual)

    # Source slot is zeroed after the move.
    zeros = Tensor.zeros((self.config.block_count, 2, 1, 16, self.config.attention_head_count_kv, self.config.key_length), dtype=dtypes.float16)
    self.assertEqual(kv.kv[:, :, 2:3].tolist(), zeros.tolist())


@pytest.mark.nv
class TestBatchedDecodeNV(unittest.TestCase):
  """Gated: run the fixed-shape batched decode contract on the NV backend."""

  def setUp(self):
    self.config = _make_config()
    self.weights = _make_weights(self.config)

  def test_batch_replay_matches_singles(self):
    model = Qwen3Model(self.weights)
    kv = ContiguousKVCache(self.config, max_context=16, num_slots=4)
    runner = Qwen3Runner(model, kv)
    runner.prefill(Tensor([[1]], dtype=dtypes.int32), slot=0).realize()
    runner.prefill(Tensor([[2]], dtype=dtypes.int32), slot=1).realize()

    batch = runner.decode_batch(
      Tensor([[10], [20]], dtype=dtypes.int32), (1, 1), (0, 1)
    ).realize()
    for i, tok in enumerate((10, 20)):
      single = model.decode(Tensor([[tok]], dtype=dtypes.int32), position=1, kv_cache=kv, slot=i).realize().tolist()[0]
      _logits_close(single, batch.tolist()[i])


if __name__ == "__main__":
  unittest.main()