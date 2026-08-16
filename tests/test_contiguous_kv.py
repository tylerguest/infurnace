import unittest
from tinygrad import Tensor, dtypes
from infurnace.executor.tinygrad.buffers import ContiguousKVCache, KVCacheError
from infurnace.models.config import Qwen3Config

def _make_config(**overrides) -> Qwen3Config:
  defaults = dict(
    architecture="qwen3", block_count=2, context_length=8, embedding_length=8,
    feed_forward_length=16, attention_head_count=2, attention_head_count_kv=1,
    key_length=4, value_length=4, rope_dimension_count=4, rope_freq_base=10000.0,
    rms_norm_epsilon=1e-6, vocab_size=16, quantization_version=2, file_type=7,
    quantization="Q8_0", qk_norm=True, attention_bias=False, mlp_bias=False,
    tied_embeddings=True, mlp_type="swiglu", tensors=(),
  )
  return Qwen3Config(**{**defaults, **overrides})

class TestContiguousKVCache(unittest.TestCase):
  def test_default_allocation_shape_and_size(self):
    config = _make_config(block_count=2, attention_head_count_kv=1, key_length=4, value_length=4)
    cache = ContiguousKVCache(config, max_context=8, num_slots=1)
    self.assertEqual(cache.shape, (2, 2, 1, 8, 1, 4))
    self.assertEqual(cache.dtype, dtypes.float16)
    self.assertEqual(cache.size_bytes, 2 * 2 * 1 * 8 * 1 * 4 * 2)

  def test_zeros_initialized(self):
    config = _make_config()
    cache = ContiguousKVCache(config, max_context=8, num_slots=1)
    self.assertEqual(cache.kv.tolist(), Tensor.zeros(cache.shape, dtype=dtypes.float16).tolist())

  def test_contiguous(self):
    config = _make_config()
    cache = ContiguousKVCache(config, max_context=8, num_slots=1)
    # Tensor was created with .contiguous().realize(); verify it has valid strides
    # and can be indexed without error.
    _ = cache.kv[0, 0, 0, 0, 0, 0]

  def test_qwen3_0_6b_1024_size(self):
    config = _make_config(
      block_count=28, attention_head_count_kv=4, key_length=128, value_length=128,
      context_length=32768,
    )
    cache = ContiguousKVCache(config, max_context=1024, num_slots=1)
    expected = 28 * 2 * 1 * 1024 * 4 * 128 * 2
    self.assertEqual(cache.size_bytes, expected)
    self.assertEqual(cache.shape, (28, 2, 1, 1024, 4, 128))

  def test_rejects_zero_max_context(self):
    config = _make_config()
    with self.assertRaises(KVCacheError):
      ContiguousKVCache(config, max_context=0, num_slots=1)

  def test_rejects_zero_slots(self):
    config = _make_config()
    with self.assertRaises(KVCacheError):
      ContiguousKVCache(config, max_context=8, num_slots=0)

  def test_rejects_context_exceeds_model_limit(self):
    config = _make_config(context_length=8)
    with self.assertRaises(KVCacheError):
      ContiguousKVCache(config, max_context=9, num_slots=1)

  def test_rejects_mismatched_key_value_length(self):
    config = _make_config(key_length=4, value_length=8)
    with self.assertRaises(KVCacheError):
      ContiguousKVCache(config, max_context=8, num_slots=1)

  def test_frozen(self):
    config = _make_config()
    cache = ContiguousKVCache(config, max_context=8, num_slots=1)
    with self.assertRaises(AttributeError):
      cache.max_context = 16

  def test_dtype_default(self):
    config = _make_config()
    cache = ContiguousKVCache(config, max_context=8, num_slots=1)
    self.assertEqual(cache.dtype, dtypes.float16)

  def test_clear_slot_zeros_contents(self):
    config = _make_config(block_count=2, attention_head_count_kv=1, key_length=4, value_length=4)
    cache = ContiguousKVCache(config, max_context=8, num_slots=2)
    # Write a non-zero value into slot 0
    ones = Tensor.ones((2, 2, 1, 8, 1, 4), dtype=dtypes.float16)
    cache.kv[:, :, 0:1].assign(ones).realize()
    self.assertNotEqual(
      cache.kv[:, :, 0:1].tolist(),
      Tensor.zeros((2, 2, 1, 8, 1, 4), dtype=dtypes.float16).tolist(),
    )
    cache.clear_slot(0)
    zeros = Tensor.zeros((2, 2, 1, 8, 1, 4), dtype=dtypes.float16)
    self.assertEqual(cache.kv[:, :, 0:1].tolist(), zeros.tolist())
    # Slot 1 should be untouched (still zeros from initialization)
    self.assertEqual(
      cache.kv[:, :, 1:2].tolist(),
      Tensor.zeros((2, 2, 1, 8, 1, 4), dtype=dtypes.float16).tolist(),
    )