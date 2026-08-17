import unittest
from tinygrad import Tensor, dtypes
from infurnace.executor.tinygrad.buffers import PagedKVCache, KVCacheError
from infurnace.kernels.kv_store import KVStoreError, decompose_slot, store_kv
from infurnace.models.config import Qwen3Config

def _make_config(**overrides) -> Qwen3Config:
  defaults = dict(
    architecture="qwen3", block_count=2, context_length=32, embedding_length=8,
    feed_forward_length=16, attention_head_count=2, attention_head_count_kv=1,
    key_length=4, value_length=4, rope_dimension_count=4, rope_freq_base=10000.0,
    rms_norm_epsilon=1e-6, vocab_size=16, quantization_version=2, file_type=7,
    quantization="Q8_0", qk_norm=True, attention_bias=False, mlp_bias=False,
    tied_embeddings=True, mlp_type="swiglu", tensors=(),
  )
  return Qwen3Config(**{**defaults, **overrides})

class TestPagedKVCache(unittest.TestCase):
  def test_allocation_shape_and_size(self):
    config = _make_config(block_count=2, attention_head_count_kv=1, key_length=4, value_length=4)
    cache = PagedKVCache(config, max_context=32, num_pages=8, page_size=4, num_dummy_pages=2)
    self.assertEqual(cache.shape, (2, 2, 10, 4, 1, 4))
    self.assertEqual(cache.dtype, dtypes.float16)
    self.assertEqual(cache.token_capacity, 32)
    self.assertEqual(cache.size_bytes, 2 * 2 * 10 * 4 * 1 * 4 * 2)

  def test_zeros_initialized(self):
    cache = PagedKVCache(_make_config(), max_context=32, num_pages=8, page_size=4, num_dummy_pages=2)
    self.assertEqual(cache.kv.tolist(), Tensor.zeros(cache.shape, dtype=dtypes.float16).tolist())

  def test_flat_slot_arithmetic(self):
    cache = PagedKVCache(_make_config(), max_context=32, num_pages=8, page_size=4, num_dummy_pages=2)
    self.assertEqual(cache.flat_slot(0, 0), 0)
    self.assertEqual(cache.flat_slot(3, 2), 14)
    with self.assertRaises(KVCacheError):
      cache.flat_slot(10, 0)
    with self.assertRaises(KVCacheError):
      cache.flat_slot(0, 4)

  def test_clear_page_zeros_contents(self):
    config = _make_config(block_count=2, attention_head_count_kv=1, key_length=4, value_length=4)
    cache = PagedKVCache(config, max_context=32, num_pages=8, page_size=4, num_dummy_pages=2)
    cache.kv[:, :, 0:1].assign(Tensor.ones((2, 2, 1, 4, 1, 4), dtype=dtypes.float16)).realize()
    cache.clear_page(0)
    self.assertEqual(cache.kv[:, :, 0:1].tolist(), Tensor.zeros((2, 2, 1, 4, 1, 4), dtype=dtypes.float16).tolist())

  def test_rejects_invalid_configuration(self):
    config = _make_config()
    with self.assertRaises(KVCacheError): PagedKVCache(config, max_context=32, num_pages=0, page_size=4)
    with self.assertRaises(KVCacheError): PagedKVCache(config, max_context=32, num_pages=8, page_size=0)
    with self.assertRaises(KVCacheError): PagedKVCache(config, max_context=32, num_pages=8, page_size=4, num_dummy_pages=-1)
    with self.assertRaises(KVCacheError): PagedKVCache(config, max_context=0, num_pages=8, page_size=4)
    with self.assertRaises(KVCacheError): PagedKVCache(config, max_context=33, num_pages=8, page_size=4)
    with self.assertRaises(KVCacheError): PagedKVCache(_make_config(key_length=4, value_length=8), max_context=32, num_pages=8, page_size=4)

class TestIndexedKVStore(unittest.TestCase):
  def _cache(self, num_pages=8, page_size=4, dummy=2):
    config = _make_config(block_count=2, attention_head_count_kv=1, key_length=4, value_length=4)
    return PagedKVCache(config, max_context=32, num_pages=num_pages, page_size=page_size, num_dummy_pages=dummy)

  def test_decompose_slot(self):
    self.assertEqual(decompose_slot(5, 4), (1, 1))
    self.assertEqual(decompose_slot(0, 4), (0, 0))
    self.assertEqual(decompose_slot(7, 4), (1, 3))
    with self.assertRaises(KVStoreError): decompose_slot(-1, 4)
    with self.assertRaises(KVStoreError): decompose_slot(1, 0)

  def test_store_single_token(self):
    cache = self._cache()
    k = Tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=dtypes.float16)
    v = Tensor([[[5.0, 6.0, 7.0, 8.0]]], dtype=dtypes.float16)
    store_kv(cache.kv, 0, k, v, [5], [True], num_pages=8, page_size=4)
    page, offset = decompose_slot(5, 4)
    self.assertEqual(cache.kv[0, 0, page, offset, 0].tolist(), [1.0, 2.0, 3.0, 4.0])
    self.assertEqual(cache.kv[0, 1, page, offset, 0].tolist(), [5.0, 6.0, 7.0, 8.0])

  def test_store_matches_dense_reference(self):
    # Core 5B gate: scattered per-token writes equal a dense per-page block write.
    cache = self._cache()
    writes = [(page, offset) for page in range(4) for offset in range(4)][:12]
    slot_mapping = [page * 4 + offset for page, offset in writes]
    k = Tensor([[[float(page * 4 + offset + 1) for _ in range(4)]] for page, offset in writes], dtype=dtypes.float16)
    v = Tensor([[[float(page * 4 + offset + 100) for _ in range(4)]] for page, offset in writes], dtype=dtypes.float16)
    store_kv(cache.kv, 1, k, v, slot_mapping, [True] * len(writes), num_pages=8, page_size=4)
    expected = Tensor.zeros(cache.shape, dtype=dtypes.float16)
    for page in range(3):
      ks = [float(page * 4 + offset + 1) for offset in range(4)]
      vs = [float(page * 4 + offset + 100) for offset in range(4)]
      k_block = Tensor([[v] * 4 for v in ks], dtype=dtypes.float16).unsqueeze(1)
      v_block = Tensor([[v] * 4 for v in vs], dtype=dtypes.float16).unsqueeze(1)
      kv_block = Tensor.stack(k_block, v_block)
      expected[1, :, page, :, :, :].assign(kv_block).realize()
    self.assertEqual(cache.kv.tolist(), expected.tolist())

  def test_masked_inactive_writes_unique_dummy(self):
    cache = self._cache(dummy=2)
    k = Tensor.arange(8, dtype=dtypes.float16).reshape(2, 1, 4)
    v = Tensor.arange(8, 16, dtype=dtypes.float16).reshape(2, 1, 4)
    store_kv(cache.kv, 0, k, v, [3, 0], [True, False], num_pages=8, page_size=4)
    self.assertEqual(cache.kv[0, 0, 0, 3, 0].tolist(), [0.0, 1.0, 2.0, 3.0])
    self.assertEqual(cache.kv[0, 0, 8, 0, 0].tolist(), [4.0, 5.0, 6.0, 7.0])
    self.assertEqual(cache.kv[0, 1, 8, 0, 0].tolist(), [12.0, 13.0, 14.0, 15.0])
    self.assertEqual(cache.kv[0, 0, 0, 0, 0].tolist(), [0.0, 0.0, 0.0, 0.0])

  def test_two_inactive_rows_do_not_alias(self):
    cache = self._cache(dummy=4)
    k = Tensor.arange(12, dtype=dtypes.float16).reshape(3, 1, 4)
    v = Tensor.arange(12, 24, dtype=dtypes.float16).reshape(3, 1, 4)
    store_kv(cache.kv, 0, k, v, [1, 2, 3], [True, False, False], num_pages=8, page_size=4)
    # Row 1 -> dummy (8, 0), row 2 -> dummy (9, 0): distinct positions.
    self.assertEqual(cache.kv[0, 0, 8, 0, 0].tolist(), [4.0, 5.0, 6.0, 7.0])
    self.assertEqual(cache.kv[0, 0, 9, 0, 0].tolist(), [8.0, 9.0, 10.0, 11.0])

  def test_fp32_input_casts_to_fp16(self):
    cache = self._cache()
    k = Tensor([[[1.0, 2.0, 3.0, 4.0]]])
    v = Tensor([[[5.0, 6.0, 7.0, 8.0]]])
    store_kv(cache.kv, 0, k, v, [5], [True], num_pages=8, page_size=4)
    self.assertEqual(cache.kv[0, 0, 1, 1, 0].tolist(), [1.0, 2.0, 3.0, 4.0])

  def test_validation_errors(self):
    cache = self._cache(dummy=4)
    k = Tensor.arange(8, dtype=dtypes.float16).reshape(2, 1, 4)
    v = Tensor.arange(8, 16, dtype=dtypes.float16).reshape(2, 1, 4)
    with self.assertRaises(KVStoreError): store_kv(cache.kv, 2, k, v, [0, 1], [True, True], num_pages=8, page_size=4)
    with self.assertRaises(KVStoreError): store_kv(cache.kv, 0, k, v, [32, 1], [True, True], num_pages=8, page_size=4)
    with self.assertRaises(KVStoreError): store_kv(cache.kv, 0, k[:, :, :3], v, [0, 1], [True, True], num_pages=8, page_size=4)
    with self.assertRaises(KVStoreError): store_kv(cache.kv, 0, k, v, [0], [True, True], num_pages=8, page_size=4)
    with self.assertRaises(KVStoreError): store_kv(cache.kv, 0, k, v, [0, 1], [True, False, True], num_pages=8, page_size=4)
    k3 = Tensor.arange(12, dtype=dtypes.float16).reshape(3, 1, 4)
    v3 = Tensor.arange(12, 24, dtype=dtypes.float16).reshape(3, 1, 4)
    with self.assertRaises(KVStoreError): store_kv(self._cache(dummy=1).kv, 0, k3, v3, [0, 1, 2], [True, False, False], num_pages=8, page_size=4)

if __name__ == "__main__":
  unittest.main()