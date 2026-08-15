import math
import unittest
from types import MappingProxyType
from tinygrad import Tensor, dtypes
from infurnace.executor.tinygrad.buffers import ContiguousKVCache
from infurnace.executor.tinygrad.model import Qwen3Model, Qwen3ModelError
from infurnace.executor.tinygrad.weights import Qwen3Weights, WeightPolicy
from infurnace.models.config import Qwen3Config, TensorSpec

def _tensor_specs(config: Qwen3Config) -> tuple[TensorSpec, ...]:
  specs = [
    TensorSpec("output_norm.weight", (config.embedding_length,), "F32", "float"),
    TensorSpec("token_embd.weight", (config.vocab_size, config.embedding_length), "Q8_0", "float"),
  ]
  qw = config.attention_head_count * config.key_length
  kw = config.attention_head_count_kv * config.key_length
  vw = config.attention_head_count_kv * config.value_length
  ow = config.attention_head_count * config.value_length
  for i in range(config.block_count):
    p = f"blk.{i}"
    specs.extend((
      TensorSpec(f"{p}.attn_k.weight", (kw, config.embedding_length), "Q8_0", "float"),
      TensorSpec(f"{p}.attn_k_norm.weight", (config.key_length,), "F32", "float"),
      TensorSpec(f"{p}.attn_norm.weight", (config.embedding_length,), "F32", "float"),
      TensorSpec(f"{p}.attn_output.weight", (config.embedding_length, ow), "Q8_0", "float"),
      TensorSpec(f"{p}.attn_q.weight", (qw, config.embedding_length), "Q8_0", "float"),
      TensorSpec(f"{p}.attn_q_norm.weight", (config.key_length,), "F32", "float"),
      TensorSpec(f"{p}.attn_v.weight", (vw, config.embedding_length), "Q8_0", "float"),
      TensorSpec(f"{p}.ffn_down.weight", (config.embedding_length, config.feed_forward_length), "Q8_0", "float"),
      TensorSpec(f"{p}.ffn_gate.weight", (config.feed_forward_length, config.embedding_length), "Q8_0", "float"),
      TensorSpec(f"{p}.ffn_norm.weight", (config.embedding_length,), "F32", "float"),
      TensorSpec(f"{p}.ffn_up.weight", (config.feed_forward_length, config.embedding_length), "Q8_0", "float"),
    ))
  return tuple(specs)

def _make_config(**overrides) -> Qwen3Config:
  defaults = dict(
    architecture="qwen3", block_count=2, context_length=16, embedding_length=8,
    feed_forward_length=16, attention_head_count=2, attention_head_count_kv=1,
    key_length=4, value_length=4, rope_dimension_count=4, rope_freq_base=10000.0,
    rms_norm_epsilon=1e-6, vocab_size=16, quantization_version=2, file_type=7,
    quantization="Q8_0", qk_norm=True, attention_bias=False, mlp_bias=False,
    tied_embeddings=True, mlp_type="swiglu", tensors=(),
  )
  config = Qwen3Config(**{**defaults, **overrides})
  if not config.tensors:
    from dataclasses import replace
    config = replace(config, tensors=_tensor_specs(config))
  return config

def _make_weights(config: Qwen3Config) -> Qwen3Weights:
  tensors = {}
  tensors["token_embd.weight"] = Tensor.arange(config.vocab_size * config.embedding_length).reshape(
    config.vocab_size, config.embedding_length
  ) * 0.01
  tensors["output_norm.weight"] = Tensor.ones(config.embedding_length)
  for i in range(config.block_count):
    p = f"blk.{i}"
    qw = config.attention_head_count * config.key_length
    kw = config.attention_head_count_kv * config.key_length
    vw = config.attention_head_count_kv * config.value_length
    tensors[f"{p}.attn_norm.weight"] = Tensor.ones(config.embedding_length)
    tensors[f"{p}.attn_q.weight"] = Tensor.ones(qw, config.embedding_length) * 0.01
    tensors[f"{p}.attn_q_norm.weight"] = Tensor.ones(config.key_length)
    tensors[f"{p}.attn_k.weight"] = Tensor.ones(kw, config.embedding_length) * 0.02
    tensors[f"{p}.attn_k_norm.weight"] = Tensor.ones(config.key_length)
    tensors[f"{p}.attn_v.weight"] = Tensor.ones(vw, config.embedding_length) * 0.03
    tensors[f"{p}.attn_output.weight"] = Tensor.ones(config.embedding_length, config.attention_head_count * config.value_length) * 0.01
    tensors[f"{p}.ffn_norm.weight"] = Tensor.ones(config.embedding_length)
    tensors[f"{p}.ffn_gate.weight"] = Tensor.zeros(config.feed_forward_length, config.embedding_length)
    tensors[f"{p}.ffn_up.weight"] = Tensor.zeros(config.feed_forward_length, config.embedding_length)
    tensors[f"{p}.ffn_down.weight"] = Tensor.zeros(config.embedding_length, config.feed_forward_length)
  tensors["output.weight"] = tensors["token_embd.weight"]
  return Qwen3Weights(config, MappingProxyType(tensors), WeightPolicy.LAZY_FP16)

def _logits_close(expected: list[float], actual: list[float], rel_tol: float = 1e-2, abs_tol: float = 1e-3) -> None:
  assert len(expected) == len(actual), f"length mismatch: {len(expected)} vs {len(actual)}"
  for ev, av in zip(expected, actual):
    assert math.isfinite(ev) and math.isfinite(av), f"non-finite values: {ev}, {av}"
    assert math.isclose(ev, av, rel_tol=rel_tol, abs_tol=abs_tol), f"logits differ: {ev} vs {av}"

class TestCachedForward(unittest.TestCase):
  def setUp(self):
    self.config = _make_config()
    self.weights = _make_weights(self.config)
    self.model = Qwen3Model(self.weights)

  def _stateless_logits(self, token_ids: list[int]) -> list[float]:
    return self.model(Tensor([token_ids], dtype=dtypes.int32)).realize().tolist()[0]

  def test_decode_matches_full_recompute(self):
    tokens = [1, 2, 3, 4, 5]
    expected = [self._stateless_logits(tokens[:i+1]) for i in range(len(tokens))]

    cache = ContiguousKVCache(self.config, max_context=16, num_slots=1)
    actual = []
    for i, token in enumerate(tokens):
      if i == 0:
        logits = self.model.prefill(Tensor([[token]], dtype=dtypes.int32), cache).realize()
      else:
        logits = self.model.decode(Tensor([[token]], dtype=dtypes.int32), position=i, kv_cache=cache).realize()
      actual.append(logits.tolist()[0])

    for e, a in zip(expected, actual):
      _logits_close(e, a)

  def test_chunked_prefill_matches_unchunked(self):
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    expected = self._stateless_logits(tokens)

    cache = ContiguousKVCache(self.config, max_context=16, num_slots=1)
    for chunk_start in range(0, len(tokens), 2):
      chunk = tokens[chunk_start:chunk_start+2]
      logits = self.model(Tensor([chunk], dtype=dtypes.int32), start_position=chunk_start, kv=cache.kv).realize()
    actual = logits.tolist()[0]

    _logits_close(expected, actual)

  def test_slot_isolation(self):
    tokens_a = [1, 2, 3]
    tokens_b = [4, 5, 6]

    cache = ContiguousKVCache(self.config, max_context=16, num_slots=2)
    self.model.prefill(Tensor([tokens_a], dtype=dtypes.int32), cache, slot=0).realize()
    self.model.prefill(Tensor([tokens_b], dtype=dtypes.int32), cache, slot=1).realize()

    logits_a = self.model.decode(Tensor([[7]], dtype=dtypes.int32), position=len(tokens_a), kv_cache=cache, slot=0).realize()
    logits_b = self.model.decode(Tensor([[8]], dtype=dtypes.int32), position=len(tokens_b), kv_cache=cache, slot=1).realize()

    cache_a = ContiguousKVCache(self.config, max_context=16, num_slots=1)
    self.model.prefill(Tensor([tokens_a], dtype=dtypes.int32), cache_a, slot=0).realize()
    expected_a = self.model.decode(Tensor([[7]], dtype=dtypes.int32), position=len(tokens_a), kv_cache=cache_a, slot=0).realize()

    cache_b = ContiguousKVCache(self.config, max_context=16, num_slots=1)
    self.model.prefill(Tensor([tokens_b], dtype=dtypes.int32), cache_b, slot=0).realize()
    expected_b = self.model.decode(Tensor([[8]], dtype=dtypes.int32), position=len(tokens_b), kv_cache=cache_b, slot=0).realize()

    _logits_close(expected_a.tolist()[0], logits_a.tolist()[0])
    _logits_close(expected_b.tolist()[0], logits_b.tolist()[0])

  def test_rejects_slot_out_of_range(self):
    cache = ContiguousKVCache(self.config, max_context=16, num_slots=1)
    with self.assertRaises(Qwen3ModelError):
      self.model.prefill(Tensor([[1]], dtype=dtypes.int32), cache, slot=1)

  def test_rejects_position_beyond_kv_context(self):
    cache = ContiguousKVCache(self.config, max_context=4, num_slots=1)
    with self.assertRaises(Qwen3ModelError):
      self.model.decode(Tensor([[1]], dtype=dtypes.int32), position=4, kv_cache=cache)

  def test_rejects_batch_size_greater_than_one_with_cache(self):
    cache = ContiguousKVCache(self.config, max_context=16, num_slots=1)
    with self.assertRaises(Qwen3ModelError):
      self.model.forward(Tensor([[1], [2]], dtype=dtypes.int32), kv=cache.kv)

  def test_stateless_path_unchanged(self):
    ids = Tensor([[0, 1, 2]], dtype=dtypes.int32)
    r1 = self.model(ids).realize()
    r2 = self.model(ids, start_position=0, kv=None).realize()
    self.assertEqual(r1.tolist(), r2.tolist())
