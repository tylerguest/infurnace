import math
import unittest
from types import MappingProxyType
from tinygrad import Tensor, dtypes
from infurnace.executor.tinygrad.buffers import ContiguousKVCache
from infurnace.executor.tinygrad.model import Qwen3Model
from infurnace.executor.tinygrad.runner import Qwen3Runner, RunnerError
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

class TestQwen3RunnerDecode(unittest.TestCase):
  def setUp(self):
    self.config = _make_config()
    self.weights = _make_weights(self.config)
    self.model = Qwen3Model(self.weights)

  def test_decode_replay_matches_eager(self):
    tokens = [1, 2, 3, 4, 5]
    eager_cache = ContiguousKVCache(self.config, max_context=16, num_slots=1)
    self.model.prefill(Tensor([tokens[:1]], dtype=dtypes.int32), eager_cache).realize()
    eager_logits = []
    for i in range(1, len(tokens)):
      logits = self.model.decode(Tensor([[tokens[i]]], dtype=dtypes.int32), position=i, kv_cache=eager_cache).realize()
      eager_logits.append(logits.tolist()[0])

    runner_cache = ContiguousKVCache(self.config, max_context=16, num_slots=1)
    runner = Qwen3Runner(self.model, runner_cache)
    runner.prefill(Tensor([tokens[:1]], dtype=dtypes.int32)).realize()
    runner_logits = []
    for i in range(1, len(tokens)):
      logits = runner.decode(Tensor([[tokens[i]]], dtype=dtypes.int32), position=i).realize()
      runner_logits.append(logits.tolist()[0])

    for e, r in zip(eager_logits, runner_logits):
      _logits_close(e, r)

  def test_decode_does_not_recompile_per_position(self):
    tokens = list(range(1, 9))  # positions 1..7
    cache = ContiguousKVCache(self.config, max_context=16, num_slots=1)
    runner = Qwen3Runner(self.model, cache)
    runner.prefill(Tensor([tokens[:1]], dtype=dtypes.int32)).realize()

    for i in range(1, len(tokens)):
      runner.decode(Tensor([[tokens[i]]], dtype=dtypes.int32), position=i).realize()

    jit = runner._decode_jit[0]
    # warmup (1) + capture (1) + 7 replays = 9 total calls
    self.assertLessEqual(jit.cnt, 9)

  def test_independent_runners_produce_same_logits(self):
    tokens = [1, 2, 3]
    cache1 = ContiguousKVCache(self.config, max_context=16, num_slots=1)
    runner1 = Qwen3Runner(self.model, cache1)
    runner1.prefill(Tensor([tokens[:1]], dtype=dtypes.int32)).realize()
    runner1.decode(Tensor([[tokens[1]]], dtype=dtypes.int32), position=1).realize()
    logits1 = runner1.decode(Tensor([[tokens[2]]], dtype=dtypes.int32), position=2).realize().tolist()[0]

    # A second runner with a fresh cache must produce identical logits. Each runner
    # captures its own JIT contract bound to its own cache buffer.
    model2 = Qwen3Model(self.weights)
    cache2 = ContiguousKVCache(self.config, max_context=16, num_slots=1)
    runner2 = Qwen3Runner(model2, cache2)
    runner2.prefill(Tensor([tokens[:1]], dtype=dtypes.int32)).realize()
    runner2.decode(Tensor([[tokens[1]]], dtype=dtypes.int32), position=1).realize()
    logits2 = runner2.decode(Tensor([[tokens[2]]], dtype=dtypes.int32), position=2).realize().tolist()[0]

    _logits_close(logits1, logits2)

  def test_model_rejects_decode_without_cache(self):
    with self.assertRaises(ValueError):
      self.model._decode_step(
        Tensor([[1]], dtype=dtypes.int32),
        Tensor.full((1, 1, 1, 16), 0.0, dtype=dtypes.float32).contiguous().realize(),
        self.model._rope[0:1].contiguous().realize(),
      )

  def test_rejects_decode_input_wrong_shape(self):
    cache = ContiguousKVCache(self.config, max_context=16, num_slots=1)
    runner = Qwen3Runner(self.model, cache)
    with self.assertRaises(RunnerError):
      runner.decode(Tensor([[1, 2]], dtype=dtypes.int32), position=0)

  def test_rejects_position_out_of_range(self):
    cache = ContiguousKVCache(self.config, max_context=16, num_slots=1)
    runner = Qwen3Runner(self.model, cache)
    with self.assertRaises(RunnerError):
      runner.decode(Tensor([[1]], dtype=dtypes.int32), position=16)


class TestQwen3RunnerStress(unittest.TestCase):
  """Phase 2D: stateful stress validation across positions, conversations, and slots."""

  def setUp(self):
    self.config = _make_config()
    self.weights = _make_weights(self.config)
    self.model = Qwen3Model(self.weights)

  def _stateless_logits(self, token_ids: list[int]) -> list[float]:
    return self.model(Tensor([token_ids], dtype=dtypes.int32)).realize().tolist()[0]

  def _run_jit_conversation(self, runner: Qwen3Runner, tokens: list[int], slot: int = 0) -> list[list[float]]:
    logits = runner.prefill(Tensor([[tokens[0]]], dtype=dtypes.int32), slot=slot).realize()
    all_logits = [logits.tolist()[0]]
    for i in range(1, len(tokens)):
      logits = runner.decode(Tensor([[tokens[i]]], dtype=dtypes.int32), position=i, slot=slot).realize()
      all_logits.append(logits.tolist()[0])
    return all_logits

  # ------------------------------------------------------------------
  # Per-position correctness
  # ------------------------------------------------------------------

  def test_decode_matches_recompute_every_position(self):
    tokens = list(range(1, 17))
    expected = [self._stateless_logits(tokens[:i+1]) for i in range(len(tokens))]

    cache = ContiguousKVCache(self.config, max_context=16, num_slots=1)
    runner = Qwen3Runner(self.model, cache)
    actual = self._run_jit_conversation(runner, tokens)

    for j, (e, a) in enumerate(zip(expected, actual)):
      _logits_close(e, a, abs_tol=2e-3)

  def test_context_boundary_decode(self):
    mc = 4
    tokens = list(range(1, mc + 1))
    expected = self._stateless_logits(tokens)

    cache = ContiguousKVCache(self.config, max_context=mc, num_slots=1)
    runner = Qwen3Runner(self.model, cache)
    _logits_close(expected, self._run_jit_conversation(runner, tokens)[-1])

    with self.assertRaises(RunnerError):
      runner.decode(Tensor([[1]], dtype=dtypes.int32), position=mc)

  def test_chunked_prefill_then_jit_decode(self):
    prefill_tokens = [1, 2, 3, 4]
    decode_tokens = [5, 6, 7]
    all_tokens = prefill_tokens + decode_tokens
    expected = [self._stateless_logits(all_tokens[:i+1]) for i in range(len(all_tokens))]

    cache = ContiguousKVCache(self.config, max_context=16, num_slots=1)
    chunk_prefill_logits = None
    for cs in range(0, len(prefill_tokens), 2):
      chunk = prefill_tokens[cs:cs+2]
      chunk_prefill_logits = self.model.forward(
        Tensor([chunk], dtype=dtypes.int32), start_position=cs, kv=cache.kv
      ).realize()
    _logits_close(expected[len(prefill_tokens)-1], chunk_prefill_logits.tolist()[0])

    runner = Qwen3Runner(self.model, cache)
    actual = [expected[len(prefill_tokens)-1]]
    for i, tok in enumerate(decode_tokens):
      pos = len(prefill_tokens) + i
      logits = runner.decode(Tensor([[tok]], dtype=dtypes.int32), position=pos).realize()
      actual.append(logits.tolist()[0])

    for j in range(len(prefill_tokens)-1, len(all_tokens)):
      _logits_close(expected[j], actual[j - (len(prefill_tokens)-1)])

  # ------------------------------------------------------------------
  # Conversation reuse and cancellation cleanup
  # ------------------------------------------------------------------

  def test_repeated_conversation_no_leak(self):
    conv_a = [1, 2, 3, 4]
    conv_b = [5, 6, 7, 8]
    expected_b = [self._stateless_logits(conv_b[:i+1]) for i in range(len(conv_b))]

    cache = ContiguousKVCache(self.config, max_context=16, num_slots=1)
    runner = Qwen3Runner(self.model, cache)
    self._run_jit_conversation(runner, conv_a)
    cache.clear_slot(0)
    actual_b = self._run_jit_conversation(runner, conv_b)

    for e, a in zip(expected_b, actual_b):
      _logits_close(e, a, abs_tol=2e-3)

  def test_cancellation_cleanup(self):
    conv_a = [1, 2, 3, 4, 5]
    conv_b = [6, 7, 8, 9, 10]
    expected_b = [self._stateless_logits(conv_b[:i+1]) for i in range(len(conv_b))]

    cache = ContiguousKVCache(self.config, max_context=16, num_slots=1)
    runner = Qwen3Runner(self.model, cache)
    # Start conversation A, decode only 2 tokens, then cancel
    runner.prefill(Tensor([[conv_a[0]]], dtype=dtypes.int32)).realize()
    for i in range(1, 3):
      runner.decode(Tensor([[conv_a[i]]], dtype=dtypes.int32), position=i).realize()
    cache.clear_slot(0)
    actual_b = self._run_jit_conversation(runner, conv_b)

    for e, a in zip(expected_b, actual_b):
      _logits_close(e, a, abs_tol=2e-3)

  def test_clear_slot_rejects_out_of_range(self):
    cache = ContiguousKVCache(self.config, max_context=8, num_slots=2)
    from infurnace.executor.tinygrad.buffers import KVCacheError
    with self.assertRaises(KVCacheError):
      cache.clear_slot(2)
    with self.assertRaises(KVCacheError):
      cache.clear_slot(-1)

  # ------------------------------------------------------------------
  # Cache replacement
  # ------------------------------------------------------------------

  def test_cache_replacement_same_output(self):
    tokens = [1, 2, 3, 4, 5]

    cache1 = ContiguousKVCache(self.config, max_context=16, num_slots=1)
    model1 = Qwen3Model(self.weights)
    runner1 = Qwen3Runner(model1, cache1)
    logits1 = self._run_jit_conversation(runner1, tokens)

    cache2 = ContiguousKVCache(self.config, max_context=16, num_slots=1)
    model2 = Qwen3Model(self.weights)
    runner2 = Qwen3Runner(model2, cache2)
    logits2 = self._run_jit_conversation(runner2, tokens)

    for l1, l2 in zip(logits1, logits2):
      _logits_close(l1, l2)

  # ------------------------------------------------------------------
  # Multi-slot isolation
  # ------------------------------------------------------------------

  def test_multi_slot_runner_isolation(self):
    tokens_a = [1, 2, 3, 4]
    tokens_b = [5, 6, 7, 8]
    expected_a = [self._stateless_logits(tokens_a[:i+1]) for i in range(len(tokens_a))]
    expected_b = [self._stateless_logits(tokens_b[:i+1]) for i in range(len(tokens_b))]

    cache = ContiguousKVCache(self.config, max_context=16, num_slots=2)
    runner = Qwen3Runner(self.model, cache)

    actual_a = self._run_jit_conversation(runner, tokens_a, slot=0)
    actual_b = self._run_jit_conversation(runner, tokens_b, slot=1)

    for e, a in zip(expected_a, actual_a):
      _logits_close(e, a, abs_tol=2e-3)
    for e, a in zip(expected_b, actual_b):
      _logits_close(e, a, abs_tol=2e-3)
