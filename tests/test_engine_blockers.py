import random
import unittest
from types import MappingProxyType
from tinygrad import Tensor
from infurnace.engine import Engine
from infurnace.engine.request import Request, SamplingParams
from infurnace.executor.tinygrad.buffers import ContiguousKVCache
from infurnace.executor.tinygrad.model import Qwen3Model
from infurnace.executor.tinygrad.runner import Qwen3Runner
from infurnace.executor.tinygrad.weights import Qwen3Weights, WeightPolicy
from infurnace.sampler import GreedySampler
from infurnace.scheduler.scheduler import Scheduler
from test_runner import _make_config


def _make_random_weights(config, seed: int = 1234) -> Qwen3Weights:
  """Synthetic weights whose greedy argmax varies with the input and KV
  content, so cross-request token corruption is observable. The constant
  weights used elsewhere in the suite always argmax to the same token and
  cannot detect silent KV or feedback divergence.
  """
  rng = random.Random(seed)
  def rnd(shape):
    return Tensor([rng.random() - 0.5 for _ in range(_prod(shape))]).reshape(shape).contiguous().realize()
  def _prod(shape):
    p = 1
    for s in shape: p *= s
    return p
  tensors: dict[str, Tensor] = {}
  tensors["token_embd.weight"] = rnd((config.vocab_size, config.embedding_length))
  tensors["output_norm.weight"] = Tensor.ones(config.embedding_length)
  for i in range(config.block_count):
    p = f"blk.{i}"
    qw = config.attention_head_count * config.key_length
    kw = config.attention_head_count_kv * config.key_length
    vw = config.attention_head_count_kv * config.value_length
    tensors[f"{p}.attn_norm.weight"] = Tensor.ones(config.embedding_length)
    tensors[f"{p}.attn_q.weight"] = rnd((qw, config.embedding_length))
    tensors[f"{p}.attn_q_norm.weight"] = Tensor.ones(config.key_length)
    tensors[f"{p}.attn_k.weight"] = rnd((kw, config.embedding_length))
    tensors[f"{p}.attn_k_norm.weight"] = Tensor.ones(config.key_length)
    tensors[f"{p}.attn_v.weight"] = rnd((vw, config.embedding_length))
    tensors[f"{p}.attn_output.weight"] = rnd((config.embedding_length, qw))
    tensors[f"{p}.ffn_norm.weight"] = Tensor.ones(config.embedding_length)
    tensors[f"{p}.ffn_gate.weight"] = rnd((config.feed_forward_length, config.embedding_length))
    tensors[f"{p}.ffn_up.weight"] = rnd((config.feed_forward_length, config.embedding_length))
    tensors[f"{p}.ffn_down.weight"] = rnd((config.embedding_length, config.feed_forward_length))
  tensors["output.weight"] = tensors["token_embd.weight"]
  return Qwen3Weights(config, MappingProxyType(tensors), WeightPolicy.LAZY_FP16)


class TestEngineCorrectnessBlockers(unittest.TestCase):
  """Pre-Phase-5C correctness blockers: fused/batched token equality across
  interleaving, slot moves, cancellation idempotency, and the async seam.
  """

  def setUp(self):
    self.config = _make_config()
    self.weights = _make_random_weights(self.config)
    self.max_context = 16

  def _build(self, num_slots: int) -> Engine:
    model = Qwen3Model(self.weights)
    kv = ContiguousKVCache(self.config, max_context=self.max_context, num_slots=num_slots)
    runner = Qwen3Runner(model, kv)
    return Engine(runner, Scheduler(num_slots=num_slots, max_context=self.max_context), GreedySampler(), None)

  def _baseline(self, prompt_token: int, max_tokens: int) -> tuple[int, ...]:
    eng = self._build(1)
    eng.add_request(Request("r1", [prompt_token], SamplingParams(max_tokens=max_tokens), self.max_context))
    while not eng.is_done():
      eng.step()
    return eng.output_tokens("r1")

  def test_interleaved_fused_batched_matches_baselines(self):
    # r1 finishes fast (frees slot 0) while r2 still decodes at slot 1; r3 is
    # then admitted into the hole at slot 0 and compaction moves r2 down.
    # r3's stream crosses batched decode, a slot move, and a fused resume —
    # every request must produce its solo-run tokens exactly.
    base_r2 = self._baseline(2, 6)
    base_r3 = self._baseline(3, 6)

    eng = self._build(2)
    eng.add_request(Request("r1", [1], SamplingParams(max_tokens=1), self.max_context))
    eng.add_request(Request("r2", [2], SamplingParams(max_tokens=6), self.max_context))
    while not eng.is_done():
      eng.step()
      if eng.scheduler.get_request("r1").state.name == "FINISHED":
        break
    eng.add_request(Request("r3", [3], SamplingParams(max_tokens=6), self.max_context))
    while not eng.is_done():
      eng.step()

    self.assertEqual(eng.output_tokens("r2"), base_r2)
    self.assertEqual(eng.output_tokens("r3"), base_r3)

  def test_fused_resume_after_batched_matches_baseline(self):
    # A second request prefills and decodes beside the first (batched), then
    # finishes; the first request's remaining decode resumes on the fused path
    # and must not read the second request's prefill seed.
    base = self._baseline(1, 8)

    eng = self._build(2)
    eng.add_request(Request("r1", [1], SamplingParams(max_tokens=8), self.max_context))
    eng.add_request(Request("r2", [2], SamplingParams(max_tokens=2), self.max_context))
    while not eng.is_done():
      eng.step()

    self.assertEqual(eng.output_tokens("r1"), base)

  def test_double_cancel_does_not_clear_reowned_slot(self):
    # Engine.cancel twice for the same request: the first frees the slot, the
    # second must be a no-op — the stale cache_slot is nulled on cancel, so a
    # slot re-owned by a live request is never clear_slot'd.
    base = self._baseline(3, 6)

    eng = self._build(2)
    eng.add_request(Request("r1", [1], SamplingParams(max_tokens=6), self.max_context))
    eng.add_request(Request("r2", [2], SamplingParams(max_tokens=6), self.max_context))
    eng.step()  # r1 prefills (slot 0)
    eng.step()  # r1 decodes (fused), r2 prefills (slot 1)
    eng.cancel("r1")  # frees slot 0, nulls r1.cache_slot
    eng.add_request(Request("r3", [3], SamplingParams(max_tokens=6), self.max_context))  # takes slot 0
    eng.cancel("r1")  # must not clear r3's slot
    while not eng.is_done():
      eng.step()

    self.assertEqual(eng.output_tokens("r3"), base)

  def test_prepare_step_applies_compaction(self):
    # The async seam (prepare/execute/commit) must restore the slot prefix
    # during plan preparation: a caller driving the seam directly must never
    # see a decode plan whose slots are not 0..B-1.
    eng = self._build(2)
    eng.add_request(Request("r1", [1], SamplingParams(max_tokens=1), self.max_context))
    eng.add_request(Request("r2", [2], SamplingParams(max_tokens=6), self.max_context))
    while eng.scheduler.get_request("r2").state.name != "DECODING":
      eng.step()  # r1 finishes (frees slot 0); r2 prefills (slot 1)
    eng.add_request(Request("r3", [3], SamplingParams(max_tokens=6), self.max_context))  # hole at 0

    from infurnace.scheduler.plan import DecodePlan
    plans = eng.prepare_step()
    decodes = [p for p in plans if isinstance(p, DecodePlan)]
    assert len(decodes) == 1
    self.assertEqual(decodes[0].slots, (0,))
    self.assertEqual(eng.scheduler.get_request("r2").cache_slot, 0)
    self.assertEqual(eng.scheduler.get_request("r3").cache_slot, 1)

    raw = eng.execute_step(plans)
    eng.commit_step(raw)
    while not eng.is_done():
      eng.step()
    self.assertEqual(eng.output_tokens("r2"), self._baseline(2, 6))
    self.assertEqual(eng.output_tokens("r3"), self._baseline(3, 6))


if __name__ == "__main__":
  unittest.main()