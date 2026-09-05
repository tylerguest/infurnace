import os
import unittest
from pathlib import Path
import pytest
from tinygrad import Tensor, dtypes

REPOSITORY_ROOT = Path(__file__).parents[1]

def _packed_blocks(ggml_type: int, n: int, size: int) -> Tensor:
  """Raw GGML packed bytes for ``n`` elements of a quant type (one 256-block repeated).

  Wraps the bytes in a padded outer buffer and slices it (like the GGUF
  whole-file buffer -> per-tensor slice), so the packed region is a real uint8
  SHRINK in the dequant graph.
  """
  count = (n // 256) * size
  outer = Tensor([(i * 7) % 256 for i in range(count + 64)], dtype=dtypes.uint8)
  return outer[:count].contiguous()

class TestPackedWeightExtraction(unittest.TestCase):
  def test_rejects_non_quantized_weights(self):
    from infurnace.kernels.quant_gemv import packed_weight
    plain = Tensor.zeros(64, 64)
    self.assertIsNone(packed_weight(plain.cast(dtypes.float16)))
    self.assertIsNone(packed_weight(plain))

  def test_rejects_weak_dtype(self):
    from infurnace.kernels.quant_gemv import packed_weight
    self.assertIsNone(packed_weight(Tensor.zeros(8, 8, dtype=dtypes.float16)))

  def test_extracts_q4_k(self):
    from tinygrad.llm.gguf import ggml_data_to_tensor
    from infurnace.kernels.quant_gemv import packed_weight, Q4_K
    n = 256 * 4
    t = _packed_blocks(Q4_K, n, 144)
    expr = ggml_data_to_tensor(t, n, Q4_K).cast(dtypes.float16)
    packed, ggml_type = packed_weight(expr)
    self.assertEqual(ggml_type, Q4_K)
    self.assertEqual(packed.dtype, dtypes.uint32)
    self.assertEqual(packed.numel(), n // 256 * 144 // 4)

  def test_extracts_q6_k(self):
    from tinygrad.llm.gguf import ggml_data_to_tensor
    from infurnace.kernels.quant_gemv import packed_weight, Q6_K
    n = 256 * 4
    t = _packed_blocks(Q6_K, n, 210)
    expr = ggml_data_to_tensor(t, n, Q6_K).cast(dtypes.float16)
    packed, ggml_type = packed_weight(expr)
    self.assertEqual(ggml_type, Q6_K)
    self.assertEqual(packed.dtype, dtypes.uint8)
    self.assertEqual(packed.numel(), n // 256 * 210)


@pytest.mark.nv
@pytest.mark.model
@pytest.mark.slow
class TestQuantGemvRealModel(unittest.TestCase):
  """Numerical and greedy agreement of the fused Q4_K/Q6_K decode GEMV."""

  @classmethod
  def setUpClass(cls):
    artifact = os.environ.get("INFURNACE_MODEL_ARTIFACT")
    if artifact is None: raise unittest.SkipTest("set INFURNACE_MODEL_ARTIFACT to the pinned GGUF")
    if os.environ.get("DEV") != "NV": raise unittest.SkipTest("requires DEV=NV")
    from infurnace.executor.tinygrad.weights import WeightPolicy, load_qwen3_checkpoint
    from infurnace.executor.tinygrad.model import Qwen3Model
    from infurnace.tokenizer import GGUFTokenizer
    from infurnace.models.manifest import load_manifest
    from tools.gguf_inspection import stable_artifact_path
    manifest = load_manifest(REPOSITORY_ROOT / "models" / "qwen3-4b-q4_k_m.json")
    with stable_artifact_path(Path(artifact)) as stable_path:
      checkpoint = load_qwen3_checkpoint(stable_path, manifest, WeightPolicy.LAZY_FP16)
    cls.checkpoint = checkpoint
    cls.model = Qwen3Model(checkpoint.weights)
    cls.tokenizer = GGUFTokenizer.from_gguf_metadata(checkpoint.metadata)

  def _engine(self):
    from infurnace.engine import Engine
    from infurnace.scheduler.scheduler import Scheduler
    from infurnace.sampler import GreedySampler
    from infurnace.executor.tinygrad.runner import Qwen3Runner
    from infurnace.executor.tinygrad.buffers import ContiguousKVCache
    kv = ContiguousKVCache(config=self.model.config, max_context=512, num_slots=1)
    runner = Qwen3Runner(self.model, kv)
    return Engine(runner, Scheduler(num_slots=1, max_context=512), GreedySampler(), self.tokenizer)

  def test_gemv_greedy_matches_generic_per_role(self):
    from infurnace.kernels.quant_gemv import packed_weight, quant_gemv
    roles = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]
    for role in roles:
      weight = self.model._w[f"blk.0.{role}.weight"]
      out_features, in_features = tuple(weight.shape)
      entry = packed_weight(weight)
      self.assertIsNotNone(entry, f"{role} should be quantized")
      packed, ggml_type = entry
      x = Tensor.randn(1, in_features, dtype=dtypes.float32).realize()
      ref = x.linear(weight.transpose()).realize()
      got = quant_gemv(x, packed, out_features=out_features, in_features=in_features, ggml_type=ggml_type).realize()
      self.assertEqual(int(got.argmax().item()), int(ref.argmax().item()), f"{role} greedy mismatch")

  def test_decode_greedy_matches_generic(self):
    from infurnace.engine.request import Request, SamplingParams
    from tinygrad import Device
    prompt = [257] + [1000 + i for i in range(15)]

    def tokens(disable: bool) -> list[int]:
      if disable: os.environ["INFURNACE_DISABLE_QUANT_GEMV"] = "1"
      else: os.environ.pop("INFURNACE_DISABLE_QUANT_GEMV", None)
      engine = self._engine()
      request = Request("t", prompt, SamplingParams(max_tokens=8), 512)
      engine.add_request(request)
      guard = 0
      while not engine.is_done():
        engine.step(); guard += 1
        self.assertLess(guard, 72, "decode did not terminate")
      Device["NV"].synchronize()
      return list(request.output_token_ids)

    try:
      generic, custom = tokens(disable=True), tokens(disable=False)
    finally:
      os.environ.pop("INFURNACE_DISABLE_QUANT_GEMV", None)
    self.assertEqual(custom, generic, "custom kernel changed greedy decode output")


if __name__ == "__main__":
  unittest.main()