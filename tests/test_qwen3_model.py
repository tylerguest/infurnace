import math
import os
import unittest
from pathlib import Path
from types import MappingProxyType

import pytest
from tinygrad import Tensor, dtypes

from infurnace.executor.tinygrad.model import Qwen3Model, Qwen3ModelError, _apply_rope, _linear, _precompute_rope, _rms_norm
from infurnace.executor.tinygrad.weights import Qwen3Weights, WeightPolicy, load_qwen3_weights
from infurnace.models.config import Qwen3Config, TensorSpec
from infurnace.models.manifest import load_manifest


REPOSITORY_ROOT = Path(__file__).parents[1]
PINNED_MANIFEST = REPOSITORY_ROOT / "models" / "qwen3-0.6b-q8_0.json"


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
    architecture="qwen3", block_count=2, context_length=8, embedding_length=8,
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


def _make_weights(config: Qwen3Config, seed: int = 42) -> Qwen3Weights:
  tensors = {}
  tensors["token_embd.weight"] = Tensor.zeros(config.vocab_size, config.embedding_length)
  tensors["output_norm.weight"] = Tensor.ones(config.embedding_length)
  for i in range(config.block_count):
    p = f"blk.{i}"
    qw = config.attention_head_count * config.key_length
    kw = config.attention_head_count_kv * config.key_length
    vw = config.attention_head_count_kv * config.value_length
    tensors[f"{p}.attn_norm.weight"] = Tensor.ones(config.embedding_length)
    tensors[f"{p}.attn_q.weight"] = Tensor.zeros(qw, config.embedding_length)
    tensors[f"{p}.attn_q_norm.weight"] = Tensor.ones(config.key_length)
    tensors[f"{p}.attn_k.weight"] = Tensor.zeros(kw, config.embedding_length)
    tensors[f"{p}.attn_k_norm.weight"] = Tensor.ones(config.key_length)
    tensors[f"{p}.attn_v.weight"] = Tensor.zeros(vw, config.embedding_length)
    tensors[f"{p}.attn_output.weight"] = Tensor.zeros(config.embedding_length, config.attention_head_count * config.value_length)
    tensors[f"{p}.ffn_norm.weight"] = Tensor.ones(config.embedding_length)
    tensors[f"{p}.ffn_gate.weight"] = Tensor.zeros(config.feed_forward_length, config.embedding_length)
    tensors[f"{p}.ffn_up.weight"] = Tensor.zeros(config.feed_forward_length, config.embedding_length)
    tensors[f"{p}.ffn_down.weight"] = Tensor.zeros(config.embedding_length, config.feed_forward_length)
  tensors["output.weight"] = tensors["token_embd.weight"]
  return Qwen3Weights(config, MappingProxyType(tensors), WeightPolicy.LAZY_FP16)


class TestRMSNorm(unittest.TestCase):
  def test_normalizes_over_last_dimension(self):
    x = Tensor([[1.0, 2.0, 3.0, 4.0]])
    weight = Tensor.ones(4)
    result = _rms_norm(x, weight, 1e-6)
    rms = math.sqrt((1 + 4 + 9 + 16) / 4 + 1e-6)
    expected = [1.0 / rms, 2.0 / rms, 3.0 / rms, 4.0 / rms]
    self.assertEqual(result.shape, (1, 4))
    for got, want in zip(result.realize().tolist()[0], expected):
      self.assertAlmostEqual(got, want, places=5)

  def test_casts_back_to_input_dtype(self):
    x = Tensor([[1.0, 2.0]], dtype=dtypes.float16)
    result = _rms_norm(x, Tensor.ones(2, dtype=dtypes.float16), 1e-6)
    self.assertEqual(result.dtype, dtypes.float16)


class TestLinear(unittest.TestCase):
  def test_gguf_out_in_orientation(self):
    x = Tensor([[1.0, 2.0, 3.0]])
    weight = Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = _linear(x, weight)
    self.assertEqual(result.shape, (1, 2))
    self.assertEqual(result.realize().tolist(), [[1.0, 2.0]])


class TestRoPE(unittest.TestCase):
  def test_position_zero_is_identity(self):
    rope = _precompute_rope(4, 8, 10000.0, "CPU")
    x = Tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    result = _apply_rope(x, rope[:1])
    self.assertEqual(result.realize().tolist(), x.realize().tolist())

  def test_half_split_convention(self):
    rope = _precompute_rope(4, 8, 10000.0, "CPU")
    # x = [0, 0, 1, 0]: x1=[0,0], x2=[1,0] in half-split
    # half-split: out = [-sin(f0), 0, cos(f0), 0]
    # adjacent-pair would give [0, 0, cos(f1), sin(f1)] — clearly different
    x = Tensor([[[[0.0, 0.0, 1.0, 0.0]]]])
    result = _apply_rope(x, rope[1:2]).realize()
    f0 = 1.0 * (10000.0 ** (0.0 / 4.0))  # = 1.0
    cos_f0, sin_f0 = math.cos(f0), math.sin(f0)
    r = result.tolist()[0][0][0]
    self.assertAlmostEqual(r[0], -sin_f0, places=5)
    self.assertAlmostEqual(r[1], 0.0, places=5)
    self.assertAlmostEqual(r[2], cos_f0, places=5)
    self.assertAlmostEqual(r[3], 0.0, places=5)


class TestQwen3Model(unittest.TestCase):
  def setUp(self):
    self.config = _make_config()
    self.weights = _make_weights(self.config)
    self.model = Qwen3Model(self.weights)

  def test_output_shape_and_dtype(self):
    ids = Tensor([[0, 1]], dtype=dtypes.int32)
    logits = self.model(ids)
    self.assertEqual(logits.shape, (1, 16))
    self.assertEqual(logits.dtype, dtypes.float32)

  def test_output_shape_batched(self):
    ids = Tensor([[0, 1], [2, 3]], dtype=dtypes.int32)
    logits = self.model(ids)
    self.assertEqual(logits.shape, (2, 16))

  def test_statelessness(self):
    ids_a = Tensor([[0, 1]], dtype=dtypes.int32)
    ids_b = Tensor([[2, 3]], dtype=dtypes.int32)
    logits_a1 = self.model(ids_a).realize()
    self.model(ids_b).realize()
    logits_a2 = self.model(ids_a).realize()
    self.assertEqual(logits_a1.tolist(), logits_a2.tolist())

  def test_deterministic_repeat(self):
    ids = Tensor([[0, 1, 2]], dtype=dtypes.int32)
    r1 = self.model(ids).realize()
    r2 = self.model(ids).realize()
    self.assertEqual(r1.tolist(), r2.tolist())

  def test_finite_output(self):
    ids = Tensor([[0, 1, 2, 3]], dtype=dtypes.int32)
    logits = self.model(ids).realize()
    self.assertTrue(all(math.isfinite(v) for row in logits.tolist() for v in row))

  def test_input_validation(self):
    with self.assertRaises(Qwen3ModelError): self.model(Tensor([0], dtype=dtypes.int32))
    with self.assertRaises(Qwen3ModelError): self.model(Tensor.zeros(1, 2, 3, dtype=dtypes.int32))
    with self.assertRaises(Qwen3ModelError): self.model(Tensor([[0.0]], dtype=dtypes.float32))
    with self.assertRaises(Qwen3ModelError): self.model(Tensor.zeros(1, 0, dtype=dtypes.int32))
    with self.assertRaises(Qwen3ModelError): self.model(Tensor.zeros(1, 9, dtype=dtypes.int32))

  def test_rejects_non_weights(self):
    with self.assertRaises(Qwen3ModelError): Qwen3Model("not weights")

  def test_rejects_untied_output(self):
    config = _make_config()
    tensors = dict(_make_weights(config).tensors)
    tensors["output.weight"] = Tensor.zeros(16, 8)
    weights = Qwen3Weights(config, MappingProxyType(tensors), WeightPolicy.LAZY_FP16)
    with self.assertRaisesRegex(Qwen3ModelError, "same object"): Qwen3Model(weights)

  def test_rejects_missing_weight(self):
    config = _make_config()
    tensors = dict(_make_weights(config).tensors)
    del tensors["blk.0.attn_q.weight"]
    weights = Qwen3Weights(config, MappingProxyType(tensors), WeightPolicy.LAZY_FP16)
    with self.assertRaisesRegex(Qwen3ModelError, "missing weight tensors"): Qwen3Model(weights)

  def test_rejects_unexpected_weight(self):
    config = _make_config()
    tensors = dict(_make_weights(config).tensors)
    tensors["extra.weight"] = Tensor.zeros(1)
    weights = Qwen3Weights(config, MappingProxyType(tensors), WeightPolicy.LAZY_FP16)
    with self.assertRaisesRegex(Qwen3ModelError, "unexpected weight tensors"): Qwen3Model(weights)


@pytest.mark.model
@pytest.mark.slow
class TestPinnedQwen3Model(unittest.TestCase):
  def test_cpu_forward_one_token(self):
    if os.environ.get("DEV") != "CPU": self.skipTest("requires DEV=CPU")
    artifact = os.environ.get("INFURNACE_MODEL_ARTIFACT")
    if artifact is None: self.skipTest("set INFURNACE_MODEL_ARTIFACT")
    weights = load_qwen3_weights(artifact, load_manifest(PINNED_MANIFEST), WeightPolicy.LAZY_FP16)
    model = Qwen3Model(weights)
    ids = Tensor([[257]], dtype=dtypes.int32)
    logits = model(ids).realize()
    self.assertEqual(logits.shape, (1, 151936))
    self.assertEqual(logits.dtype, dtypes.float32)
    self.assertTrue(all(math.isfinite(v) for v in logits.tolist()[0]))

  @pytest.mark.nv
  def test_nv_forward_lazy(self):
    if os.environ.get("DEV") != "NV": self.skipTest("requires DEV=NV")
    artifact = os.environ.get("INFURNACE_MODEL_ARTIFACT")
    if artifact is None: self.skipTest("set INFURNACE_MODEL_ARTIFACT")
    weights = load_qwen3_weights(artifact, load_manifest(PINNED_MANIFEST), WeightPolicy.LAZY_FP16)
    model = Qwen3Model(weights)
    ids = Tensor([[257] + [1000 + i for i in range(15)]], dtype=dtypes.int32)
    logits = model(ids).realize()
    from tinygrad import Device
    Device["NV"].synchronize()
    self.assertEqual(logits.shape, (1, 151936))
    self.assertEqual(logits.argmax().item(), 657)

  @pytest.mark.nv
  def test_nv_forward_realized(self):
    if os.environ.get("DEV") != "NV": self.skipTest("requires DEV=NV")
    artifact = os.environ.get("INFURNACE_MODEL_ARTIFACT")
    if artifact is None: self.skipTest("set INFURNACE_MODEL_ARTIFACT")
    weights = load_qwen3_weights(artifact, load_manifest(PINNED_MANIFEST), WeightPolicy.REALIZED_FP16)
    model = Qwen3Model(weights)
    ids = Tensor([[257] + [1000 + i for i in range(15)]], dtype=dtypes.int32)
    logits = model(ids).realize()
    from tinygrad import Device
    Device["NV"].synchronize()
    self.assertEqual(logits.shape, (1, 151936))
    self.assertEqual(logits.argmax().item(), 657)
