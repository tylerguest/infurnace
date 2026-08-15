import json
import os
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from infurnace.models import ModelConfigError, qwen3_config_from_gguf
from infurnace.models.manifest import load_manifest, verified_artifact


REPOSITORY_ROOT = Path(__file__).parents[1]
PINNED_MANIFEST = REPOSITORY_ROOT / "models" / "qwen3-0.6b-q8_0.json"
PINNED_INSPECTION = REPOSITORY_ROOT / "models" / "qwen3-0.6b-q8_0.inspection.json"


def valid_metadata():
  return {
    "general.architecture": "qwen3",
    "general.type": "model",
    "general.quantization_version": 2,
    "general.file_type": 7,
    "qwen3.block_count": 28,
    "qwen3.context_length": 40960,
    "qwen3.embedding_length": 1024,
    "qwen3.feed_forward_length": 3072,
    "qwen3.attention.head_count": 16,
    "qwen3.attention.head_count_kv": 8,
    "qwen3.rope.freq_base": 1000000.0,
    "qwen3.attention.layer_norm_rms_epsilon": 9.999999974752427e-07,
    "qwen3.attention.key_length": 128,
    "qwen3.attention.value_length": 128,
    "tokenizer.ggml.tokens": [""] * 151936,
  }


class TestQwen3Config(unittest.TestCase):
  def test_derives_exact_pinned_configuration(self):
    config = qwen3_config_from_gguf(valid_metadata())
    self.assertEqual(
      (config.architecture, config.block_count, config.context_length, config.embedding_length,
       config.feed_forward_length, config.attention_head_count, config.attention_head_count_kv),
      ("qwen3", 28, 40960, 1024, 3072, 16, 8),
    )
    self.assertEqual((config.key_length, config.value_length, config.rope_dimension_count), (128, 128, 128))
    self.assertEqual((config.rope_freq_base, config.rms_norm_epsilon), (1000000.0, 9.999999974752427e-07))
    self.assertEqual((config.vocab_size, config.quantization_version, config.file_type), (151936, 2, 7))
    self.assertEqual(config.quantization, "Q8_0")
    self.assertTrue(config.qk_norm)
    self.assertFalse(config.attention_bias)
    self.assertFalse(config.mlp_bias)
    self.assertTrue(config.tied_embeddings)
    self.assertEqual(config.mlp_type, "swiglu")

  def test_configuration_and_tensor_specs_are_immutable(self):
    config = qwen3_config_from_gguf(valid_metadata())
    with self.assertRaises(FrozenInstanceError): config.block_count = 1
    with self.assertRaises(FrozenInstanceError): config.tensors[0].shape = (1,)
    self.assertIsInstance(config.tensors, tuple)
    self.assertTrue(all(isinstance(spec.shape, tuple) for spec in config.tensors))

  def test_generates_exact_tensor_inventory(self):
    config = qwen3_config_from_gguf(valid_metadata())
    specs = {spec.name: spec for spec in config.tensors}
    self.assertEqual(len(specs), 310)
    self.assertNotIn("output.weight", specs)
    self.assertFalse(any(name.endswith(".bias") for name in specs))
    self.assertEqual(specs["token_embd.weight"].shape, (151936, 1024))
    self.assertEqual(specs["blk.0.attn_q.weight"].shape, (2048, 1024))
    self.assertEqual(specs["blk.27.attn_output.weight"].shape, (1024, 2048))
    self.assertEqual(specs["blk.27.ffn_down.weight"].shape, (1024, 3072))
    self.assertEqual(sum(spec.storage_dtype == "F32" for spec in specs.values()), 113)
    self.assertEqual(sum(spec.storage_dtype == "Q8_0" for spec in specs.values()), 197)

  def test_tensor_specs_match_pinned_inspection(self):
    config = qwen3_config_from_gguf(valid_metadata())
    expected = {spec.name: (list(spec.shape), spec.storage_dtype, spec.logical_dtype) for spec in config.tensors}
    report = json.loads(PINNED_INSPECTION.read_text(encoding="utf-8"))
    actual = {
      tensor["name"]: (tensor["logical_shape"], tensor["ggml_type"]["name"], tensor["logical_dtype"])
      for tensor in report["gguf"]["tensors"]
    }
    self.assertEqual(actual, expected)

  def test_rejects_missing_and_unknown_model_metadata(self):
    missing = valid_metadata()
    del missing["qwen3.block_count"]
    unknown = valid_metadata()
    unknown["qwen3.expert_count"] = 8
    for name, metadata in (("missing", missing), ("unknown", unknown)):
      with self.subTest(name=name), self.assertRaises(ModelConfigError): qwen3_config_from_gguf(metadata)

  def test_rejects_wrong_metadata_types_and_values(self):
    cases = (
      ("boolean count", "qwen3.block_count", True),
      ("architecture", "general.architecture", "llama"),
      ("context", "qwen3.context_length", 32768),
      ("head count", "qwen3.attention.head_count", 8),
      ("KV head count", "qwen3.attention.head_count_kv", 16),
      ("key length", "qwen3.attention.key_length", 64),
      ("value length", "qwen3.attention.value_length", 64),
      ("RoPE type", "qwen3.rope.freq_base", 1000000),
      ("RoPE value", "qwen3.rope.freq_base", float("inf")),
      ("epsilon", "qwen3.attention.layer_norm_rms_epsilon", 1e-5),
      ("file type", "general.file_type", 1),
    )
    for name, key, value in cases:
      metadata = valid_metadata()
      metadata[key] = value
      with self.subTest(name=name), self.assertRaises(ModelConfigError): qwen3_config_from_gguf(metadata)

  def test_rejects_invalid_vocabulary(self):
    cases = ([""] * 10, ["", 1], (), None)
    for tokens in cases:
      metadata = valid_metadata()
      metadata["tokenizer.ggml.tokens"] = tokens
      with self.subTest(tokens_type=type(tokens).__name__), self.assertRaises(ModelConfigError):
        qwen3_config_from_gguf(metadata)

  def test_import_does_not_import_tinygrad(self):
    code = "import sys; import infurnace.models; assert not any(x == 'tinygrad' or x.startswith('tinygrad.') for x in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    self.assertEqual(result.returncode, 0, result.stderr)


@pytest.mark.model
@pytest.mark.slow
class TestPinnedQwen3Config(unittest.TestCase):
  def test_derives_configuration_from_verified_artifact(self):
    if os.environ.get("DEV") != "CPU": self.skipTest("real GGUF configuration requires DEV=CPU")
    artifact_value = os.environ.get("INFURNACE_MODEL_ARTIFACT")
    if artifact_value is None: self.skipTest("set INFURNACE_MODEL_ARTIFACT to the pinned GGUF")

    manifest = load_manifest(PINNED_MANIFEST)
    with verified_artifact(artifact_value, manifest) as artifact:
      from tinygrad import Tensor, dtypes
      from tinygrad.llm.gguf import gguf_load
      gguf_tensor = Tensor.empty(artifact.size_bytes, dtype=dtypes.uint8, device=f"disk:{artifact.path}")
      metadata, tensors = gguf_load(gguf_tensor)
      config = qwen3_config_from_gguf(metadata)

    self.assertEqual(config.block_count, 28)
    self.assertEqual(config.vocab_size, 151936)
    self.assertEqual(len(config.tensors), 310)
    specs = {spec.name: spec for spec in config.tensors}
    self.assertEqual(tensors.keys(), specs.keys())
    for name, tensor in tensors.items():
      self.assertEqual(tuple(tensor.shape), specs[name].shape, name)
      self.assertEqual(tensor.dtype.name, specs[name].logical_dtype, name)
