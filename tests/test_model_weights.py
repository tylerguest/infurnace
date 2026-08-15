import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from infurnace.executor.tinygrad.weights import (
  WeightMappingError, WeightPolicy, load_qwen3_weights, map_qwen3_weights,
)
from infurnace.models import qwen3_config_from_gguf
from infurnace.models.manifest import CheckpointManifest, load_manifest, verified_artifact
from test_model_config import valid_metadata


REPOSITORY_ROOT = Path(__file__).parents[1]
PINNED_MANIFEST = REPOSITORY_ROOT / "models" / "qwen3-0.6b-q8_0.json"


class FakeDType:
  def __init__(self, name): self.name = name


class FakeTensor:
  def __init__(self, shape, dtype="float", cast_count=None, contiguous=False):
    self.shape, self.dtype = shape, FakeDType(dtype)
    self.cast_count, self.is_contiguous = cast_count if cast_count is not None else [0], contiguous

  def cast(self, dtype):
    self.cast_count[0] += 1
    return FakeTensor(self.shape, dtype.name, self.cast_count, self.is_contiguous)

  def contiguous(self): return FakeTensor(self.shape, self.dtype.name, self.cast_count, True)


def valid_tensors():
  specs = qwen3_config_from_gguf(valid_metadata()).tensors
  return {spec.name: FakeTensor(spec.shape, spec.logical_dtype) for spec in specs}


def manifest_for(content: bytes) -> CheckpointManifest:
  import hashlib
  return CheckpointManifest(1, "test", "owner/model", "a" * 40, "https://example.com/model.gguf", "model.gguf",
                            "GGUF", "Q8_0", len(content), hashlib.sha256(content).hexdigest(),
                            "Apache-2.0", "https://example.com/LICENSE")


class TestVerifiedArtifact(unittest.TestCase):
  def test_holds_private_verified_snapshot_after_source_mutation(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "model.gguf"
      path.write_bytes(b"verified")
      with verified_artifact(path, manifest_for(b"verified")) as artifact:
        path.write_bytes(b"mutated!")
        self.assertEqual(artifact.path.read_bytes(), b"verified")
      self.assertFalse(artifact.path.exists())
      self.assertEqual(path.read_bytes(), b"mutated!")

  def test_accepts_read_only_source(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "model.gguf"
      path.write_bytes(b"verified")
      path.chmod(0o444)
      with verified_artifact(path, manifest_for(b"verified")) as artifact:
        self.assertEqual(artifact.path.read_bytes(), b"verified")


class TestQwen3WeightMapping(unittest.TestCase):
  def test_maps_lazy_weights_and_ties_output(self):
    tensors = valid_tensors()
    weights = map_qwen3_weights(valid_metadata(), tensors, WeightPolicy.LAZY_FP16)
    self.assertEqual(weights.policy, WeightPolicy.LAZY_FP16)
    self.assertEqual(len(weights.tensors), 311)
    self.assertIs(weights.tensors["output.weight"], weights.tensors["token_embd.weight"])
    self.assertTrue(all(tensor.dtype.name == "half" for tensor in weights.tensors.values()))
    self.assertTrue(all(not tensor.is_contiguous for tensor in weights.tensors.values()))
    with self.assertRaises(TypeError): weights.tensors["extra"] = FakeTensor((1,))

  def test_realizes_contiguous_weights(self):
    tensors = valid_tensors()
    with patch("infurnace.executor.tinygrad.weights.Tensor.realize") as realize:
      weights = map_qwen3_weights(valid_metadata(), tensors)
    self.assertEqual(weights.policy, WeightPolicy.REALIZED_FP16)
    self.assertEqual(len(realize.call_args.args), 310)
    self.assertTrue(all(tensor.is_contiguous for tensor in weights.tensors.values()))

  def test_rejects_invalid_mapping_before_casting(self):
    cases = {}
    missing = valid_tensors()
    missing.pop("blk.0.attn_q.weight")
    cases["missing tensors"] = missing
    unexpected = valid_tensors()
    unexpected["output.weight"] = FakeTensor((151936, 1024))
    cases["unexpected tensors"] = unexpected
    wrong_shape = valid_tensors()
    wrong_shape["blk.0.attn_q.weight"] = FakeTensor((1, 1))
    cases["shape mismatch"] = wrong_shape
    wrong_dtype = valid_tensors()
    wrong_dtype["blk.0.attn_q.weight"] = FakeTensor((2048, 1024), "half")
    cases["dtype mismatch"] = wrong_dtype
    for message, tensors in cases.items():
      with self.subTest(message=message), self.assertRaisesRegex(WeightMappingError, message):
        map_qwen3_weights(valid_metadata(), tensors, WeightPolicy.LAZY_FP16)
      self.assertEqual(sum(tensor.cast_count[0] for tensor in tensors.values()), 0)

  def test_rejects_unknown_policy_before_casting(self):
    tensors = valid_tensors()
    with self.assertRaisesRegex(WeightMappingError, "unsupported weight policy"):
      map_qwen3_weights(valid_metadata(), tensors, "eager")
    self.assertEqual(sum(tensor.cast_count[0] for tensor in tensors.values()), 0)

  def test_loader_rejects_unpinned_manifest_before_artifact_access(self):
    with patch("infurnace.executor.tinygrad.weights.verified_artifact") as verify:
      with self.assertRaisesRegex(WeightMappingError, "not the pinned"):
        load_qwen3_weights("missing.gguf", manifest_for(b"other"))
    verify.assert_not_called()


@pytest.mark.model
@pytest.mark.slow
class TestPinnedQwen3Weights(unittest.TestCase):
  def test_loads_verified_lazy_weights(self):
    if os.environ.get("DEV") != "CPU": self.skipTest("real GGUF weight loading requires DEV=CPU")
    artifact = os.environ.get("INFURNACE_MODEL_ARTIFACT")
    if artifact is None: self.skipTest("set INFURNACE_MODEL_ARTIFACT to the pinned GGUF")
    weights = load_qwen3_weights(artifact, load_manifest(PINNED_MANIFEST), WeightPolicy.LAZY_FP16)
    self.assertEqual(len(weights.tensors), 311)
    self.assertIs(weights.tensors["output.weight"], weights.tensors["token_embd.weight"])
    self.assertTrue(all(tensor.dtype.name == "half" for tensor in weights.tensors.values()))
    self.assertTrue(all(not tensor.uop.is_realized for tensor in weights.tensors.values()))
    self.assertTrue(math.isfinite(weights.tensors["output_norm.weight"][0].item()))

  @pytest.mark.nv
  def test_loads_verified_realized_weights(self):
    if os.environ.get("DEV") != "NV": self.skipTest("realized GGUF weight loading requires DEV=NV")
    artifact = os.environ.get("INFURNACE_MODEL_ARTIFACT")
    if artifact is None: self.skipTest("set INFURNACE_MODEL_ARTIFACT to the pinned GGUF")
    weights = load_qwen3_weights(artifact, load_manifest(PINNED_MANIFEST))
    from tinygrad import Device
    Device["NV"].synchronize()
    self.assertEqual(len(weights.tensors), 311)
    self.assertIs(weights.tensors["output.weight"], weights.tensors["token_embd.weight"])
    self.assertTrue(all(tensor.dtype.name == "half" and tensor.uop.is_realized for tensor in weights.tensors.values()))
