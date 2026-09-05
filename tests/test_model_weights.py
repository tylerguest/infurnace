import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pytest
from infurnace.executor.tinygrad.weights import (
  WeightMappingError, WeightPolicy, load_qwen3_weights,
)
from infurnace.models.manifest import CheckpointManifest, load_manifest, verified_artifact

REPOSITORY_ROOT = Path(__file__).parents[1]
PINNED_MANIFEST = REPOSITORY_ROOT / "models" / "qwen3-4b-q4_k_m.json"

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