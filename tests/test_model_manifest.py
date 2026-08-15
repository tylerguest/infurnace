import copy
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from http.client import HTTPException
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError
from infurnace.models.manifest import (ArtifactError, CheckpointManifest, ManifestError, acquire_artifact, load_manifest, verify_artifact,)

REPOSITORY_ROOT = Path(__file__).parents[1]
PINNED_MANIFEST = REPOSITORY_ROOT / "models" / "qwen3-0.6b-q8_0.json"

def valid_manifest_data():
  return {
    "schema_version": 1,
    "id": "test-model-q8_0",
    "source": {
      "repository": "owner/model",
      "revision": "a" * 40,
      "url": f"https://huggingface.co/owner/model/resolve/{'a' * 40}/model-Q8_0.gguf",
    },
    "artifact": {
      "filename": "model-Q8_0.gguf",
      "format": "GGUF",
      "quantization": "Q8_0",
      "size_bytes": 3,
      "sha256": hashlib.sha256(b"abc").hexdigest(),
    },
    "license": {"spdx": "Apache-2.0", "url": "https://example.com/LICENSE"},
  }

def manifest_for(content: bytes) -> CheckpointManifest:
  return CheckpointManifest(1, "test-model", "owner/model", "a" * 40,
                            f"https://huggingface.co/owner/model/resolve/{'a' * 40}/model.gguf",
                            "model.gguf", "GGUF", "Q8_0", len(content), hashlib.sha256(content).hexdigest(),
                            "Apache-2.0", "https://example.com/LICENSE")

class TestCheckpointManifest(unittest.TestCase):
  def write_manifest(self, directory: str, data) -> Path:
    path = Path(directory) / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path

  def test_pinned_manifest_identity(self):
    manifest = load_manifest(PINNED_MANIFEST)
    self.assertEqual(manifest.revision, "23749fefcc72300e3a2ad315e1317431b06b590a")
    self.assertEqual(manifest.size_bytes, 639446688)
    self.assertEqual(manifest.sha256, "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031")

  def test_valid_manifest(self):
    with tempfile.TemporaryDirectory() as directory:
      manifest = load_manifest(self.write_manifest(directory, valid_manifest_data()))
    self.assertEqual(manifest.id, "test-model-q8_0")
    self.assertEqual(manifest.filename, "model-Q8_0.gguf")

  def test_rejects_missing_and_unknown_fields(self):
    cases = []
    missing = valid_manifest_data()
    del missing["artifact"]["sha256"]
    cases.append(missing)
    unknown = valid_manifest_data()
    unknown["source"]["branch"] = "main"
    cases.append(unknown)

    with tempfile.TemporaryDirectory() as directory:
      for index, data in enumerate(cases):
        with self.subTest(index=index), self.assertRaises(ManifestError):
          load_manifest(self.write_manifest(directory, data))

  def test_rejects_duplicate_fields(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "manifest.json"
      path.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")
      with self.assertRaisesRegex(ManifestError, "duplicate field"):
        load_manifest(path)

  def test_rejects_invalid_identity_fields(self):
    mutations = [
      ("schema", lambda data: data.update(schema_version=2)),
      ("boolean size", lambda data: data["artifact"].update(size_bytes=True)),
      ("negative size", lambda data: data["artifact"].update(size_bytes=-1)),
      ("revision", lambda data: data["source"].update(revision="main")),
      ("sha256", lambda data: data["artifact"].update(sha256="ABC")),
      ("filename", lambda data: data["artifact"].update(filename="../model.gguf")),
      ("format", lambda data: data["artifact"].update(format="safetensors")),
      ("quantization", lambda data: data["artifact"].update(quantization="q8 0")),
      ("insecure URL", lambda data: data["source"].update(url="http://example.com/model.gguf")),
      ("mutable URL", lambda data: data["source"].update(url="https://huggingface.co/owner/model/resolve/main/model-Q8_0.gguf")),
      ("malformed license URL", lambda data: data["license"].update(url="https:///LICENSE")),
      ("malformed authority", lambda data: data["license"].update(url="https://[")),
      ("invalid port", lambda data: data["license"].update(url="https://example.com:bad/LICENSE")),
    ]
    with tempfile.TemporaryDirectory() as directory:
      for name, mutate in mutations:
        data = copy.deepcopy(valid_manifest_data())
        mutate(data)
        with self.subTest(name=name), self.assertRaises(ManifestError):
          load_manifest(self.write_manifest(directory, data))

  def test_import_does_not_import_tinygrad(self):
    code = "import sys; import infurnace.models.manifest; assert 'tinygrad' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    self.assertEqual(result.returncode, 0, result.stderr)

class TestArtifactVerification(unittest.TestCase):
  def test_accepts_exact_artifact(self):
    content = b"verified artifact"
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "model.gguf"
      path.write_bytes(content)
      verify_artifact(path, manifest_for(content))

  def test_rejects_wrong_artifact(self):
    expected = b"abc"
    cases = {"truncated": b"ab", "oversized": b"abcd", "wrong hash": b"abd", "empty": b""}
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "model.gguf"
      for name, content in cases.items():
        path.write_bytes(content)
        with self.subTest(name=name), self.assertRaises(ArtifactError):
          verify_artifact(path, manifest_for(expected))

class TestArtifactAcquisition(unittest.TestCase):
  def test_downloads_and_atomically_publishes_verified_artifact(self):
    content, manifest = b"abc", manifest_for(b"abc")
    with tempfile.TemporaryDirectory() as directory:
      destination = Path(directory) / "nested" / "model.gguf"
      with patch("infurnace.models.manifest.urlopen", return_value=io.BytesIO(content)) as open_url:
        result = acquire_artifact(manifest, destination)
      self.assertEqual(result, destination)
      self.assertEqual(destination.read_bytes(), content)
      self.assertEqual(list(destination.parent.glob("*.part")), [])
      open_url.assert_called_once()

  def test_reuses_existing_verified_artifact_without_network(self):
    content, manifest = b"abc", manifest_for(b"abc")
    with tempfile.TemporaryDirectory() as directory:
      destination = Path(directory) / "model.gguf"
      destination.write_bytes(content)
      with patch("infurnace.models.manifest.urlopen", side_effect=AssertionError("network used")) as open_url:
        self.assertEqual(acquire_artifact(manifest, destination), destination)
      open_url.assert_not_called()

  def test_refuses_to_overwrite_existing_mismatched_artifact(self):
    manifest = manifest_for(b"abc")
    with tempfile.TemporaryDirectory() as directory:
      destination = Path(directory) / "model.gguf"
      destination.write_bytes(b"bad")
      with patch("infurnace.models.manifest.urlopen", side_effect=AssertionError("network used")) as open_url:
        with self.assertRaises(ArtifactError): acquire_artifact(manifest, destination)
      self.assertEqual(destination.read_bytes(), b"bad")
      open_url.assert_not_called()

  def test_rejects_invalid_downloads_and_cleans_temporary_files(self):
    expected = b"abc"
    cases = {"truncated": b"ab", "oversized": b"abcd", "wrong hash": b"abd"}
    with tempfile.TemporaryDirectory() as directory:
      for name, content in cases.items():
        destination = Path(directory) / name / "model.gguf"
        with self.subTest(name=name), patch("infurnace.models.manifest.urlopen", return_value=io.BytesIO(content)):
          with self.assertRaises(ArtifactError): acquire_artifact(manifest_for(expected), destination)
        self.assertFalse(destination.exists())
        self.assertEqual(list(destination.parent.glob("*.part")), [])

  def test_cleans_temporary_file_after_network_failure(self):
    with tempfile.TemporaryDirectory() as directory:
      destination = Path(directory) / "model.gguf"
      with patch("infurnace.models.manifest.urlopen", side_effect=URLError("offline")):
        with self.assertRaisesRegex(ArtifactError, "cannot acquire artifact"):
          acquire_artifact(manifest_for(b"abc"), destination)
      self.assertFalse(destination.exists())
      self.assertEqual(list(destination.parent.glob("*.part")), [])

  def test_cleans_temporary_file_after_response_read_failure(self):
    class FailingResponse(io.BytesIO):
      def read(self, size=-1):
        raise HTTPException("connection lost")

    with tempfile.TemporaryDirectory() as directory:
      destination = Path(directory) / "model.gguf"
      with patch("infurnace.models.manifest.urlopen", return_value=FailingResponse(b"abc")):
        with self.assertRaisesRegex(ArtifactError, "cannot acquire artifact"):
          acquire_artifact(manifest_for(b"abc"), destination)
      self.assertFalse(destination.exists())
      self.assertEqual(list(destination.parent.glob("*.part")), [])

  def test_rejects_invalid_timeout_before_network(self):
    with tempfile.TemporaryDirectory() as directory:
      for timeout in (0, -1, float("nan"), float("inf")):
        destination = Path(directory) / "model.gguf"
        with self.subTest(timeout=timeout), patch("infurnace.models.manifest.urlopen", side_effect=AssertionError("network used")) as open_url:
          with self.assertRaisesRegex(ArtifactError, "timeout must be finite and positive"):
            acquire_artifact(manifest_for(b"abc"), destination, timeout=timeout)
          open_url.assert_not_called()

  def test_refuses_dangling_destination_symlink(self):
    with tempfile.TemporaryDirectory() as directory:
      destination = Path(directory) / "model.gguf"
      destination.symlink_to(Path(directory) / "missing.gguf")
      with patch("infurnace.models.manifest.urlopen", side_effect=AssertionError("network used")) as open_url:
        with self.assertRaises(ArtifactError): acquire_artifact(manifest_for(b"abc"), destination)
      self.assertTrue(destination.is_symlink())
      open_url.assert_not_called()

  def test_concurrent_destination_is_not_overwritten(self):
    with tempfile.TemporaryDirectory() as directory:
      destination = Path(directory) / "model.gguf"

      def competing_publish(source, target):
        Path(target).write_bytes(b"competing artifact")
        raise FileExistsError

      with patch("infurnace.models.manifest.urlopen", return_value=io.BytesIO(b"abc")), \
           patch("infurnace.models.manifest.os.link", side_effect=competing_publish):
        with self.assertRaises(ArtifactError): acquire_artifact(manifest_for(b"abc"), destination)
      self.assertEqual(destination.read_bytes(), b"competing artifact")
      self.assertEqual(list(Path(directory).glob("*.part")), [])