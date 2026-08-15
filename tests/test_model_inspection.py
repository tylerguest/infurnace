import hashlib
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pytest
from infurnace.models.manifest import load_manifest, verify_artifact
from tools.gguf_inspection import (GGUFInspectionError, build_report, crosscheck_tinygrad, scan_gguf, serialize_report, stable_artifact_path,)
from tools.inspect_artifact import _write_atomic

REPOSITORY_ROOT = Path(__file__).parents[1]
PINNED_MANIFEST = REPOSITORY_ROOT / "models" / "qwen3-0.6b-q8_0.json"
PINNED_INSPECTION = REPOSITORY_ROOT / "models" / "qwen3-0.6b-q8_0.inspection.json"

def gguf_string(value: str) -> bytes:
  encoded = value.encode("utf-8")
  return struct.pack("<Q", len(encoded)) + encoded

def metadata_value(type_id: int, value) -> bytes:
  formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
  if type_id in formats: return struct.pack("<" + formats[type_id], value)
  if type_id == 8: return gguf_string(value)
  element_type, values = value
  return struct.pack("<IQ", element_type, len(values)) + b"".join(metadata_value(element_type, item) for item in values)

def build_gguf(metadata, tensors, data: bytes | None = None) -> bytes:
  descriptors = b""
  for key, type_id, value in metadata:
    descriptors += gguf_string(key) + struct.pack("<I", type_id) + metadata_value(type_id, value)
  for name, dimensions, ggml_type, offset in tensors:
    descriptors += gguf_string(name) + struct.pack("<I", len(dimensions))
    descriptors += b"".join(struct.pack("<Q", dimension) for dimension in dimensions)
    descriptors += struct.pack("<IQ", ggml_type, offset)
  header = struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(metadata))
  alignment = next((value for key, type_id, value in metadata if key == "general.alignment" and type_id == 4), 32)
  prefix = header + descriptors
  padding = b"\x00" * ((-len(prefix)) % alignment)
  if data is None:
    final_size = max((offset + (24 if ggml_type == 0 else 34) for _, _, ggml_type, offset in tensors), default=0)
    data = b"\x00" * final_size
  return prefix + padding + data

class FakeDType:
  name = "float"

class FakeTensor:
  def __init__(self, shape):
    self.shape, self.dtype = shape, FakeDType()

class TestGGUFInspection(unittest.TestCase):
  def write_gguf(self, directory: str, content: bytes) -> Path:
    path = Path(directory) / "model.gguf"
    path.write_bytes(content)
    return path

  def test_scans_metadata_and_tensor_storage(self):
    tokens = [f"token-{index}" for index in range(300)]
    metadata = [
      ("general.alignment", 4, 32),
      ("general.name", 8, "test"),
      ("tokenizer.ggml.tokens", 9, (8, tokens)),
    ]
    tensors = [("matrix", (2, 3), 0, 0), ("quantized", (32,), 8, 32)]
    with tempfile.TemporaryDirectory() as directory:
      info = scan_gguf(self.write_gguf(directory, build_gguf(metadata, tensors)))

    self.assertEqual(info.version, 3)
    self.assertEqual(info.alignment, 32)
    self.assertEqual(len(info.metadata), 3)
    self.assertEqual(info.metadata[2].element_type_name, "STRING")
    self.assertEqual(info.tensors[0].logical_shape, (3, 2))
    self.assertEqual(info.tensors[0].serialized_nbytes, 24)
    self.assertEqual(info.tensors[1].ggml_type_name, "Q8_0")
    self.assertEqual(info.tensors[1].serialized_nbytes, 34)

    tinygrad_metadata = {item.key: item.value for item in info.metadata}
    tinygrad_tensors = {item.name: FakeTensor(item.logical_shape) for item in info.tensors}
    logical_dtypes = crosscheck_tinygrad(info, tinygrad_metadata, tinygrad_tensors)
    report = build_report(info, "test", info.file_size, "a" * 64, logical_dtypes)
    token_record = next(item for item in report["gguf"]["metadata"] if item["key"] == "tokenizer.ggml.tokens")
    self.assertEqual(token_record["value"]["count"], 300)
    self.assertEqual(len(token_record["value"]["sha256"]), 64)
    self.assertEqual(json.loads(serialize_report(report)), report)

  def test_rejects_duplicate_metadata_and_tensor_names(self):
    cases = [
      build_gguf([("duplicate", 4, 1), ("duplicate", 4, 2)], []),
      build_gguf([], [("duplicate", (1,), 0, 0), ("duplicate", (1,), 0, 32)]),
    ]
    with tempfile.TemporaryDirectory() as directory:
      for index, content in enumerate(cases):
        with self.subTest(index=index), self.assertRaisesRegex(GGUFInspectionError, "duplicate"):
          scan_gguf(self.write_gguf(directory, content))

  def test_rejects_invalid_descriptors_and_data_ranges(self):
    valid = build_gguf([], [("matrix", (2, 3), 0, 0)])
    cases = {
      "magic": b"NOPE" + valid[4:],
      "truncated": valid[:20],
      "unsupported type": build_gguf([], [("matrix", (2, 3), 999, 0)]),
      "invalid quantized row": build_gguf([], [("quantized", (16, 2), 8, 0)]),
      "alignment type": build_gguf([("general.alignment", 5, 32)], []),
      "unaligned offset": build_gguf([], [("matrix", (2, 3), 0, 1)]),
      "out of bounds": build_gguf([], [("matrix", (2, 3), 0, 0)], data=b"\x00"),
      "overlap": build_gguf([], [("first", (2, 3), 0, 0), ("second", (2, 3), 0, 0)]),
    }
    with tempfile.TemporaryDirectory() as directory:
      for name, content in cases.items():
        with self.subTest(name=name), self.assertRaises(GGUFInspectionError):
          scan_gguf(self.write_gguf(directory, content))

  def test_rejects_tinygrad_crosscheck_mismatches(self):
    content = build_gguf([("general.name", 8, "test")], [("matrix", (2, 3), 0, 0)])
    with tempfile.TemporaryDirectory() as directory:
      info = scan_gguf(self.write_gguf(directory, content))
    with self.assertRaisesRegex(GGUFInspectionError, "metadata mismatch"):
      crosscheck_tinygrad(info, {}, {"matrix": FakeTensor((3, 2))})
    with self.assertRaisesRegex(GGUFInspectionError, "shape mismatch"):
      crosscheck_tinygrad(info, {"general.name": "test"}, {"matrix": FakeTensor((2, 3))})

  def test_rejects_descriptor_counts_beyond_file_bounds(self):
    cases = [
      struct.pack("<4sIQQ", b"GGUF", 3, 0, 2**63),
      struct.pack("<4sIQQ", b"GGUF", 3, 2**63, 0),
      struct.pack("<4sIQQ", b"GGUF", 3, 0, 1) + struct.pack("<Q", 2**63),
    ]
    with tempfile.TemporaryDirectory() as directory:
      for index, content in enumerate(cases):
        with self.subTest(index=index), self.assertRaises(GGUFInspectionError):
          scan_gguf(self.write_gguf(directory, content))

  def test_atomic_report_output_and_failure_cleanup(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / "nested" / "report.json"
      _write_atomic(output, "first\n")
      self.assertEqual(output.read_text(encoding="utf-8"), "first\n")
      with patch("tools.inspect_artifact.os.replace", side_effect=OSError("failed")):
        with self.assertRaises(OSError): _write_atomic(output, "second\n")
      self.assertEqual(output.read_text(encoding="utf-8"), "first\n")
      self.assertEqual(list(output.parent.glob("*.part")), [])

@pytest.mark.model
@pytest.mark.slow
class TestPinnedGGUFInspection(unittest.TestCase):
  def test_pinned_artifact_matches_tinygrad(self):
    if os.environ.get("DEV") != "CPU": self.skipTest("real GGUF inspection requires DEV=CPU")
    artifact_value = os.environ.get("INFURNACE_MODEL_ARTIFACT")
    if artifact_value is None: self.skipTest("set INFURNACE_MODEL_ARTIFACT to the pinned GGUF")

    artifact, manifest = Path(artifact_value), load_manifest(PINNED_MANIFEST)
    with stable_artifact_path(artifact) as stable_path:
      verify_artifact(stable_path, manifest)
      info = scan_gguf(stable_path)
      from tinygrad.llm.gguf import gguf_load
      metadata, tensors = gguf_load(stable_path)
      logical_dtypes = crosscheck_tinygrad(info, metadata, tensors)

    self.assertEqual(info.version, 3)
    self.assertEqual(len(info.metadata), 28)
    self.assertEqual(len(info.tensors), 310)
    self.assertEqual({item.ggml_type_name for item in info.tensors}, {"F32", "Q8_0"})
    self.assertEqual(sum(item.ggml_type_name == "F32" for item in info.tensors), 113)
    self.assertEqual(sum(item.ggml_type_name == "Q8_0" for item in info.tensors), 197)
    values = {item.key: item.value for item in info.metadata}
    self.assertEqual(values["general.architecture"], "qwen3")
    self.assertEqual(values["qwen3.context_length"], 40960)
    self.assertEqual(values["qwen3.block_count"], 28)
    self.assertEqual(len(values["tokenizer.ggml.tokens"]), 151936)
    self.assertNotIn("output.weight", {item.name for item in info.tensors})
    self.assertEqual(logical_dtypes.keys(), {item.name for item in info.tensors})
    report = build_report(info, manifest.id, manifest.size_bytes, manifest.sha256, logical_dtypes)
    self.assertEqual(report, json.loads(PINNED_INSPECTION.read_text(encoding="utf-8")))