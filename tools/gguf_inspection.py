"""Development-only GGUF descriptor inspection."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO


GGUF_VALUE_TYPES = {
  0: ("UINT8", "B"), 1: ("INT8", "b"), 2: ("UINT16", "H"), 3: ("INT16", "h"),
  4: ("UINT32", "I"), 5: ("INT32", "i"), 6: ("FLOAT32", "f"), 7: ("BOOL", "?"),
  8: ("STRING", None), 9: ("ARRAY", None), 10: ("UINT64", "Q"), 11: ("INT64", "q"), 12: ("FLOAT64", "d"),
}

# type ID: (name, elements per block, serialized bytes per block)
GGML_TYPES = {
  0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
  6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34),
  12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210),
  18: ("IQ3_XXS", 256, 98), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
  24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
  30: ("BF16", 1, 2), 39: ("MXFP4", 32, 17), 41: ("Q1_0", 128, 18),
}


class GGUFInspectionError(ValueError):
  """The GGUF descriptor stream is malformed or inconsistent."""


@dataclass(frozen=True)
class MetadataInfo:
  key: str
  type_id: int
  type_name: str
  element_type_id: int | None
  element_type_name: str | None
  value: Any


@dataclass(frozen=True)
class TensorInfo:
  name: str
  encoded_dimensions: tuple[int, ...]
  logical_shape: tuple[int, ...]
  ggml_type_id: int
  ggml_type_name: str
  data_offset: int
  serialized_nbytes: int


@dataclass(frozen=True)
class GGUFInfo:
  version: int
  alignment: int
  data_start: int
  file_size: int
  metadata: tuple[MetadataInfo, ...]
  tensors: tuple[TensorInfo, ...]


@contextmanager
def stable_artifact_path(path: str | Path):
  with Path(path).open("rb") as artifact_file:
    stable_path = Path(f"/proc/self/fd/{artifact_file.fileno()}")
    if not stable_path.exists(): raise GGUFInspectionError("inspection requires stable /proc/self/fd file descriptors")
    yield stable_path


class _Reader:
  def __init__(self, stream: BinaryIO, file_size: int): self.stream, self.file_size = stream, file_size

  def remaining(self) -> int: return self.file_size - self.stream.tell()

  def read(self, size: int) -> bytes:
    if size < 0 or size > self.remaining(): raise GGUFInspectionError(f"read exceeds GGUF bounds at byte {self.stream.tell()}")
    data = self.stream.read(size)
    if len(data) != size: raise GGUFInspectionError(f"truncated GGUF descriptor at byte {self.stream.tell()}")
    return data

  def unpack(self, fmt: str): return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))[0]
  def u32(self) -> int: return self.unpack("I")
  def u64(self) -> int: return self.unpack("Q")

  def string(self) -> str:
    size = self.u64()
    if size > self.remaining(): raise GGUFInspectionError("GGUF string exceeds artifact bounds")
    try: return self.read(size).decode("utf-8")
    except UnicodeDecodeError as error: raise GGUFInspectionError("GGUF string is not UTF-8") from error

  def value(self, type_id: int, allow_array: bool = True) -> tuple[Any, int | None]:
    if type_id not in GGUF_VALUE_TYPES: raise GGUFInspectionError(f"unsupported GGUF metadata type {type_id}")
    _, fmt = GGUF_VALUE_TYPES[type_id]
    if fmt is not None: return self.unpack(fmt), None
    if type_id == 8: return self.string(), None
    if not allow_array: raise GGUFInspectionError("nested GGUF metadata arrays are unsupported")
    element_type, count = self.u32(), self.u64()
    if element_type == 9: raise GGUFInspectionError("nested GGUF metadata arrays are unsupported")
    if element_type not in GGUF_VALUE_TYPES: raise GGUFInspectionError(f"unsupported GGUF array element type {element_type}")
    element_format = GGUF_VALUE_TYPES[element_type][1]
    minimum_size = struct.calcsize("<" + element_format) if element_format is not None else 8
    if count > self.remaining() // minimum_size: raise GGUFInspectionError("GGUF array count exceeds artifact bounds")
    return [self.value(element_type, allow_array=False)[0] for _ in range(count)], element_type


def _tensor_nbytes(dimensions: tuple[int, ...], ggml_type: int) -> int:
  if ggml_type not in GGML_TYPES: raise GGUFInspectionError(f"unsupported GGML tensor type {ggml_type}")
  elements = math.prod(dimensions)
  block_elements, block_bytes = GGML_TYPES[ggml_type][1:]
  if elements <= 0: raise GGUFInspectionError("tensor dimensions must be positive")
  if dimensions[0] % block_elements:
    raise GGUFInspectionError(f"tensor row width {dimensions[0]} is not divisible by {block_elements}")
  return elements // block_elements * block_bytes


def scan_gguf(path: str | Path) -> GGUFInfo:
  artifact_path = Path(path)
  file_size = artifact_path.stat().st_size
  with artifact_path.open("rb") as stream:
    reader = _Reader(stream, file_size)
    if reader.read(4) != b"GGUF": raise GGUFInspectionError("invalid GGUF magic")
    version, tensor_count, metadata_count = reader.u32(), reader.u64(), reader.u64()
    if version not in (2, 3): raise GGUFInspectionError(f"unsupported GGUF version {version}")
    if metadata_count > reader.remaining() // 13: raise GGUFInspectionError("metadata count exceeds artifact bounds")
    if tensor_count > reader.remaining() // 32: raise GGUFInspectionError("tensor count exceeds artifact bounds")

    metadata, metadata_keys = [], set()
    for _ in range(metadata_count):
      key, type_id = reader.string(), reader.u32()
      if key in metadata_keys: raise GGUFInspectionError(f"duplicate GGUF metadata key: {key}")
      metadata_keys.add(key)
      value, element_type_id = reader.value(type_id)
      metadata.append(MetadataInfo(key, type_id, GGUF_VALUE_TYPES[type_id][0], element_type_id,
                                   GGUF_VALUE_TYPES[element_type_id][0] if element_type_id is not None else None, value))

    tensor_descriptors, tensor_names = [], set()
    for _ in range(tensor_count):
      name, dimension_count = reader.string(), reader.u32()
      if name in tensor_names: raise GGUFInspectionError(f"duplicate GGUF tensor name: {name}")
      if dimension_count == 0 or dimension_count > 4: raise GGUFInspectionError(f"tensor {name} has invalid dimension count")
      if dimension_count > reader.remaining() // 8: raise GGUFInspectionError(f"tensor {name} dimensions exceed artifact bounds")
      tensor_names.add(name)
      dimensions = tuple(reader.u64() for _ in range(dimension_count))
      ggml_type, offset = reader.u32(), reader.u64()
      type_name = GGML_TYPES.get(ggml_type, (f"UNKNOWN_{ggml_type}",))[0]
      tensor_descriptors.append((name, dimensions, ggml_type, type_name, offset, _tensor_nbytes(dimensions, ggml_type)))

    alignment_info = next((item for item in metadata if item.key == "general.alignment"), None)
    if alignment_info is not None and alignment_info.type_id != 4:
      raise GGUFInspectionError("general.alignment must have UINT32 type")
    alignment = alignment_info.value if alignment_info is not None else 32
    if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
      raise GGUFInspectionError("general.alignment must be a positive power of two")
    data_start = (stream.tell() + alignment - 1) // alignment * alignment
    if data_start > file_size: raise GGUFInspectionError("GGUF data section starts beyond end of file")

  tensors = []
  for name, dimensions, ggml_type, type_name, offset, nbytes in tensor_descriptors:
    if offset % alignment: raise GGUFInspectionError(f"tensor {name} offset is not aligned")
    if data_start + offset + nbytes > file_size: raise GGUFInspectionError(f"tensor {name} exceeds artifact bounds")
    tensors.append(TensorInfo(name, dimensions, tuple(reversed(dimensions)), ggml_type, type_name, offset, nbytes))

  previous_end = 0
  for tensor in sorted(tensors, key=lambda item: item.data_offset):
    if tensor.data_offset < previous_end: raise GGUFInspectionError(f"tensor {tensor.name} overlaps preceding tensor data")
    previous_end = tensor.data_offset + tensor.serialized_nbytes

  return GGUFInfo(version, alignment, data_start, file_size, tuple(metadata), tuple(tensors))


def crosscheck_tinygrad(info: GGUFInfo, metadata: dict[str, Any], tensors: dict[str, Any]) -> dict[str, str]:
  scanned_metadata = {item.key: item.value for item in info.metadata}
  if scanned_metadata.keys() != metadata.keys():
    missing, unexpected = scanned_metadata.keys() - metadata.keys(), metadata.keys() - scanned_metadata.keys()
    raise GGUFInspectionError(f"tinygrad metadata mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
  for key, value in scanned_metadata.items():
    if metadata[key] != value: raise GGUFInspectionError(f"tinygrad metadata value mismatch: {key}")

  scanned_tensors = {item.name: item for item in info.tensors}
  if scanned_tensors.keys() != tensors.keys():
    missing, unexpected = scanned_tensors.keys() - tensors.keys(), tensors.keys() - scanned_tensors.keys()
    raise GGUFInspectionError(f"tinygrad tensor mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
  logical_dtypes = {}
  for name, tensor in tensors.items():
    if tuple(tensor.shape) != scanned_tensors[name].logical_shape:
      raise GGUFInspectionError(f"tinygrad tensor shape mismatch: {name}")
    logical_dtypes[name] = tensor.dtype.name
  return logical_dtypes


def _metadata_value(value: Any) -> Any:
  if isinstance(value, list) and len(value) > 256:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {"count": len(value), "sha256": hashlib.sha256(encoded).hexdigest()}
  return value


def build_report(info: GGUFInfo, artifact_id: str, artifact_size: int, artifact_sha256: str,
                 logical_dtypes: dict[str, str]) -> dict[str, Any]:
  if info.file_size != artifact_size: raise GGUFInspectionError("inspection artifact size does not match manifest")
  if logical_dtypes.keys() != {item.name for item in info.tensors}:
    raise GGUFInspectionError("logical dtype inventory does not match GGUF tensors")

  type_counts: dict[str, int] = {}
  for tensor in info.tensors: type_counts[tensor.ggml_type_name] = type_counts.get(tensor.ggml_type_name, 0) + 1
  return {
    "schema_version": 1,
    "artifact": {"id": artifact_id, "size_bytes": artifact_size, "sha256": artifact_sha256},
    "gguf": {
      "version": info.version,
      "alignment": info.alignment,
      "data_start": info.data_start,
      "metadata_count": len(info.metadata),
      "tensor_count": len(info.tensors),
      "storage_type_counts": dict(sorted(type_counts.items())),
      "metadata": [
        {
          "key": item.key,
          "type": item.type_name,
          **({"element_type": item.element_type_name} if item.element_type_name is not None else {}),
          "value": _metadata_value(item.value),
        }
        for item in info.metadata
      ],
      "tensors": [
        {
          "name": item.name,
          "encoded_dimensions": list(item.encoded_dimensions),
          "logical_shape": list(item.logical_shape),
          "ggml_type": {"id": item.ggml_type_id, "name": item.ggml_type_name},
          "logical_dtype": logical_dtypes[item.name],
          "data_offset": item.data_offset,
          "serialized_nbytes": item.serialized_nbytes,
          "layout": "gguf-reversed-dimensions",
        }
        for item in info.tensors
      ],
    },
  }


def serialize_report(report: dict[str, Any]) -> str:
  return json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
