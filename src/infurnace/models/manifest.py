"""Checkpoint identity, verification, and acquisition."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from http.client import HTTPException
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


_CHUNK_SIZE = 1024 * 1024
_EXPECTED_SCHEMA_VERSION = 1
_REPOSITORY_PATTERN = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*")
_QUANTIZATION_PATTERN = re.compile(r"[A-Z0-9_]+")


class ManifestError(ValueError):
  """The checkpoint manifest is malformed or internally inconsistent."""


class ArtifactError(RuntimeError):
  """The checkpoint artifact could not be acquired or verified."""


@dataclass(frozen=True)
class CheckpointManifest:
  schema_version: int
  id: str
  repository: str
  revision: str
  url: str
  filename: str
  format: str
  quantization: str
  size_bytes: int
  sha256: str
  license_spdx: str
  license_url: str


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
  result: dict[str, Any] = {}
  for key, value in pairs:
    if key in result: raise ManifestError(f"duplicate field: {key}")
    result[key] = value
  return result


def _object(value: Any, name: str, fields: set[str]) -> dict[str, Any]:
  if not isinstance(value, dict): raise ManifestError(f"{name} must be an object")
  missing, unknown = fields - value.keys(), value.keys() - fields
  if missing: raise ManifestError(f"{name} missing fields: {', '.join(sorted(missing))}")
  if unknown: raise ManifestError(f"{name} has unknown fields: {', '.join(sorted(unknown))}")
  return value


def _string(value: Any, name: str) -> str:
  if not isinstance(value, str) or not value: raise ManifestError(f"{name} must be a non-empty string")
  return value


def _https_url(value: Any, name: str) -> str:
  url = _string(value, name)
  try:
    parsed = urlsplit(url)
    parsed.port
  except ValueError as error:
    raise ManifestError(f"{name} is not a valid URL") from error
  if parsed.scheme != "https" or parsed.hostname is None or parsed.username is not None or parsed.password is not None:
    raise ManifestError(f"{name} must be an HTTPS URL without credentials")
  return url


def load_manifest(path: str | Path) -> CheckpointManifest:
  manifest_path = Path(path)
  try:
    data = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
  except ManifestError:
    raise
  except (OSError, UnicodeError, json.JSONDecodeError) as error:
    raise ManifestError(f"cannot read manifest {manifest_path}: {error}") from error

  root = _object(data, "manifest", {"schema_version", "id", "source", "artifact", "license"})
  source = _object(root["source"], "source", {"repository", "revision", "url"})
  artifact = _object(root["artifact"], "artifact", {"filename", "format", "quantization", "size_bytes", "sha256"})
  license_data = _object(root["license"], "license", {"spdx", "url"})

  schema_version = root["schema_version"]
  if type(schema_version) is not int or schema_version != _EXPECTED_SCHEMA_VERSION:
    raise ManifestError(f"schema_version must be {_EXPECTED_SCHEMA_VERSION}")

  checkpoint_id = _string(root["id"], "id")
  if _ID_PATTERN.fullmatch(checkpoint_id) is None: raise ManifestError("id contains unsupported characters")

  repository = _string(source["repository"], "source.repository")
  if _REPOSITORY_PATTERN.fullmatch(repository) is None: raise ManifestError("source.repository must be owner/name")
  revision = _string(source["revision"], "source.revision")
  if _REVISION_PATTERN.fullmatch(revision) is None: raise ManifestError("source.revision must be a full lowercase Git revision")

  filename = _string(artifact["filename"], "artifact.filename")
  if Path(filename).name != filename or "/" in filename or "\\" in filename:
    raise ManifestError("artifact.filename must be a basename")
  if not filename.endswith(".gguf"): raise ManifestError("artifact.filename must end with .gguf")

  url = _https_url(source["url"], "source.url")
  expected_url = f"https://huggingface.co/{repository}/resolve/{revision}/{filename}"
  if url != expected_url: raise ManifestError(f"source.url must equal {expected_url}")

  artifact_format = _string(artifact["format"], "artifact.format")
  if artifact_format != "GGUF": raise ManifestError("artifact.format must be GGUF")
  quantization = _string(artifact["quantization"], "artifact.quantization")
  if _QUANTIZATION_PATTERN.fullmatch(quantization) is None: raise ManifestError("artifact.quantization is invalid")

  size_bytes = artifact["size_bytes"]
  if type(size_bytes) is not int or size_bytes <= 0: raise ManifestError("artifact.size_bytes must be a positive integer")
  sha256 = _string(artifact["sha256"], "artifact.sha256")
  if _SHA256_PATTERN.fullmatch(sha256) is None: raise ManifestError("artifact.sha256 must be a lowercase SHA-256")

  license_spdx = _string(license_data["spdx"], "license.spdx")
  license_url = _https_url(license_data["url"], "license.url")

  return CheckpointManifest(schema_version, checkpoint_id, repository, revision, url, filename, artifact_format,
                            quantization, size_bytes, sha256, license_spdx, license_url)


def _check_identity(size_bytes: int, sha256: str, manifest: CheckpointManifest) -> None:
  if size_bytes != manifest.size_bytes:
    raise ArtifactError(f"artifact size mismatch: expected {manifest.size_bytes}, got {size_bytes}")
  if sha256 != manifest.sha256:
    raise ArtifactError(f"artifact SHA-256 mismatch: expected {manifest.sha256}, got {sha256}")


def verify_artifact(path: str | Path, manifest: CheckpointManifest) -> None:
  artifact_path = Path(path)
  digest, size_bytes = hashlib.sha256(), 0
  try:
    with artifact_path.open("rb") as artifact:
      while chunk := artifact.read(_CHUNK_SIZE):
        size_bytes += len(chunk)
        if size_bytes > manifest.size_bytes:
          raise ArtifactError(f"artifact size exceeds expected {manifest.size_bytes} bytes")
        digest.update(chunk)
  except ArtifactError:
    raise
  except OSError as error:
    raise ArtifactError(f"cannot read artifact {artifact_path}: {error}") from error
  _check_identity(size_bytes, digest.hexdigest(), manifest)


def acquire_artifact(manifest: CheckpointManifest, destination: str | Path, timeout: float = 30) -> Path:
  destination_path = Path(destination)
  if os.path.lexists(destination_path):
    verify_artifact(destination_path, manifest)
    return destination_path
  if not math.isfinite(timeout) or timeout <= 0: raise ArtifactError("timeout must be finite and positive")

  temporary_path: Path | None = None
  try:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(manifest.url, headers={"Accept-Encoding": "identity", "User-Agent": "infurnace/0.0.0"})
    with tempfile.NamedTemporaryFile(mode="wb", dir=destination_path.parent, prefix=f".{destination_path.name}.",
                                     suffix=".part", delete=False) as temporary:
      temporary_path = Path(temporary.name)
      digest, size_bytes = hashlib.sha256(), 0
      with urlopen(request, timeout=timeout) as response:
        while chunk := response.read(_CHUNK_SIZE):
          size_bytes += len(chunk)
          if size_bytes > manifest.size_bytes:
            raise ArtifactError(f"download exceeds expected {manifest.size_bytes} bytes")
          temporary.write(chunk)
          digest.update(chunk)
      _check_identity(size_bytes, digest.hexdigest(), manifest)
      temporary.flush()
      os.fsync(temporary.fileno())

    try: os.link(temporary_path, destination_path)
    except FileExistsError:
      verify_artifact(destination_path, manifest)
    return destination_path
  except ArtifactError:
    raise
  except (OSError, URLError, HTTPException) as error:
    raise ArtifactError(f"cannot acquire artifact: {error}") from error
  finally:
    if temporary_path is not None:
      try: temporary_path.unlink(missing_ok=True)
      except OSError: pass
