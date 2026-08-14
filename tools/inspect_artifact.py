#!/usr/bin/env python3
"""Inspect a verified GGUF artifact and cross-check current tinygrad."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

try:
  from tools.gguf_inspection import GGUFInspectionError, build_report, crosscheck_tinygrad, scan_gguf, serialize_report, stable_artifact_path
except ModuleNotFoundError:
  from gguf_inspection import GGUFInspectionError, build_report, crosscheck_tinygrad, scan_gguf, serialize_report, stable_artifact_path
from infurnace.models.manifest import ArtifactError, ManifestError, load_manifest, verify_artifact


def _write_atomic(path: Path, content: str) -> None:
  missing_directories, directory = [], path.parent
  while not directory.exists():
    missing_directories.append(directory)
    directory = directory.parent
  path.parent.mkdir(parents=True, exist_ok=True)
  if hasattr(os, "O_DIRECTORY"):
    for created in reversed(missing_directories):
      parent_fd = os.open(created.parent, os.O_RDONLY | os.O_DIRECTORY)
      try: os.fsync(parent_fd)
      finally: os.close(parent_fd)
  temporary_path = None
  try:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".part", delete=False) as temporary:
      temporary_path = Path(temporary.name)
      temporary.write(content)
      temporary.flush()
      os.fsync(temporary.fileno())
    os.replace(temporary_path, path)
    temporary_path = None
    if hasattr(os, "O_DIRECTORY"):
      directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
      try: os.fsync(directory_fd)
      finally: os.close(directory_fd)
  finally:
    if temporary_path is not None: temporary_path.unlink(missing_ok=True)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", required=True, type=Path, help="checkpoint manifest path")
  parser.add_argument("--artifact", required=True, type=Path, help="verified GGUF artifact path")
  action = parser.add_mutually_exclusive_group()
  action.add_argument("--output", type=Path, help="write the canonical report atomically")
  action.add_argument("--check", type=Path, help="compare with an existing canonical report")
  args = parser.parse_args()

  if os.environ.get("DEV") != "CPU":
    print("error: inspection requires DEV=CPU before tinygrad is imported", file=sys.stderr)
    return 1

  try:
    manifest = load_manifest(args.manifest)
    with stable_artifact_path(args.artifact) as stable_path:
      verify_artifact(stable_path, manifest)
      info = scan_gguf(stable_path)

      from tinygrad.llm.gguf import gguf_load
      tinygrad_metadata, tinygrad_tensors = gguf_load(stable_path)
      logical_dtypes = crosscheck_tinygrad(info, tinygrad_metadata, tinygrad_tensors)
      report = build_report(info, manifest.id, manifest.size_bytes, manifest.sha256, logical_dtypes)
      serialized = serialize_report(report)

    if args.check is not None:
      expected = args.check.read_text(encoding="utf-8")
      if expected != serialized: raise GGUFInspectionError(f"inspection differs from canonical report {args.check}")
      print(f"inspection matches {args.check}")
    elif args.output is not None:
      _write_atomic(args.output, serialized)
      print(f"wrote {args.output}")
    else:
      sys.stdout.write(serialized)
  except (ArtifactError, ManifestError, GGUFInspectionError, OSError, ValueError, RuntimeError) as error:
    print(f"error: {error}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__": raise SystemExit(main())
