#!/usr/bin/env python3
"""Acquire a checkpoint only after verifying its pinned identity."""
import argparse
import sys
from pathlib import Path
from infurnace.models.manifest import ArtifactError, ManifestError, acquire_artifact, load_manifest

def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", required=True, type=Path, help="checkpoint manifest path")
  parser.add_argument("--output", required=True, type=Path, help="verified artifact destination")
  parser.add_argument("--timeout", type=float, default=30, help="network operation timeout in seconds")
  args = parser.parse_args()

  try:
    manifest = load_manifest(args.manifest)
    destination = acquire_artifact(manifest, args.output, timeout=args.timeout)
  except (ManifestError, ArtifactError) as error:
    print(f"error: {error}", file=sys.stderr)
    return 1

  print(f"verified {destination}")
  print(f"size: {manifest.size_bytes}")
  print(f"sha256: {manifest.sha256}")
  return 0

if __name__ == "__main__": raise SystemExit(main())
