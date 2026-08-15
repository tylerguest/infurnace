#!/usr/bin/env python3
"""Compare Infurnace Qwen3Model greedy output with the tinygrad upstream baseline."""

import argparse
import json
import os
import sys
from pathlib import Path

from tinygrad import Device, Tensor, dtypes

from infurnace.executor.tinygrad.model import Qwen3Model
from infurnace.executor.tinygrad.weights import WeightPolicy, load_qwen3_weights
from infurnace.models.manifest import ArtifactError, ManifestError, load_manifest

try:
  from tools.gguf_inspection import stable_artifact_path
except ModuleNotFoundError:
  from gguf_inspection import stable_artifact_path


FIXED_PROMPT = [257] + [1000 + i for i in range(15)]


def _load_upstream_model(artifact: Path, manifest, max_context: int, realize: bool):
  from tinygrad.llm.model import Transformer
  with stable_artifact_path(artifact) as stable_path:
    # Re-verify so the tool is self-contained even when called directly.
    from infurnace.models.manifest import verify_artifact
    verify_artifact(stable_path, manifest)
    gguf_tensor = Tensor.empty(manifest.size_bytes, dtype=dtypes.uint8, device=f"disk:{stable_path}")
    os.environ["HALF"] = "1"
    model, metadata = Transformer.from_gguf(gguf_tensor, max_context, realize=realize)
  return model, metadata


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", required=True, type=Path, help="checkpoint manifest path")
  parser.add_argument("--artifact", required=True, type=Path, help="verified GGUF artifact path")
  parser.add_argument("--weight-policy", required=True, choices=("lazy-fp16", "realized-fp16"))
  parser.add_argument("--max-context", type=int, default=1024)
  args = parser.parse_args()

  if os.environ.get("DEV") != "NV":
    print("error: comparison requires DEV=NV", file=sys.stderr)
    return 1

  try:
    manifest = load_manifest(args.manifest)
    realize_upstream = args.weight_policy == "realized-fp16"

    # Infurnace stateless forward.
    inf_weights = load_qwen3_weights(args.artifact, manifest, WeightPolicy(args.weight_policy))
    inf_model = Qwen3Model(inf_weights)
    inf_input = Tensor([FIXED_PROMPT], dtype=dtypes.int32)
    inf_logits = inf_model(inf_input).realize()
    inf_token = int(inf_logits.argmax().item())

    # Upstream baseline.
    upstream_model, _ = _load_upstream_model(args.artifact, manifest, args.max_context, realize_upstream)
    upstream_model.warmup()
    upstream_tokens = []
    for token in upstream_model.generate(FIXED_PROMPT, chunk_size=32, temperature=0.0):
      upstream_tokens.append(token)
      if len(upstream_tokens) >= 1:
        break
    Device["NV"].synchronize()

    upstream_token = upstream_tokens[0] if upstream_tokens else None

    result = {
      "artifact": {"id": manifest.id, "sha256": manifest.sha256},
      "execution": {
        "device": "NV",
        "weight_policy": args.weight_policy,
        "upstream_realize": realize_upstream,
      },
      "prompt": FIXED_PROMPT,
      "infurnace": {
        "token": inf_token,
        "logits_shape": list(inf_logits.shape),
        "logits_dtype": inf_logits.dtype.name,
      },
      "upstream": {
        "token": upstream_token,
      },
      "agreement": {
        "greedy_tokens_match": inf_token == upstream_token,
      },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if inf_token == upstream_token else 1

  except (ArtifactError, ManifestError, OSError, RuntimeError, ValueError) as error:
    print(f"error: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  raise SystemExit(main())