#!/usr/bin/env python3
"""Run the pinned model through Infurnace's stateless cache-aware runner."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from tinygrad import Tensor, dtypes
from infurnace.executor.tinygrad.buffers import ContiguousKVCache
from infurnace.executor.tinygrad.runner import Qwen3Runner
from infurnace.executor.tinygrad.weights import WeightPolicy, load_qwen3_weights
from infurnace.models.manifest import ArtifactError, ManifestError, load_manifest

FIXED_PROMPT = [257] + [1000 + i for i in range(15)]


def generate(runner: Qwen3Runner, prompt: list[int], max_tokens: int) -> tuple[list[int], dict[str, float]]:
  tokens = list(prompt)
  if len(tokens) > runner.kv_cache.max_context:
    raise ValueError(f"prompt length {len(tokens)} exceeds cache max_context {runner.kv_cache.max_context}")

  timings: dict[str, float] = {}

  t0 = time.perf_counter()
  logits = runner.prefill(Tensor([tokens], dtype=dtypes.int32)).realize()
  timings["prefill_ms"] = (time.perf_counter() - t0) * 1000

  decode_times: list[float] = []
  for _ in range(max_tokens):
    token = int(logits.argmax().item())
    tokens.append(token)
    if len(tokens) >= runner.kv_cache.max_context:
      break
    t0 = time.perf_counter()
    logits = runner.decode(Tensor([[token]], dtype=dtypes.int32), position=len(tokens) - 1).realize()
    decode_times.append((time.perf_counter() - t0) * 1000)

  if decode_times:
    timings["first_decode_ms"] = decode_times[0]
    timings["avg_decode_ms"] = sum(decode_times) / len(decode_times)
    timings["decode_times_ms"] = decode_times
  return tokens, timings


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", required=True, type=Path, help="checkpoint manifest path")
  parser.add_argument("--artifact", required=True, type=Path, help="verified GGUF artifact path")
  parser.add_argument("--weight-policy", default="realized-fp16", choices=("lazy-fp16", "realized-fp16"))
  parser.add_argument("--max-context", type=int, default=1024)
  parser.add_argument("--output-tokens", type=int, default=4)
  parser.add_argument("--slot", type=int, default=0)
  args = parser.parse_args()

  if args.max_context <= 0 or args.output_tokens < 0:
    print("error: max-context must be positive and output-tokens must be non-negative", file=sys.stderr)
    return 1

  try:
    manifest = load_manifest(args.manifest)
    weights = load_qwen3_weights(args.artifact, manifest, WeightPolicy(args.weight_policy))
    from infurnace.executor.tinygrad.model import Qwen3Model
    model = Qwen3Model(weights)
    cache = ContiguousKVCache(weights.config, max_context=args.max_context, num_slots=args.slot + 1)
    runner = Qwen3Runner(model, cache)

    generated, timings = generate(runner, FIXED_PROMPT, args.output_tokens)

    result = {
      "artifact": {"id": manifest.id, "sha256": manifest.sha256},
      "execution": {
        "device": os.environ.get("DEV", "default"),
        "weight_policy": args.weight_policy,
        "max_context": args.max_context,
        "cache_slots": args.slot + 1,
        "timings_ms": timings,
      },
      "prompt": FIXED_PROMPT,
      "generated_token_ids": generated[len(FIXED_PROMPT):],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
  except (ArtifactError, ManifestError, OSError, RuntimeError, ValueError) as error:
    print(f"error: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
