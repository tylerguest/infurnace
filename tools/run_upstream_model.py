#!/usr/bin/env python3
"""Run the pinned model through tinygrad's current upstream generation path."""

import argparse
import json
import os
import sys
from pathlib import Path

from infurnace.models.manifest import ArtifactError, ManifestError, load_manifest, verify_artifact
try:
  from tools.gguf_inspection import stable_artifact_path
except ModuleNotFoundError:
  from gguf_inspection import stable_artifact_path


TEXT_PROMPT = "<|im_start|>user\nReply with exactly: hello<|im_end|>\n<|im_start|>assistant\n"


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", required=True, type=Path, help="checkpoint manifest path")
  parser.add_argument("--artifact", required=True, type=Path, help="verified GGUF artifact path")
  parser.add_argument("--weight-policy", required=True, choices=("lazy", "realized-fp16"))
  parser.add_argument("--workload", required=True, choices=("fixed", "text"))
  parser.add_argument("--max-context", type=int, default=1024)
  parser.add_argument("--chunk-size", type=int, default=32)
  parser.add_argument("--fixed-output-tokens", type=int, default=4)
  parser.add_argument("--text-output-tokens", type=int, default=16)
  args = parser.parse_args()

  if os.environ.get("DEV") != "NV":
    print("error: upstream model smoke test requires DEV=NV", file=sys.stderr)
    return 1
  output_tokens = args.fixed_output_tokens if args.workload == "fixed" else args.text_output_tokens
  if min(args.max_context, args.chunk_size, output_tokens) <= 0:
    print("error: context, chunk, and selected workload output token counts must be positive", file=sys.stderr)
    return 1

  try:
    manifest = load_manifest(args.manifest)
    os.environ.update(JIT="1", HALF="1")

    from tinygrad import Device, Tensor, dtypes
    from tinygrad.llm.cli import SimpleTokenizer
    from tinygrad.llm.model import Transformer

    with stable_artifact_path(args.artifact) as stable_path:
      verify_artifact(stable_path, manifest)
      gguf_tensor = Tensor.empty(manifest.size_bytes, dtype=dtypes.uint8, device=f"disk:{stable_path}")
      realize_weights = args.weight_policy == "realized-fp16"
      model, metadata = Transformer.from_gguf(gguf_tensor, args.max_context, realize=realize_weights)
    tokenizer = SimpleTokenizer.from_gguf_kv(metadata)
    model.warmup()

    def generate(prompt: list[int], count: int) -> list[int]:
      if len(prompt) + count > model.max_context: raise ValueError("prompt and output exceed effective model context")
      tokens, output = list(prompt), []
      generator = model.generate(tokens, chunk_size=args.chunk_size, temperature=0.0)
      for _ in range(count):
        try: token = next(generator)
        except StopIteration: break
        output.append(token)
        if tokenizer.is_end(token): break
      Device["NV"].synchronize()
      return output

    if args.workload == "fixed":
      prompt = [257] + [1000 + index for index in range(15)]
      output = generate(prompt, output_tokens)
      workload = {"name": "fixed", "prompt_token_ids": prompt, "generated_token_ids": output}
    else:
      prompt = tokenizer.encode(TEXT_PROMPT)
      output = generate(prompt, output_tokens)
      workload = {
        "name": "text",
        "prompt": TEXT_PROMPT,
        "prompt_token_ids": prompt,
        "generated_token_ids": output,
        "generated_text": tokenizer.decode(output),
      }
    result = {
      "artifact": {"id": manifest.id, "sha256": manifest.sha256},
      "execution": {
        "device": Device.DEFAULT,
        "max_context": model.max_context,
        "chunk_size": args.chunk_size,
        "weight_policy": args.weight_policy,
        "upstream_realize": realize_weights,
        "weight_dtype": "float16",
        "sampling": "greedy",
      },
      "workload": workload,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
  except (ArtifactError, ManifestError, OSError, RuntimeError, ValueError) as error:
    print(f"error: {error}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__": raise SystemExit(main())
