"""Compare wrapper overhead to direct generation using the SAME loaded transformer.

Not a comparison of independent loaders, engines, or custom kernels.
Run: DEV=NV .venv/bin/python benchmarks/wrapper.py --device NV
"""
import argparse
import json
from types import SimpleNamespace

from tinygrad import Context

from infurnace.benchmark import _measure
from infurnace.chat import prompt_ids as _prompt_ids
from infurnace.loader import load_model as _build


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", default="models/Qwen3-4B-Q4_K_M.gguf")
  parser.add_argument("--device", default="NV")
  parser.add_argument("--context", type=int, default=2048)
  parser.add_argument("--tokens", type=int, default=20)
  args = parser.parse_args()
  if args.tokens < 2: parser.error("--tokens must be at least 2 for decode timing")
  model, tokenizer = _build(args.model, args.context, args.device, nv_quant=False)
  ids = _prompt_ids(tokenizer, "Hello, explain what a GPU does.")

  def direct_generate(tokens, count):
    stream = model.transformer.generate(list(tokens))
    try:
      for _ in range(min(count, model.config.context_length-len(tokens))):
        with Context(DEV=model.device): token = next(stream)
        yield token
    finally: stream.close()

  direct = SimpleNamespace(reset=model.reset, generate=direct_generate)
  _measure(model, ids, args.tokens)
  _measure(model, ids, args.tokens)
  runs = []
  for _ in range(3):
    runs.append({"wrapper": _measure(model, ids, args.tokens), "direct_same_transformer": _measure(direct, ids, args.tokens)})
  print(json.dumps({"device": model.device, "context": model.config.context_length, "runs": runs,
                    "notes": "Same weights/backend and reset prefill. Measures wrapper overhead only; EOS ignored."}, indent=2))


if __name__ == "__main__": main()
