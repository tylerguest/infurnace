"""Plain-decoding benchmark: one cold run, two capture passes, then three warm runs, as JSON.

Run: .venv/bin/python benchmarks/decode.py --tokens 32 [--prompt-tokens 512 | --prompt-file prompt.txt]
For per-step timing use `infurnace --benchmark [PROMPT,]COUNT`. compare.py and wrapper.py share the
deterministic prompt and measurement helpers below.
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path
from tinygrad.helpers import GlobalCounters

DEFAULT_PROMPT_TOKENS = 512
_PROSE = (
  "A local language model processes a prompt before generating an answer. "
  "During prefill, it stores attention keys and values for the input tokens. "
  "During decoding, each new token uses the model weights and the growing cache. "
  "Performance depends on memory bandwidth, kernel efficiency, and active context length. "
  "A useful benchmark separates startup, first-token latency, and warm decoding, "
  "and records its model, hardware, input length, and output length.\n\n"
)

def benchmark_prompt_ids(tokenizer, prompt_tokens: int = DEFAULT_PROMPT_TOKENS, prompt_text: str | None = None) -> list[int]:
  """Exact-length repeated prose, or an unmodified raw-text prompt (no chat template).

  Synthetic length includes optional BOS. File/text prompts are never truncated.
  """
  if type(prompt_tokens) is not int or prompt_tokens <= 0: raise ValueError("prompt token count must be positive")
  prefix = [tokenizer.bos_id] if tokenizer.bos_id is not None else []
  if prompt_text is not None:
    encoded = tokenizer.encode(prompt_text)
    if not encoded: raise ValueError("benchmark prompt text must contain tokens")
    return prefix + encoded
  encoded = tokenizer.encode(_PROSE)
  if not encoded: raise ValueError("tokenizer produced an empty benchmark passage")
  length = prompt_tokens - len(prefix)
  return prefix + (encoded * ((length + len(encoded) - 1) // len(encoded)))[:length]

def validate_benchmark_context(prompt_tokens: int, output_tokens: int, context: int) -> None:
  if prompt_tokens <= 0 or output_tokens <= 0: raise ValueError("benchmark prompt and output must be nonempty")
  if prompt_tokens + output_tokens > context:
    raise ValueError(f"benchmark needs {prompt_tokens + output_tokens} context tokens "
                     f"({prompt_tokens} prompt + {output_tokens} output), but capacity is {context}")

def measure(model, ids: list[int], count: int) -> dict:
  """One synchronized generation: first-token latency and decode rate."""
  model.reset()  # include full prefill, not a cached prompt
  stream, start, first, generated = model.generate(ids, count), time.perf_counter(), None, 0
  try:
    for _ in stream:  # upstream .item() synchronizes every generated token
      if first is None: first = time.perf_counter()
      generated += 1
  finally: stream.close()
  end = time.perf_counter()
  ttft = None if first is None else first - start
  decode = 0 if first is None else end - first
  return {"prompt_tokens": len(ids), "generated_tokens": generated, "ttft_seconds": ttft, "total_seconds": end - start,
          "prefill_tokens_per_second": len(ids) / ttft if ttft else None,
          "decode_tokens_per_second": (generated - 1) / decode if generated > 1 and decode > 0 else None}

def report(model, tokenizer, count: int, load_seconds: float, prompt_tokens: int = DEFAULT_PROMPT_TOKENS,
           prompt_text: str | None = None) -> dict:
  ids = benchmark_prompt_ids(tokenizer, prompt_tokens, prompt_text)
  validate_benchmark_context(len(ids), count, model.config.context_length)
  cold = measure(model, ids, count)
  measure(model, ids, count)
  warm = [measure(model, ids, count) for _ in range(3)]
  rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  return {
    "model": model.metadata.get("general.name", "unknown"), "device": model.device, "context": model.config.context_length,
    "load_seconds": load_seconds, "prompt_tokens": len(ids), "prompt_ids": ids,
    "prompt_source": "raw text" if prompt_text is not None else "synthetic repeated prose",
    "decode_input_positions": [len(ids), len(ids) + count - 2] if count > 1 else None,
    **{name: getattr(model, name, default) for name, default in (
      ("nv_quant_linears", 0), ("nv_shared_q8_blocks", 0), ("nv_prefill", False), ("nv_prepared_weights", False),
      ("nv_prepared_weight_layout", "gguf"), ("nv_attention", False), ("nv_fused_attention", False))},
    "cold": cold, "warm_runs": warm,
    "host_peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
    "tinygrad_current_allocated_bytes": GlobalCounters.mem_used,
    "notes": "Prefill rate includes first-token work; cold includes compilation. GPU peak is not measured. EOS is ignored.",
  }

def main() -> int:
  from infurnace import loader
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--model", default=next(iter(loader.MODELS)), help="model name or GGUF path")
  ap.add_argument("--tokens", type=int, default=20, help="output tokens per run (default 20)")
  prompt = ap.add_mutually_exclusive_group()
  prompt.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS, help="synthetic prompt length including BOS")
  prompt.add_argument("--prompt-file", type=Path, help="UTF-8 raw prompt file; never truncated")
  ap.add_argument("--max-context", type=int, default=2048)
  ap.add_argument("--device")
  args = ap.parse_args()
  start = time.perf_counter()
  model, tokenizer = loader.load_model(loader.fetch_model(args.model), args.max_context, args.device)
  text = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file is not None else None
  print(json.dumps(report(model, tokenizer, args.tokens, time.perf_counter() - start, args.prompt_tokens, text), indent=2))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
