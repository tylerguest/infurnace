"""Benchmarks of plain decoding: deterministic prompts, a per-step report and a JSON summary."""
import json
import resource
import sys
import time
from tinygrad.helpers import GlobalCounters, Timing, profile_marker

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
  length = prompt_tokens-len(prefix)
  return prefix + (encoded*((length+len(encoded)-1)//len(encoded)))[:length]


def validate_benchmark_context(prompt_tokens: int, output_tokens: int, context: int) -> None:
  if prompt_tokens <= 0 or output_tokens <= 0: raise ValueError("benchmark prompt and output must be nonempty")
  if prompt_tokens + output_tokens > context:
    raise ValueError(f"benchmark needs {prompt_tokens + output_tokens} context tokens "
                     f"({prompt_tokens} prompt + {output_tokens} output), but capacity is {context}")


def parse_spec(value: str) -> tuple[int | None, int]:
  """`COUNT` or `PROMPT,COUNT` (prompt tokens, output tokens), both positive."""
  parts = value.split(",")
  if len(parts) not in (1, 2) or not all(p.strip().isdigit() and int(p) > 0 for p in parts):
    raise ValueError("benchmark must be COUNT or PROMPT,COUNT with positive integers")
  return (None, int(parts[0])) if len(parts) == 1 else (int(parts[0]), int(parts[1]))


def _measure(model, ids: list[int], count: int) -> dict:
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


def run_steps(model, tokenizer, count: int, prompt_tokens: int = DEFAULT_PROMPT_TOKENS, prompt_text: str | None = None) -> int:
  """Warm up, then print every prefill/decode step's time and tinygrad's traffic estimate."""
  ids = benchmark_prompt_ids(tokenizer, prompt_tokens, prompt_text)
  validate_benchmark_context(len(ids), count, model.config.context_length)
  model.warmup()
  model.reset()
  print(f"device={model.device}, context={model.config.context_length}, prompt={len(ids)}, outputs={count}")
  print("GB/s and traffic MB are tinygrad estimates; allocated MB is current allocation, not GPU peak.")
  stream, text, decoded, decode_seconds = model.generate(ids, count), tokenizer.stream_decoder(), 0, 0.0
  shown = ""
  try:
    for step in range(count):
      prefill = step == 0
      profile_marker(f"{'prefill' if prefill else 'decode'} @ {len(ids) if prefill else len(ids) + step - 1}")
      GlobalCounters.reset()
      timer = Timing(enabled=False)
      with timer: token = next(stream)  # upstream .item() synchronizes
      seconds, traffic, allocated = timer.et * 1e-9, GlobalCounters.global_mem, GlobalCounters.mem_used
      shown += text(token)
      if not prefill: decoded, decode_seconds = decoded + 1, decode_seconds + seconds
      label = f"prefill {len(ids):5d}" if prefill else f"decode  {step - 1:5d}"
      rate = len(ids) / seconds if prefill else 1 / seconds
      print(f"{label}: {seconds * 1000:8.2f} ms, {rate:8.2f} tok/s, {traffic / seconds / 1e9:7.2f} est GB/s, "
            f"{traffic / 1e6:.2f}/{allocated / 1e6:.2f} MB, active={len(ids) + step}  --  " + shown.replace("\n", "\\n"), flush=True)
  finally: stream.close()
  if decoded: print(f"decode aggregate: {decoded / decode_seconds:.2f} tok/s ({decoded} steps, excluding prefill)")
  return 0


def run_json(model, tokenizer, count: int, load_seconds: float, prompt_tokens: int = DEFAULT_PROMPT_TOKENS,
             prompt_text: str | None = None) -> int:
  """Cold run, two capture passes, then three warm runs, as one JSON report."""
  ids = benchmark_prompt_ids(tokenizer, prompt_tokens, prompt_text)
  validate_benchmark_context(len(ids), count, model.config.context_length)
  cold = _measure(model, ids, count)
  _measure(model, ids, count)
  warm = [_measure(model, ids, count) for _ in range(3)]
  rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  print(json.dumps({
    "model": model.metadata.get("general.name", "Qwen3"), "device": model.device, "context": model.config.context_length,
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
  }, indent=2))
  return 0
