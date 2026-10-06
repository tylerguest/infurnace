"""Single-model local CLI over tinygrad; no HTTP or scheduler."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time
from tinygrad.helpers import GlobalCounters
from infurnace.device import select_device, supports_nv_quant
from infurnace.model import Qwen3Model
from infurnace.tokenizer import Tokenizer

_MODEL_FILENAME = "Qwen3-4B-Q4_K_M.gguf"

def _default_model() -> str:
  """Prefer explicit configuration, then user storage, then an editable checkout."""
  if path := os.environ.get("INFURNACE_MODEL"): return str(Path(path).expanduser())
  data_home = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))).expanduser()
  installed_model = data_home / "infurnace/models" / _MODEL_FILENAME
  checkout_model = Path(__file__).resolve().parent.parent / "models" / _MODEL_FILENAME
  if installed_model.is_file(): return str(installed_model)
  if checkout_model.is_file(): return str(checkout_model)
  return str(installed_model)
_SINGLE_SHOT_MAX_TOKENS = 64


def _build(path: str, max_context: int, device: str | None, nv_quant: bool | None = None) -> tuple[Qwen3Model, Tokenizer]:
  device = select_device(device)
  enabled = supports_nv_quant(device) if nv_quant is None else nv_quant
  if enabled and not supports_nv_quant(device): raise ValueError("--nv-quant requires NV with the CUDA C renderer")
  kwargs = {"max_context": max_context, "device": device}
  if enabled: kwargs["nv_quant"] = True
  model = Qwen3Model.from_gguf(path, **kwargs)
  return model, Tokenizer.from_gguf(model.metadata)


def _prompt_ids(tokenizer: Tokenizer, prompt: str) -> list[int]:
  ids = tokenizer.encode(prompt)
  return ([tokenizer.bos_id] if tokenizer.bos_id is not None else []) + ids


def _stream(model: Qwen3Model, tokenizer: Tokenizer, ids: list[int], max_tokens: int | None) -> str:
  decoder, reply = tokenizer.stream_decoder(), []
  budget = model.config.context_length - len(ids) if max_tokens is None else max_tokens
  stream = model.generate(ids, budget)
  generated, ended, completed = 0, False, False
  try:
    for token in stream:
      generated += 1
      if tokenizer.is_end(token):
        ended = True
        break
      piece = decoder(token)
      reply.append(piece)
      print(piece, end="", flush=True)
    completed = True
  finally:
    stream.close()
    piece = decoder(None)
    reply.append(piece)
    print(piece, end="", flush=True)
    print()
  if completed and not ended:
    remaining = model.config.context_length - len(ids)
    if generated >= remaining:
      print("[Response stopped: context limit reached. Increase --max-context for longer answers.]", file=sys.stderr)
    else:
      print(f"[Response stopped: {budget}-token limit reached (includes thinking). Increase --max-tokens or use --no-thinking.]",
            file=sys.stderr)
  return "".join(reply)


def _run_single_shot(model: Qwen3Model, tokenizer: Tokenizer, prompt: str, max_tokens: int) -> int:
  _stream(model, tokenizer, _prompt_ids(tokenizer, prompt), max_tokens)
  return 0


def _warmup(model: Qwen3Model) -> None:
  print(f"Warming up {model.device} (first startup may take a while)...", file=sys.stderr, flush=True)
  start = time.perf_counter()
  model.warmup()
  print(f"Ready in {time.perf_counter()-start:.1f}s.", file=sys.stderr, flush=True)


def _run_chat(model: Qwen3Model, tokenizer: Tokenizer, max_tokens: int | None, max_context: int,
              enable_thinking: bool = True) -> int:
  history = []
  while True:
    try: text = input(">>> ")
    except EOFError: return 0
    context = min(max_context, model.config.context_length)
    reserve = min(max_tokens if max_tokens is not None else 1024, max(1, context // 2))
    retained, dropped = list(history), 0
    while True:
      messages = retained + [{"role": "user", "content": text}]
      ids = _prompt_ids(tokenizer, tokenizer.render_chat(messages, enable_thinking=enable_thinking))
      if len(ids) < context - reserve or not retained: break
      # CLI history consists of complete user/assistant pairs. Keep newest turns;
      # upstream token-prefix comparison safely invalidates the changed cache.
      retained = retained[2:]
      dropped += 1
    if len(ids) >= context:
      print("Message exceeds context even without history; shorten it or increase --max-context. History was not truncated.", file=sys.stderr)
      continue
    if dropped:
      print(f"[Context window: discarded {dropped} oldest turn(s); earlier details are no longer remembered.]", file=sys.stderr)
    reply = _stream(model, tokenizer, ids, max_tokens)
    history = messages + [{"role": "assistant", "content": reply}]


def _measure(model: Qwen3Model, ids: list[int], count: int) -> dict:
  model.reset()  # Include full prefill, not a cached prompt.
  stream = model.generate(ids, count)
  start = time.perf_counter()
  first = None
  actual = 0
  try:
    for _ in stream:
      now = time.perf_counter()  # Upstream .item() synchronizes each generated token.
      if first is None: first = now
      actual += 1
  finally: stream.close()
  end = time.perf_counter()
  decode_seconds = end - first if first is not None else 0
  ttft = first - start if first is not None else None
  return {
    "prompt_tokens": len(ids), "generated_tokens": actual,
    "ttft_seconds": ttft, "total_seconds": end-start,
    "prefill_tokens_per_second": len(ids)/ttft if ttft else None,
    "decode_tokens_per_second": (actual-1)/decode_seconds if actual > 1 and decode_seconds > 0 else None,
  }


def _run_benchmark(model: Qwen3Model, tokenizer: Tokenizer, count: int, load_seconds: float = 0.0) -> int:
  ids = _prompt_ids(tokenizer, "Hello, explain what a GPU does.")
  cold = _measure(model, ids, count)
  # Two complete passes capture upstream prefill/decode JIT paths before warm timing.
  _measure(model, ids, count)
  warm = [_measure(model, ids, count) for _ in range(3)]
  rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  print(json.dumps({
    "model": model.metadata.get("general.name", "Qwen3"), "device": model.device,
    "context": model.config.context_length, "load_seconds": load_seconds,
    "nv_quant_linears": getattr(model, "nv_quant_linears", 0),
    "cold": cold, "warm_runs": warm,
    "host_peak_rss_bytes": rss if sys.platform == "darwin" else rss*1024,
    "tinygrad_current_allocated_bytes": GlobalCounters.mem_used,
    "notes": "Prefill rate includes first-token work; cold includes compilation. GPU peak is not measured. EOS is ignored for benchmarking.",
  }, indent=2))
  return 0


def main() -> int:
  ap = argparse.ArgumentParser(description="infurnace local Qwen3 runner")
  ap.add_argument("--model", default=_default_model(), help="local Qwen3 GGUF (or set INFURNACE_MODEL)")
  mode = ap.add_mutually_exclusive_group()
  mode.add_argument("--prompt", help="raw single-shot prompt (not chat formatted)")
  mode.add_argument("--chat", action="store_true", help="interactive chat (default)")
  mode.add_argument("--benchmark", nargs="?", type=int, const=20, metavar="COUNT")
  ap.add_argument("--max-tokens", type=int, help="optional generation cap including thinking (chat: remaining context; raw prompt: 64)")
  ap.add_argument("--max-context", type=int, default=2048)
  ap.add_argument("--device", help="override automatic GPU/CPU selection, e.g. CPU or NV")
  acceleration = ap.add_mutually_exclusive_group()
  acceleration.add_argument("--nv-quant", dest="nv_quant", action="store_true", help="require NV quantized acceleration")
  acceleration.add_argument("--no-nv-quant", dest="nv_quant", action="store_false", help="use generic tinygrad linears")
  ap.set_defaults(nv_quant=None)
  ap.add_argument("--no-warmup", action="store_true", help="skip automatic startup warmup for interactive chat")
  ap.add_argument("--no-thinking", action="store_true", help="disable thinking in chat formatting")
  args = ap.parse_args()
  for name in ("max_context", "max_tokens", "benchmark"):
    value = getattr(args, name)
    if value is not None and value <= 0: ap.error(f"--{name.replace('_', '-')} must be positive")
  if args.no_thinking and (args.prompt is not None or args.benchmark is not None): ap.error("--no-thinking applies only to chat")
  try:
    start = time.perf_counter()
    model, tokenizer = _build(args.model, args.max_context, args.device, args.nv_quant)
    load_seconds = time.perf_counter()-start
    if args.benchmark is not None: return _run_benchmark(model, tokenizer, args.benchmark, load_seconds)
    if args.prompt is not None: return _run_single_shot(model, tokenizer, args.prompt, args.max_tokens or _SINGLE_SHOT_MAX_TOKENS)
    if not args.no_warmup: _warmup(model)
    return _run_chat(model, tokenizer, args.max_tokens, args.max_context, not args.no_thinking)
  except KeyboardInterrupt:
    print("\nInterrupted.", file=sys.stderr)
    return 130
  except MemoryError:
    print("infurnace: out of memory. Stop other model processes or reduce --max-context.", file=sys.stderr)
    return 1
  except FileNotFoundError as exc:
    print(f"infurnace: file not found: {exc.filename}\nPlace the Qwen3 GGUF in {_default_model()} or pass --model /path/to/model.gguf.",
          file=sys.stderr)
    return 1
  except (ValueError, OSError) as exc:
    print(f"infurnace: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
