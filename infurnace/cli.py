"""infurnace command line: chat (default), one raw prompt, an HTTP server, or a decode benchmark."""
from __future__ import annotations
import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Iterator
from tinygrad.helpers import GlobalCounters, Timing, profile_marker
from infurnace import loader

PROMPT_MAX_TOKENS = 64

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  ap = argparse.ArgumentParser(prog="infurnace", description="Run one local LLM fast on your GPU with tinygrad.")
  mode = ap.add_mutually_exclusive_group()
  mode.add_argument("--chat", action="store_true", help="interactive chat (default)")
  mode.add_argument("--prompt", help="generate from one raw prompt (no chat template)")
  mode.add_argument("--serve", nargs="?", type=int, const=8000, metavar="PORT", help="OpenAI-compatible HTTP server (default port 8000)")
  mode.add_argument("--benchmark", nargs="?", type=_benchmark_spec, const=(512, 20), metavar="[PROMPT,]COUNT",
                    help="time plain decoding per step: COUNT outputs (default 20) after a PROMPT-token prompt (default 512)")
  ap.add_argument("--model", "-m", default=next(iter(loader.MODELS)),
                  help=f"model name ({', '.join(loader.MODELS)}; downloaded on first use) or a path to a local GGUF file")
  ap.add_argument("--max-tokens", type=_positive, help="reply length cap, including thinking (default: chat uses the remaining "
                                                       f"context, --prompt {PROMPT_MAX_TOKENS})")
  ap.add_argument("--max-context", type=_positive, default=8192, help="context window in tokens (default 8192)")
  ap.add_argument("--temperature", type=_temperature, default=0.0, help="sampling temperature (default 0: greedy); works with DFlash")
  ap.add_argument("--no-thinking", action="store_true", help="disable thinking in chat formatting")
  ap.add_argument("--no-warmup", action="store_true", help="skip the startup warmup (the first reply is then slower)")
  ap.add_argument("--device", help="override automatic device selection, e.g. NV or CPU")
  ap.add_argument("--host", default="127.0.0.1", help="HTTP bind address for --serve (default: loopback)")
  speculation = ap.add_mutually_exclusive_group()
  speculation.add_argument("--dflash", type=Path, metavar="GGUF", help="DFlash draft for a Qwen3.5 model "
                                                                       "(used automatically when installed; or INFURNACE_DFLASH_DRAFT)")
  speculation.add_argument("--no-dflash", action="store_true", help="decode without the automatic DFlash draft")
  args = ap.parse_args(argv)

  if args.serve is not None:
    if not 1 <= args.serve <= 65535: ap.error("--serve port must be between 1 and 65535")
    if args.dflash: ap.error("--serve supports plain decoding only")
    if args.max_tokens is not None: ap.error("--serve takes max_tokens per request, not --max-tokens")
  if args.benchmark is not None:
    if args.dflash: ap.error("--benchmark measures plain decoding; use benchmarks/dflash.py for DFlash")
    args.max_context = max(args.max_context, sum(args.benchmark))  # room for the whole benchmark
  if args.no_thinking and (args.prompt is not None or args.benchmark is not None): ap.error("--no-thinking applies only to chat")
  if args.dflash is not None and not args.dflash.is_file(): ap.error("--dflash must name a local draft GGUF file")
  return args

def _temperature(value: str) -> float:
  try: temperature = float(value)
  except ValueError: temperature = -1.0
  if not 0 <= temperature < float('inf'): raise argparse.ArgumentTypeError("must be a finite number >= 0")
  return temperature

def _positive(value: str) -> int:
  if not value.isdigit() or int(value) <= 0: raise argparse.ArgumentTypeError("must be a positive integer")
  return int(value)

def _benchmark_spec(value: str) -> tuple[int, int]:
  parts = value.split(",")
  if len(parts) > 2 or not all(p.isdigit() and int(p) > 0 for p in parts):
    raise argparse.ArgumentTypeError("benchmark must be COUNT or PROMPT,COUNT with positive integers")
  return (int(parts[0]), int(parts[1])) if len(parts) == 2 else (512, int(parts[0]))

def benchmark(model, tokenizer, prompt_tokens: int, count: int, warmup: bool = True) -> int:
  """Per-step timing like tinygrad's --benchmark: the prefill step, then each decode step, with
  tinygrad's traffic estimate; long runs show only the first and last 20 steps."""
  ids = tokenizer.encode(" ".join(map(str, range(prompt_tokens))))[:prompt_tokens]
  if warmup: model.warmup()
  model.reset()
  keep, decode, text = 20, tokenizer.stream_decoder(), [""]
  stream = model.generate(ids, count)
  try:
    for i in range(count):
      profile_marker(f"decode @ {i-1}" if i else "prefill")
      if i == keep + 1 and count > 2 * keep: print("...")
      GlobalCounters.reset()
      with Timing(prefix=f"prefill {len(ids):5d}: " if i == 0 else f"decode {i-1:6d}: ", enabled=i <= keep or i >= count - keep,
                  on_exit=lambda ns: f", {(len(ids) if i == 0 else 1) * 1e9 / ns:8.2f} tok/s, {GlobalCounters.global_mem / ns:7.2f} GB/s, "
                                     f"{GlobalCounters.global_mem // 1000000}/{GlobalCounters.mem_used // 1000000} MB  --  "
                                     + text[0].replace("\n", "\\n")):
        text[0] += decode(next(stream))  # upstream .item() synchronizes each step
  finally: stream.close()
  return 0

def _dflash_draft(args: argparse.Namespace) -> tuple[Path | None, bool]:
  """The DFlash draft to use and whether it was picked automatically. Chat and --prompt on a
  Qwen3.5 model use an installed draft unless told otherwise."""
  if args.dflash is not None: return args.dflash, False
  if args.no_dflash or args.serve is not None or args.benchmark is not None: return None, False
  draft = loader.dflash_draft(args.model)  # a download URL for named models
  return (draft, True) if isinstance(draft, str) or (draft is not None and draft.is_file()) else (None, False)

def _dflash_block(dflash) -> int:
  # Parallel verification checks a whole trained block; the sequential reference checks 4 rows.
  return dflash.draft.config.block_size if dflash.verifier == 'parallel' else 4

def generator(model, tokenizer, dflash=None, temperature: float = 0.0) -> Callable[[list[int], int], Iterator[int]]:
  """generate(ids, budget): plain decoding, or DFlash speculative decoding (created for `temperature`)."""
  if dflash is None: return lambda ids, budget: model.generate(ids, budget, temperature=temperature)
  eos, block = tuple(i for i in (tokenizer.eos_id, tokenizer.eot_id) if i is not None), _dflash_block(dflash)
  return lambda ids, budget: dflash.stream(ids, max_new_tokens=budget, block_size=block, eos_token_ids=eos)

def _warm(model, dflash=None) -> None:
  # DFlash requests never use plain generation's graphs, so it captures its own.
  if dflash is not None: dflash.warmup(_dflash_block(dflash))
  else: model.warmup()

def warmup(model, dflash=None) -> None:
  print(f"Warming up {model.device} (first startup may take a while)...", file=sys.stderr, flush=True)
  start = time.perf_counter()
  _warm(model, dflash)
  print(f"Ready in {time.perf_counter() - start:.1f}s.", file=sys.stderr, flush=True)

class BackgroundWarmup:
  """Warm up on a thread while the user types the first message. tinygrad is not thread-safe, but
  nothing else touches it meanwhile: the main thread only reads the keyboard until it calls wait()."""
  def __init__(self, model, dflash=None):
    self.failure: BaseException | None = None
    def work():
      try: _warm(model, dflash)
      except BaseException as exc: self.failure = exc
    self.thread = threading.Thread(target=work, name="infurnace-warmup", daemon=True)
    self.thread.start()

  def wait(self) -> None:
    """Block until warmup has finished; re-raise its error, if any."""
    if self.thread.is_alive():
      print("Finishing warmup...", file=sys.stderr, flush=True)
      self.thread.join()
    if self.failure is not None:
      failure, self.failure = self.failure, None
      raise failure

def encode(tokenizer, text: str) -> list[int]:
  return ([tokenizer.bos_id] if tokenizer.bos_id is not None else []) + tokenizer.encode(text)

def reply(generate, tokenizer, ids: list[int], max_tokens: int | None, context: int) -> str:
  """Print a reply as it is generated and return its text. Without max_tokens it may fill the context."""
  budget = context - len(ids) if max_tokens is None else max_tokens
  decode, text, generated, ended = tokenizer.stream_decoder(), "", 0, False
  stream = generate(ids, budget)
  try:
    for token in stream:
      generated += 1
      if (ended := tokenizer.is_end(token)): break
      print(piece := decode(token), end="", flush=True)
      text += piece
  finally:
    stream.close()
    print(piece := decode(None), flush=True)
    text += piece
  if not ended and generated >= context - len(ids):
    print("[Response stopped: context limit reached. Increase --max-context for longer answers.]", file=sys.stderr)
  elif not ended:
    print(f"[Response stopped: {budget}-token limit reached (includes thinking). Increase --max-tokens or use --no-thinking.]",
          file=sys.stderr)
  return text

def chat(generate, tokenizer, context: int, max_tokens: int | None, thinking: bool = True, ready: Callable[[], None] | None = None) -> int:
  """Interactive chat. When a new message would not fit, the oldest turns are dropped.
  `ready()` is called before each reply (it waits for a background warmup)."""
  history: list[dict] = []
  reserve = min(max_tokens or 1024, max(1, context // 2))  # room left for the reply
  while True:
    try: message = {"role": "user", "content": input(">>> ")}
    except EOFError: return 0
    retained = list(history)
    while True:
      ids = encode(tokenizer, tokenizer.render_chat(retained + [message], enable_thinking=thinking))
      if len(ids) < context - reserve or not retained: break
      retained = retained[2:]  # history is user/assistant pairs; keep the newest
    if len(ids) >= context:
      print("Message exceeds context even without history; shorten it or increase --max-context. History was not truncated.",
            file=sys.stderr)
      continue
    if (dropped := (len(history) - len(retained)) // 2):
      print(f"[Context window: discarded {dropped} oldest turn(s); earlier details are no longer remembered.]", file=sys.stderr)
    if ready is not None: ready()
    history = retained + [message, {"role": "assistant", "content": reply(generate, tokenizer, ids, max_tokens, context)}]

def run(args: argparse.Namespace) -> int:
  draft, automatic = _dflash_draft(args)
  model, tokenizer = loader.load_model(loader.fetch_model(args.model), args.max_context, args.device)
  if args.benchmark is not None: return benchmark(model, tokenizer, *args.benchmark, warmup=not args.no_warmup)

  dflash = None
  if draft is not None and not (automatic and model.device.split(':')[0] != 'NV'):  # the CPU reference is slower than plain
    dflash = loader.load_dflash(model, loader.fetch_model(draft), temperature=args.temperature)
    print(f"DFlash speculative decoding{' on (--no-dflash turns it off)' if automatic else ''}.", file=sys.stderr, flush=True)
  generate, context = generator(model, tokenizer, dflash, args.temperature), model.config.context_length

  if args.prompt is not None:
    reply(generate, tokenizer, encode(tokenizer, args.prompt), args.max_tokens or PROMPT_MAX_TOKENS, context)
    return 0
  if args.serve is not None:
    from infurnace.serve import serve
    if not args.no_warmup: warmup(model, dflash)  # once per server start, before it listens
    serve(model, tokenizer, args.serve, args.host, enable_thinking=not args.no_thinking)
    return 0
  window, thinking = min(args.max_context, context), not args.no_thinking
  if args.no_warmup: return chat(generate, tokenizer, window, args.max_tokens, thinking)
  print(f"Warming up {model.device} in the background; you can type your first message now.", file=sys.stderr, flush=True)
  background = BackgroundWarmup(model, dflash)
  try: return chat(generate, tokenizer, window, args.max_tokens, thinking, ready=background.wait)
  finally: background.thread.join()  # never exit while the warmup thread is using the GPU

def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  try:
    return run(args)
  except KeyboardInterrupt:
    print("\nInterrupted.", file=sys.stderr)
    return 130
  except RuntimeError as exc:
    if 'signal wait timed out' not in str(exc) and 'hang detected' not in str(exc): raise
    print(f"infurnace: the GPU stopped responding ({exc}). Restart infurnace; if it repeats, try --no-dflash.", file=sys.stderr)
    return 1
  except MemoryError:
    print("infurnace: out of memory. Stop other model processes or reduce --max-context.", file=sys.stderr)
    return 1
  except FileNotFoundError as exc:
    print(f"infurnace: file not found: {exc.filename}\nPass --model with a name ({', '.join(loader.MODELS)}) or a path to a GGUF file.",
          file=sys.stderr)
    return 1
  except (ValueError, OSError) as exc:
    print(f"infurnace: {exc}", file=sys.stderr)
    return 1

if __name__ == "__main__":
  raise SystemExit(main())
