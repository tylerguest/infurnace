"""infurnace command line: `serve` keeps a model warm on the GPU; `run`, `ps`, `stop` and `launch`
talk to it (starting it when needed); `pull`, `list` and `rm` manage downloads.

The original flags still work: `--prompt`, `--serve [PORT]`, `--benchmark` and in-process chat (`--chat`)."""
from __future__ import annotations
import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Iterator
from tinygrad.helpers import GlobalCounters, Timing, profile_marker
from infurnace import loader
from infurnace.device import free_vram, nvidia, select_device

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
  if args.no_dflash or args.benchmark is not None: return None, False
  draft = loader.dflash_draft(args.model)  # a download URL for named models
  return (draft, True) if isinstance(draft, str) or (draft is not None and draft.is_file()) else (None, False)

def _dflash_block(dflash) -> int:
  # Parallel verification checks a whole trained block; the sequential reference checks 4 rows.
  return dflash.draft.config.block_size if dflash.verifier == 'parallel' else 4

def generator(model, tokenizer, dflash=None, temperature: float = 0.0) -> Callable[[list[int], int], Iterator[int]]:
  """generate(ids, budget): plain decoding, or DFlash speculative decoding, at `temperature`."""
  if dflash is None: return lambda ids, budget: model.generate(ids, budget, temperature=temperature)
  eos, block = tuple(i for i in (tokenizer.eos_id, tokenizer.eot_id) if i is not None), _dflash_block(dflash)
  return lambda ids, budget: dflash.stream(ids, max_new_tokens=budget, block_size=block, eos_token_ids=eos, temperature=temperature)

def _warm(model, dflash=None, prompt_graph: bool = True) -> None:
  # DFlash requests never use plain generation's graphs, so it captures its own.
  if dflash is not None: dflash.warmup(_dflash_block(dflash), prompt_graph=prompt_graph)
  else: model.warmup(prompt_graph=prompt_graph)

def warmup(model, dflash=None, prompt_graph: bool = True) -> None:
  print(f"Warming up {model.device} (first startup may take a while)...", file=sys.stderr, flush=True)
  start = time.perf_counter()
  _warm(model, dflash, prompt_graph)
  print(f"Ready in {time.perf_counter() - start:.1f}s.", file=sys.stderr, flush=True)

def capture_prompt_graph(model) -> None:
  """The server's first idle moment: the prompt-chunk graph its warmup left out (a request waits for it)."""
  start = time.perf_counter()
  if model.capture_prompt_graph():
    print(f"Prompt reading graph ready in {time.perf_counter() - start:.1f}s; {GlobalCounters.mem_used / 1e9:.2f} GB on the GPU.",
          file=sys.stderr, flush=True)

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
  if draft is not None and not (automatic and not nvidia(model.device)):  # the CPU reference is slower than plain
    dflash = loader.load_dflash(model, loader.fetch_model(draft))
    print(f"DFlash speculative decoding{' on (--no-dflash turns it off)' if automatic else ''}.", file=sys.stderr, flush=True)
  generate, context = generator(model, tokenizer, dflash, args.temperature), model.config.context_length

  if args.prompt is not None:
    reply(generate, tokenizer, encode(tokenizer, args.prompt), args.max_tokens or PROMPT_MAX_TOKENS, context)
    return 0
  if args.serve is not None:
    from infurnace.serve import serve
    # Warmup runs on a thread once the server listens (requests wait for it); the prompt graph follows in its first idle moment.
    block = _dflash_block(dflash) if dflash is not None else 16
    serve(model, tokenizer, args.serve, args.host, enable_thinking=not args.no_thinking, dflash=dflash, block_size=block,
          warmup=None if args.no_warmup else lambda: warmup(model, dflash, prompt_graph=False),
          idle=None if args.no_warmup else lambda: capture_prompt_graph(model), name=_model_name(args.model),
          info=loader.model_info(_model_name(args.model), loader.fetch_model(args.model), dflash is not None), installed=loader.installed)
    return 0
  window, thinking = min(args.max_context, context), not args.no_thinking
  if args.no_warmup: return chat(generate, tokenizer, window, args.max_tokens, thinking)
  print(f"Warming up {model.device} in the background; you can type your first message now.", file=sys.stderr, flush=True)
  background = BackgroundWarmup(model, dflash)
  try: return chat(generate, tokenizer, window, args.max_tokens, thinking, ready=background.wait)
  finally: background.thread.join()  # never exit while the warmup thread is using the GPU

def main(argv: list[str] | None = None) -> int:
  argv = sys.argv[1:] if argv is None else argv
  from infurnace.client import ServerError
  try:
    if not argv or argv[0] in COMMANDS: return command(argv or ["run"])
    return run(parse_args(argv))
  except ServerError as exc:
    print(f"infurnace: {exc}", file=sys.stderr)
    return 1
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

COMMANDS = ("serve", "run", "pull", "list", "ls", "rm", "ps", "stop", "launch")

def _server_options(parser: argparse.ArgumentParser) -> None:
  parser.add_argument("--max-context", type=_positive, help="context window in tokens (default: the most that fits in free GPU memory)")
  parser.add_argument("--no-dflash", action="store_true", help="decode without the DFlash draft")
  parser.add_argument("--device", help="override automatic device selection, e.g. NV or CPU")

def command_parser() -> argparse.ArgumentParser:
  ap = argparse.ArgumentParser(prog="infurnace", description="Run local LLMs fast on your GPU with tinygrad. "
                               "Without a command: run the default model.")
  sub = ap.add_subparsers(dest="command", required=True, metavar="COMMAND")
  default = next(iter(loader.MODELS))
  serve = sub.add_parser("serve", help="start the server: keeps the model warm; OpenAI-compatible API at /v1, infurnace API at /api")
  serve.add_argument("--model", "-m", default=default, help=f"model name ({', '.join(loader.MODELS)}) or a local GGUF file")
  serve.add_argument("--host", help="bind address (default from INFURNACE_HOST, else 127.0.0.1)")
  serve.add_argument("--port", type=int, help="port (default from INFURNACE_HOST, else 8000)")
  _server_options(serve)
  serve.add_argument("--dflash", type=Path, metavar="GGUF", help="DFlash draft file (the named model's is used automatically)")
  serve.add_argument("--no-thinking", action="store_true", help="disable thinking unless a request asks for it")
  serve.add_argument("--no-warmup", action="store_true", help="skip the warmup (the first replies are then slower)")
  run_ = sub.add_parser("run", help="chat with a model, or answer one prompt (an argument or piped input)")
  run_.add_argument("model", nargs="?", help=f"model name (default: the server's, else {default})")
  run_.add_argument("prompt", nargs="*", help="answer this and exit")
  thinking = run_.add_mutually_exclusive_group()
  thinking.add_argument("--think", dest="think", action="store_const", const=True, help="think before answering")
  thinking.add_argument("--nothink", dest="think", action="store_const", const=False, help="answer without thinking")
  run_.add_argument("--temperature", type=_temperature, help="sampling temperature (default 0: greedy)")
  run_.add_argument("--max-tokens", type=_positive, help="reply length cap, including thinking")
  _server_options(run_)
  sub.add_parser("pull", help="download a model (and its DFlash draft)").add_argument("model", help=f"one of {', '.join(loader.MODELS)}")
  sub.add_parser("list", aliases=["ls"], help="list downloaded models")
  sub.add_parser("rm", help="delete downloaded models").add_argument("models", nargs="+")
  sub.add_parser("ps", help="show the running server and its model")
  sub.add_parser("stop", help="stop the server (frees the GPU)")
  launch = sub.add_parser("launch", help="point a coding agent at the server and start it (pi)")
  launch.add_argument("integration", choices=["pi"])
  launch.add_argument("--model", "-m", help="model for a server this starts")
  launch.add_argument("--config", action="store_true", help="only write the agent's configuration")
  launch.add_argument("extra", nargs=argparse.REMAINDER, help="arguments for the agent, after --")
  _server_options(launch)
  return ap

def _forward(args: argparse.Namespace) -> list[str]:
  """Options of run/launch that shape a server they start."""
  return ((["--max-context", str(args.max_context)] if args.max_context else []) + (["--no-dflash"] if args.no_dflash else [])
          + (["--device", args.device] if args.device else []))

def _looks_like_model(value: str) -> bool:
  """`run`'s first argument names a model (a catalog name, name:tag or a GGUF file); anything else starts the prompt."""
  if "://" in value: return False  # a URL is prompt text, never a model to download
  return value in loader.MODELS or value.endswith(".gguf") or (" " not in value and ":" in value and not value.endswith(":"))

def _check_model(model: str | None) -> None:
  """A model `run` or `launch` asks for must be a catalog name or a GGUF file: say so before any server is involved."""
  if model is None or model in loader.MODELS or model.endswith(".gguf"): return
  raise ValueError(f"unknown model {model!r}; available: {', '.join(loader.MODELS)}, or a path to a GGUF file")

def _model_name(model: str) -> str:
  return model if model in loader.MODELS else Path(model).stem

def serve_command(args: argparse.Namespace) -> int:
  """Load the model and serve it; warmup runs on a thread after the server starts listening."""
  from infurnace import client
  from infurnace.serve import serve
  hostname, port = client.host().rsplit(":", 1)
  hostname, port = args.host or hostname, args.port or int(port)
  if not 1 <= port <= 65535: raise ValueError("port must be between 1 and 65535")
  args.benchmark = None
  draft, automatic = _dflash_draft(args)
  path = loader.fetch_model(args.model)
  device = select_device(args.device)
  draft_path = loader.fetch_model(draft) if draft is not None and not (automatic and not nvidia(device)) else None
  if args.max_context is None:
    free = free_vram(device)
    args.max_context = loader.fit_context(path, draft_path, free) if free else 8192
    print(f"Context: {args.max_context} tokens" + (f" (fits in {free / 1e9:.1f} GB of free GPU memory; --max-context overrides)"
                                                    if free else " (default; --max-context overrides)"), file=sys.stderr, flush=True)
  model, tokenizer = loader.load_model(path, args.max_context, device)
  dflash = loader.load_dflash(model, draft_path) if draft_path is not None else None
  def warm():
    start = time.perf_counter()
    print(f"Warming up {model.device}...", file=sys.stderr, flush=True)
    _warm(model, dflash, prompt_graph=False)
    print(f"Ready in {time.perf_counter() - start:.1f}s; {GlobalCounters.mem_used / 1e9:.2f} GB on the GPU.", file=sys.stderr, flush=True)
  name = _model_name(args.model)
  serve(model, tokenizer, port, hostname, enable_thinking=not args.no_thinking, dflash=dflash,
        block_size=_dflash_block(dflash) if dflash is not None else 16, name=name,
        warmup=None if args.no_warmup else warm, idle=None if args.no_warmup else lambda: capture_prompt_graph(model),
        info=loader.model_info(name, path, dflash is not None), installed=loader.installed)
  return 0

def command(argv: list[str]) -> int:
  from infurnace import client
  args = command_parser().parse_args(argv)
  if args.command == "serve": return serve_command(args)
  if args.command == "run":
    if args.model is not None and not _looks_like_model(args.model): args.model, args.prompt = None, [args.model, *args.prompt]
    _check_model(args.model)
    options = {**({"temperature": args.temperature} if args.temperature is not None else {}),
               **({"num_predict": args.max_tokens} if args.max_tokens else {})}
    status = client.run(args.model, " ".join(args.prompt) or None, args.think, options, _forward(args))
    if sys.stdin.isatty(): print("(The server keeps the model on the GPU; `infurnace stop` frees it.)", file=sys.stderr)
    return status
  if args.command == "pull":
    for path in loader.pull(args.model): print(f"{args.model}: {path}")
    return 0
  if args.command in ("list", "ls"):
    models = loader.installed()
    if not models: print(f"No models downloaded yet. `infurnace pull {next(iter(loader.MODELS))}` downloads one.")
    for m in models:
      print(f"{m['name']:<16} {m['details']['quantization_level']:<8} {m['size'] / 1e9:5.1f} GB  "
            f"{'with DFlash draft' if m['draft'] else ''}")
    return 0
  if args.command == "rm":
    for name in args.models:
      removed = loader.remove(name)
      print(f"Deleted {name} ({len(removed)} file{'s' if len(removed) != 1 else ''})." if removed else f"{name} has no downloads in user storage.")
    return 0
  if args.command == "ps": return client.ps()
  if args.command == "stop": return client.stop()
  if args.command == "launch":
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    _check_model(args.model)
    return client.launch(args.integration, args.model, _forward(args), extra, configure_only=args.config)
  raise AssertionError(args.command)

if __name__ == "__main__":
  raise SystemExit(main())
