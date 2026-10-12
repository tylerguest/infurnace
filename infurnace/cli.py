"""infurnace command line: chat (default), one raw prompt, an HTTP server, or a decode benchmark."""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from infurnace import benchmark, loader
from infurnace.chat import Decoder, run_chat, run_prompt

PROMPT_MAX_TOKENS = 64


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  ap = argparse.ArgumentParser(prog="infurnace", description="Run one local LLM fast on your GPU with tinygrad.")
  mode = ap.add_mutually_exclusive_group()
  mode.add_argument("--chat", action="store_true", help="interactive chat (default)")
  mode.add_argument("--prompt", help="generate from one raw prompt (no chat template)")
  mode.add_argument("--serve", nargs="?", type=int, const=8000, metavar="PORT", help="OpenAI-compatible HTTP server (default port 8000)")
  mode.add_argument("--benchmark", nargs="?", type=_benchmark_spec, const=(None, 20), metavar="[PROMPT,]COUNT",
                    help="time plain decoding: COUNT outputs (default 20) after a PROMPT-token prompt (default 512)")
  ap.add_argument("--model", default=loader.default_model(), help="GGUF model file (default: Qwen3.5-4B, else Qwen3-4B; or INFURNACE_MODEL)")
  ap.add_argument("--max-tokens", type=_positive, help="reply length cap, including thinking (default: chat uses the remaining "
                                                       f"context, --prompt {PROMPT_MAX_TOKENS})")
  ap.add_argument("--max-context", type=_positive, default=2048, help="context window in tokens (default 2048)")
  ap.add_argument("--no-thinking", action="store_true", help="disable thinking in chat formatting")
  ap.add_argument("--no-warmup", action="store_true", help="skip the startup warmup (the first reply is then slower)")
  ap.add_argument("--device", help="override automatic device selection, e.g. NV or CPU")
  ap.add_argument("--host", default="127.0.0.1", help="HTTP bind address for --serve (default: loopback)")
  ap.add_argument("--benchmark-format", choices=("text", "json"), default="text", help="per-step text or a JSON summary")
  speculation = ap.add_mutually_exclusive_group()
  speculation.add_argument("--dflash", type=Path, metavar="GGUF", help="DFlash draft for a Qwen3.5 model "
                                                                       "(used automatically when installed; or INFURNACE_DFLASH_DRAFT)")
  speculation.add_argument("--no-dflash", action="store_true", help="decode without the automatic DFlash draft")
  speculation.add_argument("--eagle3", type=Path, metavar="DIR", help="experimental EAGLE-3 checkpoint for a Qwen3 model (NV)")
  args = ap.parse_args(argv)

  if args.serve is not None:
    if not 1 <= args.serve <= 65535: ap.error("--serve port must be between 1 and 65535")
    if args.dflash or args.eagle3: ap.error("--serve supports plain decoding only")
    if args.max_tokens is not None: ap.error("--serve takes max_tokens per request, not --max-tokens")
  if args.benchmark is not None and (args.dflash or args.eagle3):
    ap.error("--benchmark measures plain decoding; use benchmarks/dflash.py or benchmarks/eagle.py for drafts")
  if args.no_thinking and (args.prompt is not None or args.benchmark is not None): ap.error("--no-thinking applies only to chat")
  if args.dflash is not None and not args.dflash.is_file(): ap.error("--dflash must name a local draft GGUF file")
  if args.eagle3 is not None:
    if not (args.eagle3 / 'config.json').is_file() or not (args.eagle3 / 'model.safetensors').is_file():
      ap.error("--eagle3 directory must contain config.json and model.safetensors")
    if args.device is not None and args.device.split(':')[0] != 'NV': ap.error("--eagle3 requires an NV device")
  return args


def _positive(value: str) -> int:
  if not value.isdigit() or int(value) <= 0: raise argparse.ArgumentTypeError("must be a positive integer")
  return int(value)


def _benchmark_spec(value: str) -> tuple[int | None, int]:
  try: return benchmark.parse_spec(value)
  except ValueError as exc: raise argparse.ArgumentTypeError(str(exc)) from None


def _dflash_draft(args: argparse.Namespace) -> tuple[Path | None, bool]:
  """The DFlash draft to use and whether it was picked automatically. Chat and --prompt on a
  Qwen3.5 model use an installed draft unless told otherwise."""
  if args.dflash is not None: return args.dflash, False
  if args.no_dflash or args.eagle3 or args.serve is not None or args.benchmark is not None: return None, False
  draft = loader.default_dflash_draft()
  return (draft, True) if draft is not None and draft.is_file() and loader.is_qwen35(args.model) else (None, False)


def run(args: argparse.Namespace) -> int:
  draft, automatic = _dflash_draft(args)
  start = time.perf_counter()
  model, tokenizer = loader.load_model(args.model, args.max_context, args.device)
  load_seconds = time.perf_counter() - start

  if args.benchmark is not None:
    prompt_tokens, count = args.benchmark
    prompt_tokens = prompt_tokens or benchmark.DEFAULT_PROMPT_TOKENS
    if args.benchmark_format == "json": return benchmark.run_json(model, tokenizer, count, load_seconds, prompt_tokens=prompt_tokens)
    return benchmark.run_steps(model, tokenizer, count, prompt_tokens=prompt_tokens)

  dflash = eagle = None
  if draft is not None and not (automatic and model.device.split(':')[0] != 'NV'):  # the CPU reference is slower than plain
    dflash = loader.load_dflash(model, draft)
    print(f"DFlash speculative decoding{' on (--no-dflash turns it off)' if automatic else ''}.", file=sys.stderr, flush=True)
  if args.eagle3 is not None:
    eagle = loader.load_eagle(model, args.eagle3)
    print("EAGLE-3 speculative decoding (experimental): the first reply captures graphs and can be slow.", file=sys.stderr, flush=True)
  decoder = Decoder(model, tokenizer, dflash=dflash, eagle=eagle)

  if args.serve is not None:
    from infurnace.serve import serve
    if not args.no_warmup: decoder.warmup()
    serve(model, tokenizer, args.serve, args.host, enable_thinking=not args.no_thinking)
    return 0
  if args.prompt is not None: return run_prompt(decoder, tokenizer, args.prompt, args.max_tokens or PROMPT_MAX_TOKENS)
  if not args.no_warmup: decoder.warmup()
  return run_chat(decoder, tokenizer, args.max_tokens, args.max_context, thinking=not args.no_thinking)


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
    print(f"infurnace: file not found: {exc.filename}\nPlace a model at {loader.default_model()} or pass --model /path/to/model.gguf.",
          file=sys.stderr)
    return 1
  except (ValueError, OSError) as exc:
    print(f"infurnace: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
