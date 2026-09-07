"""Offline driver: single-shot, interactive chat, and benchmark.

Load a Qwen3 GGUF, build the model + tokenizer, then run one of three modes
mirroring tinygrad's ``examples/llama.py``:

  * ``--prompt "..."``  — generate once and print the streamed output
  * ``--chat``          — interactive multi-turn chat (Qwen3 chat template)
  * ``--benchmark``     — measure tokens/sec

No HTTP, no batching, no serving state.
"""

from __future__ import annotations

import argparse
import os

from infurnace.gguf import load_gguf
from infurnace.model import Qwen3Model
from infurnace.tokenizer import Tokenizer

_DEFAULT_MODEL = "models/Qwen3-4B-Q4_K_M.gguf"
_SINGLE_SHOT_MAX_TOKENS = 64
_CHAT_MAX_TOKENS = 256


def _build(path: str, max_context: int, device: str | None) -> tuple[Qwen3Model, Tokenizer]:
  """Load the GGUF, build the model, and construct the tokenizer."""
  if device:
    os.environ["DEV"] = device
  metadata, _tensors = load_gguf(path)
  model = Qwen3Model.from_gguf(path, max_context=max_context)
  tokenizer = Tokenizer.from_gguf(metadata)
  return model, tokenizer


def _run_single_shot(model: Qwen3Model, tokenizer: Tokenizer, prompt: str, max_tokens: int) -> int:
  """Encode the prompt, stream generated tokens to stdout, stop at EOS."""
  raise NotImplementedError  # TODO: model.generate() + tokenizer.stream_decoder()


def _run_chat(model: Qwen3Model, tokenizer: Tokenizer, max_tokens: int, max_context: int) -> int:
  """Interactive multi-turn chat using ``tokenizer.render_chat``."""
  raise NotImplementedError  # TODO: input loop, render_chat, generate, keep history


def _run_benchmark(model: Qwen3Model, tokenizer: Tokenizer, count: int) -> int:
  """Generate ``count`` tokens and print tokens/sec."""
  raise NotImplementedError  # TODO: Timing / GlobalCounters around generate()


def main() -> int:
  ap = argparse.ArgumentParser(description="infurnace offline driver (single-shot, chat, or benchmark)")
  ap.add_argument("--model", default=_DEFAULT_MODEL, help="path to a Qwen3 GGUF file")
  ap.add_argument("--prompt", default=None, help="single-shot prompt (default: none, uses --chat)")
  ap.add_argument("--chat", action="store_true", help="interactive multi-turn chat")
  ap.add_argument("--max-tokens", type=int, default=None, help="max generated tokens")
  ap.add_argument("--max-context", type=int, default=2048, help="KV cache context length")
  ap.add_argument("--benchmark", nargs="?", type=int, const=20, metavar="COUNT", help="benchmark tokens/sec")
  ap.add_argument("--device", default=None, help="tinygrad device (sets DEV; default: auto-detect)")
  args = ap.parse_args()

  model, tokenizer = _build(args.model, args.max_context, args.device)

  if args.benchmark is not None:
    return _run_benchmark(model, tokenizer, args.benchmark)
  if args.chat or args.prompt is None:
    return _run_chat(model, tokenizer, args.max_tokens or _CHAT_MAX_TOKENS, args.max_context)
  return _run_single_shot(model, tokenizer, args.prompt, args.max_tokens or _SINGLE_SHOT_MAX_TOKENS)


if __name__ == "__main__":
  raise SystemExit(main())