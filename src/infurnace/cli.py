from __future__ import annotations
import argparse
import os
import sys

_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_SINGLE_SHOT_MAX_TOKENS = 32
_CHAT_MAX_TOKENS = 256
_SYSTEM_PROMPT = "You are a helpful assistant. Answer directly and concisely; do not use thinking or reasoning mode."

def _build_engine(artifact: str, manifest: str, *, num_slots: int, max_context: int | None, device: str | None) -> Engine:
  from infurnace.models.manifest import load_manifest
  from infurnace.executor.tinygrad.weights import load_qwen3_checkpoint
  from infurnace.executor.tinygrad.runner import Qwen3Runner
  from infurnace.tokenizer import GGUFTokenizer
  from infurnace.engine import Engine
  from infurnace.scheduler.scheduler import Scheduler
  from infurnace.sampler import GreedySampler

  if not os.path.exists(artifact):
    print(f"artifact not found: {artifact} (download/place the GGUF locally)", file=sys.stderr)
    raise SystemExit(1)
  if not os.path.exists(manifest):
    print(f"manifest not found: {manifest}", file=sys.stderr)
    raise SystemExit(1)

  checkpoint = load_qwen3_checkpoint(artifact, load_manifest(manifest))
  tokenizer = GGUFTokenizer.from_gguf_metadata(checkpoint.metadata)
  runner = Qwen3Runner.from_weights(checkpoint.weights, num_slots=num_slots, max_context=max_context, device=device,)
  scheduler = Scheduler(num_slots=num_slots, max_context=max_context or 32768)
  return Engine(runner, scheduler, GreedySampler(), tokenizer)

def _chat_template(messages: list[tuple[str, str]]) -> str:
  """Render conversation history with the Qwen3 chat template."""
  parts = [f"{_IM_START}{role}\n{text}{_IM_END}\n" for role, text in messages]
  parts.append(f"{_IM_START}assistant\n")
  return "".join(parts)

def _run_chat(eng: Engine, max_tokens: int, max_context: int) -> int:
  from infurnace.engine.request import RequestState, SamplingParams
  print(f"infurnace chat (max-context {max_context} tokens; type 'exit' or Ctrl-D to quit)", flush=True)
  messages: list[tuple[str, str]] = [("system", _SYSTEM_PROMPT)]
  while True:
    try:
      prompt = input("You: ")
    except (EOFError, KeyboardInterrupt):
      print()
      return 0
    prompt = prompt.strip()
    if not prompt:
      continue
    if prompt.lower() in ("exit", "quit"):
      return 0
    messages.append(("user", prompt))
    full_prompt = _chat_template(messages)
    stop_ids = eng.tokenizer.encode(_IM_END)
    req = eng.add_text_request(
      full_prompt,
      SamplingParams(max_tokens=max_tokens, stop_token_ids=stop_ids),
      context_limit=max_context,
    )
    if req.state is RequestState.REJECTED:
      messages.pop()
      print(f"(rejected: {req.error})", file=sys.stderr)
      continue
    chunks: list[str] = []
    while not eng.is_done():
      for text in eng.step().new_text.values():
        if text:
          sys.stdout.write(text)
          sys.stdout.flush()
          chunks.append(text)
    reply = "".join(chunks).rstrip()
    if reply.endswith(_IM_END):
      reply = reply[: -len(_IM_END)].rstrip()
    print()
    messages.append(("assistant", reply))

def main() -> int:
  ap = argparse.ArgumentParser(description="infurnace offline text driver (single-shot or interactive chat)")
  ap.add_argument("--prompt", default=None, help="single-shot prompt (required unless --chat)")
  ap.add_argument("--chat", action="store_true", help="interactive multi-turn chat")
  ap.add_argument("--max-tokens", type=int, default=None, help="max generated tokens (default: 32 single-shot, 256 chat)")
  ap.add_argument("--stop-strings", nargs="*", default=[])
  ap.add_argument("--artifact", default="artifacts/models/Qwen3-4B-Q4_K_M.gguf")
  ap.add_argument("--manifest", default="models/qwen3-4b-q4_k_m.json")
  ap.add_argument("--num-slots", type=int, default=1)
  ap.add_argument("--max-context", type=int, default=2048, help="KV cache context length (default 2048; lower saves GPU memory)")
  ap.add_argument("--device", default=None, help="tinygrad device (sets DEV); default: auto-detect (GPU if available)")
  args = ap.parse_args()

  if args.device: os.environ["DEV"] = args.device
  from infurnace.engine.request import SamplingParams
  eng = _build_engine(args.artifact, args.manifest, num_slots=args.num_slots, max_context=args.max_context, device=args.device,)
  if args.chat:
    return _run_chat(eng, args.max_tokens or _CHAT_MAX_TOKENS, args.max_context)
  if args.prompt is None:
    print("error: --prompt is required unless --chat is given", file=sys.stderr)
    return 2
  eng.add_text_request(args.prompt, SamplingParams(max_tokens=args.max_tokens or _SINGLE_SHOT_MAX_TOKENS), stop_strings=args.stop_strings,)
  out = sys.stdout
  while not eng.is_done():
    for text in eng.step().new_text.values():
      if text:
        out.write(text)
        out.flush()
  out.write("\n")
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
