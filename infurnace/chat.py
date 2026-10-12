"""Terminal generation: one raw prompt, or an interactive chat with rolling history."""
from __future__ import annotations
import sys
import time
from typing import Iterator

class Decoder:
  """How replies are generated: plain decoding, or a DFlash / EAGLE-3 speculative generator."""
  def __init__(self, model, tokenizer, *, dflash=None, eagle=None):
    self.model, self.dflash, self.eagle = model, dflash, eagle
    self.eos = tuple(i for i in (tokenizer.eos_id, tokenizer.eot_id) if i is not None)
    # Parallel DFlash verification checks a whole trained block; the sequential reference checks 4 rows.
    self.dflash_block = None if dflash is None else dflash.draft.config.block_size if dflash.verifier == 'parallel' else 4

  def stream(self, ids: list[int], budget: int) -> Iterator[int]:
    if self.dflash is not None:
      return self.dflash.stream(ids, max_new_tokens=budget, block_size=self.dflash_block, eos_token_ids=self.eos)
    if self.eagle is not None:
      return self.eagle.stream(ids, max_new_tokens=budget, draft_tokens=1, eos_token_ids=self.eos, batched=True)
    return self.model.generate(ids, budget)

  def warmup(self) -> None:
    print(f"Warming up {self.model.device} (first startup may take a while)...", file=sys.stderr, flush=True)
    start = time.perf_counter()
    # DFlash requests never use plain generation's graphs, so it captures its own.
    how = self.dflash.warmup(self.dflash_block) if self.dflash is not None else self.model.warmup()
    note = {'cached': ' (DFlash graphs from cache)', 'captured': ' (DFlash graphs cached for next time)'}.get(how, '')
    print(f"Ready in {time.perf_counter() - start:.1f}s{note}.", file=sys.stderr, flush=True)

def prompt_ids(tokenizer, text: str) -> list[int]:
  return ([tokenizer.bos_id] if tokenizer.bos_id is not None else []) + tokenizer.encode(text)

def stream_reply(decoder: Decoder, tokenizer, ids: list[int], max_tokens: int | None) -> str:
  """Print a reply as it is generated and return its text."""
  context = decoder.model.config.context_length
  budget = context - len(ids) if max_tokens is None else max_tokens
  text, pieces, generated, ended, completed = tokenizer.stream_decoder(), [], 0, False, False
  stream = decoder.stream(ids, budget)
  try:
    for token in stream:
      generated += 1
      if tokenizer.is_end(token):
        ended = True
        break
      pieces.append(text(token))
      print(pieces[-1], end="", flush=True)
    completed = True
  finally:
    stream.close()
    pieces.append(text(None))
    print(pieces[-1], flush=True)
  if completed and not ended:
    if generated >= context - len(ids):
      print("[Response stopped: context limit reached. Increase --max-context for longer answers.]", file=sys.stderr)
    else:
      print(f"[Response stopped: {budget}-token limit reached (includes thinking). Increase --max-tokens or use --no-thinking.]",
            file=sys.stderr)
  return "".join(pieces)

def run_prompt(decoder: Decoder, tokenizer, prompt: str, max_tokens: int) -> int:
  stream_reply(decoder, tokenizer, prompt_ids(tokenizer, prompt), max_tokens)
  return 0

def run_chat(decoder: Decoder, tokenizer, max_tokens: int | None, max_context: int, thinking: bool = True) -> int:
  """Interactive chat. When a new message would not fit, the oldest turns are dropped."""
  history: list[dict] = []
  context = min(max_context, decoder.model.config.context_length)
  reserve = min(max_tokens if max_tokens is not None else 1024, max(1, context // 2))  # room left for the reply
  while True:
    try: text = input(">>> ")
    except EOFError: return 0
    retained, dropped = list(history), 0
    while True:
      messages = retained + [{"role": "user", "content": text}]
      ids = prompt_ids(tokenizer, tokenizer.render_chat(messages, enable_thinking=thinking))
      if len(ids) < context - reserve or not retained: break
      retained, dropped = retained[2:], dropped + 1  # history is user/assistant pairs; keep the newest
    if len(ids) >= context:
      print("Message exceeds context even without history; shorten it or increase --max-context. History was not truncated.",
            file=sys.stderr)
      continue
    if dropped:
      print(f"[Context window: discarded {dropped} oldest turn(s); earlier details are no longer remembered.]", file=sys.stderr)
    reply = stream_reply(decoder, tokenizer, ids, max_tokens)
    history = messages + [{"role": "assistant", "content": reply}]
