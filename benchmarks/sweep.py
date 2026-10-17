"""Decoding across prompt lengths: time to first token, prompt reading and decode rate, one model load.

  python benchmarks/sweep.py --lengths 25,124,524,1024,2024,4024,8024,16024 --tokens 128 [--dflash]

The prompt is one chat message, as an agent sends after reading files: quality/eval.txt (infurnace's docs and
code, repeated as needed) cut to the length, then a request to summarize it, thinking off. The reply is a
real summary, so DFlash's speed reflects ordinary prose and code rather than a repeated pattern. After the
usual warmup, each length runs once to capture its graphs, then once measured (--once skips the first).
"""
import argparse
import json
import pathlib
import time

from infurnace import loader

TEXT = pathlib.Path(__file__).parent / "quality" / "eval.txt"
MARK = "\x00DOCUMENT\x00"

def prompt_ids(tokenizer, length: int) -> list[int]:
  """A `length`-token chat prompt: the document cut to fit, then the request."""
  head, tail = tokenizer.render_chat([{"role": "user", "content": MARK + "\n\nSummarize the text above."}],
                                     enable_thinking=False).split(MARK)
  head, tail, doc = tokenizer.encode(head), tokenizer.encode(tail), tokenizer.encode(TEXT.read_text())
  room = length - len(head) - len(tail)
  if room < 1: raise ValueError(f"prompts need at least {len(head) + len(tail) + 1} tokens")
  return head + (doc * (room // len(doc) + 1))[:room] + tail

def measure(stream, prompt: int) -> dict:
  start, first, generated = time.perf_counter(), None, 0
  for _ in stream:
    if first is None: first = time.perf_counter()
    generated += 1
  end = time.perf_counter()
  return {"prompt_tokens": prompt, "generated_tokens": generated, "ttft_seconds": first - start,
          "prefill_tokens_per_second": prompt / (first - start), "decode_tokens_per_second": (generated - 1) / (end - first)}

def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--model", default=next(iter(loader.MODELS)))
  ap.add_argument("--lengths", default="25,124,524,1024,2024,4024,8024,16024")
  ap.add_argument("--tokens", type=int, default=128, help="output tokens per run (default 128; EOS is ignored)")
  ap.add_argument("--dflash", action="store_true", help="decode with the DFlash draft, as chat and the server do by default")
  ap.add_argument("--once", action="store_true", help="skip the capture run before each measured one (long prompts)")
  ap.add_argument("--json", action="store_true", help="print JSON rows instead of a table")
  args = ap.parse_args()
  lengths = [int(n) for n in args.lengths.split(",")]
  began = time.perf_counter()
  path = loader.fetch_model(args.model)
  model, tokenizer = loader.load_model(path, max(lengths) + args.tokens + 256)  # headroom, as a real context has
  dflash = loader.load_dflash(model, loader.fetch_model(loader.dflash_draft(args.model))) if args.dflash else None
  # As chat and the server do before their first reply: captures the prompt graph (and DFlash's rounds).
  if dflash is not None: dflash.warmup()
  else: model.warmup()
  print(f"loaded and warmed up in {time.perf_counter() - began:.0f}s on {model.device}, context {model.config.context_length}"
        + (", DFlash" if dflash else ""), flush=True)
  def run(ids):
    model.reset()  # read the whole prompt: no prompt snapshot or cached prefix
    if dflash is not None: return measure(dflash.stream(ids, max_new_tokens=args.tokens), len(ids))
    return measure(model.generate(ids, args.tokens), len(ids))
  if not args.json: print(f"{'prompt':>7} {'TTFT':>8} {'prefill tok/s':>14} {'decode tok/s':>13}", flush=True)
  for n in lengths:
    ids = prompt_ids(tokenizer, n)
    if not args.once: run(ids)
    row = run(ids)
    if args.json: print(json.dumps(row), flush=True)
    else:
      print(f"{n:>7} {row['ttft_seconds'] * 1000:>6.0f}ms {row['prefill_tokens_per_second']:>14.0f} "
            f"{row['decode_tokens_per_second']:>13.1f}", flush=True)

if __name__ == "__main__":
  main()
