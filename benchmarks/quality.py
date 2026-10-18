"""Output quality: perplexity of infurnace's GPU paths on a fixed text, against llama.cpp on the same GGUF.

Follows llama.cpp's perplexity protocol: the text is cut into chunks of --ctx tokens, each read from a fresh
state, and the second half of each chunk is scored (each token predicted from all tokens before it in its chunk).

Paths (teacher-forced, infurnace's real kernels):
  one-row  decoding one token at a time: Q8-activation GEMV, fused recurrent layer, attention kernel
  16-row   reading 16 tokens at a time: tensor-core GEMM (shared with DFlash verification), gdn_prefill
  verifier DFlash's 16-row verification forward: tensor-core GEMM, gdn_chunk, the attention kernel at 16 rows,
           and the closed-form state commit that advances the recurrent layers between blocks

  python benchmarks/quality.py --llama-perplexity ~/repos/llama.cpp/build/bin/llama-perplexity
"""
from __future__ import annotations
import argparse
import math
import pathlib
import re
import subprocess
import time

from tinygrad import Tensor, TinyJit, UOp, dtypes
from infurnace import catalog, loader
from infurnace.dflash.jit import captured
from infurnace.dflash.parallel import ParallelVerifier
from infurnace.dflash.target import TargetObserver

TEXT = pathlib.Path(__file__).parent / "quality" / "eval.txt"

def chunk_nll(model, step_rows: int):
  """A captured step: rows of token ids at start_pos -> each row's negative log-likelihood of its target."""
  t = model.transformer
  def step(ids: Tensor, targets: Tensor, start_pos: UOp) -> Tensor:
    x = t.token_embd(ids).float()
    for block in t.blk: x = block(x, start_pos)
    logits = t.output(t.output_norm(x)).reshape(step_rows, -1).float()
    return -logits.log_softmax(-1).gather(-1, targets.reshape(step_rows, 1)).reshape(step_rows).contiguous()
  return TinyJit(step)

def perplexity(model, tokens: list[int], ctx: int, chunks: int, rows: int) -> tuple[float, float]:
  """(perplexity, seconds) over `chunks` chunks, reading `rows` tokens per step."""
  start_var = UOp.variable("start_pos", 0, ctx - 1)
  step = chunk_nll(model, rows)
  first, total, count, began = ctx // 2, 0.0, 0, time.perf_counter()
  for c in range(chunks):
    chunk = tokens[c * ctx:(c + 1) * ctx]
    for pos in range(0, ctx, rows):
      ids = Tensor([chunk[pos:pos + rows]], dtype=dtypes.int32, device=model.device)
      # the target of position p is token p + 1; the last position of a chunk has none (scored 0, masked below)
      targets = Tensor([chunk[p + 1] if p + 1 < ctx else 0 for p in range(pos, pos + rows)], dtype=dtypes.int32, device=model.device)
      nll = step(ids, targets, start_var.bind(pos)).tolist()
      for i, value in enumerate(nll):
        if first < pos + i + 1 < ctx: total, count = total + value, count + 1  # tokens first+1..ctx-1, as llama.cpp scores
  return math.exp(total / count), time.perf_counter() - began

def verifier_perplexity(model, tokens: list[int], ctx: int, chunks: int) -> tuple[float, float]:
  """(perplexity, seconds) through DFlash's verifier: each 16-row block is verified, then all of it committed."""
  observer = TargetObserver(model, (1,))  # the taps only feed the draft, which isn't used here
  verifier, rows = ParallelVerifier(observer, 16), 16
  def step(x: Tensor, targets: Tensor, start_pos: UOp) -> Tensor:
    logits = verifier.logits(x, start_pos)[0].float()
    nll = -logits.log_softmax(-1).gather(-1, targets.reshape(rows, 1)).reshape(rows).contiguous()
    nll.realize()
    return nll
  nll_jit = captured(step)
  first, total, count, began = ctx // 2, 0.0, 0, time.perf_counter()
  with observer:
    verifier._allocate()
    for c in range(chunks):
      chunk, observer.position = tokens[c * ctx:(c + 1) * ctx], 0  # position 0: fresh recurrent state
      for pos in range(0, ctx, rows):
        x = model.embed(chunk[pos:pos + rows]).realize()
        targets = Tensor([chunk[p + 1] if p + 1 < ctx else 0 for p in range(pos, pos + rows)], dtype=dtypes.int32, device=model.device)
        nll = nll_jit(x, targets, verifier._position.bind(pos)).tolist()
        verifier.commit(rows, [0] * rows)
        for i, value in enumerate(nll):
          if first < pos + i + 1 < ctx: total, count = total + value, count + 1
  return math.exp(total / count), time.perf_counter() - began

def llama_cpp(binary: str, model: str, ctx: int, chunks: int, text: pathlib.Path = TEXT) -> tuple[float, int]:
  """llama.cpp's perplexity and token count for the same text and chunks."""
  run = subprocess.run([binary, "-m", model, "-f", str(text), "-c", str(ctx), "--chunks", str(chunks), "-ngl", "99", "-b", str(ctx)],
                       capture_output=True, text=True, timeout=600)
  out = run.stdout + run.stderr
  ppl = re.search(r"Final estimate: PPL = ([\d.]+)", out)
  tokens = re.search(r"tokenizing the input .*?(\d+) tokens|perplexity: (\d+) tokens", out)
  if ppl is None: raise RuntimeError("llama-perplexity failed:\n" + out[-2000:])
  return float(ppl.group(1)), int(next(g for g in tokens.groups() if g)) if tokens else -1

def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--model", default="qwen3.5:4b")
  ap.add_argument("--text", type=pathlib.Path, default=TEXT, help=f"text to score (default {TEXT.name}; also quality/multilingual.txt)")
  ap.add_argument("--ctx", type=int, default=512)
  ap.add_argument("--chunks", type=int, default=8)
  ap.add_argument("--paths", default="one-row,16-row,verifier")
  ap.add_argument("--llama-perplexity", help="path to llama.cpp's llama-perplexity, for the reference number")
  args = ap.parse_args()
  path = catalog.fetch_model(args.model)
  if args.llama_perplexity:  # first, so the two models never share the GPU
    ppl, n = llama_cpp(args.llama_perplexity, str(path), args.ctx, args.chunks, args.text)
    print(f"llama.cpp          PPL {ppl:.4f}" + (f"  ({n} tokens)" if n > 0 else ""), flush=True)
  model, tokenizer = loader.load_model(path, args.ctx)
  tokens = tokenizer.encode(args.text.read_text())
  chunks = min(args.chunks, len(tokens) // args.ctx)
  print(f"{args.text.name}: {len(tokens)} tokens, {chunks} chunks of {args.ctx}, scoring the second half of each", flush=True)
  model.realize_state()
  for name in args.paths.split(","):
    if name == "verifier": ppl, seconds = verifier_perplexity(model, tokens, args.ctx, chunks)
    else: ppl, seconds = perplexity(model, tokens, args.ctx, chunks, {"one-row": 1, "16-row": 16}[name])
    print(f"infurnace {name:8s} PPL {ppl:.4f}  ({seconds:.0f}s)", flush=True)

if __name__ == "__main__":
  main()
