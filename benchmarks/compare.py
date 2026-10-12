"""Independent Infurnace/upstream comparison using tinygrad CLI step timing.

Run from the checkout with its environment: python benchmarks/compare.py --help
"""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time


def summarize(steps: list[dict]) -> dict:
  """Aggregate rate is count / elapsed time, not mean instantaneous tok/s."""
  if not steps: raise ValueError("no measured steps")
  seconds = [step["seconds"] for step in steps]
  if any(value <= 0 for value in seconds): raise ValueError("step times must be positive")
  total = sum(seconds)
  ordered = sorted(seconds)
  return {
    "tokens": len(steps), "seconds": total, "tokens_per_second": len(steps)/total,
    "latency_ms_median": statistics.median(seconds)*1000,
    "latency_ms_p95": ordered[max(0, (95*len(ordered)+99)//100-1)]*1000,
    "latency_ms_min": min(seconds)*1000, "latency_ms_max": max(seconds)*1000,
    "estimated_bytes_per_token": sum(s["estimated_bytes"] for s in steps)/len(steps),
    "estimated_gb_per_second": sum(s["estimated_bytes"] for s in steps)/total/1e9,
    "estimated_ops_per_token": sum(s["estimated_ops"] for s in steps)/len(steps),
    "tinygrad_kernel_count_per_token": sum(s["kernel_count"] for s in steps)/len(steps),
  }


def measure(stream, count: int, device: str) -> dict:
  from tinygrad import Context
  from tinygrad.helpers import GlobalCounters, Timing
  steps = []
  try:
    for index in range(count):
      GlobalCounters.reset()
      with Context(DEV=device):
        # Same timer boundary as upstream CLI: next(generate) including item sync.
        timer = Timing(enabled=False)
        with timer: token = next(stream)
      steps.append({
        "step": index, "token_id": token, "seconds": timer.et/1e9,
        "estimated_bytes": GlobalCounters.global_mem, "estimated_ops": GlobalCounters.global_ops,
        "kernel_count": GlobalCounters.kernel_count, "allocated_bytes": GlobalCounters.mem_used,
      })
  finally:
    stream.close()
  return {"ttft_seconds": steps[0]["seconds"], "all_steps": summarize(steps),
          "decode": summarize(steps[1:]), "steps": steps}


def worker(args) -> dict:
  from tinygrad import Context, Device
  from tinygrad.helpers import GlobalCounters
  from infurnace.device import select_device
  from infurnace.tokenizer import Qwen35Tokenizer
  from tinygrad.llm.model import Transformer
  from infurnace.loader import load_model
  from decode import benchmark_prompt_ids, validate_benchmark_context

  device = select_device(args.device)
  with Context(DEV=device):
    start = time.perf_counter()
    if args.worker == "infurnace":
      model, tokenizer = load_model(args.model, args.context, device,
                                    nv_quant=False if args.no_nv_quant else None, nv_prefill=args.nv_prefill)
      context = model.config.context_length
      def reset(): model.reset()
      def generate(): return model.generate(list(ids), args.tokens, chunk_size=args.chunk_size)
      name = model.metadata.get("general.name", "Qwen3.5")
      linears = getattr(model, "nv_quant_linears", 0)
    else:
      # Independent upstream loader, default lazy FP16 weights; no Infurnace adapter.
      model, metadata = Transformer.from_gguf(Path(args.model), args.context, realize=False)
      if metadata.get('general.architecture') != 'qwen35': raise ValueError('expected dense Qwen3.5 GGUF')
      # Shared Qwen3.5 codec; transformer loading/execution stays independent.
      tokenizer = Qwen35Tokenizer.from_gguf(metadata)
      context = model.max_context
      def reset(): model._cached_tokens = []
      def generate(): return model.generate(list(ids), chunk_size=args.chunk_size, temperature=0.0)
      name, linears = metadata.get("general.name", "LLM"), 0
    load_seconds = time.perf_counter()-start
    text = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file is not None else None
    ids = benchmark_prompt_ids(tokenizer, args.prompt_tokens or 512, text)
    validate_benchmark_context(len(ids), args.tokens, context)
    reset()
    cold = measure(generate(), args.tokens, device)
    # A second complete pass captures both symbolic prompt and concrete rollout paths.
    reset()
    measure(generate(), args.tokens, device)
    warm = []
    for _ in range(args.runs):
      reset()
      warm.append(measure(generate(), args.tokens, device))
    Device[device].synchronize()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    version = importlib.metadata.version("tinygrad")
    try:
      direct_url = json.loads(importlib.metadata.distribution("tinygrad").read_text("direct_url.json") or "{}")
    except (ValueError, TypeError): direct_url = {}
    return {
      "runner": args.worker, "model_name": name, "device": device, "context_capacity": context,
      "prompt_ids": ids, "prompt_tokens": len(ids),
      "prompt_source": "raw text" if text is not None else "synthetic repeated prose",
      "decode_input_positions": [len(ids), len(ids)+args.tokens-2],
      "temperature": 0.0, "chunk_size": args.chunk_size,
      "nv_quant_linears": linears, "nv_prefill": getattr(model, "nv_prefill", False),
      "load_seconds": load_seconds,
      "cold": cold, "warm_runs": warm,
      "warm_decode": summarize([s for run in warm for s in run["steps"][1:]]),
      "host_peak_rss_bytes": rss if sys.platform == "darwin" else rss*1024,
      "tinygrad_current_allocated_bytes": GlobalCounters.mem_used,
      "tinygrad_version": version, "tinygrad_source": direct_url, "python": platform.python_version(),
    }


def run_child(args, runner: str) -> dict:
  command = [sys.executable, str(Path(__file__).resolve()), "--worker", runner,
             "--model", args.model, "--device", args.device, "--context", str(args.context),
             "--tokens", str(args.tokens), "--runs", str(args.runs), "--chunk-size", str(args.chunk_size)]
  if args.prompt_file is not None: command.extend(["--prompt-file", str(args.prompt_file)])
  elif args.prompt_tokens is not None: command.extend(["--prompt-tokens", str(args.prompt_tokens)])
  if args.no_nv_quant: command.append("--no-nv-quant")
  if args.nv_prefill is not None: command.append("--nv-prefill" if args.nv_prefill else "--no-nv-prefill")
  env = dict(os.environ, DEV=args.device, VIZ="0", PROFILE="0", TRACK_MATCH_STATS="0",
             DEBUG="0", REALIZE="0", HALF="1", JIT="1")
  result = subprocess.run(command, env=env, text=True, capture_output=True)
  if result.returncode:
    raise RuntimeError(f"{runner} failed (exit {result.returncode}):\n{result.stderr}\n{result.stdout}")
  if result.stderr: print(result.stderr, file=sys.stderr, end="")
  return json.loads(result.stdout)


def parser() -> argparse.ArgumentParser:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--model", default="qwen3.5:4b", help="Qwen3.5 model name or local GGUF path")
  ap.add_argument("--device", default="NV", help="same explicit backend for both processes")
  ap.add_argument("--context", type=int, default=2048)
  ap.add_argument("--tokens", type=int, default=256, help="generated tokens per pass, including first token")
  ap.add_argument("--runs", type=int, default=3, help="measured warm passes after cold and capture passes")
  ap.add_argument("--chunk-size", type=int, default=32)
  prompt = ap.add_mutually_exclusive_group()
  prompt.add_argument("--prompt-tokens", type=int, help="synthetic prompt tokens including BOS (default: 512)")
  prompt.add_argument("--prompt-file", type=Path, help="UTF-8 raw prompt file; not truncated")
  ap.add_argument("--no-nv-quant", action="store_true", help="compare Infurnace generic operations to upstream")
  prefill = ap.add_mutually_exclusive_group()
  prefill.add_argument("--nv-prefill", dest="nv_prefill", action="store_true", help="require packed tensor-core prefill for Infurnace")
  prefill.add_argument("--no-nv-prefill", dest="nv_prefill", action="store_false", help="use generic Infurnace prefill, retaining NV decode")
  ap.set_defaults(nv_prefill=None)
  ap.add_argument("--output", type=Path, help="save complete JSON including per-step tokens/timing/counters")
  ap.add_argument("--worker", choices=("infurnace", "tinygrad"), help=argparse.SUPPRESS)
  return ap


def main() -> int:
  ap = parser()
  args = ap.parse_args()
  if args.nv_prefill and args.no_nv_quant: ap.error("--nv-prefill conflicts with --no-nv-quant")
  if args.tokens < 2: ap.error("--tokens must be at least 2")
  if args.prompt_tokens is not None and args.prompt_tokens <= 0: ap.error("--prompt-tokens must be positive")
  if args.prompt_file is None and args.context < (args.prompt_tokens or 512)+args.tokens:
    ap.error("--context must fit the prompt plus requested output")
  if args.prompt_file is not None:
    args.prompt_file = args.prompt_file.expanduser().resolve()
    if not args.prompt_file.is_file(): ap.error(f"prompt file not found: {args.prompt_file}")
  if args.runs <= 0 or args.chunk_size <= 0: ap.error("--runs and --chunk-size must be positive")
  from infurnace.loader import fetch_model
  args.model = str(fetch_model(args.model).resolve())
  if not Path(args.model).is_file(): ap.error(f"model not found: {args.model}")
  try:
    if args.worker:
      # Loader/debug output must not contaminate the child JSON protocol.
      with redirect_stdout(sys.stderr): report = worker(args)
      print(json.dumps(report))
      return 0
    print("Run sequentially with other model processes stopped; GPU peak is not measured.", file=sys.stderr)
    digest = hashlib.sha256()
    with open(args.model, "rb") as handle:
      while block := handle.read(8*1024*1024): digest.update(block)
    reports = []
    for runner in ("tinygrad", "infurnace"):
      print(f"Benchmarking {runner}...", file=sys.stderr, flush=True)
      reports.append(run_child(args, runner))
    upstream, infurnace = reports
    if upstream["prompt_ids"] != infurnace["prompt_ids"] or upstream["context_capacity"] != infurnace["context_capacity"]:
      raise ValueError("runner prompt/context mismatch; comparison rejected")
    report = {
      "model_path": args.model, "model_bytes": Path(args.model).stat().st_size, "model_sha256": digest.hexdigest(),
      "method": "shared prompt IDs, tinygrad CLI per-step GlobalCounters.reset and Timing; cold + capture + warm passes",
      "notes": ["EOS ignored; each pass resets prefix reuse. Warm decode excludes the first token.",
                "Counters estimate work, not measured DRAM bandwidth, FLOPs, or GPU peak. Custom kernel estimates may be incomplete.",
                "Independent sequential processes. GPU clocks/other workloads are not controlled.",
                "Upstream uses realize=False and FP16 views; NV linears are generic.",
                "Infurnace Q8 decode and optional FP16 packed GEMM differ from generic math; throughput is not quality parity.",
                "Both runners use the same Qwen3.5 tokenizer; independent transformer loaders, no DFlash.",
                "Upstream-equivalent benchmark methodology, not an HTTP server benchmark."],
      "results": reports,
      "warm_decode_speedup": infurnace["warm_decode"]["tokens_per_second"]/upstream["warm_decode"]["tokens_per_second"],
    }
    if args.output:
      args.output.parent.mkdir(parents=True, exist_ok=True)
      args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(f"Prompt: {upstream['prompt_tokens']} tokens ({upstream['prompt_source']}); "
          f"decode input positions: {upstream['decode_input_positions']}")
    print("Runner       Load (s)  Cold TTFT (s)  Warm decode (tok/s)  Median/p95 (ms)  Host peak (GiB)  Allocated (GiB)")
    for r in reports:
      d = r["warm_decode"]
      print(f"{r['runner']:<12} {r['load_seconds']:>8.2f} {r['cold']['ttft_seconds']:>14.2f} "
            f"{d['tokens_per_second']:>20.2f} {d['latency_ms_median']:>7.2f}/{d['latency_ms_p95']:<7.2f} "
            f"{r['host_peak_rss_bytes']/2**30:>15.2f} {r['tinygrad_current_allocated_bytes']/2**30:>16.2f}")
    print(f"Warm decode speedup: {report['warm_decode_speedup']:.2f}x")
    if args.output: print(f"Full results: {args.output}")
    return 0
  except (RuntimeError, ValueError, OSError) as exc:
    print(f"benchmark: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__": raise SystemExit(main())
