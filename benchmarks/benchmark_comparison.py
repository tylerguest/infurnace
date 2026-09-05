#!/usr/bin/env python3
"""Benchmark Infurnace against llama.cpp on the same card and checkpoint.

Runs benchmark_infurnace.py and llama-bench on the pinned artifact, computes
the head-to-head decode tok/s and achieved memory bandwidth, and prints a
comparison table. A validated combined result is written to --output.
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

BENCH_DIR = Path(__file__).parent
REPO_ROOT = Path(__file__).parents[1]
DEFAULT_LLAMA_BENCH = Path(__file__).parents[2] / "llama.cpp" / "build" / "bin" / "llama-bench"
BANDWIDTH_CEILING_GB_PER_S = 448.0

try:
  from benchmarks.common import query_device, write_result
except ModuleNotFoundError:
  from common import query_device, write_result

def run_infurnace(manifest: Path, artifact: Path, *, prompt_tokens: int, decode_tokens: int,
                  max_context: int) -> dict:
  command = [
    sys.executable, str(BENCH_DIR / "benchmark_infurnace.py"),
    "--manifest", str(manifest), "--artifact", str(artifact),
    "--max-context", str(max_context), "--prompt-tokens", str(prompt_tokens),
    "--decode-tokens", str(decode_tokens), "--memory-sample-ms", "50",
  ]
  environment = os.environ.copy()
  environment.update(DEV="NV", JIT="1")
  with tempfile.TemporaryDirectory() as directory:
    output = Path(directory) / "infurnace.json"
    process = subprocess.run(
      command + ["--output", str(output)], cwd=REPO_ROOT, env=environment,
      capture_output=True, text=True, timeout=3600,
    )
    if process.returncode != 0:
      raise RuntimeError(f"benchmark_infurnace failed:\n{process.stdout}\n{process.stderr}")
    return json.loads(output.read_text(encoding="utf-8"))

def run_llama_bench(llama_bench: Path, artifact: Path, *, prompt_tokens: int, decode_tokens: int,
                    repetitions: int, n_gpu_layers: int) -> dict:
  command = [
    str(llama_bench), "-m", str(artifact),
    "-p", str(prompt_tokens), "-n", str(decode_tokens),
    "-ngl", str(n_gpu_layers), "-r", str(repetitions), "-o", "json",
  ]
  process = subprocess.run(command, capture_output=True, text=True, timeout=3600)
  if process.returncode != 0:
    raise RuntimeError(f"llama-bench failed:\n{process.stdout}\n{process.stderr}")
  try:
    tests = json.loads(process.stdout)
  except json.JSONDecodeError as error:
    raise RuntimeError(f"llama-bench produced invalid JSON: {error}") from error
  pp = next((t for t in tests if t.get("n_prompt") and not t.get("n_gen")), None)
  tg = next((t for t in tests if t.get("n_gen")), None)
  if pp is None or tg is None:
    raise RuntimeError("llama-bench output missing prompt/generation tests")
  return {
    "pp": pp, "tg": tg, "raw_tests": tests,
  }

def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", default=str(REPO_ROOT / "models" / "qwen3-4b-q4_k_m.json"), type=Path)
  parser.add_argument("--artifact", default=str(REPO_ROOT / "artifacts" / "models" / "Qwen3-4B-Q4_K_M.gguf"), type=Path)
  parser.add_argument("--llama-bench", default=str(DEFAULT_LLAMA_BENCH), type=Path)
  parser.add_argument("--prompt-tokens", type=int, default=16)
  parser.add_argument("--decode-tokens", type=int, default=200)
  parser.add_argument("--repetitions", type=int, default=5)
  parser.add_argument("--n-gpu-layers", type=int, default=99)
  parser.add_argument("--max-context", type=int, default=512)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()

  for path, label in ((args.manifest, "manifest"), (args.artifact, "artifact"), (args.llama_bench, "llama-bench")):
    if not path.exists():
      print(f"error: {label} not found: {path}", file=sys.stderr)
      return 1

  try:
    infurnace = run_infurnace(
      args.manifest, args.artifact,
      prompt_tokens=args.prompt_tokens, decode_tokens=args.decode_tokens, max_context=args.max_context,
    )
    llama = run_llama_bench(
      args.llama_bench, args.artifact,
      prompt_tokens=args.prompt_tokens, decode_tokens=args.decode_tokens,
      repetitions=args.repetitions, n_gpu_layers=args.n_gpu_layers,
    )
  except (OSError, RuntimeError, json.JSONDecodeError, subprocess.TimeoutExpired) as error:
    print(f"error: {error}", file=sys.stderr)
    return 1

  inf_timings = infurnace["timings"]
  inf_decode_ns = inf_timings["decode_ns"]
  inf_mean_ns = statistics.fmean(inf_decode_ns)
  inf_decode_tok_s = 1e9 / inf_mean_ns
  weights_bytes = inf_timings["weights_bytes"]

  llama_pp, llama_tg = llama["pp"], llama["tg"]
  llama_pp_tok_s = llama_pp["avg_ts"]
  llama_tg_tok_s = llama_tg["avg_ts"]
  llama_tg_ms = llama_tg["avg_ns"] / 1e6
  llama_model_size = llama_tg["model_size"]

  ceiling_gb_per_s = BANDWIDTH_CEILING_GB_PER_S
  inf_gb_per_s = weights_bytes * inf_decode_tok_s / 1e9
  llama_gb_per_s = weights_bytes * llama_tg_tok_s / 1e9

  device_metadata = query_device()
  result = {
    "schema_version": 1,
    "benchmark": "comparison",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "checkpoint": {
      "id": infurnace["checkpoint"]["id"], "sha256": infurnace["checkpoint"]["sha256"],
      "size_bytes": infurnace["checkpoint"]["size_bytes"],
    },
    "system": {"python": platform.python_version(), "platform": platform.platform(), "device": device_metadata},
    "engines": {
      "infurnace": {
        "decode_tokens_per_second": inf_decode_tok_s,
        "ttft_ms": inf_timings["cold_prefill_ttft_ns"] / 1e6,
        "capture_ms": inf_timings["decode_capture_ns"] / 1e6,
        "achieved_gb_per_second": inf_gb_per_s,
        "peak_device_memory_bytes": infurnace["memory"]["device"]["sampled_peak_bytes"],
        "model_load_ms": inf_timings["model_load_ns"] / 1e6,
        "weights_bytes": weights_bytes,
      },
      "llamacpp": {
        "decode_tokens_per_second": llama_tg_tok_s,
        "decode_ms_per_token": llama_tg_ms,
        "prefill_tokens_per_second": llama_pp_tok_s,
        "achieved_gb_per_second": llama_gb_per_s,
        "model_size_bytes": llama_model_size,
        "build_commit": llama_tg["build_commit"],
        "repetitions": args.repetitions,
        "n_gpu_layers": args.n_gpu_layers,
      },
    },
    "derived": {
      "decode_ratio_infurnace_over_llamacpp": inf_decode_tok_s / llama_tg_tok_s,
      "ceiling_gb_per_second": ceiling_gb_per_s,
      "infurnace_pct_of_ceiling": inf_gb_per_s / ceiling_gb_per_s * 100.0,
      "llamacpp_pct_of_ceiling": llama_gb_per_s / ceiling_gb_per_s * 100.0,
      "notes": "decode tok/s is the tracked metric; TTFT is labeled: llama-bench pp is warm/precompiled, "
               "Infurnace cold TTFT includes first-execution compile of the eager prefill graph",
    },
  }
  write_result(result, args.output)

  def row(engine: str, tok_s: float, gb_s: float, ttft_ms: float, extra: str) -> str:
    return (f"{engine:12s} {tok_s:9.2f} tok/s  {gb_s:6.1f} GB/s  "
            f"{gb_s / ceiling_gb_per_s * 100:4.1f}%    {ttft_ms:>10.1f} ms  {extra}")

  print(f"Qwen3-4B Q4_K_M comparison on {device_metadata['name']} "
        f"({device_metadata['total_memory_bytes'] / 2**30:.0f} GiB)")
  print(f"{'engine':12s} {'decode':>9s}  {'GB/s':>6s}  {'% ceil':>6s}    {'TTFT':>10s}  notes")
  print(row("llama.cpp", llama_tg_tok_s, llama_gb_per_s, args.prompt_tokens / llama_pp_tok_s * 1e3,
            f"warm pp; {args.repetitions} reps"))
  print(row("infurnace", inf_decode_tok_s, inf_gb_per_s, inf_timings["cold_prefill_ttft_ns"] / 1e6,
            "cold; incl. compile"))
  print(f"\ninfurnace decode = {inf_decode_tok_s / llama_tg_tok_s * 100:.1f}% of llama.cpp "
        f"({llama_tg_tok_s / inf_decode_tok_s:.1f}x gap); ceiling ~{ceiling_gb_per_s:.0f} GB/s")
  print(f"peak device memory: {infurnace['memory']['device']['sampled_peak_bytes'] / 2**30:.2f} GiB")
  return 0

if __name__ == "__main__": raise SystemExit(main())