#!/usr/bin/env python3
"""Benchmark paged-KV page sizes for Phase 5A.

Computes the exact KV bytes-per-page arithmetic for Qwen3-0.6B, measures the
per-page store/read dispatch cost for candidate page sizes on a small CPU pool
(the prefill-store pattern Phase 5B mirrors), and optionally allocates the
target physical pool on DEV=NV against the device budget. Output is a small
structured result written to --output (or stdout); the decision is recorded in
docs/baselines/phase5a-page-size.md.
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
LAYERS, KV_HEADS, HEAD_DIM, DTYPE_BYTES = 28, 8, 128, 2
BYTES_PER_TOKEN = LAYERS * 2 * KV_HEADS * HEAD_DIM * DTYPE_BYTES
MIB = 1024 * 1024

def bytes_per_page(page_size: int) -> int:
  return BYTES_PER_TOKEN * page_size

def pages_for(slots: int, max_context: int, page_size: int) -> int:
  tokens = slots * max_context
  return (tokens + page_size - 1) // page_size

def positive_int(value: str) -> int:
  parsed = int(value)
  if parsed <= 0: raise argparse.ArgumentTypeError("must be positive")
  return parsed

def summary_ns(samples_ns: list[int]) -> dict[str, int | float]:
  return {"count": len(samples_ns), "min_ns": min(samples_ns),
          "max_ns": max(samples_ns), "mean_ns": statistics.fmean(samples_ns),
          "median_ns": statistics.median(samples_ns)}

def cpu_page_cost(page_size: int, repeats: int, n_pages: int) -> dict[str, int | float]:
  from tinygrad import Tensor, dtypes
  pool = Tensor.zeros(LAYERS, 2, n_pages, page_size, KV_HEADS, HEAD_DIM,
                      dtype=dtypes.float16).contiguous().realize()
  block = Tensor.ones(page_size, KV_HEADS, HEAD_DIM, dtype=dtypes.float16).realize()
  write, read = [], []
  for _ in range(repeats):
    t0 = time.perf_counter_ns()
    pool[:, :, 0, :, :, :].assign(block).realize()
    write.append(time.perf_counter_ns() - t0)
    t0 = time.perf_counter_ns()
    pool[:, :, 0, :, :, :].realize()
    read.append(time.perf_counter_ns() - t0)
  return {"write": summary_ns(write), "read": summary_ns(read)}

def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--page-sizes", default="16,32,64")
  parser.add_argument("--num-slots", type=positive_int, default=4)
  parser.add_argument("--max-context", type=positive_int, default=2048)
  parser.add_argument("--repeats", type=positive_int, default=20)
  parser.add_argument("--nv-pool-check", action="store_true",
                      help="allocate the target pool on DEV=NV and record live device bytes")
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()

  sizes = sorted(int(s) for s in args.page_sizes.split(","))
  if not sizes or any(s < 1 for s in sizes):
    print("error: --page-sizes must be a comma list of positive ints", file=sys.stderr)
    return 1

  per_size = {}
  for size in sizes:
    pages = pages_for(args.num_slots, args.max_context, size)
    per_size[size] = {
      "bytes_per_page": bytes_per_page(size),
      "pages": pages,
      "pool_bytes": pages * bytes_per_page(size),
    }
  cpu = {size: cpu_page_cost(size, args.repeats, 8) for size in sizes}

  nv = {"checked": False}
  if args.nv_pool_check:
    if os.environ.get("DEV") != "NV":
      print("error: --nv-pool-check requires DEV=NV", file=sys.stderr)
      return 1
    from tinygrad import GlobalCounters, Tensor, dtypes
    total_pages = max(pool["pages"] + 4 for pool in per_size.values())
    target = max(sizes)
    pool = Tensor.zeros(LAYERS, 2, total_pages, target, KV_HEADS, HEAD_DIM,
                        dtype=dtypes.float16).contiguous().realize()
    nv = {"checked": True, "page_size": target, "pages": total_pages,
          "live_requested_bytes": int(GlobalCounters.mem_used_per_device["NV"])}

  result = {
    "schema_version": SCHEMA_VERSION,
    "benchmark": "page_size_scan",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "checkpoint": {"id": "qwen3-0.6b-q8_0"},
    "system": {"python": platform.python_version(), "platform": platform.platform(),
               "device": os.environ.get("DEV", "CPU")},
    "model": {"layers": LAYERS, "kv_heads": KV_HEADS, "head_dim": HEAD_DIM,
              "dtype_bytes": DTYPE_BYTES, "bytes_per_token": BYTES_PER_TOKEN},
    "topology": {"num_slots": args.num_slots, "max_context": args.max_context,
                 "total_tokens": args.num_slots * args.max_context},
    "page_sizes": per_size,
    "cpu": {"repeats": args.repeats, "sizes": cpu},
    "nv": nv,
  }

  serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
  if args.output is None:
    sys.stdout.write(serialized)
  else:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")

  print(f"bytes/token={BYTES_PER_TOKEN} bytes ({BYTES_PER_TOKEN / 1024:.1f} KiB)")
  for size in sizes:
    write_ns = cpu[size]["write"]["mean_ns"]
    print(f"  page_size={size:>3}  {per_size[size]['bytes_per_page'] / MIB:7.2f} MiB/page  "
          f"{per_size[size]['pages']:>5} pages  {per_size[size]['pool_bytes'] / MIB:7.1f} MiB pool  "
          f"write={write_ns / size:8.1f} ns/token")
  print(f"provisional default: page_size={sizes[0]} "
        "(record decision in docs/baselines/phase5a-page-size.md)")
  return 0

if __name__ == "__main__":
  raise SystemExit(main())