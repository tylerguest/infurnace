#!/usr/bin/env python3
"""Benchmark upstream Qwen model loading and steady prefill TTFT on DEV=NV."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
  from benchmarks.common import NvidiaMemorySampler, read_linux_memory, sampled_peak_bytes, timing_summary, write_result
except ModuleNotFoundError:
  from common import NvidiaMemorySampler, read_linux_memory, sampled_peak_bytes, timing_summary, write_result
from infurnace.models.manifest import ArtifactError, ManifestError, load_manifest, verify_artifact
try:
  from tools.gguf_inspection import stable_artifact_path
except ModuleNotFoundError:
  sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
  from gguf_inspection import stable_artifact_path


def positive_int(value: str) -> int:
  parsed = int(value)
  if parsed <= 0: raise argparse.ArgumentTypeError("must be positive")
  return parsed


def query_device() -> dict[str, Any]:
  command = [
    "nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free",
    "--format=csv,noheader,nounits",
  ]
  result = subprocess.run(command, capture_output=True, text=True, timeout=10)
  if result.returncode != 0: raise RuntimeError(result.stderr.strip() or f"nvidia-smi exited {result.returncode}")
  rows = [line for line in result.stdout.splitlines() if line.strip()]
  if len(rows) != 1: raise RuntimeError("the initial benchmark contract requires exactly one NVIDIA GPU")
  fields = [field.strip() for field in rows[0].split(",")]
  if len(fields) != 7: raise RuntimeError(f"unexpected nvidia-smi GPU row: {rows[0]!r}")
  try: index, total_mib, used_mib, free_mib = int(fields[0]), int(fields[4]), int(fields[5]), int(fields[6])
  except ValueError as error: raise RuntimeError(f"invalid nvidia-smi GPU row: {rows[0]!r}") from error
  return {
    "index": index, "uuid": fields[1], "name": fields[2], "driver_version": fields[3],
    "total_memory_bytes": total_mib * 1024 * 1024,
    "baseline_used_memory_bytes": used_mib * 1024 * 1024,
    "baseline_free_memory_bytes": free_mib * 1024 * 1024,
  }


def make_prompt(length: int, sequence: int) -> list[int]:
  return [257 + sequence] + [1000 + (sequence * length + index) % 1000 for index in range(length - 1)]


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", required=True, type=Path)
  parser.add_argument("--artifact", required=True, type=Path)
  parser.add_argument("--weight-policy", required=True, choices=("lazy", "realized-fp16"))
  parser.add_argument("--max-context", type=positive_int, default=1024)
  parser.add_argument("--chunk-size", type=positive_int, default=32)
  parser.add_argument("--prompt-tokens", type=positive_int, default=16)
  parser.add_argument("--samples", type=positive_int, default=5)
  parser.add_argument("--memory-sample-ms", type=positive_int, default=50)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()

  if os.environ.get("DEV") != "NV":
    print("error: upstream prefill benchmark requires DEV=NV", file=sys.stderr)
    return 1
  if args.prompt_tokens + 1 > args.max_context:
    print("error: prompt and TTFT token exceed max context", file=sys.stderr)
    return 1
  if args.prompt_tokens < 2 or args.chunk_size < 2 or args.prompt_tokens > args.chunk_size:
    print("error: upstream prefill benchmark requires 2 <= prompt tokens <= chunk size", file=sys.stderr)
    return 1

  sampler: NvidiaMemorySampler | None = None
  resources = ExitStack()
  try:
    manifest = load_manifest(args.manifest)
    stable_path = resources.enter_context(stable_artifact_path(args.artifact))
    host_before_verification = read_linux_memory()
    started = time.perf_counter_ns()
    verify_artifact(stable_path, manifest)
    artifact_verification_ns = time.perf_counter_ns() - started
    host_after_verification = read_linux_memory()
    device_metadata = query_device()

    os.environ.update(JIT="1", HALF="1")
    from tinygrad import Device, GlobalCounters, Tensor, dtypes
    from tinygrad.llm.model import Transformer

    sampler = NvidiaMemorySampler(os.getpid(), args.memory_sample_ms)
    sampler.start()
    device = Device["NV"]
    device.synchronize()

    def memory_snapshot() -> dict[str, int]:
      return read_linux_memory() | {"tinygrad_live_requested_bytes": int(GlobalCounters.mem_used_per_device["NV"])}

    before_model_load = memory_snapshot()
    model_load_start = time.monotonic_ns()
    realize_weights = args.weight_policy == "realized-fp16"
    started = time.perf_counter_ns()
    gguf_tensor = Tensor.empty(manifest.size_bytes, dtype=dtypes.uint8, device=f"disk:{stable_path}")
    model, _ = Transformer.from_gguf(gguf_tensor, args.max_context, realize=realize_weights)
    device.synchronize()
    model_load_ns = time.perf_counter_ns() - started
    model_load_end = time.monotonic_ns()
    after_model_load = memory_snapshot()
    if args.prompt_tokens + 1 > model.max_context: raise ValueError("prompt and TTFT token exceed effective model context")

    def prefill(prompt: list[int]) -> tuple[int, int]:
      device.synchronize()
      started = time.perf_counter_ns()
      generated = next(model.generate(prompt, chunk_size=args.chunk_size, temperature=0.0))
      device.synchronize()
      return generated, time.perf_counter_ns() - started

    setup_timings, setup_tokens = [], []
    setup_start = time.monotonic_ns()
    for sequence in range(2):
      token, elapsed_ns = prefill(make_prompt(args.prompt_tokens, sequence))
      setup_tokens.append(token)
      setup_timings.append(elapsed_ns)
    setup_end = time.monotonic_ns()
    after_contract_setup = memory_snapshot()

    measured_timings, measured_tokens = [], []
    measurement_start = time.monotonic_ns()
    for sequence in range(2, 2 + args.samples):
      token, elapsed_ns = prefill(make_prompt(args.prompt_tokens, sequence))
      measured_tokens.append(token)
      measured_timings.append(elapsed_ns)
    measurement_end = time.monotonic_ns()
    after_measurement = memory_snapshot()
    sampler.stop()

    samples = sorted(sampler.samples, key=lambda sample: sample["query_end_ns"])
    if sampler.errors: raise RuntimeError(f"nvidia-smi sampling failed: {sampler.errors[0]}")
    peak_bytes = sampled_peak_bytes(samples)
    if peak_bytes is None: raise RuntimeError("nvidia-smi did not observe benchmark process memory")
    result = {
      "schema_version": 1,
      "benchmark": "upstream_prefill",
      "created_at_utc": datetime.now(timezone.utc).isoformat(),
      "checkpoint": {"id": manifest.id, "sha256": manifest.sha256, "size_bytes": manifest.size_bytes},
      "system": {
        "python": platform.python_version(), "platform": platform.platform(), "device": device_metadata,
      },
      "execution": {
        "device": "NV", "jit": 1, "weight_policy": args.weight_policy, "upstream_realize": realize_weights,
        "weight_dtype": "float16", "max_context": model.max_context, "chunk_size": args.chunk_size,
      },
      "workload": {
        "prompt_tokens": args.prompt_tokens, "ttft_tokens_per_sample": 1, "timed_decode_tokens_per_sample": 0,
        "total_generated_tokens": args.samples, "contract_setup_calls": 2, "measured_samples": args.samples,
        "sampling": "greedy",
      },
      "timings": {
        "artifact_verification_ns": artifact_verification_ns, "model_load_ns": model_load_ns,
        "contract_setup_ns": setup_timings, "prefill_ttft_ns": measured_timings,
        "prefill_ttft_summary": timing_summary(measured_timings),
        "prompt_tokens_per_second": [args.prompt_tokens * 1e9 / sample for sample in measured_timings],
      },
      "outputs": {"contract_setup_token_ids": setup_tokens, "measured_token_ids": measured_tokens},
      "memory": {
        "host": {
          "source": "/proc/self/status", "before_artifact_verification": host_before_verification,
          "after_artifact_verification": host_after_verification,
          "after_model_load": {key: after_model_load[key] for key in ("current_rss_bytes", "peak_rss_bytes")},
          "after_contract_setup": {key: after_contract_setup[key] for key in ("current_rss_bytes", "peak_rss_bytes")},
          "after_measurement": {key: after_measurement[key] for key in ("current_rss_bytes", "peak_rss_bytes")},
        },
        "tinygrad": {
          "source": "tinygrad.GlobalCounters.mem_used_per_device", "unit": "live_requested_bytes_not_peak",
          "before_model_load": before_model_load["tinygrad_live_requested_bytes"],
          "after_model_load": after_model_load["tinygrad_live_requested_bytes"],
          "after_contract_setup": after_contract_setup["tinygrad_live_requested_bytes"],
          "after_measurement": after_measurement["tinygrad_live_requested_bytes"],
        },
        "device": {
          "source": "nvidia-smi.compute-apps.sampled", "sample_interval_ms": args.memory_sample_ms,
          "sampled_peak_bytes": peak_bytes,
          "phase_windows_ns": {
            "model_load": {"start_ns": model_load_start, "end_ns": model_load_end},
            "contract_setup": {"start_ns": setup_start, "end_ns": setup_end},
            "measurement": {"start_ns": measurement_start, "end_ns": measurement_end},
          },
          "phase_sampled_peak_bytes": {
            "model_load": sampled_peak_bytes(samples, model_load_start, model_load_end),
            "contract_setup": sampled_peak_bytes(samples, setup_start, setup_end),
            "measurement": sampled_peak_bytes(samples, measurement_start, measurement_end),
          },
          "samples": samples,
          "limitations": [
            "Driver-reported memory is sampled, not an exact high-water mark; shorter transients may be missed.",
            "nvidia-smi reports integer MiB and includes runtime context memory not tracked by tinygrad counters.",
          ],
        },
      },
    }
    write_result(result, args.output)
  except (ArtifactError, ManifestError, OSError, RuntimeError, ValueError) as error:
    if sampler is not None and sampler.is_alive:
      try: sampler.stop()
      except RuntimeError: pass
    print(f"error: {error}", file=sys.stderr)
    return 1
  finally:
    resources.close()
  return 0


if __name__ == "__main__": raise SystemExit(main())
