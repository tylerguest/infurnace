#!/usr/bin/env python3
"""Benchmark upstream Qwen steady one-token decode on DEV=NV."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

try:
  from benchmarks.common import NvidiaMemorySampler, positive_int, query_device, read_linux_memory, sampled_peak_bytes, timing_summary, write_result
except ModuleNotFoundError:
  from common import NvidiaMemorySampler, positive_int, query_device, read_linux_memory, sampled_peak_bytes, timing_summary, write_result
from infurnace.models.manifest import ArtifactError, ManifestError, load_manifest, verify_artifact
try:
  from tools.gguf_inspection import stable_artifact_path
except ModuleNotFoundError:
  sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
  from gguf_inspection import stable_artifact_path


def make_prompt(length: int) -> list[int]:
  return [257] + [1000 + index for index in range(length - 1)]


def measure_generation(generator, device, decode_tokens: int, snapshot=lambda: None) -> dict:
  def next_token() -> tuple[int, int]:
    device.synchronize()
    started = time.perf_counter_ns()
    generated = next(generator)
    device.synchronize()
    return generated, time.perf_counter_ns() - started

  prefill_start = time.monotonic_ns()
  prefill_token, cold_prefill_ttft_ns = next_token()
  prefill_end = time.monotonic_ns()
  after_prefill = snapshot()

  setup_timings, setup_tokens = [], []
  setup_start = time.monotonic_ns()
  for _ in range(2):
    token, elapsed_ns = next_token()
    setup_tokens.append(token)
    setup_timings.append(elapsed_ns)
  setup_end = time.monotonic_ns()
  after_contract_setup = snapshot()

  measured_timings, measured_tokens = [], []
  measurement_start = time.monotonic_ns()
  for _ in range(decode_tokens):
    token, elapsed_ns = next_token()
    measured_tokens.append(token)
    measured_timings.append(elapsed_ns)
  measurement_end = time.monotonic_ns()
  after_measurement = snapshot()
  return {
    "prefill_token": prefill_token, "cold_prefill_ttft_ns": cold_prefill_ttft_ns,
    "setup_timings": setup_timings, "setup_tokens": setup_tokens,
    "measured_timings": measured_timings, "measured_tokens": measured_tokens,
    "snapshots": {
      "after_prefill": after_prefill, "after_contract_setup": after_contract_setup,
      "after_measurement": after_measurement,
    },
    "phase_windows": {
      "prefill": (prefill_start, prefill_end), "contract_setup": (setup_start, setup_end),
      "measurement": (measurement_start, measurement_end),
    },
  }


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", required=True, type=Path)
  parser.add_argument("--artifact", required=True, type=Path)
  parser.add_argument("--weight-policy", required=True, choices=("lazy", "realized-fp16"))
  parser.add_argument("--max-context", type=positive_int, default=1024)
  parser.add_argument("--chunk-size", type=positive_int, default=32)
  parser.add_argument("--prompt-tokens", type=positive_int, default=16)
  parser.add_argument("--decode-tokens", type=positive_int, default=16)
  parser.add_argument("--memory-sample-ms", type=positive_int, default=50)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()

  if os.environ.get("DEV") != "NV":
    print("error: upstream decode benchmark requires DEV=NV", file=sys.stderr)
    return 1
  if args.prompt_tokens < 2 or args.chunk_size < 2 or args.prompt_tokens > args.chunk_size:
    print("error: upstream decode benchmark requires 2 <= prompt tokens <= chunk size", file=sys.stderr)
    return 1
  total_generated_tokens = 1 + 2 + args.decode_tokens
  if args.prompt_tokens + total_generated_tokens > args.max_context:
    print("error: prompt and generated tokens exceed max context", file=sys.stderr)
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
    if args.prompt_tokens + total_generated_tokens > model.max_context:
      raise ValueError("prompt and generated tokens exceed effective model context")

    generator = model.generate(make_prompt(args.prompt_tokens), chunk_size=args.chunk_size, temperature=0.0)
    generation = measure_generation(generator, device, args.decode_tokens, memory_snapshot)
    sampler.stop()

    prefill_start, prefill_end = generation["phase_windows"]["prefill"]
    setup_start, setup_end = generation["phase_windows"]["contract_setup"]
    measurement_start, measurement_end = generation["phase_windows"]["measurement"]
    setup_timings, measured_timings = generation["setup_timings"], generation["measured_timings"]
    after_prefill = generation["snapshots"]["after_prefill"]
    after_contract_setup = generation["snapshots"]["after_contract_setup"]
    after_measurement = generation["snapshots"]["after_measurement"]

    samples = sorted(sampler.samples, key=lambda sample: sample["query_end_ns"])
    if sampler.errors: raise RuntimeError(f"nvidia-smi sampling failed: {sampler.errors[0]}")
    peak_bytes = sampled_peak_bytes(samples)
    if peak_bytes is None: raise RuntimeError("nvidia-smi did not observe benchmark process memory")
    result = {
      "schema_version": 1,
      "benchmark": "upstream_decode",
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
        "prompt_tokens": args.prompt_tokens, "ttft_tokens": 1, "contract_setup_tokens": 2,
        "timed_decode_tokens": args.decode_tokens, "total_generated_tokens": total_generated_tokens,
        "measured_samples": args.decode_tokens, "sampling": "greedy",
      },
      "timings": {
        "artifact_verification_ns": artifact_verification_ns, "model_load_ns": model_load_ns,
        "cold_prefill_ttft_ns": generation["cold_prefill_ttft_ns"], "contract_setup_ns": setup_timings,
        "decode_ns": measured_timings, "decode_summary": timing_summary(measured_timings),
        "generated_tokens_per_second": [1e9 / sample for sample in measured_timings],
      },
      "outputs": {
        "prefill_token_id": generation["prefill_token"], "contract_setup_token_ids": generation["setup_tokens"],
        "measured_token_ids": generation["measured_tokens"],
      },
      "memory": {
        "host": {
          "source": "/proc/self/status", "before_artifact_verification": host_before_verification,
          "after_artifact_verification": host_after_verification,
          "after_model_load": {key: after_model_load[key] for key in ("current_rss_bytes", "peak_rss_bytes")},
          "after_prefill": {key: after_prefill[key] for key in ("current_rss_bytes", "peak_rss_bytes")},
          "after_contract_setup": {key: after_contract_setup[key] for key in ("current_rss_bytes", "peak_rss_bytes")},
          "after_measurement": {key: after_measurement[key] for key in ("current_rss_bytes", "peak_rss_bytes")},
        },
        "tinygrad": {
          "source": "tinygrad.GlobalCounters.mem_used_per_device", "unit": "live_requested_bytes_not_peak",
          "before_model_load": before_model_load["tinygrad_live_requested_bytes"],
          "after_model_load": after_model_load["tinygrad_live_requested_bytes"],
          "after_prefill": after_prefill["tinygrad_live_requested_bytes"],
          "after_contract_setup": after_contract_setup["tinygrad_live_requested_bytes"],
          "after_measurement": after_measurement["tinygrad_live_requested_bytes"],
        },
        "device": {
          "source": "nvidia-smi.compute-apps.sampled", "sample_interval_ms": args.memory_sample_ms,
          "sampled_peak_bytes": peak_bytes,
          "phase_windows_ns": {
            "model_load": {"start_ns": model_load_start, "end_ns": model_load_end},
            "prefill": {"start_ns": prefill_start, "end_ns": prefill_end},
            "contract_setup": {"start_ns": setup_start, "end_ns": setup_end},
            "measurement": {"start_ns": measurement_start, "end_ns": measurement_end},
          },
          "phase_sampled_peak_bytes": {
            "model_load": sampled_peak_bytes(samples, model_load_start, model_load_end),
            "prefill": sampled_peak_bytes(samples, prefill_start, prefill_end),
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
  except (ArtifactError, ManifestError, OSError, RuntimeError, StopIteration, ValueError) as error:
    if sampler is not None and sampler.is_alive:
      try: sampler.stop()
      except RuntimeError: pass
    print(f"error: {error}", file=sys.stderr)
    return 1
  finally:
    resources.close()
  return 0


if __name__ == "__main__": raise SystemExit(main())
