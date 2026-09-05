#!/usr/bin/env python3
"""Benchmark Infurnace engine steady decode on DEV=NV.

Drives the offline engine (request lifecycle, scheduler, runner, greedy
sampler) on the pinned Qwen3-4B Q4_K_M checkpoint. One generation records a
cold prefill/TTFT token, two one-token decode calls that capture the TinyJit
contract, and then individually timed steady replay tokens. Reports the same
headline metrics as the llama.cpp bar (decode tok/s, TTFT) plus achieved
memory bandwidth and phase memory snapshots.
"""
from __future__ import annotations
import argparse
import json
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

def build_engine(artifact: str, manifest_path: str, *, num_slots: int, max_context: int):
  from infurnace.engine import Engine
  from infurnace.scheduler.scheduler import Scheduler
  from infurnace.sampler import GreedySampler
  from infurnace.tokenizer import GGUFTokenizer
  from infurnace.executor.tinygrad.runner import Qwen3Runner
  from infurnace.executor.tinygrad.weights import WeightPolicy, load_qwen3_checkpoint
  manifest = load_manifest(manifest_path)
  checkpoint = load_qwen3_checkpoint(artifact, manifest, WeightPolicy.LAZY_FP16)
  tokenizer = GGUFTokenizer.from_gguf_metadata(checkpoint.metadata)
  runner = Qwen3Runner.from_weights(checkpoint.weights, num_slots=num_slots, max_context=max_context)
  return Engine(runner, Scheduler(num_slots=num_slots, max_context=max_context), GreedySampler(), tokenizer)

def measure_generation(engine, prompt: list[int], max_tokens: int, max_context: int, snapshot=lambda: None) -> dict:
  from infurnace.engine.request import Request, SamplingParams
  request = Request("bench", prompt, SamplingParams(max_tokens=max_tokens), max_context)
  engine.add_request(request)

  prefill_start = time.monotonic_ns()
  step_started = time.perf_counter_ns()
  first = engine.step()
  cold_prefill_ttft_ns = time.perf_counter_ns() - step_started
  prefill_token = first.new_tokens["bench"][0] if first.new_tokens.get("bench") else None
  prefill_end = time.monotonic_ns()
  after_prefill = snapshot()

  setup_timings, setup_tokens = [], []
  setup_start = time.monotonic_ns()
  for _ in range(2):
    step_started = time.perf_counter_ns()
    result = engine.step()
    elapsed = time.perf_counter_ns() - step_started
    setup_tokens.append(result.new_tokens["bench"][0])
    setup_timings.append(elapsed)
  setup_end = time.monotonic_ns()
  after_contract_setup = snapshot()

  measured_timings, measured_tokens = [], []
  measurement_start = time.monotonic_ns()
  while not engine.is_done():
    step_started = time.perf_counter_ns()
    result = engine.step()
    measured_timings.append(time.perf_counter_ns() - step_started)
    measured_tokens.append(result.new_tokens["bench"][0])
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
  parser.add_argument("--max-context", type=positive_int, default=512)
  parser.add_argument("--chunk-size", type=positive_int, default=16)
  parser.add_argument("--prompt-tokens", type=positive_int, default=16)
  parser.add_argument("--decode-tokens", type=positive_int, default=200)
  parser.add_argument("--memory-sample-ms", type=positive_int, default=50)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()

  if os.environ.get("DEV") != "NV":
    print("error: Infurnace decode benchmark requires DEV=NV", file=sys.stderr)
    return 1
  if args.prompt_tokens < 2 or args.chunk_size < 2 or args.prompt_tokens > args.chunk_size:
    print("error: requires 2 <= prompt tokens <= chunk size", file=sys.stderr)
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

    sampler = NvidiaMemorySampler(os.getpid(), args.memory_sample_ms)
    sampler.start()
    from tinygrad import Device, GlobalCounters
    device = Device["NV"]
    device.synchronize()

    def memory_snapshot() -> dict[str, int]:
      return read_linux_memory() | {"tinygrad_live_requested_bytes": int(GlobalCounters.mem_used_per_device["NV"])}

    before_model_load = memory_snapshot()
    model_load_start = time.monotonic_ns()
    started = time.perf_counter_ns()
    engine = build_engine(stable_path, args.manifest, num_slots=1, max_context=args.max_context)
    device.synchronize()
    model_load_ns = time.perf_counter_ns() - started
    model_load_end = time.monotonic_ns()
    after_model_load = memory_snapshot()

    prompt = make_prompt(args.prompt_tokens)
    generation = measure_generation(engine, prompt, total_generated_tokens, args.max_context, memory_snapshot)
    sampler.stop()

    prefill_start, prefill_end = generation["phase_windows"]["prefill"]
    setup_start, setup_end = generation["phase_windows"]["contract_setup"]
    measurement_start, measurement_end = generation["phase_windows"]["measurement"]
    setup_timings, measured_timings = generation["setup_timings"], generation["measured_timings"]
    after_prefill = generation["snapshots"]["after_prefill"]
    after_contract_setup = generation["snapshots"]["after_contract_setup"]
    after_measurement = generation["snapshots"]["after_measurement"]
    if len(setup_timings) != 2 or len(measured_timings) != args.decode_tokens:
      raise RuntimeError(f"expected 2 setup + {args.decode_tokens} decode tokens, "
                         f"got {len(setup_timings)} + {len(measured_timings)}")

    samples = sorted(sampler.samples, key=lambda sample: sample["query_end_ns"])
    if sampler.errors: raise RuntimeError(f"nvidia-smi sampling failed: {sampler.errors[0]}")
    peak_bytes = sampled_peak_bytes(samples)
    if peak_bytes is None: raise RuntimeError("nvidia-smi did not observe benchmark process memory")

    inspection = json.loads((Path("models") / f"{manifest.id}.inspection.json").read_text(encoding="utf-8"))
    weights_bytes = sum(tensor["serialized_nbytes"] for tensor in inspection["gguf"]["tensors"])

    result = {
      "schema_version": 1,
      "benchmark": "infurnace_decode",
      "created_at_utc": datetime.now(timezone.utc).isoformat(),
      "checkpoint": {"id": manifest.id, "sha256": manifest.sha256, "size_bytes": manifest.size_bytes},
      "system": {
        "python": platform.python_version(), "platform": platform.platform(), "device": device_metadata,
      },
      "execution": {
        "device": "NV", "jit": 1, "weight_policy": "lazy", "upstream_realize": False,
        "weight_dtype": "float16", "max_context": args.max_context, "chunk_size": args.chunk_size,
      },
      "workload": {
        "prompt_tokens": args.prompt_tokens, "ttft_tokens": 1, "contract_setup_tokens": 2,
        "timed_decode_tokens": args.decode_tokens, "total_generated_tokens": total_generated_tokens,
        "measured_samples": args.decode_tokens, "sampling": "greedy",
      },
      "timings": {
        "artifact_verification_ns": artifact_verification_ns, "model_load_ns": model_load_ns,
        "cold_prefill_ttft_ns": generation["cold_prefill_ttft_ns"],
        "decode_capture_ns": setup_timings[0], "contract_setup_ns": setup_timings,
        "decode_ns": measured_timings, "decode_summary": timing_summary(measured_timings),
        "generated_tokens_per_second": [1e9 / sample for sample in measured_timings],
        "weights_bytes": weights_bytes,
        "achieved_bytes_per_second": int(weights_bytes * 1e9 / (sum(measured_timings) / len(measured_timings))),
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

    mean_tok_s = 1e9 / (sum(measured_timings) / len(measured_timings))
    print(f"TTFT (cold, {args.prompt_tokens}-token prefill): {generation['cold_prefill_ttft_ns'] / 1e6:.1f} ms")
    print(f"decode contract capture: {setup_timings[0] / 1e6:.1f} ms")
    print(f"decode: {mean_tok_s:.2f} tok/s (median {sorted(measured_timings)[len(measured_timings)//2] / 1e6:.1f} ms/token)")
    print(f"achieved bandwidth: {weights_bytes * mean_tok_s / 1e9:.1f} GB/s of ~448 GB/s ceiling")
    print(f"peak device memory: {peak_bytes / (1024 ** 3):.2f} GiB")
  except (ArtifactError, ManifestError, OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
    if sampler is not None and sampler.is_alive:
      try: sampler.stop()
      except RuntimeError: pass
    print(f"error: {error}", file=sys.stderr)
    return 1
  finally:
    resources.close()
  return 0

if __name__ == "__main__": raise SystemExit(main())