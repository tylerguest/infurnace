#!/usr/bin/env python3
"""Benchmark sequential upstream generation without claiming a server runtime."""

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


def make_prompt(length: int, sequence: int) -> list[int]:
  return [257 + sequence] + [1000 + (sequence * length + index) % 1000 for index in range(length - 1)]


def measure_generation(generator, device, output_tokens: int) -> dict:
  device.synchronize()
  generation_start = time.perf_counter_ns()
  tokens, completions = [], []
  for _ in range(output_tokens):
    tokens.append(next(generator))
    device.synchronize()
    completions.append(time.perf_counter_ns())
  ttft_ns = completions[0] - generation_start
  inter_token_ns = [current - previous for previous, current in zip(completions, completions[1:])]
  return {
    "token_ids": tokens,
    "ttft_ns": ttft_ns,
    "inter_token_ns": inter_token_ns,
    "end_to_end_ns": completions[-1] - generation_start,
  }


def measure_generations(model, device, prompts: list[list[int]], output_tokens: int, chunk_size: int) -> dict:
  results = [measure_generation(model.generate(list(prompt), chunk_size=chunk_size, temperature=0.0), device, output_tokens)
             for prompt in prompts]
  return {
    "ttft_ns": [result["ttft_ns"] for result in results],
    "inter_token_ns": [result["inter_token_ns"] for result in results],
    "end_to_end_ns": [result["end_to_end_ns"] for result in results],
    "token_ids": [result["token_ids"] for result in results],
  }


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", required=True, type=Path)
  parser.add_argument("--artifact", required=True, type=Path)
  parser.add_argument("--weight-policy", required=True, choices=("lazy", "realized-fp16"))
  parser.add_argument("--max-context", type=positive_int, default=1024)
  parser.add_argument("--chunk-size", type=positive_int, default=32)
  parser.add_argument("--prompt-tokens", type=positive_int, default=16)
  parser.add_argument("--setup-output-tokens", type=positive_int, default=3)
  parser.add_argument("--measured-generations", type=positive_int, default=5)
  parser.add_argument("--output-tokens", type=positive_int, default=16)
  parser.add_argument("--memory-sample-ms", type=positive_int, default=50)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()

  if os.environ.get("DEV") != "NV":
    print("error: upstream end-to-end benchmark requires DEV=NV", file=sys.stderr)
    return 1
  if args.prompt_tokens < 2 or args.chunk_size < 2 or args.prompt_tokens > args.chunk_size:
    print("error: upstream end-to-end benchmark requires 2 <= prompt tokens <= chunk size", file=sys.stderr)
    return 1
  if args.setup_output_tokens < 3 or args.output_tokens < 2:
    print("error: setup requires at least 3 outputs and measurement requires at least 2", file=sys.stderr)
    return 1
  if args.prompt_tokens + max(args.setup_output_tokens, args.output_tokens) > args.max_context:
    print("error: prompt and output tokens exceed max context", file=sys.stderr)
    return 1
  if 257 + 2 + args.measured_generations >= 151936:
    print("error: measured generation count exceeds the fixed Qwen3 prompt contract", file=sys.stderr)
    return 1

  sampler: NvidiaMemorySampler | None = None
  resources = ExitStack()
  startup_started = time.perf_counter_ns()
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
    if args.prompt_tokens + max(args.setup_output_tokens, args.output_tokens) > model.max_context:
      raise ValueError("prompt and output tokens exceed effective model context")

    setup_prompts = [make_prompt(args.prompt_tokens, sequence) for sequence in range(2)]
    setup_start = time.monotonic_ns()
    setup = measure_generations(model, device, setup_prompts, args.setup_output_tokens, args.chunk_size)
    setup_end = time.monotonic_ns()
    setup_elapsed_ns = setup_end - setup_start
    jit_state = {
      "prefill_count": model.prefill_jit.cnt, "prefill_captured": model.prefill_jit.captured is not None,
      "rollout_count": model.rollout_jit.cnt, "rollout_captured": model.rollout_jit.captured is not None,
    }
    if jit_state["prefill_count"] < 2 or not jit_state["prefill_captured"] \
        or jit_state["rollout_count"] < 2 or not jit_state["rollout_captured"]:
      raise RuntimeError("setup did not capture both upstream TinyJit contracts")
    startup_to_ready_ns = time.perf_counter_ns() - startup_started
    after_contract_setup = memory_snapshot()

    measured_prompts = [make_prompt(args.prompt_tokens, sequence) for sequence in range(2, 2 + args.measured_generations)]
    measurement_start = time.monotonic_ns()
    measured = measure_generations(model, device, measured_prompts, args.output_tokens, args.chunk_size)
    measurement_end = time.monotonic_ns()
    measurement_elapsed_ns = measurement_end - measurement_start
    after_measurement = memory_snapshot()
    sampler.stop()

    samples = sorted(sampler.samples, key=lambda sample: sample["query_end_ns"])
    if sampler.errors: raise RuntimeError(f"nvidia-smi sampling failed: {sampler.errors[0]}")
    peak_bytes = sampled_peak_bytes(samples)
    if peak_bytes is None: raise RuntimeError("nvidia-smi did not observe benchmark process memory")
    flat_inter_token = [value for generation in measured["inter_token_ns"] for value in generation]
    generation_rates = [args.output_tokens * 1e9 / value for value in measured["end_to_end_ns"]]
    result = {
      "schema_version": 1,
      "benchmark": "upstream_end_to_end",
      "created_at_utc": datetime.now(timezone.utc).isoformat(),
      "checkpoint": {"id": manifest.id, "sha256": manifest.sha256, "size_bytes": manifest.size_bytes},
      "system": {
        "python": platform.python_version(), "platform": platform.platform(), "device": device_metadata,
      },
      "execution": {
        "device": "NV", "jit": 1, "weight_policy": args.weight_policy, "upstream_realize": realize_weights,
        "weight_dtype": "float16", "max_context": model.max_context, "chunk_size": args.chunk_size,
        "path": "tinygrad.llm.Transformer.generate", "server_runtime": False, "transport": "none",
        "tokenization": "none", "scheduling": "none", "concurrency": 1, "jit_state": jit_state,
      },
      "workload": {
        "prompt_tokens": args.prompt_tokens, "setup_generations": 2,
        "setup_output_tokens_per_generation": args.setup_output_tokens,
        "measured_generations": args.measured_generations,
        "measured_output_tokens_per_generation": args.output_tokens,
        "total_setup_generated_tokens": 2 * args.setup_output_tokens,
        "total_measured_generated_tokens": args.measured_generations * args.output_tokens,
        "sampling": "greedy", "arrival_pattern": "closed_loop_sequential",
      },
      "timings": {
        "artifact_verification_ns": artifact_verification_ns, "model_load_ns": model_load_ns,
        "startup_to_ready_ns": startup_to_ready_ns,
        "contract_setup_elapsed_ns": setup_elapsed_ns, "measurement_elapsed_ns": measurement_elapsed_ns,
        "contract_setup": {
          "ttft_ns": setup["ttft_ns"], "inter_token_ns": setup["inter_token_ns"],
          "end_to_end_ns": setup["end_to_end_ns"],
        },
        "measured_ttft_ns": measured["ttft_ns"], "measured_ttft_summary": timing_summary(measured["ttft_ns"]),
        "measured_inter_token_ns": measured["inter_token_ns"],
        "measured_inter_token_summary": timing_summary(flat_inter_token),
        "measured_end_to_end_ns": measured["end_to_end_ns"],
        "measured_end_to_end_summary": timing_summary(measured["end_to_end_ns"]),
        "generated_tokens_per_second": generation_rates,
        "aggregate_generated_tokens_per_second": args.measured_generations * args.output_tokens * 1e9 / measurement_elapsed_ns,
      },
      "outputs": {
        "contract_setup_prompt_token_ids": setup_prompts, "contract_setup_token_ids": setup["token_ids"],
        "measured_prompt_token_ids": measured_prompts, "measured_token_ids": measured["token_ids"],
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
