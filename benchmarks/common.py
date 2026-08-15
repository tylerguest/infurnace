"""Shared contracts and telemetry helpers for development benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
MIB = 1024 * 1024


def positive_int(value: str) -> int:
  parsed = int(value)
  if parsed <= 0: raise argparse.ArgumentTypeError("must be positive")
  return parsed


def query_device() -> dict[str, Any]:
  command = [
    "nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free",
    "--format=csv,noheader,nounits",
  ]
  try: result = subprocess.run(command, capture_output=True, text=True, timeout=10)
  except subprocess.TimeoutExpired as error: raise RuntimeError("nvidia-smi device query timed out") from error
  if result.returncode != 0: raise RuntimeError(result.stderr.strip() or f"nvidia-smi exited {result.returncode}")
  rows = [line for line in result.stdout.splitlines() if line.strip()]
  if len(rows) != 1: raise RuntimeError("the initial benchmark contract requires exactly one NVIDIA GPU")
  fields = [field.strip() for field in rows[0].split(",")]
  if len(fields) != 7: raise RuntimeError(f"unexpected nvidia-smi GPU row: {rows[0]!r}")
  try: index, total_mib, used_mib, free_mib = int(fields[0]), int(fields[4]), int(fields[5]), int(fields[6])
  except ValueError as error: raise RuntimeError(f"invalid nvidia-smi GPU row: {rows[0]!r}") from error
  return {
    "index": index, "uuid": fields[1], "name": fields[2], "driver_version": fields[3],
    "total_memory_bytes": total_mib * MIB,
    "baseline_used_memory_bytes": used_mib * MIB,
    "baseline_free_memory_bytes": free_mib * MIB,
  }


def timing_summary(samples_ns: list[int]) -> dict[str, int | float]:
  if not samples_ns or any(type(sample) is not int or sample <= 0 for sample in samples_ns):
    raise ValueError("timing samples must be positive integers")
  return {
    "count": len(samples_ns),
    "min_ns": min(samples_ns),
    "max_ns": max(samples_ns),
    "mean_ns": statistics.fmean(samples_ns),
    "median_ns": statistics.median(samples_ns),
  }


def sampled_peak_bytes(samples: list[dict[str, Any]], start_ns: int | None = None, end_ns: int | None = None) -> int | None:
  values = [sample["used_bytes"] for sample in samples
            if sample["used_bytes"] is not None
            and (start_ns is None or sample["query_end_ns"] >= start_ns)
            and (end_ns is None or sample["query_start_ns"] <= end_ns)]
  return max(values) if values else None


def validate_generation_timings(group: dict[str, Any], generations: int, output_tokens: int, label: str) -> tuple[list[int], list[list[int]], list[int]]:
  if not isinstance(group, dict) or set(group) != {"ttft_ns", "inter_token_ns", "end_to_end_ns"}:
    raise ValueError(f"{label} generation timings are incomplete")
  ttft, inter_token, end_to_end = group["ttft_ns"], group["inter_token_ns"], group["end_to_end_ns"]
  if not all(isinstance(values, list) and len(values) == generations for values in (ttft, inter_token, end_to_end)):
    raise ValueError(f"{label} generation count is invalid")
  if any(type(value) is not int or value <= 0 for value in ttft + end_to_end):
    raise ValueError(f"{label} generation timing is invalid")
  for index, intervals in enumerate(inter_token):
    if not isinstance(intervals, list) or len(intervals) != output_tokens - 1 \
        or any(type(value) is not int or value <= 0 for value in intervals):
      raise ValueError(f"{label} inter-token timing is invalid")
    if end_to_end[index] != ttft[index] + sum(intervals):
      raise ValueError(f"{label} end-to-end timing does not decompose exactly")
  return ttft, inter_token, end_to_end


def parse_compute_app_rows(output: str, pid: int) -> tuple[str | None, int | None]:
  matches: list[tuple[str, int]] = []
  for line in output.splitlines():
    if not line.strip(): continue
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 3: raise ValueError(f"unexpected nvidia-smi compute-app row: {line!r}")
    try: row_pid = int(fields[0])
    except ValueError as error: raise ValueError(f"invalid nvidia-smi compute-app row: {line!r}") from error
    if row_pid != pid: continue
    try: used_mib = int(fields[2])
    except ValueError as error: raise ValueError(f"invalid nvidia-smi compute-app row: {line!r}") from error
    matches.append((fields[1], used_mib * MIB))
  if len(matches) > 1: raise ValueError(f"process {pid} appears on multiple NVIDIA devices")
  return matches[0] if matches else (None, None)


def read_linux_memory() -> dict[str, int]:
  values: dict[str, int] = {}
  with Path("/proc/self/status").open(encoding="utf-8") as status:
    for line in status:
      name, separator, value = line.partition(":")
      if separator and name in ("VmRSS", "VmHWM"):
        fields = value.split()
        if len(fields) != 2 or fields[1] != "kB": raise RuntimeError(f"unexpected /proc memory value: {line.strip()}")
        values[name] = int(fields[0]) * 1024
  if values.keys() != {"VmRSS", "VmHWM"}: raise RuntimeError("/proc/self/status does not expose VmRSS and VmHWM")
  return {"current_rss_bytes": values["VmRSS"], "peak_rss_bytes": values["VmHWM"]}


class NvidiaMemorySampler:
  """Sample driver-reported memory for one process without an NVML dependency."""

  def __init__(self, pid: int, interval_ms: int):
    if interval_ms <= 0: raise ValueError("sample interval must be positive")
    self.pid, self.interval_ms = pid, interval_ms
    self.samples: list[dict[str, Any]] = []
    self.errors: list[str] = []
    self._stop = threading.Event()
    self._lock = threading.Lock()
    self._thread = threading.Thread(target=self._run, name="nvidia-memory-sampler", daemon=True)

  def start(self) -> None:
    self._thread.start()

  def stop(self) -> None:
    self.sample_once()
    self._stop.set()
    self._thread.join(timeout=10)
    if self._thread.is_alive(): raise RuntimeError("nvidia-smi sampler did not stop")

  @property
  def is_alive(self) -> bool:
    return self._thread.is_alive()

  def sample_once(self) -> None:
    command = [
      "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
      "--format=csv,noheader,nounits",
    ]
    try:
      query_start_ns = time.monotonic_ns()
      result = subprocess.run(command, capture_output=True, text=True, timeout=5)
      query_end_ns = time.monotonic_ns()
      if result.returncode != 0: raise RuntimeError(result.stderr.strip() or f"nvidia-smi exited {result.returncode}")
      gpu_uuid, used_bytes = parse_compute_app_rows(result.stdout, self.pid)
      sample = {"query_start_ns": query_start_ns, "query_end_ns": query_end_ns, "gpu_uuid": gpu_uuid, "used_bytes": used_bytes}
      with self._lock: self.samples.append(sample)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
      with self._lock: self.errors.append(str(error))

  def _run(self) -> None:
    interval_seconds = self.interval_ms / 1000
    while not self._stop.is_set():
      started = time.monotonic()
      self.sample_once()
      self._stop.wait(max(0.0, interval_seconds - (time.monotonic() - started)))


def validate_result(result: dict[str, Any]) -> None:
  required_root = {"schema_version", "benchmark", "created_at_utc", "checkpoint", "system", "execution", "workload", "timings", "outputs", "memory"}
  if set(result) != required_root: raise ValueError("benchmark result fields are incomplete or unknown")
  if result.get("schema_version") != SCHEMA_VERSION: raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
  benchmark = result.get("benchmark")
  if benchmark not in ("upstream_prefill", "upstream_decode", "upstream_end_to_end"):
    raise ValueError("benchmark type is invalid")
  if not isinstance(result.get("created_at_utc"), str) or not result["created_at_utc"].endswith("+00:00"):
    raise ValueError("created_at_utc must be an explicit UTC timestamp")

  checkpoint = result.get("checkpoint")
  if not isinstance(checkpoint, dict) or set(checkpoint) != {"id", "sha256", "size_bytes"}:
    raise ValueError("checkpoint identity is incomplete")
  if not isinstance(checkpoint["id"], str) or not checkpoint["id"]: raise ValueError("checkpoint id must be nonempty")
  if not isinstance(checkpoint["sha256"], str) or len(checkpoint["sha256"]) != 64: raise ValueError("checkpoint sha256 is invalid")
  try: int(checkpoint["sha256"], 16)
  except ValueError as error: raise ValueError("checkpoint sha256 is invalid") from error
  if type(checkpoint["size_bytes"]) is not int or checkpoint["size_bytes"] <= 0: raise ValueError("checkpoint size is invalid")

  system = result.get("system")
  if not isinstance(system, dict) or set(system) != {"python", "platform", "device"}: raise ValueError("system metadata is incomplete")
  if any(not isinstance(system[field], str) or not system[field] for field in ("python", "platform")):
    raise ValueError("system software metadata is invalid")
  device = system["device"]
  required_device = {"index", "uuid", "name", "driver_version", "total_memory_bytes", "baseline_used_memory_bytes", "baseline_free_memory_bytes"}
  if not isinstance(device, dict) or set(device) != required_device: raise ValueError("device metadata is incomplete")
  if any(not isinstance(device[field], str) or not device[field] for field in ("uuid", "name", "driver_version")):
    raise ValueError("device identity is invalid")
  for field in ("index", "total_memory_bytes", "baseline_used_memory_bytes", "baseline_free_memory_bytes"):
    if type(device[field]) is not int or device[field] < 0: raise ValueError(f"device {field} is invalid")

  execution = result.get("execution")
  required_execution = {"device", "jit", "weight_policy", "upstream_realize", "weight_dtype", "max_context", "chunk_size"}
  if benchmark == "upstream_end_to_end":
    required_execution |= {"path", "server_runtime", "transport", "tokenization", "scheduling", "concurrency", "jit_state"}
  if not isinstance(execution, dict) or set(execution) != required_execution: raise ValueError("execution settings are incomplete")
  if execution["device"] != "NV" or type(execution["jit"]) is not int or execution["jit"] != 1:
    raise ValueError("benchmark requires DEV=NV and JIT=1")
  if execution["weight_policy"] not in ("lazy", "realized-fp16"): raise ValueError("weight policy is invalid")
  if execution["upstream_realize"] is not (execution["weight_policy"] == "realized-fp16"):
    raise ValueError("weight policy does not match upstream realization")
  if execution["weight_dtype"] != "float16": raise ValueError("weight dtype is invalid")
  for field in ("max_context", "chunk_size"):
    if type(execution[field]) is not int or execution[field] <= 0: raise ValueError(f"{field} must be positive")
  if benchmark == "upstream_end_to_end" and (
      execution["path"] != "tinygrad.llm.Transformer.generate" or execution["server_runtime"] is not False
      or execution["transport"] != "none" or execution["tokenization"] != "none"
      or execution["scheduling"] != "none" or type(execution["concurrency"]) is not int or execution["concurrency"] != 1):
    raise ValueError("end-to-end scope must not claim server behavior")
  if benchmark == "upstream_end_to_end":
    jit_state = execution["jit_state"]
    if not isinstance(jit_state, dict) or set(jit_state) != {"prefill_count", "prefill_captured", "rollout_count", "rollout_captured"} \
        or type(jit_state["prefill_count"]) is not int or jit_state["prefill_count"] < 2 \
        or type(jit_state["rollout_count"]) is not int or jit_state["rollout_count"] < 2 \
        or jit_state["prefill_captured"] is not True or jit_state["rollout_captured"] is not True:
      raise ValueError("end-to-end TinyJit contracts are not ready")

  workload, timings, outputs = result.get("workload"), result.get("timings"), result.get("outputs")
  if not isinstance(workload, dict) or not isinstance(timings, dict) or not isinstance(outputs, dict):
    raise ValueError("benchmark-specific result fields are invalid")

  if benchmark == "upstream_prefill":
    required_workload = {"prompt_tokens", "ttft_tokens_per_sample", "timed_decode_tokens_per_sample", "total_generated_tokens", "contract_setup_calls", "measured_samples", "sampling"}
    required_timings = {"artifact_verification_ns", "model_load_ns", "contract_setup_ns", "prefill_ttft_ns", "prefill_ttft_summary", "prompt_tokens_per_second"}
    if set(workload) != required_workload or set(timings) != required_timings: raise ValueError("prefill result fields are incomplete")
    for field in ("prompt_tokens", "contract_setup_calls", "measured_samples"):
      if type(workload[field]) is not int or workload[field] <= 0: raise ValueError(f"{field} must be positive")
    if workload["contract_setup_calls"] != 2 or workload["ttft_tokens_per_sample"] != 1 or workload["timed_decode_tokens_per_sample"] != 0 \
        or workload["total_generated_tokens"] != workload["contract_setup_calls"] + workload["measured_samples"] \
        or workload["sampling"] != "greedy":
      raise ValueError("prefill token accounting is invalid")
    if workload["prompt_tokens"] < 2 or workload["prompt_tokens"] > execution["chunk_size"] \
        or workload["prompt_tokens"] + workload["ttft_tokens_per_sample"] > execution["max_context"]:
      raise ValueError("prefill workload exceeds its execution contract")
    setup, measured = timings["contract_setup_ns"], timings["prefill_ttft_ns"]
    summary, rates = timings["prefill_ttft_summary"], timings["prompt_tokens_per_second"]
    rate_numerator = workload["prompt_tokens"] * 1e9
    required_outputs = {"contract_setup_token_ids", "measured_token_ids"}
    expected_output_lengths = (workload["contract_setup_calls"], workload["measured_samples"])
    host_boundaries = {"before_artifact_verification", "after_artifact_verification", "after_model_load", "after_contract_setup", "after_measurement"}
    tinygrad_boundaries = {"before_model_load", "after_model_load", "after_contract_setup", "after_measurement"}
    phases = ("model_load", "contract_setup", "measurement")
  elif benchmark == "upstream_decode":
    required_workload = {"prompt_tokens", "ttft_tokens", "contract_setup_tokens", "timed_decode_tokens", "total_generated_tokens", "measured_samples", "sampling"}
    required_timings = {"artifact_verification_ns", "model_load_ns", "cold_prefill_ttft_ns", "contract_setup_ns", "decode_ns", "decode_summary", "generated_tokens_per_second"}
    if set(workload) != required_workload or set(timings) != required_timings: raise ValueError("decode result fields are incomplete")
    for field in ("prompt_tokens", "ttft_tokens", "contract_setup_tokens", "timed_decode_tokens", "total_generated_tokens", "measured_samples"):
      if type(workload[field]) is not int or workload[field] <= 0: raise ValueError(f"{field} must be positive")
    if workload["ttft_tokens"] != 1 or workload["contract_setup_tokens"] != 2 \
        or workload["timed_decode_tokens"] != workload["measured_samples"] \
        or workload["total_generated_tokens"] != workload["ttft_tokens"] + workload["contract_setup_tokens"] + workload["timed_decode_tokens"] \
        or workload["sampling"] != "greedy":
      raise ValueError("decode token accounting is invalid")
    if workload["prompt_tokens"] < 2 or workload["prompt_tokens"] > execution["chunk_size"] \
        or workload["prompt_tokens"] + workload["total_generated_tokens"] > execution["max_context"]:
      raise ValueError("decode workload exceeds its execution contract")
    if type(timings["cold_prefill_ttft_ns"]) is not int or timings["cold_prefill_ttft_ns"] <= 0:
      raise ValueError("decode cold prefill timing is invalid")
    setup, measured = timings["contract_setup_ns"], timings["decode_ns"]
    summary, rates = timings["decode_summary"], timings["generated_tokens_per_second"]
    rate_numerator = 1e9
    required_outputs = {"prefill_token_id", "contract_setup_token_ids", "measured_token_ids"}
    expected_output_lengths = (workload["contract_setup_tokens"], workload["measured_samples"])
    host_boundaries = {"before_artifact_verification", "after_artifact_verification", "after_model_load", "after_prefill", "after_contract_setup", "after_measurement"}
    tinygrad_boundaries = {"before_model_load", "after_model_load", "after_prefill", "after_contract_setup", "after_measurement"}
    phases = ("model_load", "prefill", "contract_setup", "measurement")
  else:
    required_workload = {
      "prompt_tokens", "setup_generations", "setup_output_tokens_per_generation", "measured_generations",
      "measured_output_tokens_per_generation", "total_setup_generated_tokens", "total_measured_generated_tokens",
      "sampling", "arrival_pattern",
    }
    required_timings = {
      "artifact_verification_ns", "model_load_ns", "startup_to_ready_ns", "contract_setup",
      "contract_setup_elapsed_ns", "measurement_elapsed_ns",
      "measured_ttft_ns", "measured_ttft_summary", "measured_inter_token_ns", "measured_inter_token_summary",
      "measured_end_to_end_ns", "measured_end_to_end_summary", "generated_tokens_per_second",
      "aggregate_generated_tokens_per_second",
    }
    if set(workload) != required_workload or set(timings) != required_timings:
      raise ValueError("end-to-end result fields are incomplete")
    integer_fields = (
      "prompt_tokens", "setup_generations", "setup_output_tokens_per_generation", "measured_generations",
      "measured_output_tokens_per_generation", "total_setup_generated_tokens", "total_measured_generated_tokens",
    )
    if any(type(workload[field]) is not int or workload[field] <= 0 for field in integer_fields):
      raise ValueError("end-to-end workload values must be positive integers")
    if workload["setup_generations"] != 2 or workload["setup_output_tokens_per_generation"] < 3 \
        or workload["measured_generations"] <= 0 or workload["measured_output_tokens_per_generation"] < 2 \
        or workload["total_setup_generated_tokens"] != workload["setup_generations"] * workload["setup_output_tokens_per_generation"] \
        or workload["total_measured_generated_tokens"] != workload["measured_generations"] * workload["measured_output_tokens_per_generation"] \
        or workload["sampling"] != "greedy" or workload["arrival_pattern"] != "closed_loop_sequential":
      raise ValueError("end-to-end workload contract is invalid")
    if workload["prompt_tokens"] < 2 or workload["prompt_tokens"] > execution["chunk_size"] \
        or workload["prompt_tokens"] + max(workload["setup_output_tokens_per_generation"], workload["measured_output_tokens_per_generation"]) > execution["max_context"]:
      raise ValueError("end-to-end workload exceeds its execution contract")

    setup_group = timings["contract_setup"]
    setup_ttft, setup_inter, setup_end_to_end = validate_generation_timings(
      setup_group, workload["setup_generations"], workload["setup_output_tokens_per_generation"], "setup")
    measured_group = {
      "ttft_ns": timings["measured_ttft_ns"], "inter_token_ns": timings["measured_inter_token_ns"],
      "end_to_end_ns": timings["measured_end_to_end_ns"],
    }
    measured_ttft, measured_inter, measured_end_to_end = validate_generation_timings(
      measured_group, workload["measured_generations"], workload["measured_output_tokens_per_generation"], "measured")
    flat_inter = [value for generation in measured_inter for value in generation]
    if timing_summary(measured_ttft) != timings["measured_ttft_summary"] \
        or timing_summary(flat_inter) != timings["measured_inter_token_summary"] \
        or timing_summary(measured_end_to_end) != timings["measured_end_to_end_summary"]:
      raise ValueError("end-to-end timing summary is invalid")
    rates = timings["generated_tokens_per_second"]
    expected_rates = [workload["measured_output_tokens_per_generation"] * 1e9 / value for value in measured_end_to_end]
    if not isinstance(rates, list) or len(rates) != workload["measured_generations"] \
        or any(type(value) not in (int, float) or value <= 0 for value in rates) \
        or any(not math.isclose(actual, expected, rel_tol=1e-12) for actual, expected in zip(rates, expected_rates)):
      raise ValueError("end-to-end generation throughput is invalid")
    for field in ("contract_setup_elapsed_ns", "measurement_elapsed_ns"):
      if type(timings[field]) is not int or timings[field] <= 0: raise ValueError(f"{field} must be positive")
    if timings["contract_setup_elapsed_ns"] < sum(setup_end_to_end):
      raise ValueError("contract setup elapsed time is incomplete")
    if timings["measurement_elapsed_ns"] < sum(measured_end_to_end):
      raise ValueError("measurement elapsed time is incomplete")
    aggregate_rate = workload["total_measured_generated_tokens"] * 1e9 / timings["measurement_elapsed_ns"]
    if type(timings["aggregate_generated_tokens_per_second"]) not in (int, float) \
        or not math.isclose(timings["aggregate_generated_tokens_per_second"], aggregate_rate, rel_tol=1e-12):
      raise ValueError("aggregate generation throughput is invalid")
    if type(timings["startup_to_ready_ns"]) is not int or timings["startup_to_ready_ns"] <= 0:
      raise ValueError("startup-to-ready timing is invalid")

    required_outputs = {
      "contract_setup_prompt_token_ids", "contract_setup_token_ids",
      "measured_prompt_token_ids", "measured_token_ids",
    }
    if set(outputs) != required_outputs: raise ValueError("end-to-end outputs are incomplete")
    nested_contracts = (
      ("contract_setup_prompt_token_ids", workload["setup_generations"], workload["prompt_tokens"]),
      ("contract_setup_token_ids", workload["setup_generations"], workload["setup_output_tokens_per_generation"]),
      ("measured_prompt_token_ids", workload["measured_generations"], workload["prompt_tokens"]),
      ("measured_token_ids", workload["measured_generations"], workload["measured_output_tokens_per_generation"]),
    )
    for field, outer_length, inner_length in nested_contracts:
      values = outputs[field]
      if not isinstance(values, list) or len(values) != outer_length \
          or any(not isinstance(row, list) or len(row) != inner_length for row in values) \
          or any(type(token) is not int or token < 0 or token >= 151936 for row in values for token in row):
        raise ValueError(f"end-to-end {field} is invalid")
    prompts = outputs["contract_setup_prompt_token_ids"] + outputs["measured_prompt_token_ids"]
    if len({prompt[0] for prompt in prompts}) != len(prompts):
      raise ValueError("end-to-end prompts can reuse an upstream cached prefix")
    host_boundaries = {"before_artifact_verification", "after_artifact_verification", "after_model_load", "after_contract_setup", "after_measurement"}
    tinygrad_boundaries = {"before_model_load", "after_model_load", "after_contract_setup", "after_measurement"}
    phases = ("model_load", "contract_setup", "measurement")

  for field in ("artifact_verification_ns", "model_load_ns"):
    if type(timings[field]) is not int or timings[field] <= 0: raise ValueError(f"{field} must be positive")
  if benchmark == "upstream_end_to_end" \
      and timings["startup_to_ready_ns"] < timings["artifact_verification_ns"] + timings["model_load_ns"] + timings["contract_setup_elapsed_ns"]:
    raise ValueError("startup-to-ready does not cover required setup work")
  if benchmark != "upstream_end_to_end":
    if not isinstance(setup, list) or len(setup) != 2 or any(type(sample) is not int or sample <= 0 for sample in setup):
      raise ValueError("contract setup timings are invalid")
    if not isinstance(measured, list) or len(measured) != workload["measured_samples"]: raise ValueError("measured sample count is invalid")
    if timing_summary(measured) != summary: raise ValueError("timing summary is invalid")
    if not isinstance(rates, list) or len(rates) != len(measured) or any(type(rate) not in (int, float) or rate <= 0 for rate in rates):
      raise ValueError("throughput samples are invalid")
    expected_rates = [rate_numerator / sample for sample in measured]
    if any(not math.isclose(actual, expected, rel_tol=1e-12) for actual, expected in zip(rates, expected_rates)):
      raise ValueError("throughput does not match timing samples")
    if set(outputs) != required_outputs or not isinstance(outputs["contract_setup_token_ids"], list) \
        or not isinstance(outputs["measured_token_ids"], list):
      raise ValueError("outputs are incomplete")
    if len(outputs["contract_setup_token_ids"]) != expected_output_lengths[0] \
        or len(outputs["measured_token_ids"]) != expected_output_lengths[1]:
      raise ValueError("output count does not match timing count")
    output_tokens = outputs["contract_setup_token_ids"] + outputs["measured_token_ids"]
    if benchmark == "upstream_decode": output_tokens.append(outputs["prefill_token_id"])
    if any(type(token) is not int or token < 0 for token in output_tokens): raise ValueError("output token is invalid")

  memory = result.get("memory")
  if not isinstance(memory, dict) or set(memory) != {"host", "tinygrad", "device"}: raise ValueError("memory measurements are incomplete")
  host = memory["host"]
  if not isinstance(host, dict) or set(host) != host_boundaries | {"source"} or host.get("source") != "/proc/self/status":
    raise ValueError("host memory measurements are invalid")
  for boundary in host_boundaries:
    snapshot = host[boundary]
    if not isinstance(snapshot, dict) or set(snapshot) != {"current_rss_bytes", "peak_rss_bytes"}:
      raise ValueError(f"host memory snapshot {boundary} is invalid")
    if any(type(value) is not int or value < 0 for value in snapshot.values()): raise ValueError(f"host memory snapshot {boundary} is invalid")
    if snapshot["peak_rss_bytes"] < snapshot["current_rss_bytes"]: raise ValueError(f"host memory snapshot {boundary} has an invalid peak")

  tinygrad = memory["tinygrad"]
  if not isinstance(tinygrad, dict) or set(tinygrad) != tinygrad_boundaries | {"source", "unit"}: raise ValueError("tinygrad memory measurements are incomplete")
  if tinygrad.get("source") != "tinygrad.GlobalCounters.mem_used_per_device":
    raise ValueError("tinygrad memory source is invalid")
  if tinygrad.get("unit") != "live_requested_bytes_not_peak": raise ValueError("tinygrad memory unit is invalid")
  if any(type(tinygrad[field]) is not int or tinygrad[field] < 0 for field in tinygrad_boundaries):
    raise ValueError("tinygrad memory snapshot is invalid")

  device_memory = memory["device"]
  required_device_memory = {"source", "sample_interval_ms", "sampled_peak_bytes", "phase_windows_ns", "phase_sampled_peak_bytes", "samples", "limitations"}
  if not isinstance(device_memory, dict) or set(device_memory) != required_device_memory:
    raise ValueError("device memory measurements are incomplete")
  if device_memory.get("source") != "nvidia-smi.compute-apps.sampled": raise ValueError("device memory source is invalid")
  if type(device_memory.get("sample_interval_ms")) is not int or device_memory["sample_interval_ms"] <= 0:
    raise ValueError("device sample interval is invalid")
  samples = device_memory.get("samples")
  if not isinstance(samples, list) or not samples: raise ValueError("device memory samples are missing")
  for sample in samples:
    if not isinstance(sample, dict) or set(sample) != {"query_start_ns", "query_end_ns", "gpu_uuid", "used_bytes"}:
      raise ValueError("device memory sample is invalid")
    if type(sample["query_start_ns"]) is not int or type(sample["query_end_ns"]) is not int \
        or sample["query_start_ns"] <= 0 or sample["query_end_ns"] < sample["query_start_ns"]:
      raise ValueError("device sample timestamps are invalid")
    if sample["gpu_uuid"] is not None and (not isinstance(sample["gpu_uuid"], str) or not sample["gpu_uuid"]):
      raise ValueError("device sample UUID is invalid")
    if sample["used_bytes"] is not None and (type(sample["used_bytes"]) is not int or sample["used_bytes"] < 0):
      raise ValueError("device sample memory is invalid")
    if sample["used_bytes"] is not None and sample["gpu_uuid"] != device["uuid"]:
      raise ValueError("device sample UUID does not match execution device")
  sampled_peak = device_memory.get("sampled_peak_bytes")
  if type(sampled_peak) is not int or sampled_peak <= 0 or sampled_peak_bytes(samples) != sampled_peak:
    raise ValueError("device sampled peak is invalid")
  phase_peaks = device_memory.get("phase_sampled_peak_bytes")
  if not isinstance(phase_peaks, dict) or set(phase_peaks) != set(phases):
    raise ValueError("device phase peaks are incomplete")
  phase_windows = device_memory.get("phase_windows_ns")
  if not isinstance(phase_windows, dict) or set(phase_windows) != set(phase_peaks): raise ValueError("device phase windows are incomplete")
  for phase, window in phase_windows.items():
    if not isinstance(window, dict) or set(window) != {"start_ns", "end_ns"}: raise ValueError(f"device phase window {phase} is invalid")
    if type(window["start_ns"]) is not int or type(window["end_ns"]) is not int or window["start_ns"] <= 0 or window["end_ns"] < window["start_ns"]:
      raise ValueError(f"device phase window {phase} is invalid")
    expected_peak = sampled_peak_bytes(samples, window["start_ns"], window["end_ns"])
    if expected_peak is None or phase_peaks[phase] != expected_peak: raise ValueError(f"device phase peak {phase} is invalid")
  for previous, current in zip(phases, phases[1:]):
    if phase_windows[previous]["end_ns"] > phase_windows[current]["start_ns"]:
      raise ValueError("device phase windows are out of order")
  if benchmark == "upstream_end_to_end" \
      and (timings["contract_setup_elapsed_ns"] != phase_windows["contract_setup"]["end_ns"] - phase_windows["contract_setup"]["start_ns"]
           or timings["measurement_elapsed_ns"] != phase_windows["measurement"]["end_ns"] - phase_windows["measurement"]["start_ns"]):
    raise ValueError("end-to-end elapsed times do not match phase windows")
  if not isinstance(device_memory.get("limitations"), list) or not device_memory["limitations"]:
    raise ValueError("device memory limitations are missing")


def write_result(result: dict[str, Any], output: Path | None) -> None:
  validate_result(result)
  serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
  if output is None:
    print(serialized, end="")
    return
  output.parent.mkdir(parents=True, exist_ok=True)
  descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
  temporary = Path(temporary_name)
  try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
      destination.write(serialized)
      destination.flush()
      os.fsync(destination.fileno())
    os.replace(temporary, output)
  except BaseException:
    temporary.unlink(missing_ok=True)
    raise
