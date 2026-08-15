import copy
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.common import parse_compute_app_rows, sampled_peak_bytes, timing_summary, validate_result, write_result


def valid_result() -> dict:
  measured = [10, 20]
  return {
    "schema_version": 1,
    "benchmark": "upstream_prefill",
    "created_at_utc": "2026-08-15T00:00:00+00:00",
    "checkpoint": {"id": "model", "sha256": "a" * 64, "size_bytes": 10},
    "system": {
      "python": "3.14.4", "platform": "Linux",
      "device": {
        "index": 0, "uuid": "GPU-a", "name": "GPU", "driver_version": "1",
        "total_memory_bytes": 100, "baseline_used_memory_bytes": 10, "baseline_free_memory_bytes": 90,
      },
    },
    "execution": {
      "device": "NV", "jit": 1, "weight_policy": "lazy", "upstream_realize": False,
      "weight_dtype": "float16", "max_context": 128, "chunk_size": 32,
    },
    "workload": {
      "prompt_tokens": 16, "ttft_tokens_per_sample": 1, "timed_decode_tokens_per_sample": 0,
      "total_generated_tokens": 2, "contract_setup_calls": 2, "measured_samples": 2, "sampling": "greedy",
    },
    "timings": {
      "artifact_verification_ns": 1, "model_load_ns": 2, "contract_setup_ns": [3, 4],
      "prefill_ttft_ns": measured, "prefill_ttft_summary": timing_summary(measured),
      "prompt_tokens_per_second": [1_600_000_000.0, 800_000_000.0],
    },
    "outputs": {"contract_setup_token_ids": [1, 2], "measured_token_ids": [3, 4]},
    "memory": {
      "host": {
        "source": "/proc/self/status",
        "before_artifact_verification": {"current_rss_bytes": 1, "peak_rss_bytes": 2},
        "after_artifact_verification": {"current_rss_bytes": 1, "peak_rss_bytes": 2},
        "after_model_load": {"current_rss_bytes": 2, "peak_rss_bytes": 3},
        "after_contract_setup": {"current_rss_bytes": 3, "peak_rss_bytes": 4},
        "after_measurement": {"current_rss_bytes": 3, "peak_rss_bytes": 4},
      },
      "tinygrad": {
        "source": "tinygrad.GlobalCounters.mem_used_per_device", "unit": "live_requested_bytes_not_peak",
        "before_model_load": 0, "after_model_load": 10, "after_contract_setup": 20, "after_measurement": 20,
      },
      "device": {
        "source": "nvidia-smi.compute-apps.sampled", "sample_interval_ms": 50,
        "sampled_peak_bytes": 20,
        "phase_windows_ns": {
          "model_load": {"start_ns": 1, "end_ns": 2},
          "contract_setup": {"start_ns": 2, "end_ns": 3},
          "measurement": {"start_ns": 3, "end_ns": 4},
        },
        "phase_sampled_peak_bytes": {"model_load": 10, "contract_setup": 20, "measurement": 20},
        "samples": [
          {"query_start_ns": 1, "query_end_ns": 1, "gpu_uuid": None, "used_bytes": None},
          {"query_start_ns": 2, "query_end_ns": 2, "gpu_uuid": "GPU-a", "used_bytes": 10},
          {"query_start_ns": 3, "query_end_ns": 3, "gpu_uuid": "GPU-a", "used_bytes": 20},
        ],
        "limitations": ["sampled"],
      },
    },
  }


class TestBenchmarkCalculations(unittest.TestCase):
  def test_timing_summary(self):
    self.assertEqual(timing_summary([10, 30, 20]), {
      "count": 3, "min_ns": 10, "max_ns": 30, "mean_ns": 20.0, "median_ns": 20,
    })
    for invalid in ([], [0], [-1], [1.5]):
      with self.subTest(invalid=invalid), self.assertRaises(ValueError): timing_summary(invalid)

  def test_sampled_peak_filters_phase_and_missing_process(self):
    samples = [
      {"query_start_ns": 1, "query_end_ns": 1, "used_bytes": None},
      {"query_start_ns": 2, "query_end_ns": 2, "used_bytes": 30},
      {"query_start_ns": 3, "query_end_ns": 3, "used_bytes": 20},
    ]
    self.assertEqual(sampled_peak_bytes(samples), 30)
    self.assertEqual(sampled_peak_bytes(samples, 3, 4), 20)
    self.assertIsNone(sampled_peak_bytes(samples, 0, 1))

  def test_parse_compute_app_rows(self):
    output = "12, GPU-a, 100\n34, GPU-b, 25\n"
    self.assertEqual(parse_compute_app_rows(output, 34), ("GPU-b", 25 * 1024 * 1024))
    self.assertEqual(parse_compute_app_rows(output, 56), (None, None))
    self.assertEqual(parse_compute_app_rows("12, GPU-a, N/A\n34, GPU-b, 25\n", 34), ("GPU-b", 25 * 1024 * 1024))
    with self.assertRaises(ValueError): parse_compute_app_rows("12, malformed\n", 12)
    with self.assertRaises(ValueError): parse_compute_app_rows("12, GPU-a, N/A\n", 12)


class TestBenchmarkResultContract(unittest.TestCase):
  def test_accepts_valid_result_and_publishes_atomically(self):
    result = valid_result()
    validate_result(result)
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / "nested" / "result.json"
      write_result(result, output)
      self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)
      self.assertEqual(list(output.parent.iterdir()), [output])

  def test_rejects_policy_timing_and_peak_mismatches(self):
    mutations = [
      lambda result: result["execution"].update(weight_policy="realized-fp16"),
      lambda result: result["timings"].update(prefill_ttft_ns=[10]),
      lambda result: result["memory"]["device"].update(sampled_peak_bytes=10),
      lambda result: result["workload"].update(ttft_tokens_per_sample=2),
      lambda result: result["execution"].update(weight_dtype="float32"),
    ]
    for mutate in mutations:
      with self.subTest(mutate=mutate):
        result = copy.deepcopy(valid_result())
        mutate(result)
        with self.assertRaises(ValueError): validate_result(result)
