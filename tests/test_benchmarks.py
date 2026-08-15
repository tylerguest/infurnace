import copy
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.benchmark_decode import measure_generation
from benchmarks.common import parse_compute_app_rows, sampled_peak_bytes, timing_summary, validate_result, write_result
from benchmarks.benchmark_serving import make_prompt as make_serving_prompt
from benchmarks.benchmark_serving import measure_generation as measure_end_to_end_generation


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
      "total_generated_tokens": 4, "contract_setup_calls": 2, "measured_samples": 2, "sampling": "greedy",
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


def valid_decode_result() -> dict:
  result = copy.deepcopy(valid_result())
  measured = [10, 20]
  result["benchmark"] = "upstream_decode"
  result["workload"] = {
    "prompt_tokens": 16, "ttft_tokens": 1, "contract_setup_tokens": 2,
    "timed_decode_tokens": 2, "total_generated_tokens": 5, "measured_samples": 2, "sampling": "greedy",
  }
  result["timings"] = {
    "artifact_verification_ns": 1, "model_load_ns": 2, "cold_prefill_ttft_ns": 3,
    "contract_setup_ns": [4, 5], "decode_ns": measured, "decode_summary": timing_summary(measured),
    "generated_tokens_per_second": [100_000_000.0, 50_000_000.0],
  }
  result["outputs"] = {
    "prefill_token_id": 1, "contract_setup_token_ids": [2, 3], "measured_token_ids": [4, 5],
  }
  result["memory"]["host"]["after_prefill"] = {"current_rss_bytes": 2, "peak_rss_bytes": 3}
  result["memory"]["tinygrad"]["after_prefill"] = 15
  result["memory"]["device"].update(
    phase_windows_ns={
      "model_load": {"start_ns": 1, "end_ns": 2},
      "prefill": {"start_ns": 2, "end_ns": 3},
      "contract_setup": {"start_ns": 3, "end_ns": 4},
      "measurement": {"start_ns": 4, "end_ns": 5},
    },
    phase_sampled_peak_bytes={"model_load": 10, "prefill": 15, "contract_setup": 20, "measurement": 20},
    samples=[
      {"query_start_ns": 1, "query_end_ns": 1, "gpu_uuid": None, "used_bytes": None},
      {"query_start_ns": 2, "query_end_ns": 2, "gpu_uuid": "GPU-a", "used_bytes": 10},
      {"query_start_ns": 3, "query_end_ns": 3, "gpu_uuid": "GPU-a", "used_bytes": 15},
      {"query_start_ns": 4, "query_end_ns": 4, "gpu_uuid": "GPU-a", "used_bytes": 20},
      {"query_start_ns": 5, "query_end_ns": 5, "gpu_uuid": "GPU-a", "used_bytes": 20},
    ],
  )
  return result


def valid_end_to_end_result() -> dict:
  result = copy.deepcopy(valid_result())
  result["benchmark"] = "upstream_end_to_end"
  result["execution"].update(
    path="tinygrad.llm.Transformer.generate", server_runtime=False, transport="none",
    tokenization="none", scheduling="none", concurrency=1,
    jit_state={"prefill_count": 2, "prefill_captured": True, "rollout_count": 4, "rollout_captured": True},
  )
  result["workload"] = {
    "prompt_tokens": 2, "setup_generations": 2, "setup_output_tokens_per_generation": 3,
    "measured_generations": 2, "measured_output_tokens_per_generation": 3,
    "total_setup_generated_tokens": 6, "total_measured_generated_tokens": 6,
    "sampling": "greedy", "arrival_pattern": "closed_loop_sequential",
  }
  setup = {"ttft_ns": [10, 11], "inter_token_ns": [[2, 3], [3, 4]], "end_to_end_ns": [15, 18]}
  measured_ttft, measured_inter, measured_end_to_end = [20, 30], [[4, 5], [6, 7]], [29, 43]
  result["timings"] = {
    "artifact_verification_ns": 1, "model_load_ns": 2, "startup_to_ready_ns": 41,
    "contract_setup_elapsed_ns": 34, "measurement_elapsed_ns": 73,
    "contract_setup": setup,
    "measured_ttft_ns": measured_ttft, "measured_ttft_summary": timing_summary(measured_ttft),
    "measured_inter_token_ns": measured_inter,
    "measured_inter_token_summary": timing_summary([4, 5, 6, 7]),
    "measured_end_to_end_ns": measured_end_to_end,
    "measured_end_to_end_summary": timing_summary(measured_end_to_end),
    "generated_tokens_per_second": [3e9 / 29, 3e9 / 43],
    "aggregate_generated_tokens_per_second": 6e9 / 73,
  }
  result["outputs"] = {
    "contract_setup_prompt_token_ids": [[1, 5], [2, 5]],
    "contract_setup_token_ids": [[10, 11, 12], [13, 14, 15]],
    "measured_prompt_token_ids": [[3, 5], [4, 5]],
    "measured_token_ids": [[16, 17, 18], [19, 20, 21]],
  }
  result["memory"]["device"].update(
    phase_windows_ns={
      "model_load": {"start_ns": 1, "end_ns": 2},
      "contract_setup": {"start_ns": 10, "end_ns": 44},
      "measurement": {"start_ns": 50, "end_ns": 123},
    },
    phase_sampled_peak_bytes={"model_load": 10, "contract_setup": 15, "measurement": 20},
    samples=[
      {"query_start_ns": 1, "query_end_ns": 1, "gpu_uuid": None, "used_bytes": None},
      {"query_start_ns": 2, "query_end_ns": 2, "gpu_uuid": "GPU-a", "used_bytes": 10},
      {"query_start_ns": 20, "query_end_ns": 20, "gpu_uuid": "GPU-a", "used_bytes": 15},
      {"query_start_ns": 60, "query_end_ns": 60, "gpu_uuid": "GPU-a", "used_bytes": 20},
    ],
  )
  return result


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

  def test_decode_generation_uses_one_generator_and_synchronizes_each_token(self):
    class Device:
      synchronizations = 0
      def synchronize(self): self.synchronizations += 1

    device = Device()
    generator = iter(range(5))
    result = measure_generation(generator, device, 2)
    self.assertEqual(result["prefill_token"], 0)
    self.assertEqual(result["setup_tokens"], [1, 2])
    self.assertEqual(result["measured_tokens"], [3, 4])
    self.assertEqual(device.synchronizations, 10)
    with self.assertRaises(StopIteration): next(generator)

  def test_end_to_end_timeline_and_divergent_prompts(self):
    class Device:
      synchronizations = 0
      def synchronize(self): self.synchronizations += 1

    device = Device()
    result = measure_end_to_end_generation(iter([7, 8, 9]), device, 3)
    self.assertEqual(result["token_ids"], [7, 8, 9])
    self.assertEqual(result["end_to_end_ns"], result["ttft_ns"] + sum(result["inter_token_ns"]))
    self.assertEqual(device.synchronizations, 4)
    prompts = [make_serving_prompt(4, sequence) for sequence in range(4)]
    self.assertEqual(len({prompt[0] for prompt in prompts}), 4)
    self.assertTrue(all(len(prompt) == 4 for prompt in prompts))


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
      lambda result: result["timings"].pop("model_load_ns"),
      lambda result: result["timings"].update(contract_setup_ns=1),
    ]
    for mutate in mutations:
      with self.subTest(mutate=mutate):
        result = copy.deepcopy(valid_result())
        mutate(result)
        with self.assertRaises(ValueError): validate_result(result)

  def test_accepts_decode_result_and_rejects_accounting_mismatches(self):
    validate_result(valid_decode_result())
    mutations = [
      lambda result: result["workload"].update(total_generated_tokens=4),
      lambda result: result["workload"].update(timed_decode_tokens=3),
      lambda result: result["timings"].update(decode_ns=[10]),
      lambda result: result["timings"].update(decode_ns=1),
      lambda result: result["timings"].update(decode_ns=[0, 20]),
      lambda result: result["outputs"].update(measured_token_ids=[4]),
      lambda result: result["memory"]["device"]["phase_sampled_peak_bytes"].update(prefill=20),
    ]
    for mutate in mutations:
      with self.subTest(mutate=mutate):
        result = valid_decode_result()
        mutate(result)
        with self.assertRaises(ValueError): validate_result(result)

  def test_accepts_end_to_end_result_and_rejects_server_or_timing_claims(self):
    validate_result(valid_end_to_end_result())
    mutations = [
      lambda result: result["execution"].update(server_runtime=True),
      lambda result: result["execution"].update(concurrency=2),
      lambda result: result["execution"].update(transport="http"),
      lambda result: result["execution"]["jit_state"].update(prefill_captured=False),
      lambda result: result["workload"].update(total_measured_generated_tokens=5),
      lambda result: result["workload"].update(measured_output_tokens_per_generation=127),
      lambda result: result["timings"]["contract_setup"]["end_to_end_ns"].__setitem__(0, 14),
      lambda result: result["timings"].update(aggregate_generated_tokens_per_second=1.0),
      lambda result: result["outputs"]["measured_prompt_token_ids"][0].__setitem__(0, 1),
      lambda result: result["outputs"]["measured_token_ids"].__setitem__(0, [16, 17]),
      lambda result: result["timings"].update(measurement_elapsed_ns=71),
      lambda result: result["timings"].update(startup_to_ready_ns=36),
    ]
    for mutate in mutations:
      with self.subTest(mutate=mutate):
        result = valid_end_to_end_result()
        mutate(result)
        with self.assertRaises(ValueError): validate_result(result)
