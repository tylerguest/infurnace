import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
import pytest

REPOSITORY_ROOT = Path(__file__).parents[1]

@pytest.mark.nv
@pytest.mark.model
@pytest.mark.slow
class TestUpstreamModelRunner(unittest.TestCase):
  def run_workload(self, weight_policy: str, workload: str) -> dict:
    artifact_value = os.environ.get("INFURNACE_MODEL_ARTIFACT")
    if artifact_value is None: self.skipTest("set INFURNACE_MODEL_ARTIFACT to the pinned GGUF")

    environment = os.environ.copy()
    environment.update(DEV="NV", JIT="1")
    command = [
      sys.executable,
      str(REPOSITORY_ROOT / "tools" / "run_upstream_model.py"),
      "--manifest", str(REPOSITORY_ROOT / "models" / "qwen3-4b-q4_k_m.json"),
      "--artifact", artifact_value,
      "--weight-policy", weight_policy,
      "--workload", workload,
      "--max-context", "1024",
      "--fixed-output-tokens", "4",
      "--text-output-tokens", "16",
    ]
    result = subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, capture_output=True, text=True, timeout=600)
    self.assertEqual(result.returncode, 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    output = json.loads(result.stdout)
    self.assertEqual(output["execution"]["device"], "NV")
    self.assertEqual(output["execution"]["weight_policy"], weight_policy)
    self.assertFalse(output["execution"]["upstream_realize"])
    self.assertEqual(output["workload"]["name"], workload)
    return output

  def test_fixed_generation_for_lazy_policy(self):
    policy_outputs = [self.run_workload("lazy", "fixed") for _ in range(2)]
    self.assertEqual(policy_outputs[0]["workload"], policy_outputs[1]["workload"])
    self.assertEqual(policy_outputs[0]["workload"]["generated_token_ids"], [198, 785, 1196, 702])

  def test_text_generation_from_fresh_process(self):
    outputs = [self.run_workload("lazy", "text") for _ in range(2)]
    expected_tokens = [
      151667, 198, 32313, 11, 279, 1196, 6801, 752,
      311, 9851, 448, 6896, 330, 14990, 3263, 6771,
    ]
    self.assertEqual(outputs[0]["workload"]["generated_token_ids"], expected_tokens)
    self.assertEqual(outputs[0]["workload"]["generated_text"], ' thinking\nOkay, the user wants me to reply with exactly "hello". Let')
    self.assertNotIn("\ufffd", outputs[0]["workload"]["generated_text"])
    self.assertEqual(outputs[0]["workload"], outputs[1]["workload"])
