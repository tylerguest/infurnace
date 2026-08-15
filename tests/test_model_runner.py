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
      "--manifest", str(REPOSITORY_ROOT / "models" / "qwen3-0.6b-q8_0.json"),
      "--artifact", artifact_value,
      "--weight-policy", weight_policy,
      "--workload", workload,
      "--max-context", "1024",
      "--fixed-output-tokens", "4",
      "--text-output-tokens", "16",
    ]
    result = subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, capture_output=True, text=True, timeout=300)
    self.assertEqual(result.returncode, 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    output = json.loads(result.stdout)
    self.assertEqual(output["execution"]["device"], "NV")
    self.assertEqual(output["execution"]["weight_policy"], weight_policy)
    self.assertEqual(output["execution"]["upstream_realize"], weight_policy == "realized-fp16")
    self.assertEqual(output["workload"]["name"], workload)
    return output

  def test_fixed_generation_for_each_weight_policy(self):
    policy_outputs = {}
    for policy in ("lazy", "realized-fp16"):
      outputs = [self.run_workload(policy, "fixed") for _ in range(2)]
      self.assertEqual(outputs[0]["workload"], outputs[1]["workload"])
      self.assertEqual(outputs[0]["workload"]["generated_token_ids"], [657, 198, 9, 0])
      policy_outputs[policy] = outputs[0]["workload"]["generated_token_ids"]
    self.assertEqual(policy_outputs["lazy"], policy_outputs["realized-fp16"])

  def test_text_generation_from_fresh_process(self):
    outputs = [self.run_workload("lazy", "text") for _ in range(2)]
    expected_tokens = [
      151667, 198, 151668, 271, 31382, 124941, 124794, 124613,
      37524, 126860, 124325, 128286, 37524, 83827, 125327, 16157,
    ]
    self.assertEqual(outputs[0]["workload"]["generated_token_ids"], expected_tokens)
    self.assertEqual(outputs[0]["workload"]["generated_text"], "<think>\n</think>\n\nالسلام عليكم ورحمة الله وبركاته")
    self.assertNotIn("\ufffd", outputs[0]["workload"]["generated_text"])
    self.assertEqual(outputs[0]["workload"], outputs[1]["workload"])
