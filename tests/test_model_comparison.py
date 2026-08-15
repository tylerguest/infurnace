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
class TestModelComparison(unittest.TestCase):
  def _run_comparison(self, weight_policy: str) -> dict:
    artifact_value = os.environ.get("INFURNACE_MODEL_ARTIFACT")
    if artifact_value is None:
      self.skipTest("set INFURNACE_MODEL_ARTIFACT to the pinned GGUF")

    environment = os.environ.copy()
    environment.update(DEV="NV", JIT="1")
    command = [
      sys.executable,
      str(REPOSITORY_ROOT / "tools" / "compare_outputs.py"),
      "--manifest", str(REPOSITORY_ROOT / "models" / "qwen3-0.6b-q8_0.json"),
      "--artifact", artifact_value,
      "--weight-policy", weight_policy,
      "--max-context", "1024",
    ]
    result = subprocess.run(
      command, cwd=REPOSITORY_ROOT, env=environment,
      capture_output=True, text=True, timeout=300,
    )
    self.assertEqual(result.returncode, 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return json.loads(result.stdout)

  def test_greedy_token_agreement_for_each_weight_policy(self):
    for policy in ("lazy-fp16", "realized-fp16"):
      with self.subTest(policy=policy):
        output = self._run_comparison(policy)
        self.assertTrue(
          output["agreement"]["greedy_tokens_match"],
          f"token mismatch for {policy}: "
          f"infurnace={output['infurnace']['token']}, "
          f"upstream={output['upstream']['token']}",
        )
        self.assertEqual(output["infurnace"]["logits_shape"], [1, 151936])