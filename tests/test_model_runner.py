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
  def test_greedy_token_and_text_generation(self):
    artifact_value = os.environ.get("INFURNACE_MODEL_ARTIFACT")
    if artifact_value is None: self.skipTest("set INFURNACE_MODEL_ARTIFACT to the pinned GGUF")

    environment = os.environ.copy()
    environment.update(DEV="NV", JIT="1")
    command = [
      sys.executable,
      str(REPOSITORY_ROOT / "tools" / "run_upstream_model.py"),
      "--manifest", str(REPOSITORY_ROOT / "models" / "qwen3-0.6b-q8_0.json"),
      "--artifact", artifact_value,
      "--max-context", "1024",
      "--fixed-output-tokens", "4",
      "--text-output-tokens", "16",
    ]
    outputs = []
    for _ in range(2):
      result = subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, capture_output=True, text=True, timeout=300)
      self.assertEqual(result.returncode, 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
      outputs.append(json.loads(result.stdout))
    output = outputs[0]
    self.assertEqual(output["execution"]["device"], "NV")
    self.assertEqual(output["fixed"]["generated_token_ids"], [657, 198, 9, 0])
    self.assertEqual(output["text"]["generated_token_ids"], [
      151667, 198, 151668, 271, 31382, 124941, 124794, 124613,
      37524, 126860, 124325, 128286, 37524, 83827, 125327, 16157,
    ])
    self.assertEqual(output["text"]["generated_text"], "<think>\n</think>\n\nالسلام عليكم ورحمة الله وبركاته")
    self.assertNotIn("\ufffd", output["text"]["generated_text"])
    self.assertEqual(outputs[0]["fixed"], outputs[1]["fixed"])
    self.assertEqual(outputs[0]["text"], outputs[1]["text"])
