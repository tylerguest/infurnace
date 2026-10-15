"""Output quality gate: infurnace's decode paths match llama.cpp's perplexity on the same GGUF and text.

Opt-in (real model, ~1 minute): INFURNACE_TEST_NV=1 INFURNACE_TEST_MODEL=/path/to/Qwen3.5-4B-Q4_K_M.gguf.
Reference numbers: benchmarks/quality/reference.json (benchmarks/quality.py --llama-perplexity regenerates them)."""
import importlib.util
import json
import os
import pathlib
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.nv, pytest.mark.skipif(not os.environ.get("INFURNACE_TEST_MODEL"), reason="set INFURNACE_TEST_MODEL")]

def harness():
  spec = importlib.util.spec_from_file_location("quality", ROOT / "benchmarks" / "quality.py")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module

def test_perplexity_matches_llama_cpp():
  from infurnace import loader
  quality, reference = harness(), json.loads((ROOT / "benchmarks" / "quality" / "reference.json").read_text())
  model, tokenizer = loader.load_model(os.environ["INFURNACE_TEST_MODEL"], 512)
  tokens = tokenizer.encode(quality.TEXT.read_text())
  model.realize_state()
  expected = reference["llama.cpp"]["ppl"]["2"]
  for rows in (1, 16):
    ppl, _ = quality.perplexity(model, tokens, 512, 2, rows)
    # Within 1% of an independent engine: kernels that broke the math would move it by far more.
    assert abs(ppl - expected) / expected < 0.01, (rows, ppl, expected)
