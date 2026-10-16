"""Output quality gate: infurnace's decode paths (one-row, 16-row, DFlash verifier) match llama.cpp's perplexity on the same GGUF and texts
(English docs and code; a multilingual text whose tokens mostly sit above id 100k).

Opt-in (real model, ~3 minutes): INFURNACE_TEST_NV=1 INFURNACE_TEST_MODEL=/path/to/Qwen3.5-4B-Q4_K_M.gguf.
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

@pytest.fixture(scope="module")
def loaded():
  from infurnace import loader
  model, tokenizer = loader.load_model(os.environ["INFURNACE_TEST_MODEL"], 512)  # one model: two don't fit in 8 GB
  model.realize_state()
  return model, tokenizer

@pytest.mark.parametrize("text,chunks", [("eval.txt", 2), ("multilingual.txt", 5)])
def test_perplexity_matches_llama_cpp(loaded, text, chunks):
  (model, tokenizer), quality = loaded, harness()
  reference = json.loads((ROOT / "benchmarks" / "quality" / "reference.json").read_text())
  tokens = tokenizer.encode((ROOT / "benchmarks" / "quality" / text).read_text())
  expected = reference["ppl"][text][f"ctx 512, {chunks} chunks"]["llama.cpp"]
  results = {rows: quality.perplexity(model, tokens, 512, chunks, rows)[0] for rows in (1, 16)}
  results["verifier"] = quality.verifier_perplexity(model, tokens, 512, chunks)[0]
  for path, ppl in results.items():
    # Within 1% of an independent engine: kernels that broke the math would move it by far more.
    assert abs(ppl - expected) / expected < 0.01, (text, path, ppl, expected)
