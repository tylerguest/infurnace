import os

import pytest
from infurnace.engine import Engine
from infurnace.engine.request import RequestState, SamplingParams
from infurnace.scheduler.scheduler import Scheduler
from infurnace.sampler import GreedySampler
from infurnace.models.manifest import load_manifest
from infurnace.executor.tinygrad.weights import load_qwen3_checkpoint
from infurnace.executor.tinygrad.runner import Qwen3Runner
from infurnace.tokenizer import GGUFTokenizer

_ARTIFACT = "artifacts/models/Qwen3-0.6B-Q8_0.gguf"
_MANIFEST = "models/qwen3-0.6b-q8_0.json"
_NEEDS = pytest.mark.skipif(
    not (os.path.exists(_ARTIFACT) and os.path.exists(_MANIFEST)),
    reason="Qwen3 GGUF artifact/manifest not present",
)


def _build():
    cp = load_qwen3_checkpoint(_ARTIFACT, load_manifest(_MANIFEST))
    tok = GGUFTokenizer.from_gguf_metadata(cp.metadata)
    runner = Qwen3Runner.from_weights(cp.weights, num_slots=1, max_context=512)
    return Engine(runner, Scheduler(num_slots=1), GreedySampler(), tok)


@pytest.mark.model
@pytest.mark.slow
@_NEEDS
def test_real_runner_generates_text():
    eng = _build()
    eng.add_text_request("The capital of France is", SamplingParams(max_tokens=4))
    while not eng.is_done():
        eng.step()
    req = eng.scheduler.get_request("req-1")
    assert req.state is RequestState.FINISHED
    text = eng.final_text("req-1")
    assert isinstance(text, str)
    assert len(text) > 0


@pytest.mark.model
@pytest.mark.slow
@_NEEDS
def test_real_runner_stops_on_eos():
    eng = _build()
    # Greedy decode of a short factual prompt should hit the model EOS well
    # before a generous token budget, exercising the tokenizer.is_end stop path.
    eng.add_text_request("The capital of France is", SamplingParams(max_tokens=64))
    steps = 0
    while not eng.is_done():
        eng.step()
        steps += 1
        assert steps <= 64
    assert eng.scheduler.get_request("req-1").state is RequestState.FINISHED
