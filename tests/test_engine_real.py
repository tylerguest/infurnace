import gc
import os

import pytest
from infurnace.engine import Engine
from infurnace.engine.request import RequestState, SamplingParams
from infurnace.scheduler.scheduler import Scheduler
from infurnace.sampler import GreedySampler
from infurnace.models.manifest import load_manifest
from infurnace.executor.tinygrad.weights import load_qwen3_checkpoint, WeightPolicy
from infurnace.executor.tinygrad.runner import Qwen3Runner
from infurnace.tokenizer import GGUFTokenizer

_ARTIFACT = "artifacts/models/Qwen3-0.6B-Q8_0.gguf"
_MANIFEST = "models/qwen3-0.6b-q8_0.json"
_NEEDS = pytest.mark.skipif(
    not (os.path.exists(_ARTIFACT) and os.path.exists(_MANIFEST)),
    reason="Qwen3 GGUF artifact/manifest not present",
)


def _build(num_slots: int = 1, max_context: int = 512, policy: WeightPolicy = WeightPolicy.REALIZED_FP16):
    cp = load_qwen3_checkpoint(_ARTIFACT, load_manifest(_MANIFEST), policy=policy)
    tok = GGUFTokenizer.from_gguf_metadata(cp.metadata)
    runner = Qwen3Runner.from_weights(cp.weights, num_slots=num_slots, max_context=max_context)
    return Engine(runner, Scheduler(num_slots=num_slots, max_context=max_context), GreedySampler(), tok)


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


@pytest.mark.model
@pytest.mark.slow
@_NEEDS
def test_real_runner_multi_slot_serial_prefill():
    eng = _build(num_slots=2, max_context=256)
    eng.add_text_request("The capital of France is", SamplingParams(max_tokens=4), request_id="r1")
    eng.add_text_request("The capital of Germany is", SamplingParams(max_tokens=4), request_id="r2")
    while not eng.is_done():
        eng.step()
    assert eng.scheduler.get_request("r1").state is RequestState.FINISHED
    assert eng.scheduler.get_request("r2").state is RequestState.FINISHED
    assert len(eng.final_text("r1")) > 0
    assert len(eng.final_text("r2")) > 0


@pytest.mark.model
@pytest.mark.slow
@_NEEDS
def test_real_runner_cancel_waiting():
    eng = _build(num_slots=2, max_context=256)
    eng.add_text_request("The capital of France is", SamplingParams(max_tokens=4), request_id="r1")
    eng.add_text_request("The capital of Germany is", SamplingParams(max_tokens=4), request_id="r2")
    eng.cancel("r2")
    while not eng.is_done():
        eng.step()
    assert eng.scheduler.get_request("r1").state is RequestState.FINISHED
    assert eng.scheduler.get_request("r2").state is RequestState.CANCELLED


@pytest.mark.model
@pytest.mark.slow
@_NEEDS
def test_real_runner_stop_token_ids():
    eng = _build(num_slots=1, max_context=256)
    # Use a token id unlikely to appear early as a stop, but verify the path
    # terminates by max_tokens since the stop id is arbitrary.
    eng.add_text_request(
        "The capital of France is",
        SamplingParams(max_tokens=4, stop_token_ids=[999999]),
        request_id="r1",
    )
    while not eng.is_done():
        eng.step()
    assert eng.scheduler.get_request("r1").state is RequestState.FINISHED


@pytest.mark.model
@pytest.mark.slow
@_NEEDS
def test_real_runner_context_limit_rejection():
    eng = _build(num_slots=1, max_context=4)
    eng.add_text_request("The capital of France is", SamplingParams(max_tokens=4), request_id="r1")
    assert eng.scheduler.get_request("r1").state is RequestState.REJECTED


@pytest.mark.model
@pytest.mark.slow
@_NEEDS
def test_real_runner_batched_decode_matches_serial():
    # Phase 4C gate: greedy output is unchanged by batch membership. The same
    # two prompts produce identical per-request tokens whether they share a
    # decode batch (num_slots=2) or run alone (num_slots=1).
    def run_concurrent():
        eng = _build(num_slots=2, max_context=256)
        eng.add_text_request("The capital of France is", SamplingParams(max_tokens=4), request_id="r1")
        eng.add_text_request("The capital of Germany is", SamplingParams(max_tokens=4), request_id="r2")
        while not eng.is_done():
            eng.step()
        return eng.output_tokens("r1"), eng.output_tokens("r2")

    def run_alone(prompt):
        eng = _build(num_slots=1, max_context=256)
        eng.add_text_request(prompt, SamplingParams(max_tokens=4), request_id="r1")
        while not eng.is_done():
            eng.step()
        return eng.output_tokens("r1")

    batched_r1, batched_r2 = run_concurrent()
    assert batched_r1 == run_alone("The capital of France is")
    assert batched_r2 == run_alone("The capital of Germany is")


@pytest.mark.model
@pytest.mark.slow
@_NEEDS
def test_real_runner_ragged_batch_stable():
    # Phase 4D smoke: four concurrent requests decode in a batch of 4 that
    # shrinks ragged to 1 as requests finish. The run must reach the shape-4
    # contract, never re-capture, and match each request run alone. Lazy FP16
    # weights keep the five resident models within the 8 GB device; baselines
    # run first so the shape-4 concurrent captures are the only heavy work.
    prompts = {
        "r1": "The capital of France is",
        "r2": "The capital of Germany is",
        "r3": "The capital of Italy is",
        "r4": "The capital of Spain is",
    }
    policy = WeightPolicy.LAZY_FP16

    baseline = {}
    for rid, prompt in prompts.items():
        alone = _build(num_slots=1, max_context=256, policy=policy)
        alone.add_text_request(prompt, SamplingParams(max_tokens=6), request_id="r1")
        while not alone.is_done():
            alone.step()
        baseline[rid] = alone.output_tokens("r1")
        del alone
        gc.collect()

    eng = _build(num_slots=4, max_context=256, policy=policy)
    for rid, prompt in prompts.items():
        eng.add_text_request(prompt, SamplingParams(max_tokens=6), request_id=rid)
    while not eng.is_done():
        eng.step()
    for rid in prompts:
        assert eng.scheduler.get_request(rid).state is RequestState.FINISHED
        assert len(eng.final_text(rid)) > 0
    # The engine batch reached shape 4 (key (0, 1, 2, 3)) and used only the four
    # known contracts: no surprise shapes, no re-capture (a mismatch raises
    # JitError rather than re-capturing).
    keys = set(eng.runner._decode_batch_jit)
    assert (0, 1, 2, 3) in keys
    assert keys <= {(0,), (0, 1), (0, 1, 2, 3), (0, 1, 2, 4)}
    # Greedy output is unchanged by batch membership through the ragged shrink.
    for rid in prompts:
        assert eng.output_tokens(rid) == baseline[rid]
