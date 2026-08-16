import unittest

from infurnace.engine import Engine, EngineStepResult
from infurnace.engine.request import RequestState, SamplingParams
from infurnace.scheduler.scheduler import Scheduler
from infurnace.sampler import GreedySampler
from infurnace.tokenizer.base import Tokenizer

from fakes import FakeRunner


class FakeTokenizer(Tokenizer):
    def encode(self, text): return [ord(c) for c in text]
    def decode(self, ids): return "".join(chr(i) for i in ids)
    def is_end(self, token_id): return False


def _run_to_done(eng, max_steps=10000):
    results = []
    n = 0
    while not eng.is_done():
        results.append(eng.step())
        n += 1
        assert n <= max_steps
    return results


class TestEngineText(unittest.TestCase):
    def test_text_streams_and_matches_full_decode(self):
        eng = Engine(FakeRunner(vocab_size=300, seed=0), Scheduler(num_slots=1),
                     GreedySampler(), FakeTokenizer())
        eng.add_text_request("abc", SamplingParams(max_tokens=5))
        results = _run_to_done(eng)
        req = eng.scheduler.get_request("req-1")
        self.assertEqual(req.state, RequestState.FINISHED)
        self.assertEqual(eng.final_text("req-1"), FakeTokenizer().decode(list(req.output_token_ids)))
        self.assertTrue(any(r.new_text for r in results))

    def test_stop_string_finishes_and_excludes(self):
        eng = Engine(FakeRunner(vocab_size=300, seed=0), Scheduler(num_slots=1),
                     GreedySampler(), FakeTokenizer())
        eng.add_text_request("a", SamplingParams(max_tokens=20), stop_strings=["\x01"])
        _run_to_done(eng)
        req = eng.scheduler.get_request("req-1")
        self.assertEqual(req.state, RequestState.FINISHED)
        self.assertEqual(eng.final_text("req-1"), "\x00")

    def test_stop_string_included_in_output(self):
        eng = Engine(FakeRunner(vocab_size=300, seed=0), Scheduler(num_slots=1),
                     GreedySampler(), FakeTokenizer())
        sp = SamplingParams(max_tokens=20, include_stop_str_in_output=True)
        eng.add_text_request("a", sp, stop_strings=["\x01"])
        _run_to_done(eng)
        self.assertEqual(eng.final_text("req-1"), "\x00\x01")

    def test_no_malformed_output_per_step(self):
        eng = Engine(FakeRunner(vocab_size=300, seed=0), Scheduler(num_slots=1),
                     GreedySampler(), FakeTokenizer())
        eng.add_text_request("hi", SamplingParams(max_tokens=4))
        results = _run_to_done(eng)
        joined = "".join(t for r in results for t in r.new_text.values())
        self.assertEqual(joined, eng.final_text("req-1"))

    def test_logprobs_stub_present(self):
        eng = Engine(FakeRunner(vocab_size=300, seed=0), Scheduler(num_slots=1),
                     GreedySampler(), FakeTokenizer())
        eng.add_text_request("a", SamplingParams(max_tokens=1))
        res = eng.step()
        self.assertIsInstance(res, EngineStepResult)
        self.assertEqual(dict(res.logprobs), {})
