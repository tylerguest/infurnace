import unittest
from infurnace.engine import Engine, EngineStepResult
from infurnace.engine.request import Request, RequestState, SamplingParams
from infurnace.scheduler.scheduler import Scheduler
from fakes import FakeRunner


def _req(rid, prompt_len=3, sp=None, eos=None):
    sp = sp or SamplingParams(max_tokens=4)
    return Request(rid, list(range(prompt_len)), sp, 1024, eos_token_ids=eos or [])


def _run_to_done(engine, max_steps=1000):
    results = []
    n = 0
    while not engine.is_done():
        results.append(engine.step())
        n += 1
        assert n <= max_steps, "engine did not terminate"
    return results


class RaisingRunner(FakeRunner):
    def prefill(self, input_ids, slot=0, start_position=0):
        raise RuntimeError("boom")


class TestEngineSingleRequest(unittest.TestCase):
    def test_finishes_at_max_tokens(self):
        runner = FakeRunner(vocab_size=50, seed=0)
        eng = Engine(runner, Scheduler(num_slots=1))
        eng.add_request(_req("r1", prompt_len=3, sp=SamplingParams(max_tokens=4)))
        _run_to_done(eng)
        req = eng.scheduler.get_request("r1")
        self.assertEqual(req.state, RequestState.FINISHED)
        self.assertEqual(eng.output_tokens("r1"), (0, 1, 2, 3))
        self.assertEqual(runner.cleared_slots, [0])  # slot freed on finish

    def test_eos_terminates(self):
        runner = FakeRunner(vocab_size=50, seed=0)
        eng = Engine(runner, Scheduler(num_slots=1))
        eng.add_request(_req("r1", prompt_len=3, sp=SamplingParams(max_tokens=10), eos=[2]))
        _run_to_done(eng)
        req = eng.scheduler.get_request("r1")
        self.assertEqual(req.state, RequestState.FINISHED)
        self.assertEqual(eng.output_tokens("r1"), (0, 1, 2))  # stops at eos token 2

    def test_min_tokens_floor_blocks_early_eos(self):
        runner = FakeRunner(vocab_size=50, seed=0)
        eng = Engine(runner, Scheduler(num_slots=1))
        sp = SamplingParams(max_tokens=4, min_tokens=2)
        eng.add_request(_req("r1", prompt_len=3, sp=sp, eos=[0]))
        _run_to_done(eng)
        # eos token 0 appears first but min_tokens=2 forces continuation
        self.assertEqual(eng.output_tokens("r1"), (0, 1, 2, 3))

    def test_decode_feeds_generated_token_at_correct_position(self):
        runner = FakeRunner(vocab_size=50, seed=0)
        eng = Engine(runner, Scheduler(num_slots=1))
        eng.add_request(_req("r1", prompt_len=3, sp=SamplingParams(max_tokens=4)))
        _run_to_done(eng)
        decode_calls = [c for c in runner.calls if c[0] == "decode_batch"]
        # First token comes from prefill; 3 decode steps follow.
        self.assertEqual(len(decode_calls), 3)
        self.assertEqual([c[2]["positions"][0] for c in decode_calls], [3, 4, 5])
        self.assertEqual([c[1][0] for c in decode_calls], [0, 1, 2])  # fed-back tokens

    def test_metrics_recorded(self):
        runner = FakeRunner(vocab_size=50, seed=0)
        eng = Engine(runner, Scheduler(num_slots=1))
        eng.add_request(_req("r1", prompt_len=3, sp=SamplingParams(max_tokens=4)))
        results = _run_to_done(eng)
        m = eng.metrics_of("r1")
        self.assertIsNotNone(m.arrival_time)
        self.assertIsNotNone(m.first_token_time)
        self.assertIsNotNone(m.completion_time)
        self.assertLessEqual(m.first_token_time, m.completion_time)
        final = results[-1]
        self.assertIn("r1", final.metrics)


class TestEngineFIFOAndConcurrency(unittest.TestCase):
    def test_two_requests_batch_concurrently(self):
        # num_slots=2 admits both; r2's prefill overlaps r1's decode so the
        # engine produces a real decode batch containing both requests.
        runner = FakeRunner(vocab_size=50, seed=0, num_slots=2)
        eng = Engine(runner, Scheduler(num_slots=2))
        eng.add_request(_req("r1", prompt_len=3, sp=SamplingParams(max_tokens=3)))
        eng.add_request(_req("r2", prompt_len=2, sp=SamplingParams(max_tokens=3)))
        _run_to_done(eng)
        self.assertEqual(eng.scheduler.get_request("r1").state, RequestState.FINISHED)
        self.assertEqual(eng.scheduler.get_request("r2").state, RequestState.FINISHED)
        # A shared decode batch carries one row per active request, in slot order.
        shared = [c for c in runner.calls if c[0] == "decode_batch" and len(c[1]) == 2]
        self.assertTrue(shared)
        self.assertEqual(shared[0][1], (1, 2))       # row tokens for r1, r2
        self.assertEqual(shared[0][2]["slots"], (0, 1))

    def test_rejection_on_capacity(self):
        runner = FakeRunner(vocab_size=50, num_slots=1)
        eng = Engine(runner, Scheduler(num_slots=1, max_context=runner.max_context))
        eng.add_request(_req("r1"))
        eng.add_request(_req("r2"))  # no free slot -> REJECTED, no raise
        self.assertEqual(eng.scheduler.get_request("r1").state, RequestState.WAITING)
        # r2 was rejected but remains observable via the scheduler
        req2 = eng.scheduler.get_request("r2")
        self.assertEqual(req2.state, RequestState.REJECTED)
        self.assertIsNotNone(req2.error)

    def test_rejection_on_context_limit(self):
        runner = FakeRunner(vocab_size=50, num_slots=1, max_context=10)
        eng = Engine(runner)
        req = _req("r1", prompt_len=11)
        eng.add_request(req)
        self.assertEqual(req.state, RequestState.REJECTED)
        self.assertEqual(eng.scheduler.get_request("r1").state, RequestState.REJECTED)

    def test_slot_reuse_assigns_lowest_free_slot(self):
        runner = FakeRunner(vocab_size=50, seed=0, num_slots=2)
        eng = Engine(runner, Scheduler(num_slots=2))
        eng.add_request(_req("r1", sp=SamplingParams(max_tokens=1)))
        eng.add_request(_req("r2", sp=SamplingParams(max_tokens=3)))
        _run_to_done(eng)
        # r1 finishes first and frees slot 0; a new request reuses the lowest slot.
        eng.add_request(_req("r3", sp=SamplingParams(max_tokens=1)))
        self.assertEqual(eng.scheduler.get_request("r3").cache_slot, 0)


class TestEngineBatchedDecode(unittest.TestCase):
    """Phase 4C: decode plans execute through decode_batch and requests in one
    batch finish independently."""

    def test_batch_requests_finish_independently(self):
        runner = FakeRunner(vocab_size=50, seed=0, num_slots=2)
        eng = Engine(runner, Scheduler(num_slots=2))
        eng.add_request(_req("r1", prompt_len=3, sp=SamplingParams(max_tokens=2)))
        eng.add_request(_req("r2", prompt_len=3, sp=SamplingParams(max_tokens=5)))
        _run_to_done(eng)
        self.assertEqual(eng.scheduler.get_request("r1").state, RequestState.FINISHED)
        self.assertEqual(eng.scheduler.get_request("r2").state, RequestState.FINISHED)
        self.assertEqual(len(eng.output_tokens("r1")), 2)
        self.assertEqual(len(eng.output_tokens("r2")), 5)

    def test_cancel_one_of_batched_requests(self):
        runner = FakeRunner(vocab_size=50, seed=0, num_slots=2)
        eng = Engine(runner, Scheduler(num_slots=2))
        eng.add_request(_req("r1", prompt_len=3, sp=SamplingParams(max_tokens=10)))
        eng.add_request(_req("r2", prompt_len=3, sp=SamplingParams(max_tokens=10)))
        eng.step()  # r1 prefill
        eng.step()  # r1 decode + r2 prefill
        eng.step()  # r1+r2 batched decode
        self.assertEqual(eng.scheduler.get_request("r1").state, RequestState.DECODING)
        self.assertEqual(eng.scheduler.get_request("r2").state, RequestState.DECODING)
        calls_before = len(runner.calls)
        eng.cancel("r1")
        _run_to_done(eng)
        self.assertEqual(eng.scheduler.get_request("r1").state, RequestState.CANCELLED)
        self.assertEqual(eng.scheduler.get_request("r2").state, RequestState.FINISHED)
        self.assertIn(0, runner.cleared_slots)  # r1's slot cleared on cancel
        after = [c for c in runner.calls[calls_before:] if c[0] == "decode_batch"]
        self.assertTrue(all(len(c[1]) == 1 for c in after))  # r2 continues alone

    def test_new_request_joins_active_batch(self):
        runner = FakeRunner(vocab_size=50, seed=0, num_slots=2)
        eng = Engine(runner, Scheduler(num_slots=2))
        eng.add_request(_req("r1", prompt_len=3, sp=SamplingParams(max_tokens=1)))
        eng.add_request(_req("r2", prompt_len=3, sp=SamplingParams(max_tokens=8)))
        eng.step()  # r1 prefill
        eng.step()  # r1 decode + r2 prefill; r1 finishes and frees slot 0
        eng.add_request(_req("r3", prompt_len=3, sp=SamplingParams(max_tokens=4)))
        self.assertEqual(eng.scheduler.get_request("r3").cache_slot, 0)  # reuses slot 0
        _run_to_done(eng)
        self.assertEqual(eng.scheduler.get_request("r3").state, RequestState.FINISHED)
        # r2 and r3 decode together after r3's prefill overlaps r2's decode.
        shared = [c for c in runner.calls if c[0] == "decode_batch" and len(c[1]) == 2]
        self.assertTrue(shared)


class TestEngineCancellation(unittest.TestCase):
    def test_cancel_waiting_is_cancelled_no_runner_calls(self):
        runner = FakeRunner(vocab_size=50, seed=0, num_slots=2)
        eng = Engine(runner, Scheduler(num_slots=2))
        eng.add_request(_req("r1", prompt_len=3, sp=SamplingParams(max_tokens=6)))
        eng.add_request(_req("r2", prompt_len=3, sp=SamplingParams(max_tokens=6)))
        eng.cancel("r2")  # cancel while queued
        self.assertEqual(eng.scheduler.get_request("r2").state, RequestState.CANCELLED)
        _run_to_done(eng)
        self.assertEqual(eng.scheduler.get_request("r1").state, RequestState.FINISHED)
        # r2 produced no runner calls; its slot was cleared
        self.assertTrue(all(c[1] != () for c in runner.calls))  # sanity
        self.assertIn(1, runner.cleared_slots)

    def test_cancel_decoding_suppresses_further_output(self):
        runner = FakeRunner(vocab_size=50, seed=0, num_slots=1)
        eng = Engine(runner, Scheduler(num_slots=1))
        eng.add_request(_req("r1", prompt_len=3, sp=SamplingParams(max_tokens=99)))
        eng.step()  # prefill -> decoding, one token emitted
        self.assertEqual(eng.scheduler.get_request("r1").state, RequestState.DECODING)
        calls_before = len(runner.calls)
        cleared_before = len(runner.cleared_slots)
        eng.cancel("r1")
        eng.step()  # should be a no-op (idle)
        self.assertEqual(eng.scheduler.get_request("r1").state, RequestState.CANCELLED)
        self.assertEqual(len(runner.calls), calls_before)  # no further decode
        self.assertEqual(len(runner.cleared_slots), cleared_before + 1)


class TestEngineFailure(unittest.TestCase):
    def test_runner_failure_marks_failed_and_frees_slot(self):
        runner = RaisingRunner(vocab_size=50, seed=0, num_slots=1)
        eng = Engine(runner, Scheduler(num_slots=1))
        eng.add_request(_req("r1", prompt_len=3))
        eng.step()
        req = eng.scheduler.get_request("r1")
        self.assertEqual(req.state, RequestState.FAILED)
        self.assertEqual(req.error, "boom")
        self.assertEqual(eng.output_tokens("r1"), ())
        self.assertEqual(runner.cleared_slots, [0])

    def test_failure_releases_slot_and_engine_goes_idle(self):
        runner = RaisingRunner(vocab_size=50, seed=0, num_slots=1)
        eng = Engine(runner, Scheduler(num_slots=1))
        eng.add_request(_req("r1", prompt_len=3))
        _run_to_done(eng)  # must terminate despite the failure
        self.assertEqual(eng.scheduler.get_request("r1").state, RequestState.FAILED)
        self.assertTrue(eng.is_done())
        self.assertEqual(eng.scheduler.num_free_slots, 1)


class TestEngineStepAPI(unittest.TestCase):
    def test_prepare_execute_commit_equals_step(self):
        runner = FakeRunner(vocab_size=50, seed=0)
        eng = Engine(runner, Scheduler(num_slots=1))
        eng.add_request(_req("r1", prompt_len=3, sp=SamplingParams(max_tokens=2)))
        composed = eng.step()
        # Reset and replay via the split API
        runner2 = FakeRunner(vocab_size=50, seed=0)
        eng2 = Engine(runner2, Scheduler(num_slots=1))
        eng2.add_request(_req("r1", prompt_len=3, sp=SamplingParams(max_tokens=2)))
        raw = eng2.execute_step(eng2.prepare_step())
        split = eng2.commit_step(raw)
        self.assertEqual(composed.new_tokens, split.new_tokens)
        self.assertEqual(composed.completed_requests, split.completed_requests)

    def test_step_result_fields(self):
        runner = FakeRunner(vocab_size=50, seed=0)
        eng = Engine(runner, Scheduler(num_slots=1))
        eng.add_request(_req("r1", prompt_len=3, sp=SamplingParams(max_tokens=1)))
        result = eng.step()
        self.assertIsInstance(result, EngineStepResult)
        self.assertEqual(result.completed_requests, ("r1",))
        self.assertEqual(result.new_tokens["r1"], (0,))
        self.assertEqual(result.cache_slots_freed, (0,))


if __name__ == "__main__":
    unittest.main()
