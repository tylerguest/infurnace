import unittest
from dataclasses import FrozenInstanceError
from tinygrad import Tensor, dtypes
from infurnace.engine.request import Request, RequestState, SamplingParams
from infurnace.scheduler.scheduler import Scheduler, SchedulerError
from infurnace.scheduler.plan import PrefillPlan, DecodePlan
from infurnace.scheduler.policy import chunk_prompt
from fakes import FakeRunner


def _req(rid: str, prompt_len: int = 5, **overrides) -> Request:
    sp = SamplingParams(**{**dict(max_tokens=10), **overrides.pop("sp", {})})
    return Request(rid, list(range(prompt_len)), sp, 1024, **overrides)


def _tids(tokens: list[int]) -> Tensor:
    return Tensor([tokens], dtype=dtypes.int32)


class TestChunkPrompt(unittest.TestCase):
    def test_unchunked_short(self):
        self.assertEqual(chunk_prompt([1, 2, 3], 512), [(0, 3)])

    def test_chunked(self):
        self.assertEqual(
            chunk_prompt(list(range(1500)), 512),
            [(0, 512), (512, 1024), (1024, 1500)],
        )

    def test_empty(self):
        self.assertEqual(chunk_prompt([], 512), [(0, 0)])

    def test_exact_multiple(self):
        self.assertEqual(
            chunk_prompt(list(range(1024)), 512),
            [(0, 512), (512, 1024)],
        )


class TestPlanImmutability(unittest.TestCase):
    def test_prefill_plan_frozen(self):
        plan = PrefillPlan("r1", (1, 2, 3), 0, ((0, 3),))
        with self.assertRaises(FrozenInstanceError):
            plan.request_id = "r2"

    def test_decode_plan_frozen(self):
        plan = DecodePlan(("r1",), (7,), (0,), (0,), (True,))
        with self.assertRaises(FrozenInstanceError):
            plan.request_ids = ("r2",)


class TestSchedulerAdmission(unittest.TestCase):
    def test_add_and_capacity(self):
        s = Scheduler(num_slots=1)
        s.add_request(_req("r1"))
        with self.assertRaises(SchedulerError):
            s.add_request(_req("r2"))  # only 1 slot, r1 still live
        self.assertEqual(s.num_free_slots, 0)

    def test_duplicate_request_id(self):
        s = Scheduler(num_slots=2)
        s.add_request(_req("r1"))
        with self.assertRaises(SchedulerError):
            s.add_request(_req("r1"))

    def test_invalid_num_slots(self):
        with self.assertRaises(SchedulerError):
            Scheduler(num_slots=0)


class TestSchedulerFIFOAndOrder(unittest.TestCase):
    def test_fifo_admission(self):
        s = Scheduler(num_slots=3)
        for rid in ("r1", "r2", "r3"):
            s.add_request(_req(rid))
        # Admit one at a time; FIFO order preserved
        admitted = []
        for _ in range(3):
            plans = s.schedule()
            self.assertEqual(len(plans), 1)
            self.assertIsInstance(plans[0], PrefillPlan)
            admitted.append(plans[0].request_id)
            rid = admitted[-1]
            s.mark_prefill_chunk_done(rid, 0)
            s.mark_finished(rid)
        self.assertEqual(admitted, ["r1", "r2", "r3"])

    def test_waiting_stays_queued_until_slot_free(self):
        s = Scheduler(num_slots=2)
        s.add_request(_req("r1"))
        s.add_request(_req("r2"))
        plans = s.schedule()
        self.assertEqual(plans[0].request_id, "r1")
        # r2 still queued (FIFO), even though a slot is free
        self.assertEqual(s._requests["r2"].state, RequestState.WAITING)
        s.mark_prefill_chunk_done("r1", 0)
        self.assertEqual(s._requests["r2"].state, RequestState.WAITING)
        s.mark_finished("r1")
        plans = s.schedule()
        self.assertEqual(plans[0].request_id, "r2")


class TestSchedulerChunkedPrefill(unittest.TestCase):
    def test_chunk_bounds_correct(self):
        s = Scheduler(num_slots=1, max_chunk_tokens=512)
        s.add_request(_req("r1", prompt_len=1500))
        plans = s.schedule()
        self.assertEqual(plans[0].chunk_bounds, ((0, 512), (512, 1024), (1024, 1500)))

    def test_chunked_matches_unchunked_via_fake_runner(self):
        runner = FakeRunner(vocab_size=2000, seed=0)
        s = Scheduler(num_slots=1, max_chunk_tokens=512)
        s.add_request(_req("r1", prompt_len=1500))
        plan = s.schedule()[0]
        # Execute chunks in order through the fake runner, record token sequence
        seen = []
        for idx, (start, end) in enumerate(plan.chunk_bounds):
            chunk = list(plan.input_ids[start:end])
            seen.extend(chunk)
            runner.prefill(_tids(chunk))
            s.mark_prefill_chunk_done("r1", idx)
        self.assertEqual(seen, list(range(1500)))
        # Chunked prefill feeds the identical token sequence as a single pass
        chunked_tokens = [tok for call in runner.calls for tok in call[1]]
        self.assertEqual(chunked_tokens, list(range(1500)))


class TestSchedulerCancellation(unittest.TestCase):
    def test_cancel_waiting_not_scheduled(self):
        s = Scheduler(num_slots=2)
        s.add_request(_req("r1"))
        s.add_request(_req("r2"))
        s.cancel("r2")
        plans = s.schedule()
        self.assertEqual(plans[0].request_id, "r1")
        # r2 cancelled while waiting: never scheduled, its slot freed
        self.assertEqual(s.num_free_slots, 1)  # r1 holds one slot, r2 freed

    def test_cancel_active_frees_slot_and_suppresses_output(self):
        runner = FakeRunner(vocab_size=50)
        s = Scheduler(num_slots=1)
        s.add_request(_req("r1", prompt_len=4))
        plan = s.schedule()[0]
        runner.prefill(_tids(list(plan.input_ids)), slot=plan.cache_slot)
        s.mark_prefill_chunk_done("r1", 0)
        # Now decoding
        self.assertEqual(s._requests["r1"].state, RequestState.DECODING)
        s.cancel("r1")
        self.assertEqual(s._requests["r1"].state, RequestState.CANCELLED)
        self.assertEqual(s.num_free_slots, 1)
        # Cancelled request must not produce decode calls
        self.assertEqual(len(runner.calls), 1)  # only the prefill

    def test_cancel_unknown_is_noop(self):
        s = Scheduler(num_slots=1)
        s.cancel("nope")  # should not raise


class TestSchedulerDecodePlan(unittest.TestCase):
    def test_decode_plan_structure(self):
        s = Scheduler(num_slots=1)
        s.add_request(_req("r1", prompt_len=3))
        s.schedule()  # prefill
        s.mark_prefill_chunk_done("r1", 0)
        # Engine samples a token, appends it, then schedules decode
        s._requests["r1"].append_output_token(7)
        plans = s.schedule()
        self.assertEqual(len(plans), 1)
        self.assertIsInstance(plans[0], DecodePlan)
        dp = plans[0]
        self.assertEqual(dp.request_ids, ("r1",))
        self.assertEqual(dp.input_ids, (7,))  # most recent token (written this step)
        self.assertEqual(dp.positions, (3,))  # len(all_token_ids)-1 = 4-1
        self.assertEqual(dp.slots, (0,))
        self.assertEqual(dp.active_mask, (True,))
        self.assertEqual(dp.sampling_params, (SamplingParams(max_tokens=10),))
        self.assertEqual(dp.block_tables, ())

    def test_decode_plan_fields_via_fake_runner(self):
        runner = FakeRunner(vocab_size=50)
        s = Scheduler(num_slots=1)
        s.add_request(_req("r1", prompt_len=4))
        plan = s.schedule()[0]
        runner.prefill(_tids(list(plan.input_ids)), slot=plan.cache_slot)
        s.mark_prefill_chunk_done("r1", 0)
        s._requests["r1"].append_output_token(42)
        dp = s.schedule()[0]
        self.assertIsInstance(dp, DecodePlan)
        self.assertEqual(dp.request_ids, ("r1",))
        self.assertEqual(dp.input_ids, (42,))
        # Runner must receive the new token at its absolute position
        logits = runner.decode(_tids(list(dp.input_ids)), position=dp.positions[0], slot=dp.slots[0])
        self.assertEqual(logits.shape, (1, 50))


if __name__ == "__main__":
    unittest.main()
