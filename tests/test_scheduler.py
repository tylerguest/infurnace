import unittest
from dataclasses import FrozenInstanceError
from tinygrad import Tensor, dtypes
from infurnace.engine.request import Request, RequestState, SamplingParams
from infurnace.scheduler.scheduler import Scheduler, SchedulerError
from infurnace.scheduler.plan import PrefillPlan, DecodePlan
from infurnace.scheduler.policy import chunk_prompt
from fakes import FakeRunner


def _req(rid: str, prompt_len: int = 5, **overrides) -> Request:
    sp = overrides.pop("sp", None)
    if sp is None:
        sp = SamplingParams(max_tokens=10)
    elif isinstance(sp, dict):
        sp = SamplingParams(**{**dict(max_tokens=10), **sp})
    context_limit = overrides.pop("context_limit", 1024)
    return Request(rid, list(range(prompt_len)), sp, context_limit, **overrides)


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
        r2 = _req("r2")
        s.add_request(r2)  # only 1 slot, r1 still live -> REJECTED, no raise
        self.assertEqual(s.num_free_slots, 0)
        self.assertEqual(r2.state, RequestState.REJECTED)
        self.assertEqual(s.get_request("r2").state, RequestState.REJECTED)

    def test_duplicate_request_id(self):
        s = Scheduler(num_slots=2)
        s.add_request(_req("r1"))
        with self.assertRaises(SchedulerError):
            s.add_request(_req("r1"))

    def test_invalid_num_slots(self):
        with self.assertRaises(SchedulerError):
            Scheduler(num_slots=0)

    def test_reject_prompt_exceeds_scheduler_max_context(self):
        s = Scheduler(num_slots=1, max_context=10)
        req = _req("r1", prompt_len=11)
        s.add_request(req)
        self.assertEqual(req.state, RequestState.REJECTED)
        self.assertIn("context limit", req.error)

    def test_reject_prompt_exceeds_request_context_limit(self):
        s = Scheduler(num_slots=1, max_context=1024)
        req = _req("r1", prompt_len=6, context_limit=5)
        s.add_request(req)
        self.assertEqual(req.state, RequestState.REJECTED)

    def test_reject_when_max_tokens_exceeds_available_context(self):
        s = Scheduler(num_slots=1, max_context=10)
        sp = SamplingParams(max_tokens=4)
        req = _req("r1", prompt_len=7, sp=sp)
        s.add_request(req)
        self.assertEqual(req.state, RequestState.REJECTED)

    def test_admit_at_exact_boundary(self):
        s = Scheduler(num_slots=1, max_context=10)
        s.add_request(_req("r1", prompt_len=10, sp=SamplingParams(max_tokens=0)))
        self.assertEqual(s.num_free_slots, 0)


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
        s.add_request(_req("r1", prompt_len=1500, context_limit=1500, sp=dict(max_tokens=0)))
        plans = s.schedule()
        self.assertEqual(plans[0].chunk_bounds, ((0, 512), (512, 1024), (1024, 1500)))

    def test_chunked_matches_unchunked_via_fake_runner(self):
        runner = FakeRunner(vocab_size=2000, seed=0)
        s = Scheduler(num_slots=1, max_chunk_tokens=512)
        s.add_request(_req("r1", prompt_len=1500, context_limit=1500, sp=dict(max_tokens=0)))
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


class TestSchedulerSlotManagement(unittest.TestCase):
  """Phase 4A: persistent slot state — lowest-slot admission, compaction, and
  decode-plan slot ordering."""

  def test_admission_assigns_lowest_free_slot(self):
    s = Scheduler(num_slots=3)
    s.add_request(_req("r1"))
    s.add_request(_req("r2"))
    s.add_request(_req("r3"))
    self.assertEqual([s._requests[r].cache_slot for r in ("r1", "r2", "r3")], [0, 1, 2])
    s.schedule()  # r1 prefilling
    s.mark_prefill_chunk_done("r1", 0)  # r1 decoding
    s.mark_finished("r1")  # frees slot 0
    s.add_request(_req("r4"))
    self.assertEqual(s._requests["r4"].cache_slot, 0)

  def _set_decoding(self, s: Scheduler, rid: str):
    """Force a waiting request into an active DECODING state for compaction
    tests (serial admission keeps one request active via ``schedule``)."""
    req = s._requests[rid]
    req.state = RequestState.DECODING
    s._active[rid] = req

  def test_compact_single_hole(self):
    s = Scheduler(num_slots=2)
    s.add_request(_req("r1"))
    s.add_request(_req("r2"))
    self._set_decoding(s, "r1")  # r1@0
    self._set_decoding(s, "r2")  # r2@1
    s.mark_finished("r1")  # frees slot 0; r2 still active at slot 1
    self.assertEqual(s.compact(), [("r2", 1, 0)])

  def test_compact_no_moves_when_already_prefix(self):
    s = Scheduler(num_slots=2)
    s.add_request(_req("r1"))
    s.add_request(_req("r2"))
    self._set_decoding(s, "r1")
    self._set_decoding(s, "r2")
    self.assertEqual(s.compact(), [])

  def test_compact_multiple_holes(self):
    s = Scheduler(num_slots=4)
    for rid in ("r1", "r2", "r3", "r4"):
      s.add_request(_req(rid))
    for rid in ("r1", "r2", "r3", "r4"):
      self._set_decoding(s, rid)
    s.mark_finished("r1")
    s.mark_finished("r2")  # holes at 0 and 1
    self.assertEqual(s.compact(), [("r4", 3, 0), ("r3", 2, 1)])

  def test_compact_ignores_terminal_requests(self):
    s = Scheduler(num_slots=2)
    s.add_request(_req("r1"))
    s.add_request(_req("r2"))
    self._set_decoding(s, "r1")
    self._set_decoding(s, "r2")
    s.mark_finished("r1")
    s.mark_finished("r2")  # nothing active
    self.assertEqual(s.compact(), [])

  def test_decode_plan_slots_sorted_by_slot(self):
    s = Scheduler(num_slots=2)
    r1, r2 = _req("r1"), _req("r2")
    r1.cache_slot, r2.cache_slot = 1, 0
    dp = s._make_decode_plan([r1, r2])
    self.assertEqual(dp.slots, (0, 1))
    self.assertEqual(dp.request_ids, ("r2", "r1"))

  def test_free_slot_reclaims_vacated_slot(self):
    s = Scheduler(num_slots=4)
    for rid in ("r1", "r2", "r3", "r4"):
      s.add_request(_req(rid))
    # All four slots reserved at admission; a compaction move vacates a slot
    # that the engine returns to the free set (Phase 4D regression).
    self.assertEqual(s.num_free_slots, 0)
    s.free_slot(3)
    self.assertEqual(s.num_free_slots, 1)
    s.free_slot(3)  # idempotent
    self.assertEqual(s.num_free_slots, 1)
    with self.assertRaises(SchedulerError):
      s.free_slot(4)


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
