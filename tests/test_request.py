import unittest
from infurnace.engine.request import (
    Request, RequestState, SamplingParams, RequestMetrics,
)


def _make(**overrides) -> Request:
    sp = SamplingParams(**{**dict(max_tokens=10), **overrides.pop("sp", {})})
    return Request("r1", [1, 2, 3], sp, 1024, **overrides)


class TestRequestLifecycle(unittest.TestCase):
    def _make(self, **overrides) -> Request:
        return _make(**overrides)

    def test_initial_waiting(self):
        req = _make()
        self.assertEqual(req.state, RequestState.WAITING)
        self.assertFalse(req.is_terminal())

    def test_valid_path_waiting_to_finished(self):
        req = _make()
        req.transition(RequestState.PREFILLING)
        req.transition(RequestState.DECODING)
        req.transition(RequestState.FINISHED)
        self.assertTrue(req.is_terminal())
        self.assertIsNotNone(req.metrics.completion_time)

    def test_invalid_transition_rejected(self):
        req = _make()
        req.transition(RequestState.PREFILLING)
        with self.assertRaises(ValueError):
            req.transition(RequestState.WAITING)  # backward

    def test_cancel_from_any_nonterminal(self):
        # Each path ends with CANCELLED from a different nonterminal state
        for path in (
            [RequestState.CANCELLED],
            [RequestState.PREFILLING, RequestState.CANCELLED],
            [RequestState.PREFILLING, RequestState.DECODING, RequestState.CANCELLED],
        ):
            req = _make()
            for state in path:
                req.transition(state)
            self.assertEqual(req.state, RequestState.CANCELLED)

    def test_terminal_is_absorbing(self):
        req = _make()
        req.transition(RequestState.PREFILLING)
        req.transition(RequestState.DECODING)
        req.transition(RequestState.FINISHED)
        with self.assertRaises(ValueError):
            req.transition(RequestState.CANCELLED)


class TestStopConditions(unittest.TestCase):
    def test_eos_stops(self):
        req = _make(eos_token_ids=[151643])
        req.append_output_token(5)
        self.assertEqual(req.check_finished(151643), "eos")

    def test_stop_token_ids(self):
        req = _make(sp=dict(stop_token_ids=[999]))
        req.append_output_token(1)
        self.assertEqual(req.check_finished(999), "stop_token_ids")

    def test_min_tokens_forces_continuation(self):
        req = _make(sp=dict(min_tokens=5, max_tokens=10, stop_token_ids=[999]))
        for i in range(4):
            req.append_output_token(i)
        self.assertIsNone(req.check_finished(999))  # ignored by min_tokens
        req.append_output_token(4)
        self.assertEqual(req.check_finished(999), "stop_token_ids")

    def test_max_tokens_stops(self):
        req = _make(sp=dict(max_tokens=3))
        for i in range(3):
            req.append_output_token(i)
        self.assertEqual(req.check_finished(7), "max_tokens")


class TestMetrics(unittest.TestCase):
    def test_arrival_set_on_init(self):
        req = _make()
        self.assertIsNotNone(req.metrics.arrival_time)

    def test_first_token_on_first_output(self):
        req = _make()
        req.transition(RequestState.PREFILLING)
        req.transition(RequestState.DECODING)
        # TTFT is recorded at the first *generated* token, not at prefill end.
        self.assertIsNone(req.metrics.first_token_time)
        req.append_output_token(7)
        self.assertIsNotNone(req.metrics.first_token_time)
        self.assertIsNotNone(req.metrics.ttft)


class TestTokenViews(unittest.TestCase):
    def test_readonly_views(self):
        req = _make()
        req.append_output_token(7)
        self.assertEqual(req.output_token_ids, (7,))
        self.assertEqual(req.all_token_ids, (1, 2, 3, 7))
        # tuple is immutable by construction

    def test_computed_tokens_increments(self):
        req = _make()
        req.append_output_token(7)
        req.append_output_token(8)
        self.assertEqual(req.computed_tokens, 2)


if __name__ == "__main__":
    unittest.main()
