from __future__ import annotations
from tinygrad import Tensor, dtypes
from infurnace.executor.runner import Runner


class FakeRunner(Runner):
    """Deterministic runner for scheduler/engine tests.

    Produces one-hot logits at a per-call counter so sampling is reproducible.
    Records every call so tests can assert schedule order and that cancellation
    suppresses in-flight output. Also tracks ``clear_slot`` invocations so tests
    can verify cache clearance on terminal paths.
    """

    def __init__(self, vocab_size: int = 100, seed: int = 0, num_slots: int = 1, max_context: int = 1024):
        self.vocab_size = vocab_size
        self.calls: list[tuple[str, tuple, dict]] = []
        self._counter = seed
        self._num_slots = num_slots
        self._max_context = max_context
        self.cleared_slots: list[int] = []

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def max_context(self) -> int:
        return self._max_context

    def prefill(self, input_ids: Tensor, slot: int = 0, start_position: int = 0) -> Tensor:
        self.calls.append(("prefill", tuple(input_ids.tolist()[0]), {"slot": slot, "start_position": start_position}))
        return self._fake_logits()

    def decode(self, input_ids: Tensor, position: int, slot: int = 0) -> Tensor:
        tok = input_ids.tolist()[0][0]
        self.calls.append(("decode", (tok,), {"position": position, "slot": slot}))
        return self._fake_logits()

    def decode_batch(self, input_ids: Tensor, positions, slots) -> Tensor:
        rows = input_ids.tolist()
        toks = tuple(r[0] for r in rows)
        self.calls.append(("decode_batch", toks, {"positions": tuple(positions), "slots": tuple(slots)}))
        data = [[0.0] * self.vocab_size for _ in rows]
        for i in range(len(rows)):
            t = self._counter % self.vocab_size
            self._counter += 1
            data[i][t] = 1.0
        return Tensor(data, dtype=dtypes.float32)

    def clear_slot(self, slot: int) -> None:
        self.cleared_slots.append(slot)

    def move_slot(self, from_slot: int, to_slot: int) -> None:
        self.calls.append(("move_slot", (), {"from_slot": from_slot, "to_slot": to_slot}))

    def _fake_logits(self) -> Tensor:
        t = self._counter % self.vocab_size
        self._counter += 1
        data = [[0.0] * self.vocab_size]
        data[0][t] = 1.0
        return Tensor(data, dtype=dtypes.float32)
