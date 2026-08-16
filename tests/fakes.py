from __future__ import annotations
from tinygrad import Tensor, dtypes
from infurnace.executor.runner import Runner


class FakeRunner(Runner):
    """Deterministic runner for scheduler/engine tests.

    Produces one-hot logits at a per-call counter so sampling is reproducible.
    Records every call so tests can assert schedule order and that cancellation
    suppresses in-flight output.
    """

    def __init__(self, vocab_size: int = 100, seed: int = 0):
        self.vocab_size = vocab_size
        self.calls: list[tuple[str, tuple, dict]] = []
        self._counter = seed

    def prefill(self, input_ids: Tensor, slot: int = 0) -> Tensor:
        self.calls.append(("prefill", tuple(input_ids.tolist()[0]), {"slot": slot}))
        return self._fake_logits()

    def decode(self, input_ids: Tensor, position: int, slot: int = 0) -> Tensor:
        tok = input_ids.tolist()[0][0]
        self.calls.append(("decode", (tok,), {"position": position, "slot": slot}))
        return self._fake_logits()

    def _fake_logits(self) -> Tensor:
        t = self._counter % self.vocab_size
        self._counter += 1
        data = [[0.0] * self.vocab_size]
        data[0][t] = 1.0
        return Tensor(data, dtype=dtypes.float32)
