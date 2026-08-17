import gc
import unittest

import pytest
from tinygrad import Tensor, dtypes
from tinygrad.helpers import GlobalCounters

from infurnace.executor.tinygrad.buffers import ContiguousKVCache
from infurnace.executor.tinygrad.model import Qwen3Model
from infurnace.executor.tinygrad.runner import Qwen3Runner

from test_runner import _make_config, _make_weights, _logits_close

_EXPECTED_KEYS = {(0,), (0, 1), (0, 1, 2, 3), (0, 1, 2, 4)}


class TestSteadyStateStability(unittest.TestCase):
    """Phase 4D: steady decode preserves allocations and never re-captures."""

    def setUp(self):
        self.config = _make_config(context_length=64)
        self.weights = _make_weights(self.config)

    def _build(self, num_slots: int = 4, max_context: int = 64):
        model = Qwen3Model(self.weights)
        kv = ContiguousKVCache(self.config, max_context=max_context, num_slots=num_slots)
        model.kv_cache = kv
        runner = Qwen3Runner(model, kv)
        for slot in range(num_slots):
            runner.prefill(Tensor([[slot + 1]], dtype=dtypes.int32), slot=slot).realize()
        return runner, kv

    @staticmethod
    def _mem() -> int:
        return GlobalCounters.mem_used

    def _warm_up(self, runner: Qwen3Runner) -> None:
        # One decode_batch per supported contract: shapes 1, 2, 4 and the shape-4
        # padded tail (B=3 writes the reserved dummy slot).
        for B, slots in ((1, (0,)), (2, (0, 1)), (3, (0, 1, 2)), (4, (0, 1, 2, 3))):
            runner.decode_batch(
                Tensor([[9] * B], dtype=dtypes.int32).reshape(B, 1), (1,) * B, slots
            ).realize()

    @staticmethod
    def _snapshot(runner: Qwen3Runner):
        return {k: (jit.captured, jit.cnt) for k, jit in runner._decode_batch_jit.items()}

    def test_steady_shapes_flat_and_no_recapture(self):
        runner, _kv = self._build()
        self._warm_up(runner)
        self.assertEqual(set(runner._decode_batch_jit), _EXPECTED_KEYS)
        gc.collect()
        mem_before = self._mem()
        before = self._snapshot(runner)

        n = 30
        for step in range(n):
            pos = 1 + step
            for B, slots in ((1, (0,)), (2, (0, 1)), (3, (0, 1, 2)), (4, (0, 1, 2, 3))):
                runner.decode_batch(
                    Tensor([[9] * B], dtype=dtypes.int32).reshape(B, 1), (pos,) * B, slots
                ).realize()

        gc.collect()
        self.assertEqual(self._mem(), mem_before)  # no new persistent device buffers
        for k, jit in runner._decode_batch_jit.items():
            captured, cnt = before[k]
            self.assertIs(jit.captured, captured)   # never re-captured
            self.assertEqual(jit.cnt - cnt, n)      # exactly one replay per step

    def test_ragged_trajectory_no_new_contracts(self):
        runner, kv = self._build()
        self._warm_up(runner)
        keys_before = set(runner._decode_batch_jit)
        before = self._snapshot(runner)
        gc.collect()
        mem_before = self._mem()

        # Active rows shrink 4 -> 3 -> 2 -> 1; every step reuses a captured
        # contract and still matches independent single-request eager decode.
        steps = [
            (4, (0, 1, 2, 3), (10, 11, 12, 13), (1, 1, 1, 1)),
            (3, (0, 1, 2), (20, 21, 22), (2, 2, 2)),
            (2, (0, 1), (30, 31), (3, 3)),
            (1, (0,), (40,), (4,)),
        ]
        for B, slots, toks, positions in steps:
            batch = runner.decode_batch(
                Tensor([[t] for t in toks], dtype=dtypes.int32), positions, slots
            ).realize()
            self.assertEqual(batch.shape, (B, self.config.vocab_size))
            for i, (tok, pos, slot) in enumerate(zip(toks, positions, slots)):
                single = runner.model.decode(
                    Tensor([[tok]], dtype=dtypes.int32), position=pos, kv_cache=kv, slot=slot
                ).realize().tolist()[0]
                _logits_close(single, batch.tolist()[i])

        self.assertEqual(set(runner._decode_batch_jit), keys_before)  # no new contracts
        for k, jit in runner._decode_batch_jit.items():
            captured, _ = before[k]
            self.assertIs(jit.captured, captured)
        gc.collect()
        self.assertEqual(self._mem(), mem_before)


@pytest.mark.nv
class TestSteadyStateStabilityNV(unittest.TestCase):
    """Phase 4D on NV: the same invariants against real device buffer accounting."""

    def setUp(self):
        self.config = _make_config(context_length=64)
        self.weights = _make_weights(self.config)

    def _build(self, num_slots: int = 4, max_context: int = 64):
        model = Qwen3Model(self.weights)
        kv = ContiguousKVCache(self.config, max_context=max_context, num_slots=num_slots)
        model.kv_cache = kv
        runner = Qwen3Runner(model, kv)
        for slot in range(num_slots):
            runner.prefill(Tensor([[slot + 1]], dtype=dtypes.int32), slot=slot).realize()
        return runner

    @staticmethod
    def _mem() -> int:
        return GlobalCounters.mem_used_per_device.get("NV", 0)

    def test_steady_shapes_flat_and_no_recapture(self):
        runner = self._build()
        for B, slots in ((1, (0,)), (2, (0, 1)), (3, (0, 1, 2)), (4, (0, 1, 2, 3))):
            runner.decode_batch(
                Tensor([[9] * B], dtype=dtypes.int32).reshape(B, 1), (1,) * B, slots
            ).realize()
        self.assertEqual(set(runner._decode_batch_jit), _EXPECTED_KEYS)
        gc.collect()
        mem_before = self._mem()
        before = {k: (jit.captured, jit.cnt) for k, jit in runner._decode_batch_jit.items()}

        n = 20
        for step in range(n):
            pos = 1 + step
            for B, slots in ((1, (0,)), (2, (0, 1)), (3, (0, 1, 2)), (4, (0, 1, 2, 3))):
                runner.decode_batch(
                    Tensor([[9] * B], dtype=dtypes.int32).reshape(B, 1), (pos,) * B, slots
                ).realize()

        gc.collect()
        self.assertEqual(self._mem(), mem_before)
        for k, jit in runner._decode_batch_jit.items():
            captured, cnt = before[k]
            self.assertIs(jit.captured, captured)
            self.assertEqual(jit.cnt - cnt, n)


if __name__ == "__main__":
    unittest.main()