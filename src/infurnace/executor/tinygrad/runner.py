from __future__ import annotations
from tinygrad import Tensor, TinyJit, UOp, Variable, dtypes
from tinygrad.helpers import Context
from .buffers import ContiguousKVCache
from .model import Qwen3Model

class RunnerError(ValueError):
  """Runner inputs or configuration do not satisfy the execution contract."""

class Qwen3Runner:
  """Serving runner with TinyJit-captured decode contracts.

  Prefill currently uses the eager stateless forward (Phase 2B). Decode is
  captured per slot with a symbolic ``Variable("position")``; the JIT contract
  uses ``@function`` + SSA cache writes so one compiled program replays for
  every token position without recompilation.
  """

  def __init__(self, model: Qwen3Model, kv_cache: ContiguousKVCache):
    if not isinstance(model, Qwen3Model): raise RunnerError("model must be a Qwen3Model")
    if not isinstance(kv_cache, ContiguousKVCache): raise RunnerError("kv_cache must be a ContiguousKVCache")
    self.model = model
    self.kv_cache = kv_cache
    self.model.kv_cache = kv_cache
    self._decode_jit: dict[int, TinyJit] = {}
    for slot in range(kv_cache.num_slots):
      self._decode_jit[slot] = self._capture_decode(slot)

  def prefill(self, input_ids: Tensor, slot: int = 0) -> Tensor:
    """Eager prefill at position 0."""
    return self.model.prefill(input_ids, self.kv_cache, slot=slot)

  def decode(self, input_ids: Tensor, position: int, slot: int = 0) -> Tensor:
    if input_ids.shape != (1, 1):
      raise RunnerError(f"decode input_ids must have shape (1, 1), got {input_ids.shape}")
    if slot < 0 or slot >= self.kv_cache.num_slots:
      raise RunnerError(f"slot {slot} out of range [0, {self.kv_cache.num_slots})")
    if position < 0 or position >= self.kv_cache.max_context:
      raise RunnerError(f"position {position} out of range [0, {self.kv_cache.max_context})")

    ids = input_ids if input_ids.uop.is_realized else input_ids.contiguous().realize()
    pos_var = Variable("position", 0, self.kv_cache.max_context - 1).bind(position)
    logits = self._decode_jit[slot](ids, pos_var)
    return logits

  def _capture_decode(self, slot: int) -> TinyJit:
    @TinyJit
    def _jit_decode(input_ids: Tensor, position: UOp) -> Tensor:
      return self.model._decode_step(input_ids, position, slot)

    max_context = self.kv_cache.max_context
    warmup_ids = Tensor([[0]], dtype=dtypes.int32).contiguous().realize()
    warmup_pos = Variable("position", 0, max_context - 1).bind(0)

    with Context(BEAM=0):
      _jit_decode(warmup_ids, warmup_pos)
      _jit_decode(warmup_ids, warmup_pos)

    return _jit_decode