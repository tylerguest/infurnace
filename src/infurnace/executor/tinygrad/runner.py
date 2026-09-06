from __future__ import annotations
from typing import TYPE_CHECKING, Sequence
from tinygrad import Tensor, TinyJit, UOp, Variable, dtypes
from tinygrad.helpers import Context
from .buffers import ContiguousKVCache
from .model import Qwen3Model
from ..runner import Runner

if TYPE_CHECKING:
  from .weights import Qwen3Weights

SUPPORTED_DECODE_SHAPES: tuple[int, ...] = (1, 2, 4)

class RunnerError(ValueError):
  """Runner inputs or configuration do not satisfy the execution contract."""

class Qwen3Runner(Runner):
  """Serving runner with TinyJit-captured decode contracts.

  Prefill currently uses the eager stateless forward (Phase 2B). Single-request
  decode is captured per slot with a symbolic ``Variable("position")`` and SSA
  ``uop.store`` / ``uop.after`` cache writes. Fixed-shape batched decode
  (Phase 4B) is captured per slot configuration with per-row symbolic
  positions; padded/inactive rows write the reserved dummy slot.
  """

  def __init__(self, model: Qwen3Model, kv_cache: ContiguousKVCache):
    if not isinstance(model, Qwen3Model): raise RunnerError("model must be a Qwen3Model")
    if not isinstance(kv_cache, ContiguousKVCache): raise RunnerError("kv_cache must be a ContiguousKVCache")
    self.model = model
    self.kv_cache = kv_cache
    self.model.kv_cache = kv_cache
    # Single-decode and batched-decode contracts are captured lazily on first
    # use so a runner that never decodes through a given path does not pay the
    # compile/workspace cost (keeps NV engine memory within budget).
    self._decode_jit: dict[int, TinyJit] = {}
    self._decode_batch_jit: dict[tuple[int, ...], TinyJit] = {}
    # Phase 5D fused decode state: two persistent [1, 1] token buffers the
    # decode graph reads from and writes to on device (closure constants), plus
    # the alternating read/write role between two captures. Variable-indexed
    # feedback slices on a closure buffer lag by one replay on the NV HCQ path,
    # so the buffers are swapped by selecting which capture runs, not by a
    # bound index.
    self._feedback_read = Tensor.zeros(1, 1, dtype=dtypes.int32, device=kv_cache.kv.device).contiguous().realize()
    self._feedback_write = Tensor.zeros(1, 1, dtype=dtypes.int32, device=kv_cache.kv.device).contiguous().realize()
    self._fused_toggle = 0
    self._fused_seed = 0
    self._decode_fused_jit: dict[int, tuple[TinyJit, TinyJit]] = {}

  @property
  def num_slots(self) -> int:
    return self.kv_cache.num_slots

  @property
  def max_context(self) -> int:
    return self.kv_cache.max_context

  def clear_slot(self, slot: int) -> None:
    self.kv_cache.clear_slot(slot)

  def move_slot(self, from_slot: int, to_slot: int) -> None:
    """Copy slot KV and zero the source, enabling prefix compaction.

    The scheduler compacts active requests into slots ``0..B-1`` so each
    batched-decode row stays bound to its physical slot.
    """
    if from_slot < 0 or from_slot >= self.kv_cache.num_slots:
      raise RunnerError(f"move_slot source {from_slot} out of range [0, {self.kv_cache.num_slots})")
    if to_slot < 0 or to_slot >= self.kv_cache.num_slots:
      raise RunnerError(f"move_slot destination {to_slot} out of range [0, {self.kv_cache.num_slots})")
    kv = self.kv_cache.kv
    kv[:, :, to_slot:to_slot+1].assign(kv[:, :, from_slot:from_slot+1]).realize()
    self.kv_cache.clear_slot(from_slot)

  @classmethod
  def from_weights(cls, weights: "Qwen3Weights", *, num_slots: int = 1,
                   max_context: int | None = None, device: str | None = None) -> "Qwen3Runner":
    """Build a runner from validated weights: allocate KV cache + JIT capture.

    ``max_context`` defaults to the model's ``context_length``. ``device`` is the
    tinygrad device for the KV cache (None lets tinygrad auto-detect).
    """
    from .buffers import ContiguousKVCache
    from .model import Qwen3Model
    config = weights.config
    if max_context is None:
      max_context = config.context_length
    kv = ContiguousKVCache(config=config, max_context=max_context, num_slots=num_slots, device=device)
    return cls(Qwen3Model(weights), kv)

  def prefill(self, input_ids: Tensor, slot: int = 0, start_position: int = 0) -> Tensor:
    """Eager prefill. ``start_position`` is the KV position of the chunk's first token."""
    return self.model.prefill(input_ids, self.kv_cache, slot=slot, start_position=start_position)

  def decode(self, input_ids: Tensor, position: int, slot: int = 0) -> Tensor:
    if input_ids.shape != (1, 1):
      raise RunnerError(f"decode input_ids must have shape (1, 1), got {input_ids.shape}")
    if slot < 0 or slot >= self.kv_cache.num_slots:
      raise RunnerError(f"slot {slot} out of range [0, {self.kv_cache.num_slots})")
    if position < 0 or position >= self.kv_cache.max_context:
      raise RunnerError(f"position {position} out of range [0, {self.kv_cache.max_context})")

    ids = input_ids if input_ids.uop.is_realized else input_ids.contiguous().realize()
    jit = self._decode_jit.get(slot)
    if jit is None:
      jit = self._capture_decode(slot)
      self._decode_jit[slot] = jit
    pos_var = Variable("position", 0, self.kv_cache.max_context - 1).bind(position)
    logits = jit(ids, pos_var)
    return logits

  def decode_batch(self, input_ids: Tensor, positions: Sequence[int], slots: Sequence[int]) -> Tensor:
    """Batched decode for ``B`` active rows, padded to the next supported shape.

    The caller keeps active requests compacted in slots ``0..B-1`` so row ``i``
    maps to a fixed physical slot (a Python-constant store target). Padded rows
    write the reserved dummy slot at distinct positions and their logits are
    discarded; only the first ``B`` rows are returned.
    """
    if input_ids.ndim != 2 or input_ids.shape[1] != 1:
      raise RunnerError(f"decode_batch input_ids must have shape [B, 1], got {tuple(input_ids.shape)}")
    B = input_ids.shape[0]
    if len(positions) != B or len(slots) != B:
      raise RunnerError("decode_batch positions and slots must have one entry per row")
    if tuple(slots) != tuple(range(B)):
      raise RunnerError(f"decode_batch requires active slots 0..{B-1} (compaction), got {slots}")
    S = next((s for s in SUPPORTED_DECODE_SHAPES if s >= B), None)
    if S is None or S > self.kv_cache.num_slots:
      raise RunnerError(f"no supported decode shape for batch size {B} (shapes {SUPPORTED_DECODE_SHAPES})")
    for pos in positions:
      if pos < 0 or pos >= self.kv_cache.max_context:
        raise RunnerError(f"position {pos} out of range [0, {self.kv_cache.max_context})")

    dummy = self.kv_cache.dummy_slot
    padded_slots = tuple(range(B)) + (dummy,) * (S - B)
    padded_ids = Tensor(
      input_ids.tolist() + [[0]] * (S - B), dtype=dtypes.int32
    ).contiguous().realize()
    padded_positions = list(positions) + list(range(S - B))
    bound = [
      Variable(f"position_{i}", 0, self.kv_cache.max_context - 1).bind(p)
      for i, p in enumerate(padded_positions)
    ]
    jit = self._decode_batch_jit.get(padded_slots)
    if jit is None:
      jit = self._capture_decode_batch(padded_slots)
      self._decode_batch_jit[padded_slots] = jit
    logits = jit(padded_ids, *bound)
    return logits[:B]

  def initialize_fused_decode(self, seed_token: int) -> None:
    """Seed the fused-decode feedback buffers before a request's first decode.

    The prefill token is sampled on the host, so both buffers are written with
    it once per request. If the fused contracts are already captured, the
    buffers are written immediately; otherwise the first ``decode_fused`` call
    captures first and re-seeds after the capture warmups consumed the seed.
    """
    if type(seed_token) is not int or seed_token < 0:
      raise RunnerError(f"seed_token must be a non-negative int, got {seed_token!r}")
    self._fused_seed = seed_token
    if self._decode_fused_jit:
      self._seed_feedback()

  def _seed_feedback(self) -> None:
    self._feedback_read.assign(Tensor([[self._fused_seed]], dtype=dtypes.int32)).realize()
    self._feedback_write.assign(Tensor([[self._fused_seed]], dtype=dtypes.int32)).realize()
    self._fused_toggle = 0

  def decode_fused(self, position: int, slot: int = 0) -> int:
    """Single-token decode with in-graph greedy argmax and device feedback.

    The captured graph reads the previous sampled token from a persistent
    on-device buffer, decodes, argmaxes on device, and writes the next token
    back to the paired buffer — no per-step host roundtrip or eager ops.
    Returns the sampled token id as a host int.
    """
    if slot < 0 or slot >= self.kv_cache.num_slots:
      raise RunnerError(f"slot {slot} out of range [0, {self.kv_cache.num_slots})")
    if position < 0 or position >= self.kv_cache.max_context:
      raise RunnerError(f"position {position} out of range [0, {self.kv_cache.max_context})")
    pair = self._decode_fused_jit.get(slot)
    if pair is None:
      pair = self._capture_decode_fused(slot)
      self._decode_fused_jit[slot] = pair
      self._seed_feedback()  # capture warmups consumed the seed; restore it
    self._fused_toggle = 1 - self._fused_toggle
    jit = pair[self._fused_toggle]
    pos_var = Variable("position", 0, self.kv_cache.max_context - 1).bind(position)
    token = jit(pos_var)
    return int(token.tolist()[0][0])

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

  def _capture_decode_batch(self, slots: tuple[int, ...]) -> TinyJit:
    @TinyJit
    def _jit_decode_batch(input_ids: Tensor, *positions: UOp) -> Tensor:
      return self.model._decode_batch_step(input_ids, positions, slots)

    S = len(slots)
    max_context = self.kv_cache.max_context
    warmup_ids = Tensor([[0]] * S, dtype=dtypes.int32).contiguous().realize()
    warmup_positions = [
      Variable(f"position_{i}", 0, max_context - 1).bind(0) for i in range(S)
    ]

    with Context(BEAM=0):
      _jit_decode_batch(warmup_ids, *warmup_positions)
      _jit_decode_batch(warmup_ids, *warmup_positions)

    return _jit_decode_batch

  def _capture_decode_fused(self, slot: int) -> tuple[TinyJit, TinyJit]:
    """Capture the two paired decode contracts: read A -> write B and read B -> write A."""
    @TinyJit
    def _jit_read_a(position: UOp) -> Tensor:
      return self.model._decode_step_fused(self._feedback_read, position, self._feedback_write, slot)

    @TinyJit
    def _jit_read_b(position: UOp) -> Tensor:
      return self.model._decode_step_fused(self._feedback_write, position, self._feedback_read, slot)

    max_context = self.kv_cache.max_context
    # Warmup at the last cache position so the capture's KV stores and reads
    # never touch the request's real prefix (positions 0..max_context-2).
    warmup_pos = Variable("position", 0, max_context - 1).bind(max_context - 1)

    with Context(BEAM=0):
      _jit_read_a(warmup_pos)
      _jit_read_a(warmup_pos)
      _jit_read_b(warmup_pos)
      _jit_read_b(warmup_pos)

    return _jit_read_a, _jit_read_b