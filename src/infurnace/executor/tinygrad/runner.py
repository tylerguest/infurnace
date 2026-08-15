from __future__ import annotations
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.helpers import Context
from .buffers import ContiguousKVCache
from .model import Qwen3Model

class RunnerError(ValueError):
  """Runner inputs or configuration do not satisfy the execution contract."""

class Qwen3Runner:
  """Serving runner with TinyJit-captured decode contracts.

  Prefill currently uses the eager stateless forward. Decode is captured per slot
  with a symbolic position Variable so the same compiled program replays for every
  token position without recompilation.
  """

  def __init__(self, model: Qwen3Model, kv_cache: ContiguousKVCache):
    if not isinstance(model, Qwen3Model): raise RunnerError("model must be a Qwen3Model")
    if not isinstance(kv_cache, ContiguousKVCache): raise RunnerError("kv_cache must be a ContiguousKVCache")
    self.model = model
    self.kv_cache = kv_cache
    self.model.kv_cache = kv_cache
    self._decode_jit: dict[int, TinyJit] = {}
    # Capture the decode contract immediately on the production cache. Warmup writes
    # a dummy token at position 0, which the subsequent prefill overwrites.
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
    mask = self._decode_mask(position)
    rope = self.model._rope[position:position+1].contiguous().realize()
    if not rope.uop.is_realized: rope = rope.clone()
    logits, k_stores, v_stores = self._decode_jit[slot](ids, mask, rope)
    self._write_kv_stores(k_stores, v_stores, position, slot)
    return logits

  def _decode_mask(self, position: int) -> Tensor:
    mc = self.kv_cache.max_context
    device = self.kv_cache.kv.device
    idx = Tensor.arange(mc + 1, dtype=dtypes.int32).to(device)
    keep = (idx < position) | (idx == mc)
    zero = Tensor.zeros(1, dtype=dtypes.float32).to(device)
    negi = Tensor.full((1,), float("-inf"), dtype=dtypes.float32).to(device)
    return keep.where(zero, negi).reshape(1, 1, 1, mc + 1).contiguous().realize()

  def _write_kv_stores(self, k_stores: Tensor, v_stores: Tensor, position: int, slot: int) -> None:
    kv = self.kv_cache.kv
    writes = []
    for i in range(self.model.config.block_count):
      k_write = kv[i, 0, slot, position:position+1, :, :].assign(k_stores[i])
      v_write = kv[i, 1, slot, position:position+1, :, :].assign(v_stores[i])
      writes.extend([k_write, v_write])
    Tensor.realize(*writes)

  def _capture_decode(self, slot: int) -> TinyJit:
    @TinyJit
    def _jit_decode(input_ids: Tensor, attn_mask: Tensor, rope: Tensor) -> tuple[Tensor, Tensor, Tensor]:
      return self.model._decode_step(input_ids, attn_mask, rope, slot)

    max_context = self.kv_cache.max_context
    warmup_ids = Tensor([[0]], dtype=dtypes.int32).contiguous().realize()
    warmup_mask = self._decode_mask(0)
    warmup_rope = self.model._rope[0:1].contiguous().realize()
    if not warmup_rope.uop.is_realized: warmup_rope = warmup_rope.clone()

    with Context(BEAM=0):
      _jit_decode(warmup_ids, warmup_mask, warmup_rope)
      _jit_decode(warmup_ids, warmup_mask, warmup_rope)

    return _jit_decode
