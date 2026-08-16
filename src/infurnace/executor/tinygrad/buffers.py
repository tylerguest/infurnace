from __future__ import annotations
from dataclasses import dataclass, field
from tinygrad import Tensor, dtypes
from infurnace.models.config import Qwen3Config

class KVCacheError(ValueError):
  """KV cache configuration or allocation does not satisfy the contract."""

@dataclass(frozen=True)
class ContiguousKVCache:
  """Persistent external contiguous KV storage for a single model instance.

  Axis order: [layer, K_or_V, slot, token_position, kv_head, head_dim].
  The tensor is allocated with zeros, made contiguous, and realized so it can
  be passed as a stable TinyJit input and replaced by a compatible buffer later.

  One extra physical slot (index ``num_slots``) is reserved as the ``dummy_slot``:
  a harmless write target for padded/inactive rows in fixed-shape batched decode.
  It is never assigned to a request and never cleared.
  """
  config: Qwen3Config
  max_context: int
  num_slots: int
  dtype = dtypes.float16
  device: str | None = None
  kv: Tensor = field(init=False)

  def __post_init__(self):
    if not isinstance(self.config, Qwen3Config):
      raise KVCacheError("config must be a Qwen3Config instance")
    if self.config.key_length != self.config.value_length:
      raise KVCacheError("ContiguousKVCache requires key_length == value_length")
    if self.max_context < 1:
      raise KVCacheError(f"max_context must be positive, got {self.max_context}")
    if self.num_slots < 1:
      raise KVCacheError(f"num_slots must be positive, got {self.num_slots}")
    if self.max_context > self.config.context_length:
      raise KVCacheError(f"max_context {self.max_context} exceeds model context_length {self.config.context_length}")

    shape = (
      self.config.block_count,
      2,  # K and V
      self.num_slots + 1,  # real slots plus the reserved dummy slot
      self.max_context,
      self.config.attention_head_count_kv,
      self.config.key_length,
    )
    kv = Tensor.zeros(*shape, dtype=self.dtype, device=self.device).contiguous().realize()
    object.__setattr__(self, "kv", kv)

  @property
  def dummy_slot(self) -> int:
    """Physical slot index reserved for padded/inactive batched-decode writes."""
    return self.num_slots

  @property
  def shape(self) -> tuple[int, ...]:
    return tuple(self.kv.shape)

  @property
  def size_bytes(self) -> int:
    return self.kv.numel() * self.kv.dtype.itemsize

  @property
  def num_layers(self) -> int:
    return self.config.block_count

  @property
  def num_kv_heads(self) -> int:
    return self.config.attention_head_count_kv

  @property
  def head_dim(self) -> int:
    return self.config.key_length

  def clear_slot(self, slot: int) -> None:
    """Zero out all KV entries for *slot*, enabling reuse without stale state.

    After clearing, the slot is in the same state as a freshly allocated cache.
    The caller is responsible for re-prefilling before decoding.
    """
    if slot < 0 or slot >= self.num_slots:
      raise KVCacheError(f"slot {slot} out of range [0, {self.num_slots})")
    shape = (
      self.config.block_count, 2, 1, self.max_context,
      self.config.attention_head_count_kv, self.config.key_length,
    )
    self.kv[:, :, slot:slot+1].assign(
      Tensor.zeros(*shape, dtype=self.dtype, device=self.kv.device)
    ).realize()