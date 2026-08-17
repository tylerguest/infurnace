from __future__ import annotations
from typing import Sequence
from tinygrad import Tensor, dtypes

class KVStoreError(ValueError):
  """Indexed KV-store inputs or state do not satisfy the contract."""

def decompose_slot(slot: int, page_size: int) -> tuple[int, int]:
  """Split a flat pool slot into ``(physical_page, page_offset)``."""
  if slot < 0:
    raise KVStoreError(f"slot must be non-negative, got {slot}")
  if page_size < 1:
    raise KVStoreError(f"page_size must be positive, got {page_size}")
  return divmod(slot, page_size)

def store_kv(
  pool: Tensor,
  layer: int,
  new_k: Tensor,
  new_v: Tensor,
  slot_mapping: Sequence[int],
  active_mask: Sequence[bool],
  *,
  num_pages: int,
  page_size: int,
) -> Tensor:
  """Eager indexed KV store into a paged pool (Phase 5B).

  ``pool`` is a realized ``[layers, 2, pages, page_size, kv_heads, head_dim]``
  fp16 tensor. Each row writes its new K/V into
  ``pool[layer, :, page, offset:offset+1]`` where
  ``(page, offset) = divmod(slot_mapping[i], page_size)``. Inactive rows write
  the unique dummy position ``(num_pages + i, 0)`` so padded decode rows never
  alias live or shared state; the pool must reserve at least as many dummy
  pages as there are inactive rows.

  Returns the (realized) pool after the stores so callers can chain reads; the
  writes are applied in row order. New K/V of any float dtype are cast to the
  pool dtype, matching the model's cache-store behavior.
  """
  if not isinstance(pool, Tensor) or pool.ndim != 6:
    raise KVStoreError(f"pool must be a rank-6 tensor, got {pool}")
  if pool.dtype != dtypes.float16:
    raise KVStoreError(f"pool dtype must be float16, got {pool.dtype}")
  if layer < 0 or layer >= pool.shape[0]:
    raise KVStoreError(f"layer {layer} out of range [0, {pool.shape[0]})")
  if num_pages < 1 or num_pages > pool.shape[2]:
    raise KVStoreError(f"num_pages {num_pages} out of range [1, {pool.shape[2]}]")
  if page_size < 1 or page_size != pool.shape[3]:
    raise KVStoreError(f"page_size {page_size} does not match pool {pool.shape[3]}")

  batch = new_k.shape[0]
  if new_k.shape != new_v.shape or tuple(new_k.shape) != (batch, pool.shape[4], pool.shape[5]):
    raise KVStoreError(
      f"new_k/new_v must be [B, {pool.shape[4]}, {pool.shape[5]}], "
      f"got {tuple(new_k.shape)} / {tuple(new_v.shape)}"
    )
  if len(slot_mapping) != batch or len(active_mask) != batch:
    raise KVStoreError("slot_mapping and active_mask must have one entry per row")

  real_slots = num_pages * page_size
  for slot in slot_mapping:
    if slot < 0 or slot >= real_slots:
      raise KVStoreError(f"slot {slot} out of range [0, {real_slots})")
  inactive = sum(1 for active in active_mask if not active)
  if inactive > pool.shape[2] - num_pages:
    raise KVStoreError(f"{inactive} inactive rows exceed {pool.shape[2] - num_pages} dummy pages")

  dummy_index = 0
  for i in range(batch):
    if active_mask[i]:
      page, offset = divmod(slot_mapping[i], page_size)
    else:
      page, offset = num_pages + dummy_index, 0
      dummy_index += 1
    kv_pair = Tensor.stack(new_k[i].cast(pool.dtype), new_v[i].cast(pool.dtype))
    pool[layer, :, page, offset:offset+1].assign(kv_pair.unsqueeze(1)).realize()
  return pool