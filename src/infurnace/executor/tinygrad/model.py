from __future__ import annotations
import math
from typing import Sequence
from tinygrad import Tensor, UOp, dtypes
from infurnace.models.config import Qwen3Config
from .buffers import ContiguousKVCache
from .weights import Qwen3Weights

class Qwen3ModelError(ValueError):
  """Inputs or weights do not satisfy the Qwen3 forward contract."""

def _rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
  xf = x.float()
  return (xf * (xf.square().mean(-1, keepdim=True) + eps).rsqrt()).cast(x.dtype) * weight

def _linear(x: Tensor, weight: Tensor) -> Tensor:
  return x.linear(weight.transpose())

def _precompute_rope(dim: int, context_length: int, theta: float, device) -> Tensor:
  inv_freq = 1.0 / (theta ** (Tensor.arange(0, dim, 2, dtype=dtypes.float) / dim))
  positions = Tensor.arange(context_length, dtype=dtypes.float)
  freqs = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)
  return freqs.cos().cat(freqs.sin(), dim=-1).to(device).contiguous().realize()

def _apply_rope(x: Tensor, rope: Tensor) -> Tensor:
  cos, sin = rope.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

class Qwen3Model:
  def __init__(self, weights: Qwen3Weights):
    if not isinstance(weights, Qwen3Weights): raise Qwen3ModelError("weights must be a Qwen3Weights instance")
    config, tensors = weights.config, weights.tensors

    if "output.weight" not in tensors: raise Qwen3ModelError("missing tied output.weight")
    if tensors["output.weight"] is not tensors["token_embd.weight"]:
      raise Qwen3ModelError("output.weight must be the same object as token_embd.weight")

    expected_names = {spec.name for spec in config.tensors} | {"output.weight"}
    actual_names = set(tensors.keys())
    if missing := expected_names - actual_names:
      raise Qwen3ModelError(f"missing weight tensors: {', '.join(sorted(missing))}")
    if unexpected := actual_names - expected_names:
      raise Qwen3ModelError(f"unexpected weight tensors: {', '.join(sorted(unexpected))}")

    self.config = config
    self._w = tensors
    self.kv_cache: ContiguousKVCache | None = None
    device = tensors["token_embd.weight"].device
    self._rope = _precompute_rope(config.rope_dimension_count, config.context_length, config.rope_freq_base, device)

  def _validate_forward_inputs(self, input_ids: Tensor, start_position: int, kv: Tensor | None, slot: int) -> tuple[int, int]:
    config = self.config

    if not isinstance(input_ids, Tensor): raise Qwen3ModelError("input_ids must be a Tensor")
    if input_ids.ndim != 2: raise Qwen3ModelError(f"input_ids must be rank 2 [B, T], got rank {input_ids.ndim}")
    batch, seq_len = input_ids.shape
    if batch < 1: raise Qwen3ModelError("batch must be at least 1")
    if seq_len < 1: raise Qwen3ModelError("sequence length must be at least 1")
    if input_ids.dtype not in dtypes.ints: raise Qwen3ModelError(f"input_ids must have integer dtype, got {input_ids.dtype}")

    if start_position < 0: raise Qwen3ModelError(f"start_position must be non-negative, got {start_position}")
    end_position = start_position + seq_len
    if end_position > config.context_length:
      raise Qwen3ModelError(f"end_position {end_position} exceeds context length {config.context_length}")

    if kv is not None:
      if not isinstance(kv, Tensor): raise Qwen3ModelError("kv must be a Tensor or None")
      if kv.ndim != 6:
        raise Qwen3ModelError(f"kv must be rank 6 [layer, 2, slot, position, head, dim], got rank {kv.ndim}")
      if tuple(kv.shape[:2]) != (config.block_count, 2):
        raise Qwen3ModelError(f"kv leading shape mismatch: expected ({config.block_count}, 2), got {tuple(kv.shape[:2])}")
      if tuple(kv.shape[4:]) != (config.attention_head_count_kv, config.key_length):
        raise Qwen3ModelError(
          f"kv trailing shape mismatch: expected ({config.attention_head_count_kv}, {config.key_length}), "
          f"got {tuple(kv.shape[4:])}"
        )
      if kv.dtype != dtypes.float16:
        raise Qwen3ModelError(f"kv dtype must be float16, got {kv.dtype}")
      num_slots = kv.shape[2] - 1  # last physical slot is the reserved dummy
      if slot < 0 or slot >= num_slots:
        raise Qwen3ModelError(f"slot {slot} out of range [0, {num_slots})")
      max_context = kv.shape[3]
      if end_position > max_context:
        raise Qwen3ModelError(f"end_position {end_position} exceeds kv max_context {max_context}")
      if batch != 1:
        raise Qwen3ModelError("cached forward requires batch size 1")

    return batch, seq_len

  def forward(self, input_ids: Tensor, start_position: int = 0, kv: Tensor | None = None, slot: int = 0) -> Tensor:
    config, w = self.config, self._w
    batch, seq_len = self._validate_forward_inputs(input_ids, start_position, kv, slot)
    end_position = start_position + seq_len

    x = w["token_embd.weight"][input_ids].float()
    rope = self._rope[start_position:end_position]

    for i in range(config.block_count):
      p = f"blk.{i}"

      h = _rms_norm(x, w[f"{p}.attn_norm.weight"], config.rms_norm_epsilon)
      q = _linear(h, w[f"{p}.attn_q.weight"])
      k = _linear(h, w[f"{p}.attn_k.weight"])
      v = _linear(h, w[f"{p}.attn_v.weight"])

      q = q.reshape(batch, seq_len, config.attention_head_count, config.key_length).transpose(1, 2)
      k = k.reshape(batch, seq_len, config.attention_head_count_kv, config.key_length).transpose(1, 2)
      v = v.reshape(batch, seq_len, config.attention_head_count_kv, config.value_length).transpose(1, 2)

      q = _rms_norm(q, w[f"{p}.attn_q_norm.weight"], config.rms_norm_epsilon)
      k = _rms_norm(k, w[f"{p}.attn_k_norm.weight"], config.rms_norm_epsilon)

      q = _apply_rope(q, rope)
      k = _apply_rope(k, rope)

      if kv is not None:
        if start_position == 0:
          attn = q.scaled_dot_product_attention(k, v, is_causal=True, enable_gqa=True)
        else:
          cached_k = kv[i, 0, slot, :start_position].permute(1, 0, 2).unsqueeze(0)
          cached_v = kv[i, 1, slot, :start_position].permute(1, 0, 2).unsqueeze(0)
          full_k = cached_k.cat(k, dim=2)
          full_v = cached_v.cat(v, dim=2)

          if seq_len == 1:
            attn = q.scaled_dot_product_attention(full_k, full_v, is_causal=False, enable_gqa=True)
          else:
            total_len = start_position + seq_len
            key_pos = Tensor.arange(total_len, dtype=dtypes.int32).to(q.device)
            query_pos = Tensor.arange(start_position, end_position, dtype=dtypes.int32).to(q.device)
            mask = (key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)).reshape(1, 1, seq_len, total_len)
            attn = q.scaled_dot_product_attention(full_k, full_v, attn_mask=mask, is_causal=False, enable_gqa=True)

        k_store = k[0].permute(1, 0, 2).contiguous().cast(kv.dtype)
        v_store = v[0].permute(1, 0, 2).contiguous().cast(kv.dtype)
        k_write = kv[i, 0, slot, start_position:end_position].assign(k_store)
        v_write = kv[i, 1, slot, start_position:end_position].assign(v_store)
        Tensor.realize(k_write, v_write)
      else:
        attn = q.scaled_dot_product_attention(k, v, is_causal=True, enable_gqa=True)

      attn = attn.transpose(1, 2).reshape(batch, seq_len, -1)
      x = x + _linear(attn, w[f"{p}.attn_output.weight"])

      h = _rms_norm(x, w[f"{p}.ffn_norm.weight"], config.rms_norm_epsilon)
      gate = _linear(h, w[f"{p}.ffn_gate.weight"])
      up = _linear(h, w[f"{p}.ffn_up.weight"])
      x = (x + _linear(gate.silu().contiguous() * up, w[f"{p}.ffn_down.weight"])).contiguous()

    x = _rms_norm(x[:, -1:], w["output_norm.weight"], config.rms_norm_epsilon)
    return _linear(x, w["output.weight"])[:, -1, :]

  def _decode_step(self, input_ids: Tensor, position: UOp, slot: int = 0) -> Tensor:
    """Single-token decode using SSA cache writes inside the TinyJit graph.

    Uses `uop.store` / `uop.after` to write new K/V into the cache at a
    symbolic position, then reads positions [0:position+1] back. The cache
    buffer is captured by TinyJit as a closure buffer (like the model weights)
    so its mutations persist across JIT replays. No `@function` boundary —
    TinyJit captures the whole graph.

    ``position`` is a bound ``UOp`` (Variable); the JIT captures its range so
    one compiled program replays for every position.
    """
    if self.kv_cache is None: raise Qwen3ModelError("kv_cache must be attached to the model before decode")
    kv = self.kv_cache.kv
    config, w = self.config, self._w
    rope = self._rope[position:position+1]
    x = w["token_embd.weight"][input_ids].float()

    for i in range(config.block_count):
      p = f"blk.{i}"

      h = _rms_norm(x, w[f"{p}.attn_norm.weight"], config.rms_norm_epsilon)
      q = _linear(h, w[f"{p}.attn_q.weight"])
      k = _linear(h, w[f"{p}.attn_k.weight"])
      v = _linear(h, w[f"{p}.attn_v.weight"])

      q = q.reshape(1, 1, config.attention_head_count, config.key_length).transpose(1, 2)
      k = k.reshape(1, 1, config.attention_head_count_kv, config.key_length).transpose(1, 2)
      v = v.reshape(1, 1, config.attention_head_count_kv, config.value_length).transpose(1, 2)

      q = _rms_norm(q, w[f"{p}.attn_q_norm.weight"], config.rms_norm_epsilon)
      k = _rms_norm(k, w[f"{p}.attn_k_norm.weight"], config.rms_norm_epsilon)

      q = _apply_rope(q, rope)
      k = _apply_rope(k, rope)

      # SSA store k/v into cache at symbolic position, then observe the
      # cache after the store to read positions [0:position+1].
      k_for_store = k[0].permute(1, 0, 2).contiguous().cast(kv.dtype)
      v_for_store = v[0].permute(1, 0, 2).contiguous().cast(kv.dtype)
      store_uop = kv[i, :, slot, position:position+1, :, :].uop.store(
        Tensor.stack(k_for_store, v_for_store).uop
      )
      assigned = Tensor(kv[i, :, slot].uop.after(store_uop))
      cached_k = assigned[0, 0:position+1, :, :].permute(1, 0, 2).unsqueeze(0).float()
      cached_v = assigned[1, 0:position+1, :, :].permute(1, 0, 2).unsqueeze(0).float()

      attn = q.scaled_dot_product_attention(cached_k, cached_v, is_causal=False, enable_gqa=True)
      attn = attn.transpose(1, 2).reshape(1, 1, -1)
      x = x + _linear(attn, w[f"{p}.attn_output.weight"])

      h = _rms_norm(x, w[f"{p}.ffn_norm.weight"], config.rms_norm_epsilon)
      gate = _linear(h, w[f"{p}.ffn_gate.weight"])
      up = _linear(h, w[f"{p}.ffn_up.weight"])
      x = (x + _linear(gate.silu().contiguous() * up, w[f"{p}.ffn_down.weight"])).contiguous()

    x = _rms_norm(x[:, -1:], w["output_norm.weight"], config.rms_norm_epsilon)
    return _linear(x, w["output.weight"])[:, -1, :]

  def _decode_batch_step(self, input_ids: Tensor, positions: Sequence[int | UOp], slots: Sequence[int]) -> Tensor:
    """Batched single-token decode for a fixed shape ``[S, 1]``.

    Row ``i`` writes its K/V into ``kv[.., slots[i], ..]`` at the row's symbolic
    ``positions[i]`` and attends over ``kv[.., slots[i], 0:positions[i]+1]``, so
    each row reads only O(position) cache entries (no full-cache mask). Store
    targets are Python constants (``slots``), so a fixed-shape contract can be
    captured once by TinyJit. The reserved dummy slot is the write target for
    padded/inactive rows.
    """
    if self.kv_cache is None: raise Qwen3ModelError("kv_cache must be attached to the model before decode")
    kv = self.kv_cache.kv
    if input_ids.ndim != 2 or input_ids.shape[1] != 1:
      raise Qwen3ModelError(f"decode_batch input_ids must have shape [S, 1], got {tuple(input_ids.shape)}")
    if input_ids.dtype not in dtypes.ints:
      raise Qwen3ModelError("decode_batch input_ids must have integer dtype")
    S = input_ids.shape[0]
    if S < 1: raise Qwen3ModelError("decode_batch requires at least one row")
    if len(positions) != S or len(slots) != S:
      raise Qwen3ModelError("decode_batch positions and slots must have one entry per row")
    max_slot = kv.shape[2] - 1
    for i, (pos, slot) in enumerate(zip(positions, slots)):
      if slot < 0 or slot > max_slot:
        raise Qwen3ModelError(f"slot {slot} out of range [0, {max_slot}]")
      if isinstance(pos, int) and (pos < 0 or pos >= self.kv_cache.max_context):
        raise Qwen3ModelError(f"position {pos} out of range [0, {self.kv_cache.max_context})")

    config, w = self.config, self._w
    x = w["token_embd.weight"][input_ids].float()  # [S, 1, dim]

    for l in range(config.block_count):
      p = f"blk.{l}"

      h = _rms_norm(x, w[f"{p}.attn_norm.weight"], config.rms_norm_epsilon)
      q = _linear(h, w[f"{p}.attn_q.weight"])
      k = _linear(h, w[f"{p}.attn_k.weight"])
      v = _linear(h, w[f"{p}.attn_v.weight"])

      q = q.reshape(S, 1, config.attention_head_count, config.key_length).transpose(1, 2)
      k = k.reshape(S, 1, config.attention_head_count_kv, config.key_length).transpose(1, 2)
      v = v.reshape(S, 1, config.attention_head_count_kv, config.value_length).transpose(1, 2)

      q = _rms_norm(q, w[f"{p}.attn_q_norm.weight"], config.rms_norm_epsilon)
      k = _rms_norm(k, w[f"{p}.attn_k_norm.weight"], config.rms_norm_epsilon)

      row_q = [_apply_rope(q[i:i+1], self._rope[positions[i]:positions[i]+1]) for i in range(S)]
      row_k = [_apply_rope(k[i:i+1], self._rope[positions[i]:positions[i]+1]) for i in range(S)]

      stores = []
      for i in range(S):
        k_for_store = row_k[i][0].permute(1, 0, 2).contiguous().cast(kv.dtype)
        v_for_store = v[i].permute(1, 0, 2).contiguous().cast(kv.dtype)
        stores.append(
          kv[l, :, slots[i], positions[i]:positions[i]+1, :, :].uop.store(
            Tensor.stack(k_for_store, v_for_store).uop
          )
        )

      row_attn = []
      for i in range(S):
        # Observe every store in the layer so the scheduler sees one AFTER per
        # buffer slice with a single superseding write set (no WAR cycle).
        assigned = Tensor(kv[l, :, slots[i]].uop.after(*stores))
        cached_k = assigned[0, 0:positions[i]+1, :, :].permute(1, 0, 2).unsqueeze(0).float()
        cached_v = assigned[1, 0:positions[i]+1, :, :].permute(1, 0, 2).unsqueeze(0).float()
        attn = row_q[i].scaled_dot_product_attention(cached_k, cached_v, is_causal=False, enable_gqa=True)
        row_attn.append(attn.transpose(1, 2).reshape(1, 1, -1))

      attn = row_attn[0].stack(*row_attn[1:], dim=0).reshape(S, 1, -1)  # [S, 1, out_dim]
      x = x + _linear(attn, w[f"{p}.attn_output.weight"])

      h = _rms_norm(x, w[f"{p}.ffn_norm.weight"], config.rms_norm_epsilon)
      gate = _linear(h, w[f"{p}.ffn_gate.weight"])
      up = _linear(h, w[f"{p}.ffn_up.weight"])
      x = (x + _linear(gate.silu().contiguous() * up, w[f"{p}.ffn_down.weight"])).contiguous()

    x = _rms_norm(x[:, -1:], w["output_norm.weight"], config.rms_norm_epsilon)
    return _linear(x, w["output.weight"])[:, -1, :]  # [S, V]

  def prefill(self, input_ids: Tensor, kv_cache: ContiguousKVCache, slot: int = 0, start_position: int = 0) -> Tensor:
    if not isinstance(kv_cache, ContiguousKVCache): raise Qwen3ModelError("kv_cache must be a ContiguousKVCache")
    return self.forward(input_ids, start_position=start_position, kv=kv_cache.kv, slot=slot)

  def decode(self, input_ids: Tensor, position: int, kv_cache: ContiguousKVCache, slot: int = 0) -> Tensor:
    if not isinstance(kv_cache, ContiguousKVCache): raise Qwen3ModelError("kv_cache must be a ContiguousKVCache")
    return self.forward(input_ids, start_position=position, kv=kv_cache.kv, slot=slot)

  def __call__(self, input_ids: Tensor, **kwargs) -> Tensor:
    return self.forward(input_ids, **kwargs)
