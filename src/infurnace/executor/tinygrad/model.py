from __future__ import annotations
import math
from tinygrad import Tensor, dtypes
from infurnace.models.config import Qwen3Config
from .weights import Qwen3Weights

class Qwen3ModelError(ValueError):
  """Inputs or weights do not satisfy the stateless Qwen3 forward contract."""

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
    device = tensors["token_embd.weight"].device
    self._rope = _precompute_rope(config.rope_dimension_count, config.context_length, config.rope_freq_base, device)

  def forward(self, input_ids: Tensor) -> Tensor:
    config, w = self.config, self._w

    if not isinstance(input_ids, Tensor): raise Qwen3ModelError("input_ids must be a Tensor")
    if input_ids.ndim != 2: raise Qwen3ModelError(f"input_ids must be rank 2 [B, T], got rank {input_ids.ndim}")
    batch, seq_len = input_ids.shape
    if batch < 1: raise Qwen3ModelError("batch must be at least 1")
    if seq_len < 1: raise Qwen3ModelError("sequence length must be at least 1")
    if seq_len > config.context_length:
      raise Qwen3ModelError(f"sequence length {seq_len} exceeds context length {config.context_length}")
    if input_ids.dtype not in dtypes.ints: raise Qwen3ModelError(f"input_ids must have integer dtype, got {input_ids.dtype}")

    x = w["token_embd.weight"][input_ids].float()
    rope = self._rope[:seq_len]

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

      attn = q.scaled_dot_product_attention(k, v, is_causal=True, enable_gqa=True)
      attn = attn.transpose(1, 2).reshape(batch, seq_len, -1)
      x = x + _linear(attn, w[f"{p}.attn_output.weight"])

      h = _rms_norm(x, w[f"{p}.ffn_norm.weight"], config.rms_norm_epsilon)
      gate = _linear(h, w[f"{p}.ffn_gate.weight"])
      up = _linear(h, w[f"{p}.ffn_up.weight"])
      x = (x + _linear(gate.silu().contiguous() * up, w[f"{p}.ffn_down.weight"])).contiguous()

    x = _rms_norm(x[:, -1:], w["output_norm.weight"], config.rms_norm_epsilon)
    return _linear(x, w["output.weight"])[:, -1, :]

  def __call__(self, input_ids: Tensor) -> Tensor:
    return self.forward(input_ids)