from __future__ import annotations
import math
from collections.abc import Mapping
from typing import Any
from .config import ModelConfigError, Qwen3Config, TensorSpec

_MODEL_KEYS = {
  "qwen3.block_count", "qwen3.context_length", "qwen3.embedding_length", "qwen3.feed_forward_length",
  "qwen3.attention.head_count", "qwen3.attention.head_count_kv", "qwen3.rope.freq_base",
  "qwen3.attention.layer_norm_rms_epsilon", "qwen3.attention.key_length", "qwen3.attention.value_length",
}
_EXPECTED = {
  "general.architecture": (str, "qwen3"),
  "general.type": (str, "model"),
  "qwen3.block_count": (int, 28),
  "qwen3.context_length": (int, 40960),
  "qwen3.embedding_length": (int, 1024),
  "qwen3.feed_forward_length": (int, 3072),
  "qwen3.attention.head_count": (int, 16),
  "qwen3.attention.head_count_kv": (int, 8),
  "qwen3.rope.freq_base": (float, 1000000.0),
  "qwen3.attention.layer_norm_rms_epsilon": (float, 9.999999974752427e-07),
  "qwen3.attention.key_length": (int, 128),
  "qwen3.attention.value_length": (int, 128),
  "general.quantization_version": (int, 2),
  "general.file_type": (int, 7),
}

def _expected(metadata: Mapping[str, Any], key: str) -> Any:
  if key not in metadata: raise ModelConfigError(f"missing GGUF metadata: {key}")
  expected_type, expected_value = _EXPECTED[key]
  value = metadata[key]
  if type(value) is not expected_type: raise ModelConfigError(f"{key} must be {expected_type.__name__}")
  if value != expected_value: raise ModelConfigError(f"unsupported {key}: expected {expected_value!r}, got {value!r}")
  return value

def _tensor_specs(blocks: int, hidden: int, intermediate: int, heads: int, kv_heads: int,
                  key_length: int, value_length: int, vocab_size: int) -> tuple[TensorSpec, ...]:
  query_width, key_width = heads * key_length, kv_heads * key_length
  value_width, output_width = kv_heads * value_length, heads * value_length
  specs = [
    TensorSpec("output_norm.weight", (hidden,), "F32", "float"),
    TensorSpec("token_embd.weight", (vocab_size, hidden), "Q8_0", "float"),
  ]
  for block in range(blocks):
    prefix = f"blk.{block}"
    specs.extend((
      TensorSpec(f"{prefix}.attn_k.weight", (key_width, hidden), "Q8_0", "float"),
      TensorSpec(f"{prefix}.attn_k_norm.weight", (key_length,), "F32", "float"),
      TensorSpec(f"{prefix}.attn_norm.weight", (hidden,), "F32", "float"),
      TensorSpec(f"{prefix}.attn_output.weight", (hidden, output_width), "Q8_0", "float"),
      TensorSpec(f"{prefix}.attn_q.weight", (query_width, hidden), "Q8_0", "float"),
      TensorSpec(f"{prefix}.attn_q_norm.weight", (key_length,), "F32", "float"),
      TensorSpec(f"{prefix}.attn_v.weight", (value_width, hidden), "Q8_0", "float"),
      TensorSpec(f"{prefix}.ffn_down.weight", (hidden, intermediate), "Q8_0", "float"),
      TensorSpec(f"{prefix}.ffn_gate.weight", (intermediate, hidden), "Q8_0", "float"),
      TensorSpec(f"{prefix}.ffn_norm.weight", (hidden,), "F32", "float"),
      TensorSpec(f"{prefix}.ffn_up.weight", (intermediate, hidden), "Q8_0", "float"),
    ))
  return tuple(specs)

def qwen3_config_from_gguf(metadata: Mapping[str, Any]) -> Qwen3Config:
  """Derive the pinned Qwen3-0.6B contract from tinygrad GGUF metadata."""
  if not isinstance(metadata, Mapping): raise ModelConfigError("GGUF metadata must be a mapping")
  if any(type(key) is not str for key in metadata): raise ModelConfigError("GGUF metadata keys must be strings")
  unknown_model_keys = {key for key in metadata if isinstance(key, str) and key.startswith("qwen3.")} - _MODEL_KEYS
  if unknown_model_keys: raise ModelConfigError(f"unsupported Qwen3 metadata: {', '.join(sorted(unknown_model_keys))}")

  values = {key: _expected(metadata, key) for key in _EXPECTED}
  tokens = metadata.get("tokenizer.ggml.tokens")
  if type(tokens) is not list or not tokens: raise ModelConfigError("tokenizer.ggml.tokens must be a non-empty array")
  if any(type(token) is not str for token in tokens): raise ModelConfigError("tokenizer.ggml.tokens must contain strings")

  heads, kv_heads = values["qwen3.attention.head_count"], values["qwen3.attention.head_count_kv"]
  key_length, value_length = values["qwen3.attention.key_length"], values["qwen3.attention.value_length"]
  if heads % kv_heads: raise ModelConfigError("attention head count must be divisible by KV head count")
  if key_length != value_length: raise ModelConfigError("Qwen3 key and value head dimensions must match")
  if key_length % 2: raise ModelConfigError("Qwen3 RoPE dimension must be even")
  for key in ("qwen3.rope.freq_base", "qwen3.attention.layer_norm_rms_epsilon"):
    if not math.isfinite(values[key]) or values[key] <= 0: raise ModelConfigError(f"{key} must be finite and positive")

  blocks, hidden = values["qwen3.block_count"], values["qwen3.embedding_length"]
  intermediate, vocab_size = values["qwen3.feed_forward_length"], len(tokens)
  if vocab_size != 151936: raise ModelConfigError(f"unsupported vocabulary size: expected 151936, got {vocab_size}")
  tensors = _tensor_specs(blocks, hidden, intermediate, heads, kv_heads, key_length, value_length, vocab_size)
  return Qwen3Config(
    architecture=values["general.architecture"], block_count=blocks,
    context_length=values["qwen3.context_length"], embedding_length=hidden,
    feed_forward_length=intermediate, attention_head_count=heads, attention_head_count_kv=kv_heads,
    key_length=key_length, value_length=value_length, rope_dimension_count=key_length,
    rope_freq_base=values["qwen3.rope.freq_base"],
    rms_norm_epsilon=values["qwen3.attention.layer_norm_rms_epsilon"], vocab_size=vocab_size,
    quantization_version=values["general.quantization_version"], file_type=values["general.file_type"],
    quantization="Q8_0", qk_norm=True, attention_bias=False, mlp_bias=False, tied_embeddings=True,
    mlp_type="swiglu", tensors=tensors,
  )