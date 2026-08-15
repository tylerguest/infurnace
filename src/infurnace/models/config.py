from __future__ import annotations

from dataclasses import dataclass


class ModelConfigError(ValueError):
  """GGUF metadata does not satisfy a supported model contract."""


@dataclass(frozen=True, slots=True)
class TensorSpec:
  name: str
  shape: tuple[int, ...]
  storage_dtype: str
  logical_dtype: str


@dataclass(frozen=True, slots=True)
class Qwen3Config:
  architecture: str
  block_count: int
  context_length: int
  embedding_length: int
  feed_forward_length: int
  attention_head_count: int
  attention_head_count_kv: int
  key_length: int
  value_length: int
  rope_dimension_count: int
  rope_freq_base: float
  rms_norm_epsilon: float
  vocab_size: int
  quantization_version: int
  file_type: int
  quantization: str
  qk_norm: bool
  attention_bias: bool
  mlp_bias: bool
  tied_embeddings: bool
  mlp_type: str
  tensors: tuple[TensorSpec, ...]
