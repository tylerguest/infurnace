from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping
from tinygrad import Tensor, dtypes  # noqa: F401  # used by forward / KV cache (TODO)
from infurnace.gguf import load_gguf
from infurnace.kernels.quant_gemv import quant_gemv, packed_weight  # noqa: F401  # used by _decode_linear (TODO)


@dataclass(frozen=True)
class Qwen3Config:
  """Shape/rope/norm hyperparameters derived from GGUF metadata."""
  block_count: int
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
  context_length: int


def config_from_gguf(metadata: Mapping) -> Qwen3Config:
  """Read the ``qwen3.*`` GGUF metadata into a config (no strict pinning)."""
  raise NotImplementedError  # TODO: qwen3.block_count, embedding_length, feed_forward_length, heads, ...


class Qwen3Model:
  def __init__(self, config: Qwen3Config, weights: Mapping[str, Tensor]):
    """``weights``: lazy fp16 views keyed by GGUF tensor name.
    Sanity checks the required tensor names/shapes, precomputes RoPE freqs,
    and allocates the contiguous fp16 KV cache ``[layers, 2, context, kv_heads, head_dim]``.
    """
    raise NotImplementedError  # TODO: validate tensors, rope, KV cache, per-weight packed cache

  @classmethod
  def from_gguf(cls, path: str | Path, max_context: int | None = None) -> "Qwen3Model":
    """Load a GGUF, derive the config, and build the model. ``max_context`` caps
    the KV cache (defaults to the model's context length)."""
    metadata, tensors = load_gguf(path)
    config = config_from_gguf(metadata)
    return cls(config, tensors)

  def _decode_linear(self, x: Tensor, name: str) -> Tensor:
    """Decode matmul for weight ``name``: packed Q4_K/Q6_K GEMV when possible,
    generic fp16 ``x.linear(w.transpose())`` fallback otherwise."""
    raise NotImplementedError  # TODO: packed_weight + quant_gemv, fall back to _linear

  def forward(self, input_ids: Tensor, start_position: int = 0) -> Tensor:
    """Eager prefill: embed -> blocks -> logits for the last token, writing
    K/V into the internal cache at ``[start_position, start_position+seq_len)``."""
    raise NotImplementedError  # TODO: rmsnorm, rope, scaled_dot_product_attention, KV writes

  def decode(self, input_ids: Tensor, position: int) -> Tensor:
    """Single-token decode at cache ``position``. TinyJit-captured with a
    symbolic ``Variable("position")`` and SSA ``uop.store``/``uop.after`` KV
    writes so one compiled program replays for every position."""
    raise NotImplementedError  # TODO: _decode_step + TinyJit capture (port from old executor runner)

  def generate(self, tokens: list[int], max_new_tokens: int, temperature: float = 0.0) -> Iterator[int]:
    """Prefill ``tokens`` then yield generated token ids until ``max_new_tokens``.

    Sampling: greedy argmax for ``temperature == 0``, otherwise Gumbel-max
    (same as ``tinygrad.llm.model.Transformer.forward``). Callers check
    ``tokenizer.is_end`` on yielded ids to stop at EOS.
    """
    raise NotImplementedError  # TODO: prefill -> sample -> decode loop