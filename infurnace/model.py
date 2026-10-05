from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, TypeVar
import math
from tinygrad import Tensor, dtypes  # noqa: F401  # used by forward / KV cache (TODO)
from infurnace.gguf import PackedTensor, parse_gguf
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

  @property
  def kv_cache_bytes(self) -> int:
    """FP16 K and V storage; excludes weights, activations, and scratch buffers."""
    return self.block_count * self.context_length * self.attention_head_count_kv * (self.key_length + self.value_length) * 2


def config_from_gguf(metadata: Mapping, max_context: int | None = None) -> Qwen3Config:
  """Validate the supported dense Qwen3 contract before allocating model state."""
  if metadata.get("general.architecture") != "qwen3": raise ValueError("only dense Qwen3 models are supported")
  for feature in ("expert_count", "expert_used_count", "nextn_predict_layers", "attention.sliding_window"):
    if metadata.get(f"qwen3.{feature}", 0): raise ValueError(f"unsupported Qwen3 feature: {feature}")
  scaling = metadata.get("qwen3.rope.scaling.type", "none")
  if scaling != "none": raise ValueError(f"unsupported Qwen3 RoPE scaling: {scaling!r}")
  if any(k.startswith("qwen3.rope.scaling.") and k != "qwen3.rope.scaling.type" for k in metadata):
    raise ValueError("unsupported Qwen3 RoPE scaling parameters")

  def positive_int(key: str, default=None) -> int:
    value = metadata.get(key, default)
    if type(value) is not int or value <= 0: raise ValueError(f"{key} must be a positive integer")
    return value

  def positive_float(key: str) -> float:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
      raise ValueError(f"{key} must be a positive finite number")
    return float(value)

  dim = positive_int("qwen3.embedding_length")
  heads = positive_int("qwen3.attention.head_count")
  kv_heads = positive_int("qwen3.attention.head_count_kv")
  if "qwen3.attention.key_length" not in metadata and dim % heads:
    raise ValueError("cannot derive an integral attention head size")
  key_length = positive_int("qwen3.attention.key_length", dim // heads)
  value_length = positive_int("qwen3.attention.value_length", key_length)
  rope_dim = positive_int("qwen3.rope.dimension_count", key_length)
  if heads % kv_heads: raise ValueError("query head count must be divisible by KV head count")
  if value_length != key_length: raise ValueError("Qwen3 requires equal K/V head sizes")
  if rope_dim > key_length or rope_dim % 2: raise ValueError("RoPE dimension must be even and no larger than head size")
  context = positive_int("qwen3.context_length")
  if max_context is not None:
    if type(max_context) is not int or max_context <= 0: raise ValueError("max_context must be a positive integer")
    context = min(context, max_context)
  tokens = metadata.get("tokenizer.ggml.tokens")
  if not isinstance(tokens, list) or not tokens or any(not isinstance(token, str) for token in tokens):
    raise ValueError("missing or invalid tokenizer vocabulary")
  return Qwen3Config(
    block_count=positive_int("qwen3.block_count"), embedding_length=dim,
    feed_forward_length=positive_int("qwen3.feed_forward_length"), attention_head_count=heads,
    attention_head_count_kv=kv_heads, key_length=key_length, value_length=value_length, rope_dimension_count=rope_dim,
    rope_freq_base=positive_float("qwen3.rope.freq_base"),
    rms_norm_epsilon=positive_float("qwen3.attention.layer_norm_rms_epsilon"), vocab_size=len(tokens), context_length=context,
  )


def expected_weight_shapes(config: Qwen3Config) -> dict[str, tuple[int, ...]]:
  """Linear weights are [out_features, in_features], independent of storage format."""
  dim, hidden = config.embedding_length, config.feed_forward_length
  query_width = config.attention_head_count * config.key_length
  key_width = config.attention_head_count_kv * config.key_length
  value_width = config.attention_head_count_kv * config.value_length
  shapes = {"token_embd.weight": (config.vocab_size, dim), "output_norm.weight": (dim,), "output.weight": (config.vocab_size, dim)}
  for layer in range(config.block_count):
    layer_shapes = {
      "attn_norm.weight": (dim,), "attn_q.weight": (query_width, dim), "attn_k.weight": (key_width, dim),
      "attn_v.weight": (value_width, dim), "attn_q_norm.weight": (config.key_length,), "attn_k_norm.weight": (config.key_length,),
      "attn_output.weight": (dim, query_width), "ffn_norm.weight": (dim,), "ffn_gate.weight": (hidden, dim),
      "ffn_up.weight": (hidden, dim), "ffn_down.weight": (dim, hidden),
    }
    shapes.update({f"blk.{layer}.{name}": shape for name, shape in layer_shapes.items()})
  return shapes


_Weight = TypeVar("_Weight", Tensor, PackedTensor)

def prepare_weights(config: Qwen3Config, weights: Mapping[str, _Weight]) -> dict[str, _Weight]:
  """Validate tensors or packed descriptors without executing or mutating them."""
  prepared = dict(weights)
  if "output.weight" not in prepared:
    if "token_embd.weight" not in prepared: raise ValueError("missing required weight: token_embd.weight")
    prepared["output.weight"] = prepared["token_embd.weight"]
  shapes = expected_weight_shapes(config)
  for name, shape in shapes.items():
    if name not in prepared: raise ValueError(f"missing required weight: {name}")
    if prepared[name].shape != shape: raise ValueError(f"{name}: expected {shape}, got {prepared[name].shape}")
  # Unscaled Qwen3 uses metadata-derived frequencies. Accept only the known auxiliary table.
  if "rope_freqs.weight" in prepared:
    if prepared["rope_freqs.weight"].shape != (config.rope_dimension_count // 2,):
      raise ValueError("rope_freqs.weight has invalid shape")
    shapes["rope_freqs.weight"] = prepared["rope_freqs.weight"].shape
  extra = set(prepared) - set(shapes)
  if extra: raise ValueError(f"unsupported Qwen3 weights: {', '.join(sorted(extra))}")
  return prepared


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
    metadata, entries = parse_gguf(path)
    config = config_from_gguf(metadata, max_context)
    entries = prepare_weights(config, entries)
    tensors = {}
    for name, entry in entries.items():
      # Preserve the alias rather than copying/decoding tied embedding bytes twice.
      if name == "output.weight" and entry is entries["token_embd.weight"]:
        tensors[name] = tensors["token_embd.weight"]
      else:
        tensors[name] = entry.to().decode()
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