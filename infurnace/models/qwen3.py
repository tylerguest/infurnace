from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, TypeVar
import math
from tinygrad import Tensor, Context, dtypes
from tinygrad.llm.model import Transformer, TransformerConfig, TransformerBlock, apply_rope
from tinygrad.nn.state import load_state_dict
from tinygrad.llm.gguf import ggml_data_to_tensor
from infurnace.gguf import PackedTensor, parse_gguf
from infurnace.kernels.quant_gemv import (quant_gemv, q8_quantize, prepare_q6, prepare_q4, dequant_prepared_q4, dequant_prepared_q6,
                                         QUANT_SIZES)
from infurnace.kernels.quant_gemm import quant_gemm, nv_prefill_supported
from infurnace.kernels.attention import decode_attention, fused_decode_attention, nv_attention_supported

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

def transformer_config(config: Qwen3Config) -> TransformerConfig:
  return TransformerConfig(
    num_blocks=config.block_count, dim=config.embedding_length, hidden_dim=config.feed_forward_length,
    n_heads=config.attention_head_count, n_kv_heads=config.attention_head_count_kv, norm_eps=config.rms_norm_epsilon,
    vocab_size=config.vocab_size, head_dim=config.key_length, v_head_dim=config.value_length,
    rope_theta=config.rope_freq_base, rope_dim=config.rope_dimension_count, qk_norm=config.key_length,
    max_context=config.context_length,
  )

class _NVAttentionBlock(TransformerBlock):
  """Per-model decode-only adapter; upstream owns prefill and unsupported shapes.

  The pin has no dispatch hook. Keep this small supported decode setup here
  until an upstream hook exists; no dependency or global Tensor method is patched.
  """
  fused_attention = False
  # Keep packing local to each invocation: no activation cache survives a token,
  # reset or JIT capture. The upstream function boundary and norm math stay intact.
  share_q8 = True

  def _feed_forward(self, x):
    if (self.share_q8 and all(isinstance(s, int) for s in x.shape[:2]) and x.shape[:2] == (1, 1) and
        all(isinstance(getattr(self, name, None), _NVLinear) for name in ('ffn_gate', 'ffn_up'))):
      q8 = q8_quantize(x, 1, self.config.dim)
      return self.ffn_down(self.ffn_gate.decode(x, q8).silu().contiguous()*self.ffn_up.decode(x, q8))
    return super()._feed_forward(x)

  def _attention(self, x, start_pos):
    if not all(isinstance(s, int) for s in x.shape[:2]) or x.shape[:2] != (1, 1):
      return super()._attention(x, start_pos)
    if self.share_q8 and all(isinstance(getattr(self, name), _NVLinear) for name in ('attn_q', 'attn_k', 'attn_v')):
      q8 = q8_quantize(x, 1, self.config.dim)
      q, k, v = (getattr(self, name).decode(x, q8) for name in ('attn_q', 'attn_k', 'attn_v'))
    else:
      q, k, v = self.attn_q(x), self.attn_k(x), self.attn_v(x)
    q = self.attn_q_norm(q.reshape(1, 1, self.config.n_heads, 128).transpose(1, 2))
    k = self.attn_k_norm(k.reshape(1, 1, self.config.n_kv_heads, 128).transpose(1, 2))
    v = v.reshape(1, 1, self.config.n_kv_heads, 128).transpose(1, 2)
    frequencies = self.freqs_cis[start_pos:start_pos+1]
    q, k = apply_rope(q, frequencies), apply_rope(k, frequencies)
    store = self.cache_kv[:, :, :, start_pos:start_pos+1, :].uop.store(Tensor.stack(k, v).cast(self.cache_kv.dtype).uop)
    assigned_kv = Tensor(self.cache_kv.uop.after(store))
    attention = fused_decode_attention if self.fused_attention else decode_attention
    attn = attention(q, assigned_kv, start_pos+1)
    return self.attn_output(attn.transpose(1, 2).reshape(1, 1, -1))

def enable_decode_attention(blocks, *, fused: bool = False) -> int:
  """Adapt existing supported blocks in place, preserving weights/state identities."""
  count = 0
  for block in blocks:
    config = block.config
    if (type(block) not in (TransformerBlock, _NVAttentionBlock) or config.head_dim != 128 or config.rope_dim != 128
        or config.qk_norm != 128 or config.attn_output_gate or config.n_heads%config.n_kv_heads
        or (fused and config.n_heads != config.n_kv_heads*4)):
      continue
    block.__class__ = _NVAttentionBlock
    block.fused_attention = fused
    count += 1
  return count

class _NVLinear:
  """Single-token fast path using explicit GGUF identity, never graph-size guessing."""
  def __init__(self, original, entry: PackedTensor, *, prefill: bool = False, q6_aligned: bool = False,
               q6_swizzled: bool = True, q4_swizzled: bool = True, realize: bool = True):
    self.original = original
    if prefill and entry.ggml_type not in (12, 13, 14): raise ValueError("NV packed prefill supports only Q4_K/Q5_K/Q6_K")
    self.prefill = prefill
    if prefill and not nv_prefill_supported(entry.data.device):
      raise ValueError("NV prefill requires CUDA-C FP16 tensor cores")
    self.in_features, self.out_features = entry.shape[1], entry.shape[0]
    self.ggml_type = entry.ggml_type
    dtype = dtypes.uint32 if entry.ggml_type in (12, 13) else dtypes.uint16
    # Typed preparation may allocate new storage on NV. Generic consumers must
    # use this same storage; _enable_nv_quant also redirects any tied embedding.
    self.q4_swizzled = q4_swizzled and q6_aligned and entry.ggml_type == 12
    self.q6_aligned = q6_aligned and entry.ggml_type == 14
    self.q6_swizzled = self.q6_aligned and q6_swizzled
    # realize=False leaves `packed` lazy so a caller can realize many adapters in one schedule.
    self.packed = (prepare_q4(entry.data, in_features=self.in_features, realize=realize) if self.q4_swizzled else
                   prepare_q6(entry.data, in_features=self.in_features, swizzled=self.q6_swizzled, realize=realize) if self.q6_aligned else
                   entry.data.contiguous().bitcast(dtype).flatten())
    if realize: self.packed.realize()
    # Generic consumers decode fields directly from the same padded allocation.
    # Neither the original GGUF bytes nor a second dense weight copy is retained.
    decoded = (dequant_prepared_q4(self.packed, in_features=self.in_features) if self.q4_swizzled else
               dequant_prepared_q6(self.packed, in_features=self.in_features, swizzled=self.q6_swizzled) if self.q6_aligned else
               ggml_data_to_tensor(self.packed.bitcast(dtypes.uint8), self.in_features * self.out_features, self.ggml_type))
    original.weight = decoded.reshape(entry.shape).cast(dtypes.float16)
    self.weight, self.bias = original.weight, original.bias

  def decode(self, x: Tensor, q8: tuple[Tensor, Tensor, Tensor] | None = None) -> Tensor:
    result = quant_gemv(x, self.packed, out_features=self.out_features, in_features=self.in_features, ggml_type=self.ggml_type,
                        q6_aligned=self.q6_aligned, q6_swizzled=self.q6_swizzled, q4_swizzled=self.q4_swizzled, q8=q8)
    return result if self.bias is None else result+self.bias

  def __call__(self, x: Tensor) -> Tensor:
    if isinstance(x.numel(), int) and x.numel() == self.in_features:
      return self.decode(x)
    if self.prefill:
      # Upstream prefill uses a symbolic token count. Pad to its fixed maximum
      # for capture, then slice valid rows; the kernel handles the final 16-row tile.
      symbolic = not isinstance(x.numel(), int)
      activation = x.pad_to(x.max_shape) if symbolic else x
      result = quant_gemm(activation, self.packed, out_features=self.out_features,
                          in_features=self.in_features, ggml_type=self.ggml_type, q6_aligned=self.q6_aligned, q6_swizzled=self.q6_swizzled,
                          q4_swizzled=self.q4_swizzled)
      if symbolic: result = result.shrink(tuple((0, size) for size in (*x.shape[:-1], self.out_features)))
      return result if self.bias is None else result + self.bias
    return self.original(x)

class Qwen3Model:
  """Thin local runner: tinygrad owns transformer math, cache, generation and JIT."""
  def __init__(self, config: Qwen3Config, weights: Mapping[str, Tensor]):
    weights = prepare_weights(config, weights)
    devices = {weight.device for weight in weights.values()}
    if len(devices) != 1: raise ValueError("all model weights must be on the same compute device")
    self.device = next(iter(devices))
    if not isinstance(self.device, str) or self.device.startswith("DISK:"):
      raise ValueError("model weights must be on a single compute device")
    self.config, self.metadata = config, {}
    self._generating = False
    with Context(DEV=self.device):
      self.transformer = Transformer(transformer_config(config))
      # Cast each identity once: embedding/output stay tied even after dtype conversion.
      casts = {}
      state = {}
      for name, weight in weights.items():
        if name == "rope_freqs.weight": continue
        if id(weight) not in casts: casts[id(weight)] = weight.cast(dtypes.float16)
        state[name] = casts[id(weight)]
      load_state_dict(self.transformer, state, verbose=False, realize=False)
      # Baseline uses generic tinygrad linears; custom NV integration is a later phase.
      self.transformer.output.use_custom_quant = False
      for block in self.transformer.blk:
        for name in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"):
          getattr(block, name).use_custom_quant = False

  @classmethod
  def from_gguf(cls, path: str | Path, max_context: int | None = None, device: str | None = None, *,
                nv_quant: bool = False, nv_prefill: bool = False, nv_attention: bool = False,
                nv_fused_attention: bool = False, nv_prepared_weights: bool | None = None) -> "Qwen3Model":
    """Load a GGUF, derive the config, and build the model. ``max_context`` caps
    the KV cache (defaults to the model's context length). ``nv_prefill`` opts
    into FP16 tensor-core packed prefill and requires ``nv_quant``.
    ``nv_attention`` opts into experimental parallel-softmax NV decode attention;
    ``nv_fused_attention`` selects the fused four-query-head GQA path instead.
    ``nv_prepared_weights`` defaults on with ``nv_quant``: value-preserving
    tiled/swizzled Q4/Q6 blocks shared by decode, prefill and tied embedding consumers.
    Pass False to retain native GGUF Q4/Q6 layouts."""
    if nv_prepared_weights is None: nv_prepared_weights = nv_quant
    if nv_prepared_weights and not nv_quant: raise ValueError("prepared weights require NV quantized integration")
    if nv_prefill and not nv_quant: raise ValueError("NV prefill requires NV quantized integration")
    if (nv_attention or nv_fused_attention) and not nv_quant: raise ValueError("NV attention requires NV quantized integration")
    metadata, entries = parse_gguf(path)
    config = config_from_gguf(metadata, max_context)
    entries = prepare_weights(config, entries)
    tensors, resident = {}, {}
    for name, entry in entries.items():
      # Preserve the alias rather than copying/decoding tied embedding bytes twice.
      if name == "output.weight" and entry is entries["token_embd.weight"]:
        tensors[name] = tensors["token_embd.weight"]
        resident[name] = resident["token_embd.weight"]
      else:
        resident[name] = entry.to(device) if device is not None else entry.to()
        tensors[name] = resident[name].decode()
    result = cls(config, tensors)
    result.metadata = metadata
    if nv_quant: result._enable_nv_quant(resident, nv_prefill=nv_prefill, nv_prepared_weights=nv_prepared_weights)
    if nv_attention or nv_fused_attention:
      if not nv_attention_supported(result.device): raise ValueError("NV attention requires NV with the CUDA C renderer")
      result.nv_attention_blocks = enable_decode_attention(result.transformer.blk, fused=nv_fused_attention)
      if not result.nv_attention_blocks: raise ValueError("no supported blocks for NV decode attention")
      result.nv_attention = True
      result.nv_fused_attention = nv_fused_attention
      result.nv_shared_q8_blocks = sum(block.share_q8 and all(isinstance(getattr(block, name, None), _NVLinear)
        for name in ('attn_q', 'attn_k', 'attn_v', 'ffn_gate', 'ffn_up'))
        for block in result.transformer.blk if isinstance(block, _NVAttentionBlock))
    return result

  def _enable_nv_quant(self, entries: Mapping[str, PackedTensor], *, nv_prefill: bool = False,
                       nv_prepared_weights: bool = False) -> None:
    from tinygrad import Device
    from tinygrad.renderer.cstyle import CUDARenderer
    if self.device.split(":")[0] != "NV" or not isinstance(Device[self.device].renderer, CUDARenderer):
      raise ValueError("--nv-quant requires NV with the CUDA C renderer")
    if nv_prefill and not nv_prefill_supported(self.device): raise ValueError("NV prefill requires CUDA-C FP16 tensor cores")
    self.nv_prefill = nv_prefill
    self.nv_prepared_weights = nv_prepared_weights
    self.nv_prepared_weight_layout = "q4_tile4_quad_q6_tile4_pair" if nv_prepared_weights else "gguf"
    self.nv_quant_linears = 0
    for name, entry in entries.items():
      if entry.ggml_type not in QUANT_SIZES or len(entry.shape) != 2 or name == "token_embd.weight": continue
      if entry.shape[1] % 256 or entry.shape[0] % 16: continue
      if name == "output.weight": owner, attribute = self.transformer, "output"
      elif name.startswith("blk."):
        _, layer, attribute, _ = name.split(".")
        owner = self.transformer.blk[int(layer)]
      else: continue
      adapter = _NVLinear(getattr(owner, attribute), entry, prefill=nv_prefill, q6_aligned=nv_prepared_weights)
      setattr(owner, attribute, adapter)
      if name == "output.weight" and entry is entries["token_embd.weight"]:
        # Typed packed preparation can allocate new storage. Keep tied consumers
        # on its same lazy decoded graph so the old embedding bytes can be freed.
        self.transformer.token_embd.weight = adapter.weight
      self.nv_quant_linears += 1

  def warmup(self) -> None:
    """Prime upstream prefill/decode JIT using the same chunk size as normal generation."""
    if self.config.context_length < 2: return
    prompt = [0] * min(2, self.config.context_length - 1)
    count = min(4, self.config.context_length - len(prompt))
    try:
      # First calls execute, second calls capture, third calls verify replay.
      for _ in range(3):
        stream = self.generate(prompt, count, reuse_cache=False)
        try:
          for _ in stream: pass
        finally: stream.close()
    finally:
      self.reset()  # No dummy tokens become conversation history or reusable prefix.

  def reset(self) -> None:
    """Invalidate prefix reuse without reallocating tinygrad's cache or JIT."""
    if self._generating: raise RuntimeError("cannot reset during generation")
    self.transformer._cached_tokens = []

  def generate(self, tokens: list[int], max_new_tokens: int, temperature: float = 0.0, *,
               chunk_size: int = 32, reuse_cache: bool = True) -> Iterator[int]:
    """Bound upstream generation; callers check EOS and must close abandoned streams.

    Sampling and prefix reuse are upstream behavior. Context exhaustion bounds
    output; an already-full prompt is rejected, never silently truncated.
    """
    if self._generating: raise RuntimeError("only one generation may run at a time")
    if not isinstance(tokens, list) or not tokens: raise ValueError("prompt must contain at least one token")
    if any(type(t) is not int or not 0 <= t < self.config.vocab_size for t in tokens): raise ValueError("invalid prompt token ID")
    if type(max_new_tokens) is not int or max_new_tokens < 0: raise ValueError("max_new_tokens must be a nonnegative integer")
    if type(chunk_size) is not int or chunk_size <= 0: raise ValueError("chunk_size must be a positive integer")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
      raise ValueError("temperature must be finite and nonnegative")
    if len(tokens) >= self.config.context_length: raise ValueError("prompt exceeds available context; shorten history or increase --max-context")
    if not reuse_cache: self.reset()
    count = min(max_new_tokens, self.config.context_length - len(tokens))
    if count == 0: return
    self._generating = True
    stream = self.transformer.generate(list(tokens), chunk_size=chunk_size, temperature=float(temperature))
    try:
      for _ in range(count):
        # Do not leave a device context active while yielding control to a caller.
        with Context(DEV=self.device): token = next(stream)
        yield token
    finally:
      stream.close()
      self._generating = False
