"""Packed NV projection adapter shared by the target models and the DFlash draft."""
from tinygrad import Tensor, dtypes
from tinygrad.llm.gguf import ggml_data_to_tensor
from infurnace.gguf import PackedTensor
from infurnace.kernels.quant_gemv import (quant_gemv, prepare_q2, prepare_q6, prepare_q4, dequant_prepared_q2, dequant_prepared_q4,
                                          dequant_prepared_q6, untile)
from infurnace.kernels.quant_gemm import quant_gemm, nv_prefill_supported, q2_int8

KERNEL_TYPES = (12, 13, 14, 42)  # Q4_K, Q5_K, Q6_K, Q2_0: the one-row GEMV, the tensor-core GEMM and the row gather

class _NVLinear:
  """Single-token fast path using explicit GGUF identity, never graph-size guessing.

  `rotation` (hadamard.py) is the transform a Hadamard-folded weight's input needs."""
  rotation = None

  def __init__(self, original, entry: PackedTensor, *, prefill: bool = False, q6_aligned: bool = False,
               q6_tiled: bool = True, q4_tiled: bool = True, realize: bool = True):
    self.original = original
    if prefill and entry.ggml_type not in KERNEL_TYPES: raise ValueError("NV packed prefill supports only Q4_K/Q5_K/Q6_K/Q2_0")
    self.prefill = prefill
    if prefill and not nv_prefill_supported(entry.data.device):
      raise ValueError("NV prefill requires CUDA-C FP16 tensor cores")
    self.in_features, self.out_features = entry.shape[1], entry.shape[0]
    self.ggml_type = entry.ggml_type
    dtype = dtypes.uint32 if entry.ggml_type in (12, 13) else dtypes.uint16
    # Typed preparation may allocate new storage on NV. Generic consumers and
    # tied embeddings must use this same storage.
    # Q4_K, Q5_K and Q6_K go in 16-row tiles (prepare_q4, prepare_q6_tiled) when the rows allow it; every kernel
    # reads them. q4_tiled covers Q4_K and Q5_K, which share their layout code.
    self.q4_tiled = q4_tiled and q6_aligned and entry.ggml_type in (12, 13) and self.out_features % 16 == 0
    self.q6_aligned = q6_aligned and entry.ggml_type == 14
    self.q6_tiled = self.q6_aligned and q6_tiled and self.out_features % 16 == 0
    # realize=False leaves `packed` lazy so a caller can realize many adapters in one schedule.
    self.packed = (prepare_q2(entry.data, in_features=self.in_features, realize=realize) if self.ggml_type == 42 else
                   prepare_q4(entry.data, in_features=self.in_features, realize=realize, ggml_type=entry.ggml_type) if self.q4_tiled else
                   prepare_q6(entry.data, in_features=self.in_features, tiled=self.q6_tiled, realize=realize) if self.q6_aligned else
                   entry.data.contiguous().bitcast(dtype).flatten())
    if realize: self.packed.realize()
    # Generic consumers decode fields directly from the same padded allocation.
    # Neither the original GGUF bytes nor a second dense weight copy is retained.
    decoded = (dequant_prepared_q2(self.packed, in_features=self.in_features) if self.ggml_type == 42 else
               dequant_prepared_q4(self.packed, in_features=self.in_features) if self.q4_tiled and self.ggml_type == 12 else
               ggml_data_to_tensor(untile(self.packed, in_features=self.in_features, block_bytes=176), self.in_features*self.out_features,
                                   self.ggml_type) if self.q4_tiled else
               dequant_prepared_q6(self.packed, in_features=self.in_features, tiled=self.q6_tiled) if self.q6_aligned else
               ggml_data_to_tensor(self.packed.bitcast(dtypes.uint8), self.in_features * self.out_features, self.ggml_type))
    original.weight = decoded.reshape(entry.shape).cast(dtypes.float16)
    self.weight, self.bias = original.weight, original.bias

  def decode(self, x: Tensor, q8: tuple[Tensor, Tensor, Tensor] | None = None) -> Tensor:
    """With `q8`, the input already quantized (in this weight's basis: one rotation for the projections sharing it)."""
    if q8 is None and self.rotation is not None: q8 = self.rotation.quantized(x, 32)  # rotated and quantized in one kernel
    result = quant_gemv(x, self.packed, out_features=self.out_features, in_features=self.in_features, ggml_type=self.ggml_type,
                        q6_aligned=self.q6_aligned, q6_tiled=self.q6_tiled, q4_tiled=self.q4_tiled, q8=q8)
    return result if self.bias is None else result+self.bias

  @property
  def rows_quantized(self) -> bool:
    """Whether multirow inputs are rotated and quantized in one kernel (rows_q8), Hadamard-folded Q2_0 on int8 tensor cores."""
    return self.prefill and self.rotation is not None and q2_int8(self.ggml_type)

  def rows_q8(self, x: Tensor, up: Tensor | None = None, pair: bool = False, gated=None) -> tuple[Tensor, Tensor]:
    """The int8 multirow input quant_gemm takes as `q8`: rows padded to a multiple of 16, rotated and quantized
    (with `up`, of silu(x) * up; with `pair`, x is gate then up along its rows; with `gated` (gate rows, offset,
    weight, eps), of the gated RMSNorm of x, kernels/hadamard.fwht_q8)."""
    def padded(t: Tensor) -> Tensor:
      activation = t.pad_to(t.max_shape) if not isinstance(t.numel(), int) else t
      tokens = activation.numel() // t.shape[-1]
      return activation.reshape(tokens, t.shape[-1]).pad(((0, -tokens % 16), (0, 0)))
    if gated is not None: gated = (padded(gated[0]), *gated[1:])
    return self.rotation.quantized(padded(x), 64, up=padded(up) if up is not None else None, pair=pair, gated=gated)

  def __call__(self, x: Tensor, q8: tuple[Tensor, Tensor] | None = None) -> Tensor:
    """`q8`: rows_q8(x), shared by projections of one input."""
    if isinstance(x.numel(), int) and x.numel() == self.in_features:
      return self.decode(x)
    if self.rows_quantized and q8 is None: q8 = self.rows_q8(x)
    if self.rotation is not None and q8 is None: x = self.rotation(x)
    if self.prefill:
      # Upstream prefill uses a symbolic token count. Pad to its fixed maximum
      # for capture, then slice valid rows; the kernel handles the final 16-row tile.
      symbolic = not isinstance(x.numel(), int)
      activation = x.pad_to(x.max_shape) if symbolic else x
      result = quant_gemm(activation, self.packed, out_features=self.out_features,
                          in_features=self.in_features, ggml_type=self.ggml_type, q6_aligned=self.q6_aligned, q6_tiled=self.q6_tiled,
                          q4_tiled=self.q4_tiled, q8=q8)
      if symbolic: result = result.shrink(tuple((0, size) for size in (*x.shape[:-1], self.out_features)))
      return result if self.bias is None else result + self.bias
    return self.original(x)

  @classmethod
  def merged(cls, layers: tuple['_NVLinear', ...]) -> '_NVLinear':
    """One Q2_0 linear of the rows of `layers` (projections of one input, rotation and flags): prepare_q2's 8-row tiles
    are contiguous, so their words concatenate, and the projections take one launch instead of one each. Each layer
    keeps a view of its rows of the merged words, so no weight is stored twice, and a dense decode graph only if it
    kept one (a live lazy graph is walked on every realize: tens of ms each for the model's)."""
    merged = object.__new__(cls)
    merged.__dict__.update(layers[0].__dict__)
    merged.packed = Tensor.cat(*(layer.packed for layer in layers)).contiguous().realize()
    merged.out_features = sum(layer.out_features for layer in layers)
    merged.original = merged.weight = merged.bias = None
    offset = 0
    for layer in layers:
      layer.packed = merged.packed[offset:offset + layer.packed.numel()]
      offset += layer.packed.numel()
      if layer.weight is None: continue
      decoded = dequant_prepared_q2(layer.packed, in_features=layer.in_features)
      layer.original.weight = layer.weight = decoded.reshape(layer.out_features, layer.in_features).cast(dtypes.float16)
    return merged
