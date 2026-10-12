"""Packed NV projection adapter shared by the Qwen3.5 target and DFlash draft."""
from tinygrad import Tensor, dtypes
from tinygrad.llm.gguf import ggml_data_to_tensor
from infurnace.gguf import PackedTensor
from infurnace.kernels.quant_gemv import (quant_gemv, prepare_q6, prepare_q4, dequant_prepared_q4, dequant_prepared_q6)
from infurnace.kernels.quant_gemm import quant_gemm, nv_prefill_supported

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
    # Typed preparation may allocate new storage on NV. Generic consumers and
    # tied embeddings must use this same storage.
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
