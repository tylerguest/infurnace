"""Weights to Q4_K at load time (on the device, in tinygrad ops), for GGUFs shipped in BF16/F16 that infurnace's
packed kernels should run: today the DFlash2 draft, 1.56 GB in BF16 against 0.44 GB in Q4_K.

Round to nearest, per GGUF's block_q4_K: each 256-value block has 8 sub-blocks of 32, value = d*sc*q - dmin*m with
q in 0..15 and 6-bit sc and m. A sub-block's range [min(x, 0), max(x)] sets its scale and offset; d and dmin set
the 6-bit steps from the block's largest. No search (llama.cpp's make_qkx2_quants refines both): a draft's errors
only cost acceptance, never correctness, since the target checks every token.
"""
from tinygrad import Tensor, dtypes

Q4_K, Q4_K_BYTES = 12, 144

def _u8(x: Tensor) -> Tensor:
  return x.cast(dtypes.int32).cast(dtypes.uint8)

def quantize_q4_k(weight: Tensor) -> Tensor:
  """(rows, width) float weights, width a multiple of 256, to GGUF Q4_K blocks: rows * width / 256 * 144 bytes."""
  rows, width = weight.shape
  if width % 256: raise ValueError("Q4_K needs a width that is a multiple of 256")
  x = weight.float().reshape(-1, 8, 32)                                                      # (blocks, sub-blocks, values)
  low, high = x.min(-1).minimum(0), x.max(-1).maximum(0)
  scale, offset = (high - low) / 15, -low                                                    # value = scale*q - offset
  d, dmin = scale.max(-1, keepdim=True) / 63, offset.max(-1, keepdim=True) / 63
  sc = (d > 0).where(scale / d.maximum(1e-30), 0).round().clip(0, 63)
  m = (dmin > 0).where(offset / dmin.maximum(1e-30), 0).round().clip(0, 63)
  # The stored half-precision steps are what decoding uses: quantize against them
  d16, dmin16 = d.cast(dtypes.float16), dmin.cast(dtypes.float16)
  step, base = d16.float() * sc, dmin16.float() * m
  q = (step > 0).unsqueeze(-1).where((x + base.unsqueeze(-1)) / step.maximum(1e-30).unsqueeze(-1), 0).round().clip(0, 15)
  # 12 scale bytes: sub-blocks 0-3 keep 6 bits in bytes 0-7 (scale, then offset), with the high 2 bits of sub-blocks
  # 4-7 on top; bytes 8-11 hold sub-blocks 4-7's low 4 bits (scale | offset << 4). See _q4_scales.
  sc, m = sc.cast(dtypes.int32), m.cast(dtypes.int32)
  scales = Tensor.cat(sc[:, :4] + (sc[:, 4:] // 16) * 64, m[:, :4] + (m[:, 4:] // 16) * 64, (sc[:, 4:] % 16) + (m[:, 4:] % 16) * 16, dim=1)
  # 128 code bytes: each 32-byte run holds sub-block 2i in its low nibbles and 2i+1 in its high nibbles
  q = q.cast(dtypes.int32).reshape(-1, 4, 2, 32)
  codes = (q[:, :, 0] + q[:, :, 1] * 16).reshape(-1, 128)
  header = d16.bitcast(dtypes.uint16).cat(dmin16.bitcast(dtypes.uint16), dim=1).bitcast(dtypes.uint8).reshape(-1, 4)
  return header.cat(_u8(scales), _u8(codes), dim=1).flatten().contiguous()
