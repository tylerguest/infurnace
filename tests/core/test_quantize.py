"""Q4_K quantization at load time: the bytes decode (tinygrad's GGUF decoder) to within one quantization step."""
from tinygrad import Tensor

from infurnace import gguf
from infurnace.quantize import Q4_K, Q4_K_BYTES, quantize_q4_k

def test_q4_k_round_trip():
  Tensor.manual_seed(3)
  weight = Tensor.randn(6, 512) * Tensor([[0.02], [0.5], [3.0], [0.0], [1.0], [0.1]])  # a zero row and assorted scales
  weight = weight.cat(Tensor.rand(2, 512) + 1, dim=0)                                    # all-positive rows (offset 0)
  packed = quantize_q4_k(weight)
  assert packed.numel() == 8 * 512 // 256 * Q4_K_BYTES
  decoded = gguf.decode(packed, 8 * 512, Q4_K).reshape(8, 512)
  blocks = weight.reshape(-1, 8, 32)
  # one step of a sub-block is its range / 15; 6-bit scales and FP16 steps add a little
  step = (blocks.max(-1).maximum(0) - blocks.min(-1).minimum(0)) / 15
  error = (decoded.reshape(-1, 8, 32) - blocks).abs().max(-1)
  assert (error <= step * 0.75 + 1e-4).all().item()
  assert decoded[3].abs().max().item() == 0
