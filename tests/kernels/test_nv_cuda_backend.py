"""infurnace's CUDA C kernels on tinygrad's CUDA backend (the CUDA driver API, also on Windows) match NV."""
import pytest
from tinygrad import Context, Tensor, dtypes
from infurnace.kernels.attention import attention
from infurnace.kernels.quant_gemm import quant_gemm
from infurnace.kernels.quant_gemv import prepare_q4, quant_gemv
from infurnace.kernels.sampling import Sampler

pytestmark = pytest.mark.nv

def q4_weights(rows, width):
  Tensor.manual_seed(0)
  raw = Tensor.randint(rows * width // 256 * 144, low=0, high=255, dtype=dtypes.uint8, device='CPU').reshape(-1, 144)
  header = Tensor([0x00, 0x20, 0x00, 0x18], dtype=dtypes.uint8, device='CPU').reshape(1, 4).expand(raw.shape[0], 4)
  return header.cat(raw[:, 4:], dim=1).flatten().realize().tolist()

def run(device, raw, x, q, kv, logits):
  with Context(DEV=device):
    packed = prepare_q4(Tensor(raw, dtype=dtypes.uint8), in_features=2560)
    xs = Tensor(x)
    one = quant_gemv(xs[0], packed, out_features=256, in_features=2560, ggml_type=12, q4_tiled=True)
    rows = quant_gemm(xs.reshape(1, 16, 2560), packed, out_features=256, in_features=2560, ggml_type=12, q4_tiled=True)
    attended = attention(Tensor(q), Tensor(kv, dtype=dtypes.float16), 300, window=128)
    sampled = Sampler(device, 5)(Tensor(logits), Tensor([0.0]))
    return [t.tolist() for t in (one, rows, attended, sampled)]

def test_kernels_match_nv():
  Tensor.manual_seed(1)
  raw, x = q4_weights(256, 2560), Tensor.randn(16, 2560, device='CPU').tolist()
  q, kv = Tensor.randn(1, 16, 16, 256, device='CPU').tolist(), (Tensor.randn(2, 1, 4, 1024, 256, device='CPU') * 0.5).tolist()
  logits = (Tensor.randn(16, 3000, device='CPU') * 2).tolist()
  try: cuda = run('CUDA', raw, x, q, kv, logits)
  except (RuntimeError, OSError) as exc: pytest.skip(f'CUDA backend unavailable: {exc}')
  # Same CUDA C kernels, same launch shapes: bit-identical results.
  assert cuda == run('NV', raw, x, q, kv, logits)
