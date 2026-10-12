
import pytest
from tinygrad import Tensor, TinyJit, UOp

from infurnace.kernels.attention import fused_chain_attention, fused_decode_attention

pytestmark = pytest.mark.nv

@pytest.mark.parametrize('count', [2, 3, 4, 8])
def test_chain_attention_exact_replay_boundaries_and_future_mask(count):
  capacity = 256
  position = UOp.variable('chain_position', 0, capacity-count)
  jit = TinyJit(lambda q, kv, pos: fused_chain_attention(q, kv, pos))
  for start in (5, 15, 16, 62, 63, 64, 120, capacity-count):
    q = Tensor.randn(1, 32, count, 128, device='NV').realize()
    kv = Tensor.randn(2, 1, 8, capacity, 128, device='NV').half().realize()
    if start+count < capacity: kv[:, :, :, start+count:].assign(float('nan')).realize()
    actual = jit(q, kv, position.bind(start)).clone().realize()
    expected = Tensor.cat(*(fused_decode_attention(q[:, :, i:i+1].contiguous(), kv, start+i+1)
                            for i in range(count)), dim=2).realize()
    assert (actual-expected).abs().max().item() == 0
