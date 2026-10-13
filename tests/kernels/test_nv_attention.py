import pytest
from tinygrad import Context, Tensor, TinyJit, UOp
from infurnace.kernels.attention import attention

pytestmark = pytest.mark.nv

def reference(q, kv, start, causal=True, window=0):
  """Plain softmax attention over the whole cache with each row's visible key range."""
  _, heads, rows, dim = q.shape
  kv_heads, capacity = kv.shape[2], kv.shape[3]
  keys = Tensor.arange(capacity).reshape(1, capacity)
  position = Tensor.arange(rows).reshape(rows, 1) + start
  hi = position + 1 if causal else Tensor.full((rows, 1), start + rows)
  if window and not causal: hi = hi.minimum(position + window)
  visible = keys < hi
  if window: visible = visible & (keys >= position - window + 1)
  k = kv[0, 0].float().repeat_interleave(heads // kv_heads, dim=0)
  v = kv[1, 0].float().repeat_interleave(heads // kv_heads, dim=0)
  scores = (q.float()[0] @ k.transpose(-1, -2)) / dim**0.5
  return (visible.where(scores, float('-inf')).softmax(-1) @ v).unsqueeze(0)

@pytest.mark.parametrize('heads,kv_heads,rows,dim,capacity,causal,window', [
  (16, 4, 16, 256, 1024, True, 0),      # Qwen3.5-4B target verify: 16 rows, GQA 4:1, head 256
  (16, 4, 1, 256, 1024, True, 0),       # one-row decode
  (32, 8, 16, 128, 1040, True, 2048),   # DFlash draft: sliding causal layers
  (32, 8, 5, 128, 1040, False, 0),      # DFlash draft: final full non-causal layer, partial block
  (32, 8, 3, 128, 1040, False, 300),    # non-causal sliding window
])
def test_matches_reference_across_segments(heads, kv_heads, rows, dim, capacity, causal, window):
  with Context(DEV='NV'):
    Tensor.manual_seed(3)
    kv = (Tensor.randn(2, 1, kv_heads, capacity, dim) * 0.5).half().contiguous().realize()
    q = Tensor.randn(1, heads, rows, dim).realize()
    for start in (0, 7, 250, 513, capacity - rows):   # within, across and at the end of 256-key segments
      out = attention(q, kv, start, causal=causal, window=window).realize()
      ref = reference(q, kv, start, causal, window)
      assert (out - ref).abs().max().item() < 2e-3, start

def test_one_captured_graph_serves_every_length():
  with Context(DEV='NV'):
    Tensor.manual_seed(4)
    kv = (Tensor.randn(2, 1, 4, 2048, 256) * 0.5).half().contiguous().realize()
    q = Tensor.randn(1, 16, 16, 256).realize()
    variable = UOp.variable('start_pos', 0, 2048 - 16)
    step = TinyJit(lambda q, start: attention(q, kv, start).realize())
    for start in (5, 300, 1100, 2032, 40):
      out = step(q, variable.bind(start))
      assert (out - reference(q, kv, start)).abs().max().item() < 2e-3, start
