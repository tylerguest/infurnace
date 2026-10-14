import pytest
from tinygrad import Context, Tensor, TinyJit, UOp, dtypes
from infurnace.kernels.attention import attention, store_rows

pytestmark = pytest.mark.nv

def reference(q, kv, start, causal=True, window=0, floor=0):
  """Plain softmax attention over the whole cache with each row's visible key range."""
  _, heads, rows, dim = q.shape
  kv_heads, capacity = kv.shape[2], kv.shape[3]
  keys = Tensor.arange(capacity).reshape(1, capacity)
  position = Tensor.arange(rows).reshape(rows, 1) + start
  hi = position + 1 if causal else Tensor.full((rows, 1), start + rows)
  if window and not causal: hi = hi.minimum(position + window)
  visible = keys < hi
  if window: visible = visible & (keys >= position - window + 1)
  visible = visible & (keys >= floor)
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

def ring_of(kv, top, length):
  """The ring cache of `length` slots holding positions top-length+1..top of a full cache."""
  slots = Tensor.arange(length)
  positions = top - (top - slots + length) % length
  return kv[:, :, :, positions.clip(0, kv.shape[3] - 1)].contiguous().realize()

@pytest.mark.parametrize('heads,kv_heads,rows,dim,causal,window', [
  (32, 8, 16, 128, True, 300),    # DFlash draft sliding layers, at a toy window
  (32, 8, 5, 128, True, 300),
  (32, 8, 3, 128, False, 300),
])
def test_ring_matches_the_full_cache(heads, kv_heads, rows, dim, causal, window):
  length, capacity = window + 2 * 16, 2048
  with Context(DEV='NV'):
    Tensor.manual_seed(5)
    kv = (Tensor.randn(2, 1, kv_heads, capacity, dim) * 0.5).half().contiguous().realize()
    q = Tensor.randn(1, heads, rows, dim).realize()
    for start in (0, 40, 333, 700, 1500, capacity - rows):  # before, at and after the first wrap
      ring = ring_of(kv, start + rows - 1, length)
      out = attention(q, ring, start, causal=causal, window=window, ring=True).realize()
      assert (out - reference(q, kv, start, causal, window)).abs().max().item() < 2e-3, start

def test_floor_hides_older_keys_inside_one_graph():
  with Context(DEV='NV'):
    Tensor.manual_seed(6)
    kv = (Tensor.randn(2, 1, 8, 1024, 128) * 0.5).half().contiguous().realize()
    q = Tensor.randn(1, 32, 16, 128).realize()
    floor = Tensor([0], dtype=dtypes.int32).contiguous().realize()
    variable = UOp.variable('start_pos', 0, 1024 - 16)
    step = TinyJit(lambda q, start: attention(q, kv, start, window=300, floor=floor).realize())
    for start, lowest in ((500, 0), (500, 400), (600, 300), (700, 650), (500, 0)):
      Tensor(floor.uop.after(floor.uop.store(Tensor([lowest], dtype=dtypes.int32).uop))).realize()
      out = step(q, variable.bind(start))
      assert (out - reference(q, kv, start, window=300, floor=lowest)).abs().max().item() < 2e-3, (start, lowest)

def test_ring_store_wraps():
  length, rows = 40, 16
  with Context(DEV='NV'):
    Tensor.manual_seed(7)
    for start in (0, 30, 75):                  # inside, across the end, past one lap
      cache = Tensor.zeros(2, 1, 2, length, 32, dtype=dtypes.float16).contiguous().realize()
      values = Tensor.randn(2, 1, 2, rows, 32).half().realize()
      out = store_rows(cache, values, start, ring=True).realize().tolist()
      expected = values.tolist()
      for r in range(rows):
        assert out[1][0][1][(start + r) % length] == expected[1][0][1][r]
