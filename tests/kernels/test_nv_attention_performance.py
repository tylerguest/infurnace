"""Opt-in relative gate for the current 32Q/8KV/128 decode softmax fix.

Cache-resident isolated attention is not a DRAM bandwidth or whole-model test.
Use an idle RTX 5060; profiling must be disabled for timing.
"""
import os
import statistics
import time

import pytest
from tinygrad import Device, Tensor, TinyJit, UOp, function

from infurnace.kernels.attention import decode_attention, fused_decode_attention


@pytest.mark.skipif(not os.environ.get("INFURNACE_TEST_NV_PERF"), reason="set INFURNACE_TEST_NV_PERF=1 on an idle NV GPU")
@pytest.mark.parametrize("mode", ["parallel", "fused"])
def test_attention_relative_latency(mode):
  q = Tensor.randn(1, 32, 1, 128, device="NV").realize()
  cache = Tensor.randn(2, 1, 8, 4096, 128, device="NV").half().realize()
  length = UOp.variable("perf_attention_length", 1, 4096).bind(2048)
  @function
  def legacy(q, kv, length):
    return q.scaled_dot_product_attention(kv[0, :, :, :length, :], kv[1, :, :, :length, :], enable_gqa=True)
  @function(allow_implicit=True)
  def optimized(q, kv, length):
    return (fused_decode_attention if mode == "fused" else decode_attention)(q, kv, length)
  old, new = TinyJit(legacy), TinyJit(optimized)
  for jit in (old, new):
    for _ in range(3): jit(q, cache, length).realize()
  reference, result = old(q, cache, length).tolist(), new(q, cache, length).tolist()
  for expected, actual in zip(reference[0], result[0]):
    assert actual[0] == pytest.approx(expected[0], rel=3e-4, abs=3e-5)
  timings = {"legacy": [], "optimized": []}
  for repeat in range(6):
    order = [("legacy", old), ("optimized", new)]
    if repeat%2: order.reverse()
    for name, jit in order:
      Device["NV"].synchronize()
      start = time.perf_counter()
      for _ in range(16): jit(q, cache, length).realize()
      Device["NV"].synchronize()
      timings[name].append((time.perf_counter()-start)/16)
  baseline, current = (statistics.median(timings[name]) for name in ("legacy", "optimized"))
  ratio = current/baseline
  print(f"attention {mode}: legacy={baseline*1e3:.3f} ms, optimized={current*1e3:.3f} ms, ratio={ratio:.3f}")
  assert ratio <= 0.95, timings
