"""Opt-in relative performance regression, not a portable tok/s assertion.

INFURNACE_TEST_NV_PERF=1 INFURNACE_TEST_MODEL=models/Qwen3-4B-Q4_K_M.gguf
DEV=NV python -m pytest -q -s tests/test_nv_performance.py
"""
import importlib
import os
import statistics
import time

import pytest
from tinygrad import Device, Tensor, TinyJit, UOp, dtypes
from tinygrad.uop.ops import KernelInfo

from infurnace.gguf import parse_gguf

kernels = importlib.import_module("infurnace.kernels.quant_gemv")

def legacy_q6_kernel(out, raw, xq, xd, xs, out_features, in_features):
  """Frozen pre-optimization arithmetic + one-warp layout. Test oracle only.

  Both paths deliberately share CURRENT Q8 packing, isolating Q6 arithmetic/layout.
  """
  def group_dot(token, output, group):
    block, subgroup = group//8, group%8
    base = (output*in_features//256+block)*210
    xwords = kernels._load_lanes(xq[token, group, 0], 8)
    dots = [UOp.const(0, dtypes.int32)]*2
    for word_idx in range(8):
      pos = subgroup*32+word_idx*4
      within = pos%128
      low = kernels._load_lanes(raw[base+(pos//128)*64+within%64], 4) >> ((within//64)*4).cast(dtypes.uint8)
      high = kernels._load_lanes(raw[base+128+(pos//128)*32+within%32], 4) >> ((within//32)*2).cast(dtypes.uint8)
      quant = ((low & 15) | ((high & 3) << 4)).bitcast(dtypes.int8)-32
      word = sum((quant[i].cast(dtypes.uint8).cast(dtypes.uint32) << (i*8) for i in range(4)), UOp.const(0, dtypes.uint32))
      dots[word_idx//4] = kernels._dp4a(word, xwords[word_idx], dots[word_idx//4])
    scales = [raw[base+192+subgroup*2+i].cast(dtypes.uint8).bitcast(dtypes.int8).float() for i in range(2)]
    dbits = raw[base+208].cast(dtypes.uint16) | (raw[base+209].cast(dtypes.uint16) << 8)
    return (dots[0].float()*scales[0]+dots[1].float()*scales[1])*xd[token, group]*kernels._half(dbits)
  # Independent old grid, so changing production layout cannot silently change the baseline.
  from tinygrad.uop.ops import AxisType
  chunks = (in_features+1023)//1024
  block = UOp.range(out_features*chunks, 0)
  lane = UOp.range(32, -1, axis_type=AxisType.WARP)
  output, chunk = block//chunks, block%chunks
  group = lane+chunk*32
  value = (group < in_features//32).where(
    group_dot(0, output, group.minimum(in_features//32-1)), UOp.const(0, dtypes.float32))
  total = kernels.warp_reduce(value, full_wave=True)
  return out[0, output, chunk.valid(lane.eq(0))].store(total).end(block, lane).sink(
    arg=KernelInfo(name="legacy_q6_regression", opts_to_apply=()))

def legacy_q6(x, packed, rows, width):
  xq, xd, xs = kernels.q8_quantize(x, 1, width)
  out = Tensor.empty(1, rows, (width+1023)//1024, dtype=dtypes.float32, device=x.device).uop
  sources = (out, packed.uop, xq.uop, xd.uop, xs.uop)
  params = [UOp.placeholder_like(s, slot=i) for i, s in enumerate(sources)]
  call = legacy_q6_kernel(*params, out_features=rows, in_features=width).call(*sources)
  return Tensor(out.after(call)).sum(-1)

def batch_ms(jit, x, repeats=10):
  Device["NV"].synchronize()
  start = time.perf_counter_ns()
  for _ in range(repeats): jit(x)
  Device["NV"].synchronize()
  return (time.perf_counter_ns()-start)/1e6/repeats

@pytest.mark.skipif(not (os.environ.get("INFURNACE_TEST_NV_PERF") and os.environ.get("INFURNACE_TEST_MODEL")),
                    reason="set INFURNACE_TEST_NV_PERF=1 and INFURNACE_TEST_MODEL on an idle RTX 5060")
def test_q6_output_head_retains_speedup():
  if os.environ.get("VIZ", "0") != "0" or os.environ.get("PROFILE", "0") != "0":
    pytest.skip("performance test requires profiling disabled")
  _, entries = parse_gguf(os.environ["INFURNACE_TEST_MODEL"])
  entry = entries.get("output.weight", entries["token_embd.weight"])
  if entry.ggml_type != 14 or entry.shape != (151936, 2560):
    pytest.skip("performance gate is calibrated only for the tested Q6 output-head shape")
  rows, width = entry.shape
  raw = entry.to("NV").data
  packed = raw.bitcast(dtypes.uint16).flatten().realize()
  x = Tensor([[(i%127-63)/64 for i in range(width)]], device="NV", dtype=dtypes.float32).realize()
  baseline = TinyJit(lambda x: legacy_q6(x, raw, rows, width).realize())
  optimized = TinyJit(lambda x: kernels.quant_gemv(x, packed, out_features=rows, in_features=width, ggml_type=14).realize())
  for _ in range(3):
    reference, actual = baseline(x), optimized(x)
  assert actual.tolist()[0] == pytest.approx(reference.tolist()[0], rel=3e-3, abs=2e-3)
  before, after = [], []
  # Alternate order to reduce thermal/clock/order bias; loading/compilation excluded.
  for pair in range(6):
    if pair%2:
      after.append(batch_ms(optimized, x))
      before.append(batch_ms(baseline, x))
    else:
      before.append(batch_ms(baseline, x))
      after.append(batch_ms(optimized, x))
  ratio = statistics.median(after)/statistics.median(before)
  print(f"Q6 head: legacy {statistics.median(before):.3f} ms, optimized {statistics.median(after):.3f} ms, ratio {ratio:.3f}")
  # Conservative relative gate, not a fixed model tok/s number; rerun on an idle GPU on failure.
  assert ratio <= 0.95, f"Q6 speedup regressed: ratio={ratio:.3f}; before={before}, after={after}"
