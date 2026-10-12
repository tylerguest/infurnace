import random
import struct

import pytest
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.uop.ops import Ops

from infurnace.kernels.quant_gemv import QuantGemvError, q8_quantize, quant_gemv

def packed(typ, rows, width, device):
  size = 144 if typ == 12 else 210
  rng = random.Random(53)
  raw = bytearray(rng.getrandbits(8) for _ in range(rows*width//256*size))
  for i in range(0, len(raw), size):
    if typ == 12: raw[i:i+4] = struct.pack('<ee', 0.001, 0.0002)
    else: raw[i+208:i+210] = struct.pack('<e', 0.001)
  return Tensor(bytes(raw), dtype=dtypes.uint8, device=device).bitcast(dtypes.uint32 if typ == 12 else dtypes.uint16).realize()

def test_invalid_shared_q8_contract():
  x = Tensor.zeros(1, 256, device='CPU')
  w = Tensor.zeros(16*36, dtype=dtypes.uint32, device='CPU')
  wrong = (Tensor.zeros(1, 8, 8, dtype=dtypes.int32, device='CPU'),
           Tensor.zeros(1, 8, device='CPU'), Tensor.zeros(1, 8, 2, dtype=dtypes.int32, device='CPU'))
  with pytest.raises(QuantGemvError, match='shared Q8'):
    quant_gemv(x, w, out_features=16, in_features=256, ggml_type=12, q8=wrong)

@pytest.mark.nv
@pytest.mark.parametrize('typ', [12, 14])
def test_shared_q8_projection_parity_and_graph(typ):
  rows, width = 16, 1280
  w = packed(typ, rows, width, 'NV')
  def run(x):
    shared = q8_quantize(x, 1, width)
    values = [quant_gemv(x, w, out_features=rows, in_features=width, ggml_type=typ, q8=shared) for _ in range(3)]
    return tuple(values)
  jit = TinyJit(run)
  for _ in range(4):
    x = Tensor.randn(1, 1, width, device='NV').realize()
    graph = run(x)
    calls = [u for u in Tensor.stack(*graph).uop.toposort() if u.op is Ops.CALL and u.body.op is Ops.SINK
             and u.body.arg.name == 'q8_quantize']
    assert len(calls) == 1
    reference = quant_gemv(x, w, out_features=rows, in_features=width, ggml_type=typ)
    for actual in jit(x):
      assert (actual-reference).abs().max().item() == 0

@pytest.mark.nv
def test_block_shared_packing_prefill_jit_and_reset():
  import copy
  from tinygrad import Context, UOp
  from tinygrad.llm.model import TransformerBlock, TransformerConfig
  from infurnace.gguf import PackedTensor
  from infurnace.models.qwen3 import _NVLinear, enable_decode_attention
  config = TransformerConfig(num_blocks=1, dim=256, hidden_dim=512, n_heads=8, n_kv_heads=2,
    norm_eps=1e-6, vocab_size=256, head_dim=128, v_head_dim=128, rope_theta=10000,
    rope_dim=128, qk_norm=128, max_context=32)
  with Context(DEV='NV'):
    baseline = TransformerBlock(config)
    for name in ('attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate', 'ffn_up', 'ffn_down'):
      original = getattr(baseline, name)
      rows, width = original.weight.shape
      raw = packed(12, rows, width, 'NV').bitcast(dtypes.uint8)
      setattr(baseline, name, _NVLinear(original, PackedTensor(raw, (rows, width), 12), prefill=True, q6_aligned=True))
    candidate = copy.copy(baseline)
    assert enable_decode_attention([baseline, candidate], fused=True) == 2
    baseline.share_q8, candidate.share_q8 = False, True
    x = Tensor.randn(1, 2, config.dim, device='NV').realize()
    assert (baseline(x, 0)-candidate(x, 0)).abs().max().item() == 0
    jit_baseline, jit_candidate = TinyJit(baseline), TinyJit(candidate)
    position = UOp.variable('packing_position', 0, 31)
    for pos in (2, 3, 4, 5, 0, 1):
      x = Tensor.randn(1, 1, config.dim, device='NV').realize()
      assert (jit_baseline(x, position.bind(pos))-jit_candidate(x, position.bind(pos))).abs().max().item() == 0
    assert baseline.cache_kv.uop.buffer is not candidate.cache_kv.uop.buffer
