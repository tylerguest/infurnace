import random
import struct

import pytest
from tinygrad import Tensor, TinyJit, dtypes

from infurnace.gguf import _dequant_q6_k
from infurnace.kernels.quant_gemv import prepare_q6, quant_gemv, Q6_PREPARED_BYTES
from infurnace.kernels.quant_gemm import quant_gemm

def data(rows, width):
  rng = random.Random(43)
  raw = bytearray(rng.getrandbits(8) for _ in range(rows*width//256*210))
  for offset in range(0, len(raw), 210): raw[offset+208:offset+210] = struct.pack('<e', 0.001)
  return bytes(raw)

@pytest.mark.parametrize('rows,width', [(1, 256), (1, 768), (16, 1280), (2, 2560)])
@pytest.mark.parametrize('swizzled', [False, True])
def test_q6_preparation_preserves_blocks(rows, width, swizzled):
  raw = data(rows, width)
  packed = prepare_q6(Tensor(raw, dtype=dtypes.uint8, device='CPU'), in_features=width, swizzled=swizzled)
  physical = packed.bitcast(dtypes.uint8).tolist()
  blocks = []
  for row in range(rows):
    for block in range(width//256):
      logical = []
      for word in range(54):
        if swizzled:
          tile, within = divmod(block, 4)
          live = min(4, width//256-tile*4)
          index = row*(width//256)*54+tile*216+(word//2)*live*2+within*2+word%2
        else: index = (row*(width//256)+block)*54+word
        logical.extend(physical[index*4:index*4+4])
      blocks.append(logical)
  assert packed.dtype == dtypes.uint32
  assert packed.numel()*4 == len(raw)//210*Q6_PREPARED_BYTES
  for i, block in enumerate(blocks):
    assert bytes(block[:210]) == raw[i*210:(i+1)*210]
    assert block[210:] == [0]*(Q6_PREPARED_BYTES-210)
  native = _dequant_q6_k(Tensor(raw, dtype=dtypes.uint8, device='CPU'))
  prepared = _dequant_q6_k(packed.bitcast(dtypes.uint8), block_bytes=Q6_PREPARED_BYTES,
                          swizzled_width=width//256 if swizzled else 0)
  assert (native-prepared).abs().max().item() == 0

@pytest.mark.nv
@pytest.mark.parametrize('width', [256, 1280, 2560, 9728])
@pytest.mark.parametrize('swizzled', [False, True])
@pytest.mark.parametrize('warps', [1, 8])
def test_q6_prepared_decode_and_prefill(width, swizzled, warps, monkeypatch):
  import importlib
  monkeypatch.setattr(importlib.import_module('infurnace.kernels.quant_gemv'), '_decode_warps', lambda *args: warps)
  rows = 16
  raw = Tensor(data(rows, width), dtype=dtypes.uint8, device='NV').realize()
  native = raw.bitcast(dtypes.uint16).flatten().realize()
  prepared = prepare_q6(raw, in_features=width, swizzled=swizzled)
  def run(x):
    return quant_gemv(x, prepared, out_features=rows, in_features=width, ggml_type=14,
                      q6_aligned=True, q6_swizzled=swizzled)
  jit = TinyJit(run)
  for _ in range(4):
    x = Tensor.randn(2, width, device='NV').realize()
    reference = quant_gemv(x, native, out_features=rows, in_features=width, ggml_type=14).realize()
    assert (jit(x)-reference).abs().max().item() < 1e-5
  x = Tensor.randn(3, width, device='NV').realize()
  baseline = quant_gemm(x, native, out_features=rows, in_features=width, ggml_type=14)
  actual = quant_gemm(x, prepared, out_features=rows, in_features=width, ggml_type=14,
                      q6_aligned=True, q6_swizzled=swizzled)
  # Prepared Q6 uses the paired kernel, which sums the same products in a
  # permuted k order: compare relative to the output scale, not bitwise.
  assert ((baseline-actual).abs().max()/baseline.abs().max().maximum(1)).item() < 1e-5

@pytest.mark.parametrize('override,expected', [(None, True), (True, True), (False, False)])
def test_cli_prepared_default_and_override(monkeypatch, override, expected):
  from types import SimpleNamespace
  from infurnace import loader
  seen = {}
  def load(path, **kwargs):
    seen.update(kwargs)
    return SimpleNamespace(metadata={})
  monkeypatch.setattr(loader, 'select_device', lambda device: 'NV')
  monkeypatch.setattr(loader, 'supports_nv_quant', lambda device: True)
  monkeypatch.setattr(loader.Qwen3Model, 'from_gguf', load)
  monkeypatch.setattr(loader.Tokenizer, 'from_gguf', lambda metadata: None)
  loader.load_model('unused', 64, 'NV', nv_prefill=False, nv_prepared_weights=override)
  assert seen['nv_prepared_weights'] is expected
