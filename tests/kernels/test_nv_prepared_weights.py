import random
import struct

import pytest
from tinygrad import Tensor, TinyJit, dtypes

from tinygrad.llm.gguf import ggml_data_to_tensor
from infurnace.kernels.quant_gemv import dequant_prepared_q6, prepare_q6, quant_gemv, Q6_PREPARED_BYTES
from infurnace.kernels.quant_gemm import quant_gemm

def data(rows, width):
  rng = random.Random(43)
  raw = bytearray(rng.getrandbits(8) for _ in range(rows*width//256*210))
  for offset in range(0, len(raw), 210): raw[offset+208:offset+210] = struct.pack('<e', 0.001)
  return bytes(raw)

@pytest.mark.parametrize('rows,width,tiled', [(1, 256, False), (1, 768, False), (16, 1280, False), (2, 2560, False),
                                              (16, 256, True), (32, 1280, True), (16, 2560, True)])
def test_q6_preparation_preserves_blocks(rows, width, tiled):
  raw = data(rows, width)
  packed = prepare_q6(Tensor(raw, dtype=dtypes.uint8, device='CPU'), in_features=width, tiled=tiled)
  physical = packed.bitcast(dtypes.uint8).tolist()
  blocks = []
  for row in range(rows):
    for block in range(width//256):
      logical = []
      for word in range(54):
        if tiled: index = ((row//16*(width//256)+block)*16+row%16)*54+word  # 16-row tiles per block contiguous
        else: index = (row*(width//256)+block)*54+word
        logical.extend(physical[index*4:index*4+4])
      blocks.append(logical)
  assert packed.dtype == dtypes.uint32
  assert packed.numel()*4 == len(raw)//210*Q6_PREPARED_BYTES
  for i, block in enumerate(blocks):
    assert bytes(block[:210]) == raw[i*210:(i+1)*210]
    assert block[210:] == [0]*(Q6_PREPARED_BYTES-210)
  native = ggml_data_to_tensor(Tensor(raw, dtype=dtypes.uint8, device='CPU'), len(raw)//210*256, 14).flatten()
  prepared = dequant_prepared_q6(packed, in_features=width, tiled=tiled)
  assert (native-prepared).abs().max().item() == 0

@pytest.mark.nv
@pytest.mark.parametrize('width', [256, 1280, 2560, 9728])
@pytest.mark.parametrize('tiled', [False, True])
@pytest.mark.parametrize('warps', [1, 8])
def test_q6_prepared_decode_and_prefill(width, tiled, warps, monkeypatch):
  import importlib
  monkeypatch.setattr(importlib.import_module('infurnace.kernels.quant_gemv'), '_decode_warps', lambda *args: warps)
  rows = 16
  raw = Tensor(data(rows, width), dtype=dtypes.uint8, device='NV').realize()
  native = raw.bitcast(dtypes.uint16).flatten().realize()
  prepared = prepare_q6(raw, in_features=width, tiled=tiled)
  def run(x):
    return quant_gemv(x, prepared, out_features=rows, in_features=width, ggml_type=14,
                      q6_aligned=True, q6_tiled=tiled)
  jit = TinyJit(run)
  for _ in range(4):
    x = Tensor.randn(2, width, device='NV').realize()
    reference = quant_gemv(x, native, out_features=rows, in_features=width, ggml_type=14).realize()
    assert (jit(x)-reference).abs().max().item() < 1e-5
  x = Tensor.randn(3, width, device='NV').realize()
  baseline = quant_gemm(x, native, out_features=rows, in_features=width, ggml_type=14)
  actual = quant_gemm(x, prepared, out_features=rows, in_features=width, ggml_type=14,
                      q6_aligned=True, q6_tiled=tiled)
  # Prepared Q6 sums the same products in a permuted k order; the tiled kernel also rounds d*scale to FP16
  # (as Marlin does): compare relative to the output scale, not bitwise.
  assert ((baseline-actual).abs().max()/baseline.abs().max().maximum(1)).item() < (1e-3 if tiled else 1e-5)
