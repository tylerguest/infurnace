import os
import random
import struct

import pytest
from tinygrad import Tensor, TinyJit, dtypes

from infurnace.gguf import PackedTensor
from tinygrad.llm.gguf import ggml_data_to_tensor as _dequant
from infurnace.models.qwen3 import _NVLinear, Qwen3Model
from infurnace.kernels.quant_gemv import quant_gemv, QuantGemvError

def test_submission_link_reinitializes_reused_signal_storage():
  # Eager HCQ submissions borrow ring storage that the hardware mutates. A
  # cached "already wrote this blob" marker must not skip zeroing on reuse.
  # ce9c84605 fails this invariant; upstream bc5a2d393 removes that shortcut.
  from tinygrad import UOp
  from tinygrad.device import Buffer
  from tinygrad.runtime.support.hcq2 import fold_binary
  from tinygrad.uop.ops import Ops
  storage = Buffer('CPU', 4, dtypes.uint64).ensure_allocated()
  view, initial = UOp.from_buffer(storage, 'CPU'), UOp(Ops.BINARY, arg=bytes(32))
  fold_binary(view, initial)
  for repeat in range(4):
    storage.host.view(fmt='Q')[repeat] = 999999999  # Simulated queue/timeline signal.
    fold_binary(view, initial)
    assert list(storage.host.view(fmt='Q')[:]) == [0]*4

def test_llm_imports():
  # This failed in the old wheel because tinygrad.llm.kernels was omitted.
  from tinygrad.llm.cli import SimpleTokenizer
  from tinygrad.llm.model import Transformer
  assert callable(SimpleTokenizer.from_gguf_kv)
  assert callable(Transformer.from_gguf)

@pytest.mark.parametrize("width,rows,typ", [(32, 16, 12), (256, 1, 12), (256, 16, 99)])
def test_kernel_invalid_geometry(width, rows, typ):
  with pytest.raises(QuantGemvError):
    quant_gemv(Tensor.ones(1, width, device="CPU"), Tensor.empty(1, dtype=dtypes.uint32, device="CPU"),
               out_features=rows, in_features=width, ggml_type=typ)

def test_q6_rejects_byte_storage():
  with pytest.raises(QuantGemvError, match="dtype/size"):
    quant_gemv(Tensor.ones(1, 256, device="CPU"), Tensor.empty(16*210, dtype=dtypes.uint8, device="CPU"),
               out_features=16, in_features=256, ggml_type=14)

def test_kernel_rejects_packed_size():
  with pytest.raises(QuantGemvError, match="dtype/size"):
    quant_gemv(Tensor.ones(1, 256, device="CPU"), Tensor.empty(1, dtype=dtypes.uint32, device="CPU"),
               out_features=16, in_features=256, ggml_type=12)

@pytest.mark.skipif(not os.environ.get("INFURNACE_TEST_NV"), reason="set INFURNACE_TEST_NV=1 on an NV GPU")
@pytest.mark.parametrize("typ,size", [(12, 144), (14, 210)])
@pytest.mark.parametrize("tied", [True, False])
def test_nv_adapter_preserves_embedding_ties(typ, size, tied):
  from types import SimpleNamespace
  from tinygrad import nn
  raw = Tensor(bytes(16*size), dtype=dtypes.uint8, device="NV").realize()
  output_entry = PackedTensor(raw, (16, 256), typ)
  embedding_entry = output_entry if tied else PackedTensor(
    Tensor(bytes(16*size), dtype=dtypes.uint8, device="NV").realize(), (16, 256), typ)
  embedding = SimpleNamespace(weight=embedding_entry.decode().half())
  old_embedding = embedding.weight
  output = nn.Linear(256, 16, bias=False)
  output.weight = old_embedding if tied else output_entry.decode().half()
  runner = object.__new__(Qwen3Model)
  runner.device = "NV"
  runner.transformer = SimpleNamespace(output=output, token_embd=embedding, blk=[])
  runner._enable_nv_quant({"token_embd.weight": embedding_entry, "output.weight": output_entry})
  adapter = runner.transformer.output
  if tied:
    assert embedding.weight.uop is adapter.weight.uop
    # Every resident buffer reachable through the embedding is on adapter storage.
    from tinygrad.uop.ops import Ops
    bases = {u.buffer.base for u in embedding.weight.uop.toposort() if u.op is Ops.BUFFER}
    assert bases == {adapter.packed.uop.buffer.base}
  else:
    assert embedding.weight is old_embedding
    assert embedding.weight.uop is not adapter.weight.uop

@pytest.mark.skipif(not os.environ.get("INFURNACE_TEST_NV"), reason="set INFURNACE_TEST_NV=1 on an NV GPU")
@pytest.mark.parametrize("typ,size", [(12, 144), (14, 210)])
@pytest.mark.parametrize("width", [256, 1280, 2560])
def test_nv_quant_eager_and_jit(typ, size, width):
  rng = random.Random(7)
  rows = 16
  block_count = rows * width // 256
  data = bytearray(rng.getrandbits(8) for _ in range(block_count*size))
  for b in range(block_count):
    if typ == 12: data[b*size:b*size+4] = struct.pack("<ee", 0.001, 0.0002)
    else: data[(b+1)*size-2:(b+1)*size] = struct.pack("<e", 0.001)
  # Explicit packed storage; copy offset bytes safely rather than use graph extraction.
  raw = Tensor(bytes(4)+bytes(data), dtype=dtypes.uint8, device="NV").realize()[4:]
  packed_dtype = dtypes.uint32 if typ == 12 else dtypes.uint16
  packed = raw.contiguous().bitcast(packed_dtype).clone().flatten().realize()
  weights = _dequant(Tensor(bytes(data), dtype=dtypes.uint8, device="CPU"), rows*width, typ).reshape(rows, width).tolist()
  def run(x): return quant_gemv(x, packed, out_features=rows, in_features=width, ggml_type=typ)
  jit = TinyJit(run)
  for step in range(4):
    values = [rng.uniform(-1, 1) for _ in range(width)] if step != 3 else [0.0]*width
    rounded = []
    for i in range(0, width, 32):
      group = values[i:i+32]
      scale = max(max(map(abs, group))/127, 1e-8)
      rounded.extend(round(v/scale)*scale for v in group)
    expected = [sum(x*w for x, w in zip(rounded, row)) for row in weights]
    x = Tensor([values], dtype=dtypes.float32, device="NV").realize()
    assert run(x).tolist()[0] == pytest.approx(expected, rel=3e-3, abs=2e-3)
    assert jit(x).tolist()[0] == pytest.approx(expected, rel=3e-3, abs=2e-3)

  # Exercise the adapter with independent resident loader storage (no offset).
  from tinygrad import nn
  resident = Tensor(bytes(data), dtype=dtypes.uint8, device="NV").realize()
  original = nn.Linear(width, rows, bias=False)
  original.weight = _dequant(resident, rows*width, typ).reshape(rows, width).half()
  adapter = _NVLinear(original, PackedTensor(resident, (rows, width), typ))
  assert adapter.packed.dtype == packed_dtype
  assert adapter.packed.numel()*adapter.packed.dtype.itemsize == len(data)  # No Q6 padding/repacking.
  assert adapter.packed.bitcast(dtypes.uint8).tolist() == list(data)
  assert adapter(Tensor.ones(1, width, device="NV")).tolist()[0] == pytest.approx(
    [sum(row) for row in weights], rel=3e-3, abs=2e-3)
  x = Tensor.ones(2, width, device="NV")
  assert adapter(x).tolist() == original(x).tolist()  # Multi-token prefill fallback.
  assert adapter.original.weight.uop is adapter.weight.uop
