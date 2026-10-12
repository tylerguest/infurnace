import math
import os
from pathlib import Path
import random
import struct

import pytest
from tinygrad import Tensor, dtypes
from tinygrad.llm.gguf import ggml_data_to_tensor, gguf_load
from tinygrad.uop.ops import Ops

import infurnace.gguf as gguf

def random_blocks(typ, count=3):
  rng = random.Random(42)
  size = gguf.QUANT_BLOCK_BYTES[typ]
  data = bytearray(rng.getrandbits(8) for _ in range(size * count))
  for start in range(0, len(data), size):
    if typ == gguf.Q4_K: data[start:start+4] = struct.pack("<ee", 0.01, 0.002)
    else: data[start+208:start+210] = struct.pack("<e", 0.01)
  return bytes(data)

def scalar_decode(data, typ):
  """Independent scalar interpretation in logical weight order."""
  result = []
  size = gguf.QUANT_BLOCK_BYTES[typ]
  for start in range(0, len(data), size):
    block = data[start:start+size]
    if typ == gguf.Q4_K:
      d, dmin = struct.unpack("<ee", block[:4])
      s = block[4:16]
      for group in range(8):
        scale = s[group] & 63 if group < 4 else (s[group+4] & 15) | ((s[group-4] >> 6) << 4)
        minimum = s[group+4] & 63 if group < 4 else (s[group+4] >> 4) | ((s[group] >> 6) << 4)
        for lane in range(32):
          q = (block[16 + (group//2)*32 + lane] >> (4*(group % 2))) & 15
          result.append(d * scale * q - dmin * minimum)
    else:
      d = struct.unpack("<e", block[208:210])[0]
      scales = struct.unpack("<16b", block[192:208])
      for index in range(256):
        half, within = divmod(index, 128)
        low = (block[half*64 + within % 64] >> (4*(within//64))) & 15
        high = (block[128 + half*32 + within % 32] >> (2*(within//32))) & 3
        result.append(d * scales[index//16] * ((low | (high << 4)) - 32))
  return result

@pytest.mark.parametrize("typ", [gguf.Q4_K, gguf.Q6_K])
@pytest.mark.parametrize("offset", [0, 4, 32])
def test_quant_scalar_and_tinygrad_parity(typ, offset):
  data = random_blocks(typ)
  raw = Tensor(b"\x00" * offset + data, dtype=dtypes.uint8, device="CPU").realize()[offset:]
  actual = gguf._dequant(raw, 3*256, typ)
  assert actual.shape == (768,)
  assert actual.dtype == dtypes.float32
  assert actual.tolist() == pytest.approx(scalar_decode(data, typ), rel=1e-5, abs=1e-6)
  reference = ggml_data_to_tensor(raw, 768, typ).flatten().tolist()
  assert actual.tolist() == pytest.approx(reference, rel=1e-5, abs=1e-6)

def test_q4_known_groups():
  scales = bytes([1, 2, 3, 4] + [1]*4 + [0x15, 0x16, 0x17, 0x18])
  qs = b"".join(bytes([low | high << 4])*32 for low, high in [(1, 2), (3, 4), (5, 6), (7, 8)])
  raw = Tensor(struct.pack("<ee", 1, 0.5) + scales + qs, dtype=dtypes.uint8, device="CPU")
  assert gguf._dequant(raw, 256, gguf.Q4_K).tolist() == [float(g*g)-0.5 for g in range(1, 9) for _ in range(32)]

def test_q6_signed_scales():
  scales = [1, -1]*8
  data = bytes(192) + struct.pack("<16b", *scales) + struct.pack("<e", 0.5)
  raw = Tensor(data, dtype=dtypes.uint8, device="CPU")
  assert gguf._dequant(raw, 256, gguf.Q6_K).tolist() == [-16.0*s for s in scales for _ in range(16)]

@pytest.mark.parametrize("typ", [gguf.Q4_K, gguf.Q6_K])
def test_zero_blocks(typ):
  raw = Tensor(bytes(gguf.QUANT_BLOCK_BYTES[typ]), dtype=dtypes.uint8, device="CPU")
  assert gguf._dequant(raw, 256, typ).tolist() == [0.0]*256

@pytest.mark.parametrize("typ", [0, gguf.Q4_K, gguf.Q6_K])
def test_reject_wrong_packed_size(typ):
  raw = Tensor(bytes(3), dtype=dtypes.uint8, device="CPU")
  with pytest.raises(ValueError, match="packed bytes"): gguf._dequant(raw, 256, typ)

def test_reject_wrong_dtype():
  with pytest.raises(ValueError, match="uint8"):
    gguf._dequant(Tensor([1.0], device="CPU"), 1, 0)

def build_file(path):
  def string(s): return struct.pack("<Q", len(s)) + s.encode()
  data4, data6 = random_blocks(gguf.Q4_K, 4), random_blocks(gguf.Q6_K, 6)
  offset6 = (len(data4) + 31)//32*32
  offset32 = (offset6 + len(data6) + 31)//32*32
  specs = [("q4", (512, 2), gguf.Q4_K, 0), ("q6", (512, 3), gguf.Q6_K, offset6), ("f32", (2,), 0, offset32)]
  header = b"GGUF" + struct.pack("<IQQ", 3, len(specs), 0)
  for name, dims, typ, offset in specs:
    header += string(name) + struct.pack("<I", len(dims)) + struct.pack("<" + "Q"*len(dims), *dims) + struct.pack("<iQ", typ, offset)
  payload = data4 + bytes(offset6-len(data4)) + data6
  payload += bytes(offset32-len(payload)) + struct.pack("<ff", 1.5, -2.0)
  path.write_bytes(header + bytes(-len(header) % 32) + payload)
  return {"q4": data4, "q6": data6, "f32": struct.pack("<ff", 1.5, -2.0)}

def test_parse_stage_and_decode(tmp_path):
  path = tmp_path / "mixed.gguf"
  expected = build_file(path)
  metadata, entries = gguf.parse_gguf(path)
  assert metadata == {}
  assert entries["q4"].shape == (2, 512)
  assert entries["q6"].shape == (3, 512)
  for name, entry in entries.items():
    assert entry.data.device.startswith("DISK:")
    assert entry.data.numel() == len(expected[name])
    with pytest.raises(ValueError, match="compute device"): entry.decode()
    resident = entry.to("CPU")
    assert resident.data.device == "CPU"
    assert bytes(resident.data.tolist()) == expected[name]
    decoded = resident.decode()
    if name != "f32":
      # Only packed storage has been realized, not the floating-point output.
      assert decoded.uop.op is not Ops.BUFFER
      assert not any(u.op is Ops.BUFFER and u.dtype in dtypes.floats for u in decoded.uop.toposort())
      assert decoded.flatten().tolist() == pytest.approx(scalar_decode(expected[name], entry.ggml_type), rel=1e-5, abs=1e-6)
  _, tensors = gguf.load_gguf(path, device="CPU")
  assert tensors["q4"].shape == (2, 512)
  assert tensors["q6"].shape == (3, 512)
  assert tensors["f32"].tolist() == [1.5, -2.0]
  reference_metadata, reference_tensors = gguf_load(Tensor(path).to("CPU"))
  assert reference_metadata == metadata
  assert reference_tensors.keys() == tensors.keys()
  for name, tensor in tensors.items():
    assert tensor.shape == reference_tensors[name].shape
    assert tensor.flatten().tolist() == pytest.approx(reference_tensors[name].flatten().tolist(), rel=1e-5, abs=1e-6)
  for name in ("q4", "q6"):
    assert tensors[name].device == "CPU"
    assert tensors[name].flatten().tolist() == pytest.approx(scalar_decode(expected[name], entries[name].ggml_type), rel=1e-5, abs=1e-6)

def test_decode_consumer(tmp_path):
  path = tmp_path / "mixed.gguf"
  expected = build_file(path)
  _, tensors = gguf.load_gguf(path, device="CPU")
  x = Tensor.ones(512, device="CPU")
  rows = scalar_decode(expected["q4"], gguf.Q4_K)
  assert (tensors["q4"] @ x).tolist() == pytest.approx([sum(rows[:512]), sum(rows[512:])], rel=1e-5, abs=1e-5)
  assert math.isfinite(tensors["q6"][1, :16].sum().item())

@pytest.mark.skipif(not os.environ.get("INFURNACE_TEST_MODEL"), reason="set INFURNACE_TEST_MODEL to a local GGUF path")
def test_local_model():
  path = Path(os.environ["INFURNACE_TEST_MODEL"])
  metadata, entries = gguf.parse_gguf(path)
  loaded_metadata, tensors = gguf.load_gguf(path, device="CPU")
  assert loaded_metadata == metadata
  assert tensors.keys() == entries.keys()
  sampled = set()
  for name, entry in entries.items():
    decoded = tensors[name]
    assert decoded.shape == entry.shape
    assert decoded.dtype == dtypes.float32
    assert decoded.device == "CPU"
    assert entry.data.numel() == gguf._tensor_nbytes(math.prod(entry.shape), entry.ggml_type)
    if entry.ggml_type not in gguf.QUANT_BLOCK_BYTES: continue
    assert not any(u.op is Ops.BUFFER and u.dtype in dtypes.floats for u in decoded.uop.toposort())
    if entry.ggml_type in sampled: continue
    sampled.add(entry.ggml_type)
    size = gguf.QUANT_BLOCK_BYTES[entry.ggml_type]
    # Sample packed blocks directly, avoiding a full large-weight reference realization.
    for offset in (0, entry.data.numel()//size//2*size, entry.data.numel()-size):
      packed = entry.data[offset:offset+size].to("CPU").realize()
      actual = gguf._dequant(packed, 256, entry.ggml_type).tolist()
      reference = ggml_data_to_tensor(packed, 256, entry.ggml_type).flatten().tolist()
      assert actual == pytest.approx(reference, rel=1e-5, abs=1e-6)
      assert actual == pytest.approx(scalar_decode(bytes(packed.tolist()), entry.ggml_type), rel=1e-5, abs=1e-6)
  assert sampled == {gguf.Q4_K, gguf.Q6_K}
