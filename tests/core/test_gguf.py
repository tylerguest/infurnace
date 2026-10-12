import os
from pathlib import Path
import random
import struct

import pytest
from tinygrad import dtypes
from tinygrad.uop.ops import Ops

import infurnace.gguf as gguf

BLOCK_BYTES = {gguf.Q4_K: 144, gguf.Q6_K: 210}

def random_blocks(typ, count=3):
  rng = random.Random(42)
  size = BLOCK_BYTES[typ]
  data = bytearray(rng.getrandbits(8) for _ in range(size * count))
  for start in range(0, len(data), size):
    if typ == gguf.Q4_K: data[start:start+4] = struct.pack("<ee", 0.01, 0.002)
    else: data[start+208:start+210] = struct.pack("<e", 0.01)
  return bytes(data)

def scalar_decode(data, typ):
  """Independent scalar interpretation in logical weight order."""
  result = []
  size = BLOCK_BYTES[typ]
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

def test_parse_and_decode(tmp_path):
  path = tmp_path / "mixed.gguf"
  expected = build_file(path)
  metadata, entries = gguf.parse_gguf(path)
  assert metadata == {}
  assert (entries["q4"].shape, entries["q6"].shape, entries["f32"].shape) == ((2, 512), (3, 512), (2,))
  for name, entry in entries.items():
    assert entry.data.device.startswith("DISK:") and entry.data.numel() == len(expected[name])
    resident = entry.to("CPU")
    assert bytes(resident.data.tolist()) == expected[name]
    decoded = resident.decode()
    if name == "f32": assert decoded.tolist() == [1.5, -2.0]
    else:
      # Only packed storage has been realized, not the floating-point output.
      assert not any(u.op is Ops.BUFFER and u.dtype in dtypes.floats for u in decoded.uop.toposort())
      assert decoded.flatten().tolist() == pytest.approx(scalar_decode(expected[name], entry.ggml_type), rel=1e-5, abs=1e-6)

def test_metadata_is_a_fresh_dict(tmp_path):
  path = tmp_path / "mixed.gguf"
  build_file(path)
  gguf.read_gguf_metadata(path)["added"] = 1
  assert gguf.read_gguf_metadata(path) == {}

def test_truncated_file_rejected(tmp_path):
  path = tmp_path / "cut.gguf"
  build_file(path)
  path.write_bytes(path.read_bytes()[:-16])
  with pytest.raises(ValueError, match="truncated"): gguf.parse_gguf(path)

@pytest.mark.skipif(not os.environ.get("INFURNACE_TEST_MODEL"), reason="set INFURNACE_TEST_MODEL to a local GGUF path")
def test_local_model_samples_match_scalar_decode():
  _, entries = gguf.parse_gguf(Path(os.environ["INFURNACE_TEST_MODEL"]))
  sampled = set()
  for entry in entries.values():
    if entry.ggml_type not in (gguf.Q4_K, gguf.Q6_K) or entry.ggml_type in sampled: continue
    sampled.add(entry.ggml_type)
    size = BLOCK_BYTES[entry.ggml_type]
    for offset in (0, entry.data.numel()//size//2*size, entry.data.numel()-size):  # sample blocks, not whole weights
      packed = gguf.PackedTensor(entry.data[offset:offset+size], (256,), entry.ggml_type).to("CPU")
      assert packed.decode().tolist() == pytest.approx(scalar_decode(bytes(packed.data.tolist()), entry.ggml_type), rel=1e-5, abs=1e-6)
  assert sampled == {gguf.Q4_K, gguf.Q6_K}
