import io, struct, pytest
import infurnace.gguf as gguf

from tinygrad import Tensor, dtypes

def test_read_unpack_uint32():
  handle = io.BytesIO(struct.pack("<I", 123))
  assert gguf._read_unpack("<I", 4, handle) == 123

def test_read_str():
  value = "hello GGUF"
  encoded = value.encode("utf-8")
  handle = io.BytesIO(struct.pack("<Q", len(encoded)) + encoded)
  assert gguf._read_str(handle) == value

def test_read_arr():
  data = b"".join([
    struct.pack("<i", 4),
    struct.pack("<Q", 3),
    struct.pack("<III", 10, 20, 30),
  ])
  assert gguf._read_arr(io.BytesIO(data)) == [10, 20, 30]

def test_read_value_uint32():
  handle = io.BytesIO(struct.pack("I", 123))
  assert gguf._read_value(handle, 4) == 123

def test_read_value_float32():
  handle = io.BytesIO(struct.pack("<f", 1.25))
  assert gguf._read_value(handle, 6) == 1.25

def test_read_value_bool():
  handle = io.BytesIO(struct.pack("<?", True))
  assert gguf._read_value(handle, 7) is True

def test_read_value_int32_array():
  data = b"".join([
    struct.pack("<i", 5),
    struct.pack("<Q", 3),
    struct.pack("<iii", 10, 20, 30),
  ])
  assert gguf._read_value(io.BytesIO(data), 9) == [10, 20, 30]

def test_read_value_string_array():
  values = ["one", "two"]
  data = b"".join([
    struct.pack("<i", 8),
    struct.pack("<Q", len(values)),
  ])
  for value in values:
    encoded = value.encode("utf-8")
    data += struct.pack("<Q", len(encoded)) + encoded
  assert gguf._read_value(io.BytesIO(data), 9) == values

def test_read_empty_array():
  data = struct.pack("<iQ", 4, 0)
  assert gguf._read_value(io.BytesIO(data), 9) == []

def test_read_header():
  data = b"".join([
    b"GGUF",
    struct.pack("<I", 3),
    struct.pack("<Q", 12),
    struct.pack("<Q", 34),
  ])
  assert gguf._read_header(io.BytesIO(data)) == (3, 12, 34)

def test_read_metadata():
  key = "general.architecture"
  value = "qwen3"
  data = b"".join([
    struct.pack("<Q", len(key)),
    key.encode("utf-8"),
    struct.pack("<i", 8),
    struct.pack("<Q", len(value)),
    value.encode("utf-8"),
  ])
  assert gguf._read_metadata(io.BytesIO(data), 1) == { key: value, }

def test_read_tensor_info():
  name = "test.weight"
  data = b"".join([
    struct.pack("Q", len(name)),
    name.encode("utf-8"),
    struct.pack("<I", 2),
    struct.pack("<QQ", 64, 128),
    struct.pack("<i", 0),
    struct.pack("<Q", 256),
  ])
  assert gguf._read_tensor_info(io.BytesIO(data)) == ("test.weight", (64, 128), 0, 256)

def test_read_multi_tensor_info():
  def encode_info(name, dims, ggml_type, offset):
    return b"".join([
      struct.pack("<Q", len(name)),
      name.encode("utf-8"),
      struct.pack("<I", len(dims)),
      struct.pack("<" + "Q" * len(dims), *dims),
      struct.pack("<i", ggml_type),
      struct.pack("<Q", offset),
    ])
  data = encode_info("a", (4,), 0, 0) + encode_info("b", (2, 3), 12, 64)
  assert gguf._read_multi_tensor_info(io.BytesIO(data), 2) == [
    ("a", (4,), 0, 0),
    ("b", (2, 3), 12, 64),
  ]

def test_data_start_uses_default_alignment():
  handle = io.BytesIO(b"\x00" * 37)
  handle.seek(37)
  assert gguf._data_start(handle, {}) == 64

def test_data_start_uses_metadata_alignment():
  handle = io.BytesIO(b"\x00" * 37)
  handle.seek(37)
  assert gguf._data_start(handle, {"general.alignment": 16}) == 48

def encode_string(value: str) -> bytes:
  encoded = value.encode("utf-8")
  return struct.pack("<Q", len(encoded)) + encoded

def test_read_gguf_structure():
  metadata = b"".join([
    encode_string("general.alignment"),
    struct.pack("<i", 4),
    struct.pack("<I", 32),
    ])

  tensor_info = b"".join([
    encode_string("test_weight"),
    struct.pack("<I", 1),
    struct.pack("<Q", 4),
    struct.pack("<i", 0),
    struct.pack("<Q", 0),
  ])

  header = b"".join([
    b"GGUF",
    struct.pack("<I", 3),
    struct.pack("<Q", 1),
    struct.pack("<Q", 1),
  ])

  data = header+metadata+tensor_info
  data += b"\x00"*((gguf._round_up(len(data), 32)) - len(data))
  data += struct.pack("<ffff", 1.0, 2.0, 3.0, 4.0)
  handle = io.BytesIO(data)
  _, tensor_count, metadata_count = gguf._read_header(handle)
  parsed_metadata = gguf._read_metadata(handle, metadata_count)
  infos = gguf._read_multi_tensor_info(handle, tensor_count)
  data_start = gguf._data_start(handle, parsed_metadata)

  assert parsed_metadata == {"general.alignment": 32}
  assert infos == [("test_weight", (4,), 0, 0)]
  assert data_start == 128

def test_tensor_nbytes_f32():
  assert gguf._tensor_nbytes(4, 0) == 16

def test_tensor_nbytes_q4_k():
  assert gguf._tensor_nbytes(256, 12) == 144
  assert gguf._tensor_nbytes(512, 12) == 288

def test_tensor_nbytes_q6_k():
  assert gguf._tensor_nbytes(256, 14) == 210

def test_tensor_nbytes_rejects_partial_quant_block():
  with pytest.raises(ValueError, match="whole number"): gguf._tensor_nbytes(255, 12)

def test_tensor_range():
  info = ("test_weight", (4,), 0, 64)
  assert gguf._tensor_range(info, data_start=128, file_size=1000) == (192, 208)

def test_tensor_range_rejects_out_of_bounds_tensor():
  info = ("test_weight", (4,), 0, 64)
  with pytest.raises(ValueError, match="extends beyond"):
    gguf._tensor_range(info, data_start=128, file_size=200)

def test_dequant_f32():
  raw = Tensor(struct.pack("<ffff", 1.0, 2.0, 3.0, 4.0), dtype=dtypes.uint8, device="CPU")
  result = gguf._dequant(raw, 4, 0)
  assert result.tolist() == [1.0, 2.0, 3.0, 4.0]

def test_tensor_from_info_f32():
  raw = Tensor(b"\x00" * 16 + struct.pack("<ffff", 1.0, 2.0, 3.0, 4.0), dtype=dtypes.uint8, device="CPU")
  info = ("test_weight", (2, 2), 0, 16)
  result = gguf._tensor_from_info(raw, info, data_start=0)
  assert result.shape == (2, 2)
  assert result.tolist() == [[1.0, 2.0], [3.0, 4.0]]

def test_tensor_from_info_f32_square():
  raw = Tensor(b"\x00"*24 + struct.pack("<ffffff", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0), dtype=dtypes.uint8, device="CPU")
  info = ("test_weight", (2, 3), 0, 24)
  result = gguf._tensor_from_info(raw, info, data_start=0)
  assert result.shape == (3, 2)
  assert result.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

def test_load_gguf_f32(tmp_path):
  metadata = b"".join([
    encode_string("general.alignment"),
    struct.pack("<i", 4),
    struct.pack("<I", 32),
  ])

  tensor_info = b"".join([
    encode_string("test_weight"),
    struct.pack("<I", 1),
    struct.pack("<Q", 4),
    struct.pack("<i", 0),
    struct.pack("<Q", 0),
  ])

  header = b"".join([
    b"GGUF",
    struct.pack("<I", 3),
    struct.pack("<Q", 1),
    struct.pack("<Q", 1),
  ])

  data = header+metadata+tensor_info
  data += b"\x00" * (gguf._round_up(len(data), 32) - len(data))
  data += struct.pack("<ffff", 1.0, 2.0, 3.0, 4.0)
  path = tmp_path / "test.gguf"
  path.write_bytes(data)
  metadata, tensors = gguf.load_gguf(path)

  assert metadata == {"general.alignment": 32}
  assert list(tensors) == ["test_weight"]
  assert tensors["test_weight"].shape == (4,)
  assert tensors["test_weight"].tolist() == [1.0, 2.0, 3.0, 4.0]

def test_q_to_uint8_q4():
  raw = Tensor([0x21], dtype=dtypes.uint8, device="CPU")
  assert gguf._q_to_uint8(raw, 4).tolist() == [1, 2]

def test_q4_values_unpack():
  raw = Tensor([0x21, 0x43], dtype=dtypes.uint8, device="CPU")
  values = gguf._q_to_uint8(raw, 4)
  assert values.tolist() == [1, 3, 2, 4]