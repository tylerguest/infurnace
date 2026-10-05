import io
import struct

import pytest

import infurnace.gguf as gguf


def string(value):
  data = value.encode()
  return struct.pack("<Q", len(data)) + data


def fixture(infos, metadata=(), version=3, payload=b"\x00" * 128):
  data = b"GGUF" + struct.pack("<IQQ", version, len(infos), len(metadata))
  for key, typ, value in metadata:
    data += string(key) + struct.pack("<i", typ) + value
  for name, dims, typ, offset in infos:
    data += string(name) + struct.pack("<I", len(dims))
    data += struct.pack("<" + "Q" * len(dims), *dims) + struct.pack("<iQ", typ, offset)
  return data + b"\x00" * (-len(data) % 32) + payload


@pytest.mark.parametrize("size", range(24))
def test_truncated_header(size):
  header = b"GGUF" + struct.pack("<IQQ", 3, 0, 0)
  with pytest.raises(ValueError, match="truncated"):
    gguf._read_header(io.BytesIO(header[:size]))


def test_truncated_string():
  with pytest.raises(ValueError, match="truncated"):
    gguf._read_str(io.BytesIO(struct.pack("<Q", 5) + b"abc"))


def test_invalid_utf8():
  with pytest.raises(ValueError, match="UTF-8"):
    gguf._read_str(io.BytesIO(struct.pack("<Q", 1) + b"\xff"))


@pytest.mark.parametrize("array", [False, True])
def test_unknown_metadata_type(array):
  with pytest.raises(ValueError, match="metadata type"):
    if array: gguf._read_arr(io.BytesIO(struct.pack("<iQ", 99, 0)))
    else: gguf._read_value(io.BytesIO(), 99)


def test_nested_array():
  data = struct.pack("<iQ", 9, 2) + struct.pack("<iQii", 5, 2, -1, 2) + struct.pack("<iQ", 8, 1) + string("hi")
  assert gguf._read_arr(io.BytesIO(data)) == [[-1, 2], ["hi"]]


@pytest.mark.parametrize("alignment", [0, -1, 3, 12, True, 32.0, "32"])
def test_invalid_alignment(alignment):
  with pytest.raises(ValueError, match="alignment"):
    gguf._data_start(io.BytesIO(), {"general.alignment": alignment})


@pytest.mark.parametrize("dims", [(), (0,), (1, 0), (1, 1, 1, 1, 1)])
def test_invalid_dimensions(dims):
  with pytest.raises(ValueError, match="dimensions"):
    gguf._tensor_range(("a", dims, 0, 0), 0, 128)


def test_quantized_row_width():
  # Total count is one complete block, but neither row is valid.
  with pytest.raises(ValueError, match="row width"):
    gguf._tensor_range(("a", (128, 2), gguf.Q4_K, 0), 0, 144)


@pytest.mark.parametrize("infos,metadata,error", [
  ([("a", (4,), 0, 0), ("a", (4,), 0, 32)], (), "duplicate tensor"),
  ([("a", (16,), 0, 0), ("b", (4,), 0, 32)], (), "overlap"),
  ([("a", (4,), 0, 4)], (), "aligned"),
  ([("a", (256,), 99, 0)], (), "unsupported GGML"),
  ([("a", (64,), 0, 0)], (), "beyond"),
  ([], [("split.count", 4, struct.pack("<I", 2))], "split"),
  ([], [("general.alignment", 4, struct.pack("<I", 0))], "alignment"),
  ([], [("a", 4, struct.pack("<I", 1)), ("a", 4, struct.pack("<I", 2))], "duplicate GGUF metadata"),
])
def test_reject_before_tensor_allocation(tmp_path, monkeypatch, infos, metadata, error):
  path = tmp_path / "invalid.gguf"
  path.write_bytes(fixture(infos, metadata))
  def unexpected(*args, **kwargs):
    pytest.fail("allocated tensors before validating the file")
  monkeypatch.setattr(gguf.Tensor, "empty", unexpected)
  with pytest.raises(ValueError, match=error): gguf.load_gguf(path)


@pytest.mark.parametrize("version", [2, 3])
def test_multiple_tensors_nonzero_offset(tmp_path, version):
  path = tmp_path / "valid.gguf"
  payload = struct.pack("<ff", 1, 2) + b"\x00" * 24 + struct.pack("<ffffff", 3, 4, 5, 6, 7, 8)
  path.write_bytes(fixture([("a", (2,), 0, 0), ("b", (2, 3), 0, 32)], version=version, payload=payload))
  _, tensors = gguf.load_gguf(path)
  assert tensors["a"].tolist() == [1, 2]
  assert tensors["b"].shape == (3, 2)
  assert tensors["b"].tolist() == [[3, 4], [5, 6], [7, 8]]
