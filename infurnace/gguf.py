import struct, functools, math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from tinygrad import Tensor, dtypes

Q8_0, Q4_K, Q5_K, Q6_K = 8, 12, 13, 14
GGML_BLOCK_SIZE = 256
QUANT_BLOCK_BYTES = {Q8_0: 34, Q4_K: 144, Q5_K: 176, Q6_K: 210}
QUANT_BLOCK_VALUES = {Q8_0: 32, Q4_K: 256, Q5_K: 256, Q6_K: 256}
Q4_K_BLOCK_VALUES = 256
Q4_K_BLOCK_BYTES = 144
_NATIVE_DTYPES = {0: dtypes.float32, 1: dtypes.float16}

def _read_exact(handle, n: int) -> bytes:
  offset = handle.tell()
  data = handle.read(n)
  if len(data) != n: raise ValueError(f"truncated GGUF at byte {offset}: expected {n} bytes, got {len(data)}")
  return data

def _read_unpack(fmt: str, n: int, handle):
  return struct.unpack(fmt, _read_exact(handle, n))[0]

def _read_str(handle) -> str:
  length = _read_unpack("<Q", 8, handle)
  try: return _read_exact(handle, length).decode("utf-8")
  except UnicodeDecodeError as exc: raise ValueError("invalid GGUF UTF-8 string") from exc

def _read_arr(handle) -> list[Any]:
  value_type = _read_unpack("<i", 4, handle)
  reader = _value_reader(value_type)
  return [reader(handle) for _ in range(_read_unpack("<Q", 8, handle))]

_READERS = {
  0: functools.partial(_read_unpack, "<B", 1),  # uint8
  1: functools.partial(_read_unpack, "<b", 1),  # int8
  2: functools.partial(_read_unpack, "<H", 2),  # uint16
  3: functools.partial(_read_unpack, "<h", 2),  # int16
  4: functools.partial(_read_unpack, "<I", 4),  # uint32
  5: functools.partial(_read_unpack, "<i", 4),  # int32
  6: functools.partial(_read_unpack, "<f", 4),  # float32
  7: functools.partial(_read_unpack, "<?", 1),  # bool
  8: _read_str,                                 # string
  9: _read_arr,                                 # array
  10: functools.partial(_read_unpack, "<Q", 8), # uint64
  11: functools.partial(_read_unpack, "<q", 8), # int64
  12: functools.partial(_read_unpack, "<d", 8), # float64
}

def _round_up(value: int, alignment: int) -> int:
  return (value + alignment - 1) // alignment * alignment

def _value_reader(value_type: int):
  if value_type not in _READERS: raise ValueError(f"unsupported GGUF metadata type: {value_type}")
  return _READERS[value_type]

def _read_value(handle, value_type: int) -> Any:
  return _value_reader(value_type)(handle)

def _read_header(handle) -> tuple[int, int, int]:
  magic = _read_exact(handle, 4)
  if magic != b"GGUF": raise ValueError(f"invalid GGUF magic: {magic!r}")
  version = _read_unpack("<I", 4, handle)
  if version not in (2, 3): raise ValueError(f"unsupported GGUF version: {version}")
  tensor_count = _read_unpack("<Q", 8, handle)
  metadata_count = _read_unpack("<Q", 8, handle)
  return version, tensor_count, metadata_count

def _read_metadata(handle, count: int) -> dict[str, Any]:
  metadata = {}
  for _ in range(count):
    key = _read_str(handle)
    value_type = _read_unpack("<i", 4, handle)
    if key in metadata: raise ValueError(f"duplicate GGUF metadata key: {key!r}")
    metadata[key] = _read_value(handle, value_type)
  return metadata

def _read_tensor_info(handle) -> tuple[str, tuple[int, ...], int, int]:
  name = _read_str(handle)
  dim_count = _read_unpack("<I", 4, handle)
  if not 1 <= dim_count <= 4: raise ValueError(f"tensor {name!r} has invalid rank: {dim_count}")
  dims = tuple(_read_unpack("<Q", 8, handle) for _ in range(dim_count))
  ggml_type = _read_unpack("<i", 4, handle)
  offset = _read_unpack("<Q", 8, handle)
  return name, dims, ggml_type, offset

def _read_multi_tensor_info(handle, count: int) -> list[tuple[str, tuple[int, ...], int, int]]:
  return [_read_tensor_info(handle) for _ in range(count)]

def _alignment(metadata: dict[str, Any]) -> int:
  alignment = metadata.get("general.alignment", 32)
  if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
    raise ValueError(f"invalid GGUF alignment: {alignment!r}; expected a positive power of two")
  return alignment

def _data_start(handle, metadata: dict[str, Any]) -> int:
  return _round_up(handle.tell(), _alignment(metadata))

def _tensor_nbytes(n: int, ggml_type: int) -> int:
  if ggml_type in _NATIVE_DTYPES: return n * _NATIVE_DTYPES[ggml_type].itemsize
  if ggml_type in QUANT_BLOCK_BYTES:
    block_values = QUANT_BLOCK_VALUES[ggml_type]
    if n % block_values != 0:
      raise ValueError(f"tensor has {n} values, not a whole number of quant blocks")
    return n // block_values * QUANT_BLOCK_BYTES[ggml_type]
  raise ValueError(f"unsupported GGML type: {ggml_type}")

def _tensor_range(info: tuple[str, tuple[int, ...], int, int], data_start: int, file_size: int,) -> tuple[int, int]:
  name, dims, ggml_type, offset = info
  if not 1 <= len(dims) <= 4 or any(type(d) is not int or d <= 0 for d in dims):
    raise ValueError(f"tensor {name!r} has invalid dimensions: {dims}")
  if offset < 0: raise ValueError(f"tensor {name!r} has negative offset")
  if ggml_type in QUANT_BLOCK_BYTES and dims[0] % QUANT_BLOCK_VALUES[ggml_type]:
    raise ValueError(f"tensor {name!r} row width {dims[0]} is not a multiple of {QUANT_BLOCK_VALUES[ggml_type]}")
  n = math.prod(dims)
  nbytes = _tensor_nbytes(n, ggml_type)
  start = data_start+offset
  end = start+nbytes
  if end > file_size: raise ValueError(f"tensor {name!r} extends beyond the GGUF file")
  return start, end

def _q_to_uint8(data: Tensor, bits: int) -> Tensor:
  shifts = Tensor.const(tuple(2**(i*bits) for i in range(8//bits)), dtype=data.dtype)
  bitmask = 0xFF >> (8-bits)
  return (data.unsqueeze(-1).div(shifts, rounding_mode="trunc").bitwise_and(bitmask).transpose(-1, -2).flatten(-2))

def _q4_scales(blocks:Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  d = blocks[:, 0:2].bitcast(dtypes.float16).cast(dtypes.float32)
  dmin = blocks[:, 2:4].bitcast(dtypes.float16).cast(dtypes.float32)
  packed = blocks[:, 4:16]
  scales = packed[:, 0:4].bitwise_and(63).cat(packed[:, 8:12].bitwise_and(15).bitwise_or(packed[:, 0:4].rshift(6).lshift(4),), dim = -1)
  minimums = packed[:, 4:8].bitwise_and(63).cat(packed[:, 8:12].rshift(4).bitwise_or(packed[:, 4:8].rshift(6).lshift(4),), dim = -1)
  return d, dmin, scales, minimums

def _dequant_q4_k(data: Tensor, *, swizzled_width: int = 0) -> Tensor:
  if swizzled_width:
    words = data.bitcast(dtypes.uint32).reshape(-1, swizzled_width*36)
    full, tail = swizzled_width//4*4, swizzled_width%4
    parts = []
    if full:
      parts.append(words[:, :full*36].reshape(-1, full//4, 9, 4, 4).permute(0, 1, 3, 2, 4).reshape(-1, full, 36))
    if tail:
      parts.append(words[:, full*36:].reshape(-1, 9, tail, 4).permute(0, 2, 1, 3).reshape(-1, tail, 36))
    ordered = parts[0] if len(parts) == 1 else parts[0].cat(parts[1], dim=1)
    blocks = ordered.reshape(-1, 36).bitcast(dtypes.uint8)
  else:
    blocks = data.reshape(-1, QUANT_BLOCK_BYTES[Q4_K]).contiguous()
  d, dmin, scales, minimums = _q4_scales(blocks)
  # Each 32-byte chunk encodes a low-nibble group followed by a high-nibble group.
  qs = blocks[:, 16:144].reshape(-1, 4, 32)
  q = Tensor.stack(qs.bitwise_and(15), qs.rshift(4), dim=2).reshape(-1, 8, 32)
  return (d.unsqueeze(-1) * scales.unsqueeze(-1) * q - dmin.unsqueeze(-1) * minimums.unsqueeze(-1)).flatten()

def _dequant_q5_k(data: Tensor) -> Tensor:
  blocks = data.reshape(-1, QUANT_BLOCK_BYTES[Q5_K]).contiguous()
  d, dmin, scales, minimums = _q4_scales(blocks)
  qs = blocks[:, 48:176].reshape(-1, 4, 32)
  low = Tensor.stack(qs.bitwise_and(15), qs.rshift(4), dim=2).reshape(-1, 8, 32)
  high = blocks[:, 16:48].unsqueeze(1).rshift(Tensor.arange(8).to(data.device).reshape(1, 8, 1))
  q = low.bitwise_or(high.bitwise_and(1).lshift(4))
  return (d.unsqueeze(-1) * scales.unsqueeze(-1) * q - dmin.unsqueeze(-1) * minimums.unsqueeze(-1)).flatten()


def _dequant_q8_0(data: Tensor) -> Tensor:
  blocks = data.reshape(-1, 34).contiguous()
  d = blocks[:, :2].bitcast(dtypes.float16).float()
  return (d * blocks[:, 2:].bitcast(dtypes.int8).float()).flatten()


def _dequant_q6_k(data: Tensor, *, block_bytes: int = 210, swizzled_width: int = 0) -> Tensor:
  # Internal prepared NV storage may add alignment bytes per block.
  # Decode fields in place, without reconstructing a second tight GGUF buffer.
  if block_bytes not in (210, 212, 216): raise ValueError("Q6 block stride must be 210, 212 or 216 bytes")
  if swizzled_width:
    if block_bytes != 216: raise ValueError("swizzled Q6 requires 216-byte prepared blocks")
    # Inverse views of [pair, block, element], not a realized tight GGUF copy.
    words = data.bitcast(dtypes.uint32).reshape(-1, swizzled_width*54)
    full, tail = swizzled_width//4*4, swizzled_width%4
    parts = []
    if full:
      parts.append(words[:, :full*54].reshape(-1, full//4, 27, 4, 2).permute(0, 1, 3, 2, 4).reshape(-1, full, 54))
    if tail:
      parts.append(words[:, full*54:].reshape(-1, 27, tail, 2).permute(0, 2, 1, 3).reshape(-1, tail, 54))
    ordered = parts[0] if len(parts) == 1 else parts[0].cat(parts[1], dim=1)
    blocks = ordered.reshape(-1, 54).bitcast(dtypes.uint8)
  else:
    blocks = data.reshape(-1, block_bytes).contiguous()
  low = _q_to_uint8(blocks[:, :128].reshape(-1, 2, 64), 4)
  high = _q_to_uint8(blocks[:, 128:192].reshape(-1, 2, 32), 2).lshift(4)
  q = (low.bitwise_or(high).bitcast(dtypes.int8) - 32).reshape(-1, 256)
  scales = blocks[:, 192:208].bitcast(dtypes.int8).unsqueeze(-1).expand(-1, 16, 16).reshape(-1, 256)
  d = blocks[:, 208:210].bitcast(dtypes.float16).cast(dtypes.float32)
  return (d * q * scales).flatten()

def _dequant(data: Tensor, n: int, ggml_type: int) -> Tensor:
  """Build decoding math over exact resident packed bytes; do not realize weights."""
  expected = _tensor_nbytes(n, ggml_type)
  if data.dtype != dtypes.uint8: raise ValueError("packed GGUF data must have uint8 dtype")
  if data.numel() != expected: raise ValueError(f"expected {expected} packed bytes, got {data.numel()}")
  if data.device.split(":")[0].upper() == "DISK": raise ValueError("transfer packed GGUF bytes to a compute device before decoding")
  if ggml_type in _NATIVE_DTYPES: return data.contiguous().bitcast(_NATIVE_DTYPES[ggml_type])
  if ggml_type == Q4_K: return _dequant_q4_k(data)
  if ggml_type == Q5_K: return _dequant_q5_k(data)
  if ggml_type == Q6_K: return _dequant_q6_k(data)
  if ggml_type == Q8_0: return _dequant_q8_0(data)
  raise ValueError(f"unsupported GGML type: {ggml_type}")

@dataclass(frozen=True)
class PackedTensor:
  """Bounded storage plus explicit GGML identity; shape is in logical tensor order."""
  data: Tensor
  shape: tuple[int, ...]
  ggml_type: int

  def to(self, device: str | None = None, *, realize: bool = True) -> "PackedTensor":
    data = self.data.to(device)
    return PackedTensor(data.realize() if realize else data, self.shape, self.ggml_type)

  def decode(self) -> Tensor:
    return _dequant(self.data, math.prod(self.shape), self.ggml_type).reshape(self.shape)

def _tensor_from_info(raw: Tensor, info: tuple[str, tuple[int, ...], int, int], data_start: int,
                      device: str | None = None) -> Tensor:
  _, dims, ggml_type, _ = info
  start, end = _tensor_range(info, data_start, raw.numel())
  return PackedTensor(raw[start:end], tuple(reversed(dims)), ggml_type).to(device).decode()

def _validate_tensor_infos(infos, metadata: dict[str, Any], data_start: int, file_size: int) -> None:
  if metadata.get("split.count", 1) != 1 or metadata.get("split.no", 0) != 0:
    raise ValueError("split GGUF files are not supported")
  alignment = _alignment(metadata)
  names, ranges = set(), []
  for info in infos:
    name, _, _, offset = info
    if name in names: raise ValueError(f"duplicate tensor name: {name!r}")
    names.add(name)
    if offset % alignment: raise ValueError(f"tensor {name!r} offset {offset} is not aligned to {alignment}")
    start, end = _tensor_range(info, data_start, file_size)
    ranges.append((start, end, name))
  ranges.sort()
  for previous, current in zip(ranges, ranges[1:]):
    if current[0] < previous[1]: raise ValueError(f"tensor ranges overlap: {previous[2]!r} and {current[2]!r}")

@functools.lru_cache(maxsize=8)
def _read_gguf(path: Path, file_size: int, mtime_ns: int) -> tuple[dict[str, Any], list, int]:
  # Metadata holds the whole vocabulary and merges (~0.5M strings for Qwen3.5); one
  # load reads it from several places, so parse each file version once.
  with path.open("rb") as handle:
    _, tensor_count, metadata_count = _read_header(handle)
    metadata = _read_metadata(handle, metadata_count)
    infos = _read_multi_tensor_info(handle, tensor_count)
    data_start = _data_start(handle, metadata)
  _validate_tensor_infos(infos, metadata, data_start, file_size)
  return metadata, infos, data_start


def _cached_gguf(path: str | Path) -> tuple[Path, int, dict[str, Any], list, int]:
  path = Path(path).resolve()
  stat = path.stat()
  metadata, infos, data_start = _read_gguf(path, stat.st_size, stat.st_mtime_ns)
  return path, stat.st_size, dict(metadata), infos, data_start  # a fresh dict: callers may add keys


def read_gguf_metadata(path: str | Path) -> dict[str, Any]:
  """Read metadata only, for architecture dispatch before weight allocation."""
  return _cached_gguf(path)[2]


def parse_gguf(path: str | Path) -> tuple[dict[str, Any], dict[str, PackedTensor]]:
  """Validate a single GGUF and return bounded DISK views without transferring weights."""
  path, file_size, metadata, infos, data_start = _cached_gguf(path)
  raw = Tensor.empty(file_size, dtype=dtypes.uint8, device=f"disk:{path}")
  entries = {}
  for info in infos:
    name, dims, ggml_type, _ = info
    start, end = _tensor_range(info, data_start, file_size)
    entries[name] = PackedTensor(raw[start:end], tuple(reversed(dims)), ggml_type)
  return metadata, entries

def load_gguf(path: str | Path, device: str | None = None) -> tuple[dict[str, Any], dict[str, Tensor]]:
  """Stage packed bytes once on the compute device and return lazy FP32 decoding views.

  Native F32 remains F32. Only packed bytes are realized here, never decoded weights.
  Use parse_gguf() when explicit packed descriptors are needed for custom kernels.
  """
  metadata, entries = parse_gguf(path)
  return metadata, {name: entry.to(device).decode() for name, entry in entries.items()}
