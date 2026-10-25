"""GGUF files, kept packed: our NV kernels read the quantized blocks directly.

The header is read here, with tinygrad's readers, so that types tinygrad doesn't know (Q2_0, Bonsai's ternary
weights) still get their tensor views; decode() handles those and leaves the rest to tinygrad."""
import functools
import io
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from tinygrad import Tensor, dtypes
from tinygrad.helpers import round_up
from tinygrad.llm.gguf import _GGML_NATIVE, _GGML_QUANT, TensorIO, ggml_data_to_tensor, read_int32, read_int64, read_str, read_uint32, \
  read_uint64, readers

Q8_0, Q4_K, Q5_K, Q6_K, Q2_0 = 8, 12, 13, 14, 42
# Types beyond tinygrad's: (values per block, bytes per block)
_EXTRA_QUANT = {Q2_0: (64, 18)}  # FP16 scale, then 64 two-bit codes, four per byte from the low bits: 0 -> -1, 1 -> 0, 2 -> +1

def block_size(ggml_type: int) -> tuple[int, int]:
  """(values per block, bytes per block) of a GGML type."""
  if ggml_type in _GGML_NATIVE: return 1, _GGML_NATIVE[ggml_type].itemsize
  if ggml_type in _GGML_QUANT: return _GGML_QUANT[ggml_type]
  if ggml_type in _EXTRA_QUANT: return _EXTRA_QUANT[ggml_type]
  raise ValueError(f"GGML type {ggml_type} is not supported")

def decode(data: Tensor, n: int, ggml_type: int) -> Tensor:
  """`n` values of packed GGML bytes as a float tensor (tinygrad's decoders, plus the types it lacks)."""
  if ggml_type == Q2_0:
    blocks = data[:n // 64 * 18].reshape(-1, 18).contiguous()  # bitcasting a strided or lazy view isn't safe
    scale = blocks[:, :2].bitcast(dtypes.float16).float()
    codes = (blocks[:, 2:].unsqueeze(-1).cast(dtypes.int32) // Tensor([1, 4, 16, 64], dtype=dtypes.int32, device=data.device)) % 4
    return ((codes.flatten(-2) - 1).float() * scale).flatten()[:n]
  return ggml_data_to_tensor(data, n, ggml_type)

@dataclass(frozen=True)
class PackedTensor:
  """A tensor's raw GGUF bytes (a DISK view until moved), its logical shape and GGML type."""
  data: Tensor
  shape: tuple[int, ...]
  ggml_type: int

  def to(self, device: str | None = None, *, realize: bool = True) -> "PackedTensor":
    data = self.data.to(device)
    return PackedTensor(data.realize() if realize else data, self.shape, self.ggml_type)

  def decode(self) -> Tensor:
    return decode(self.data, math.prod(self.shape), self.ggml_type).reshape(self.shape)

def _gguf_parse(path: Path) -> tuple[dict[str, Any], dict[str, tuple[Tensor, tuple[int, ...], int]]]:
  """tinygrad's gguf_parse, with the types it lacks: metadata from the first file, tensors from every file of a
  split model (name-00001-of-0000N.gguf and its parts), as tinygrad merges them."""
  files, metadata, entries = [path], None, {}
  for i, file in enumerate(files):
    tensor = Tensor(file)
    r = io.BufferedReader(TensorIO(tensor), 1_000_000)
    magic, version, n_tensors, n_kv = r.read(4), read_int32(r), read_int64(r), read_int64(r)
    if magic != b"GGUF" or version not in (2, 3): raise ValueError(f"invalid GGUF file: {file}")
    kv = {}
    for _ in range(n_kv):
      key, typ = read_str(r), read_int32(r)
      kv[key] = readers[typ](r)
    if i == 0:
      metadata = kv
      if (total := kv.get('split.count', 1)) > 1:
        if kv.get('split.no', 0) != 0: raise ValueError(f"load a split GGUF from its first part, not part {kv['split.no'] + 1}: {file}")
        if not (m := re.match(r"^(.*)-00001-of-\d{5}\.gguf$", str(file))):
          raise ValueError(f"a split GGUF's first part must end in -00001-of-NNNNN.gguf: {file}")
        files.extend(Path(f"{m.group(1)}-{part:05d}-of-{total:05d}.gguf") for part in range(2, total + 1))
    infos = [(read_str(r), tuple(read_uint64(r) for _ in range(read_uint32(r))), read_int32(r), read_uint64(r)) for _ in range(n_tensors)]
    start = round_up(r.tell(), kv.get("general.alignment", 32))
    for name, dims, typ, offset in infos:
      values, nbytes = block_size(typ)
      begin, size = start + offset, math.prod(dims) // values * nbytes
      entries[name] = (tensor[begin:begin + size], tuple(reversed(dims)), typ)
  return metadata, entries

@functools.lru_cache(maxsize=8)
def _parse(path: Path, size: int, mtime_ns: int) -> tuple[dict[str, Any], dict[str, tuple[Tensor, tuple[int, ...], int]]]:
  # Metadata holds the whole vocabulary (~0.5M strings for Qwen3.5) and one load reads it from several places.
  kv, entries = _gguf_parse(path)
  for name, (data, shape, typ) in entries.items():  # tensor slices silently stop at the end of a truncated file
    values, nbytes = block_size(typ)
    if data.numel() != math.prod(shape) // values * nbytes: raise ValueError(f"truncated GGUF: tensor {name!r} extends beyond {path}")
  return kv, entries

def parse_gguf(path: str | Path) -> tuple[dict[str, Any], dict[str, PackedTensor]]:
  """Metadata (a fresh dict: callers may add keys) and packed DISK views; no weights are read."""
  path = Path(path).resolve()
  stat = path.stat()
  kv, entries = _parse(path, stat.st_size, stat.st_mtime_ns)
  return dict(kv), {name: PackedTensor(*entry) for name, entry in entries.items()}

def read_gguf_metadata(path: str | Path) -> dict[str, Any]:
  return parse_gguf(path)[0]
