"""GGUF files through tinygrad's parser, kept packed: our NV kernels read the quantized blocks directly."""
import functools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from tinygrad import Tensor
from tinygrad.llm.gguf import _GGML_NATIVE, _GGML_QUANT, ggml_data_to_tensor, gguf_parse

Q8_0, Q4_K, Q5_K, Q6_K = 8, 12, 13, 14

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
    return ggml_data_to_tensor(self.data, math.prod(self.shape), self.ggml_type).reshape(self.shape)

@functools.lru_cache(maxsize=8)
def _parse(path: Path, size: int, mtime_ns: int) -> tuple[dict[str, Any], dict[str, tuple[Tensor, tuple[int, ...], int]]]:
  # Metadata holds the whole vocabulary (~0.5M strings for Qwen3.5) and one load reads it from several places.
  kv, entries = gguf_parse(path)
  for name, (data, shape, typ) in entries.items():  # tinygrad's slices silently stop at the end of a truncated file
    block, size = (1, _GGML_NATIVE[typ].itemsize) if typ in _GGML_NATIVE else _GGML_QUANT[typ]
    if data.numel() != math.prod(shape) // block * size: raise ValueError(f"truncated GGUF: tensor {name!r} extends beyond {path}")
  return kv, entries

def parse_gguf(path: str | Path) -> tuple[dict[str, Any], dict[str, PackedTensor]]:
  """Metadata (a fresh dict: callers may add keys) and packed DISK views; no weights are read."""
  path = Path(path).resolve()
  stat = path.stat()
  kv, entries = _parse(path, stat.st_size, stat.st_mtime_ns)
  return dict(kv), {name: PackedTensor(*entry) for name, entry in entries.items()}

def read_gguf_metadata(path: str | Path) -> dict[str, Any]:
  return parse_gguf(path)[0]
