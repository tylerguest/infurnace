"""The parameter cache: weights prepared for the GPU (packed, tiled, rotated), saved so the next load reads them
back in one piece instead of preparing them again."""
import functools
import hashlib
import json
import os
import struct
from pathlib import Path
from tinygrad import Tensor, UOp, dtypes
from tinygrad.device import Buffer

# The parameter cache lives under $XDG_CACHE_HOME/infurnace/params and is
# keyed on everything that changes what it holds. A key mismatch, missing file or read error is a miss.
def cache_root() -> Path:
  return Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))).expanduser() / 'infurnace'

@functools.cache
def source_hash() -> str:
  """Hash of the tinygrad and infurnace sources: any code change selects new entries."""
  import tinygrad, infurnace
  digest = hashlib.sha256()
  for module in (tinygrad, infurnace):
    package = Path(module.__file__).parent
    for path in sorted(package.rglob('*.py')):
      digest.update(str(path.relative_to(package)).encode())
      digest.update(path.read_bytes())
  return digest.hexdigest()

def file_identity(path: str | Path) -> str:
  stat = Path(path).stat()
  return f'{Path(path).resolve()} {stat.st_size} {stat.st_mtime_ns}'

def prune_cache(directory: Path, keep: int) -> None:
  # Finished entries only: a .tmp may be another process's write in progress.
  for old in sorted(directory.glob('*.bin'), key=lambda p: p.stat().st_mtime)[:-keep]: old.unlink(missing_ok=True)

_ALIGN, _MAGIC = 256, b'INFTENS1'

def save_tensors(path: Path, tensors: dict[str, Tensor]) -> None:
  """Realized tensors as one blob, each 256-byte aligned, plus a name/dtype/shape/offset index."""
  index, chunks, offset = [], [], 0
  for name, tensor in tensors.items():
    data = bytes(tensor.data())
    index.append({'name': name, 'dtype': tensor.dtype.name, 'shape': list(tensor.shape), 'offset': offset, 'nbytes': len(data)})
    chunks.append(data + bytes(-len(data) % _ALIGN))
    offset += len(chunks[-1])
  header = json.dumps(index).encode()
  path.parent.mkdir(parents=True, exist_ok=True)
  partial = path.with_suffix('.tmp')
  with partial.open('wb') as f:
    f.write(_MAGIC + struct.pack('<Q', len(header)) + header)
    f.write(bytes(-f.tell() % _ALIGN))
    for chunk in chunks: f.write(chunk)
  partial.replace(path)

def load_tensors(path: Path, device: str) -> dict[str, Tensor] | None:
  """One host read and one copy to `device`; each tensor is a view into that buffer."""
  try:
    raw = path.read_bytes()
    if raw[:8] != _MAGIC: return None
    size = struct.unpack_from('<Q', raw, 8)[0]
    index = json.loads(raw[16:16 + size])
    blob = raw[16 + size + (-(16 + size) % _ALIGN):]
    if any(entry['offset'] + entry['nbytes'] > len(blob) for entry in index): return None
    base = Buffer(device, max(len(blob), 1), initial_value=blob)
    return {entry['name']: Tensor(UOp.from_buffer(Buffer(device, entry['nbytes'], base=base, offset=entry['offset']).ensure_allocated(),
                                                  getattr(dtypes, entry['dtype']))).reshape(entry['shape']) for entry in index}
  except (OSError, ValueError, KeyError, TypeError, AttributeError, struct.error):
    return None
