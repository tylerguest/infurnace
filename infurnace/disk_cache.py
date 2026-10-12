"""Small on-disk caches, keyed on everything that changes what they hold.

Entries live under $XDG_CACHE_HOME/infurnace (default ~/.cache/infurnace). A key
mismatch, a missing file or any read error is simply a miss.
"""
from __future__ import annotations
import functools, hashlib, json, os, struct
from pathlib import Path
from tinygrad import Tensor, dtypes
from tinygrad.device import Buffer
from tinygrad.uop.ops import UOp

_ALIGN = 256
_MAGIC = b'INFTENS1'

def root() -> Path:
  return Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))).expanduser() / 'infurnace'

@functools.cache
def sources() -> str:
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

def prune(directory: Path, keep: int) -> None:
  for old in sorted(directory.glob('*'), key=lambda p: p.stat().st_mtime)[:-keep]: old.unlink(missing_ok=True)

def save_tensors(path: Path, tensors: dict[str, Tensor]) -> None:
  """Realized tensors as one aligned blob plus a name/dtype/shape/offset index."""
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
    start = 16 + size + (-(16 + size) % _ALIGN)
    blob = raw[start:]
    if any(entry['offset'] + entry['nbytes'] > len(blob) for entry in index): return None
    base = Buffer(device, max(len(blob), 1), initial_value=blob)
    result = {}
    for entry in index:
      dtype = getattr(dtypes, entry['dtype'])
      view = Buffer(device, entry['nbytes'], base=base, offset=entry['offset']).ensure_allocated()
      result[entry['name']] = Tensor(UOp.from_buffer(view, dtype)).reshape(entry['shape'])
    return result
  except (OSError, ValueError, KeyError, TypeError, AttributeError, struct.error):
    return None
