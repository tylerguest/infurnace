"""Truncated copies of real GGUF models for the NV model tests, as tinygrad tests its LLM code: small, real-shaped models
by default, the full ones only when asked for (INFURNACE_TEST_SLOW=1, see tests/conftest.py).

A copy keeps the real embedding, output head, tokenizer, chat template and layer shapes, so every kernel runs at the
shapes it runs at in use, but only the first few layers: a model's graph work (capture, warmup, eager test steps)
grows with its layer count, and 36 layers cost minutes per load. The header is copied record by record, changing only
the layer count and per-layer lists; tensor bytes are copied as they are, never decoded or requantized.

Copies are made once per source file and kept under $XDG_CACHE_HOME/infurnace-tests, outside the tests' temporary
cache directories (tests/conftest.py)."""
import hashlib
import os
import re
import struct
from pathlib import Path
from infurnace.weights.gguf import block_size

ROOT = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))).expanduser() / 'infurnace-tests'
_SCALARS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
_STRING, _ARRAY = 8, 9

class _Reader:
  def __init__(self, file): self.file = file
  def unpack(self, fmt: str): return struct.unpack('<' + fmt, self.file.read(struct.calcsize('<' + fmt)))[0]
  def string(self) -> str: return self.file.read(self.unpack('Q')).decode()
  def value(self, kind: int):
    if kind == _STRING: return self.string()
    if kind == _ARRAY:
      item, count = self.unpack('I'), self.unpack('Q')
      return item, [self.value(item) for _ in range(count)]
    return self.unpack(_SCALARS[kind])

def _string(value: str) -> bytes:
  encoded = value.encode()
  return struct.pack('<Q', len(encoded)) + encoded

def _value(kind: int, value) -> bytes:
  if kind == _STRING: return _string(value)
  if kind == _ARRAY:
    item, values = value
    return struct.pack('<IQ', item, len(values)) + b''.join(_value(item, v) for v in values)
  return struct.pack('<' + _SCALARS[kind], value)

def _layer(name: str) -> int | None:
  return int(match.group(1)) if (match := re.match(r'blk\.(\d+)\.', name)) else None

def truncate(source: str | Path, layers: int, *, taps: int | None = None) -> Path:
  """A copy of the GGUF `source` with its first `layers` blocks. A DFlash draft also keeps its first `taps` target
  layers (its fc input columns), renumbered 1..taps so they fit a target truncated to more than `taps` layers."""
  source = Path(source).resolve()
  stat = source.stat()
  key = hashlib.sha256(f'{source} {stat.st_size} {stat.st_mtime_ns} {layers} {taps}'.encode()).hexdigest()[:12]
  out = ROOT / f'{source.stem}-{layers}L-{key}.gguf'
  if out.is_file(): return out
  with open(source, 'rb') as file:
    r = _Reader(file)
    if file.read(4) != b'GGUF' or (version := r.unpack('I')) not in (2, 3): raise ValueError(f'not a GGUF file: {source}')
    n_tensors, n_kv = r.unpack('Q'), r.unpack('Q')
    records = []
    for _ in range(n_kv):
      key_name, kind = r.string(), r.unpack('I')
      records.append((key_name, kind, r.value(kind)))
    infos = []
    for _ in range(n_tensors):
      name = r.string()
      dims = tuple(r.unpack('Q') for _ in range(r.unpack('I')))
      infos.append((name, dims, r.unpack('I'), r.unpack('Q')))
    metadata = {k: v for k, _, v in records}
    alignment = metadata.get('general.alignment', 32)
    start = -(-file.tell() // alignment) * alignment
    arch = metadata['general.architecture']
    kept_taps = None
    if taps is not None:
      _, features = metadata[f'{arch}.target_layers']
      if taps > len(features): raise ValueError(f'{source.name} has only {len(features)} target layers')
      kept_taps = taps
    def rewrite(key_name, value):
      if key_name == f'{arch}.block_count': return value if value < layers else layers
      if key_name == 'prism.hadamard.weight_names':
        item, names = value
        return item, [n for n in names if (i := _layer(n)) is None or i < layers]
      if key_name == f'{arch}.attention.sliding_window_pattern':
        item, pattern = value
        return item, pattern[:layers - 1] + pattern[-1:] if layers < len(pattern) else pattern  # keep the last kind too
      if key_name == f'{arch}.target_layers' and kept_taps is not None: return value[0], list(range(1, kept_taps + 1))
      return value
    if metadata[f'{arch}.block_count'] < layers: raise ValueError(f'{source.name} has fewer than {layers} layers')
    header = b'GGUF' + struct.pack('<IQQ', version, 0, len(records))  # tensor count patched below
    for key_name, kind, value in records: header += _string(key_name) + struct.pack('<I', kind) + _value(kind, rewrite(key_name, value))
    # Each kept tensor: (name, dims, type, source byte ranges); the draft's fc keeps its first taps column blocks per row.
    kept, offset = [], 0
    for name, dims, kind, at in infos:
      if (i := _layer(name)) is not None and i >= layers: continue
      values, nbytes = block_size(kind)
      count = 1  # GGUF dims list the row width first
      for d in dims: count *= d
      begin = start + at
      if name == 'fc.weight' and kept_taps is not None:
        width, rows = dims[0], dims[1]
        if width % (taps_total := len(metadata[f'{arch}.target_layers'][1])): raise ValueError('fc.weight is not one block per tap')
        keep = width // taps_total * kept_taps
        row, part = width // values * nbytes, keep // values * nbytes
        ranges, dims = [(begin + j * row, part) for j in range(rows)], (keep, rows)
      else: ranges = [(begin, count // values * nbytes)]
      kept.append((name, dims, kind, ranges, offset))
      offset += -(-sum(n for _, n in ranges) // alignment) * alignment
    header = header[:4] + struct.pack('<IQQ', version, len(kept), len(records)) + header[24:]
    for name, dims, kind, _, at in kept:
      header += _string(name) + struct.pack(f'<I{len(dims)}QIQ', len(dims), *dims, kind, at)
    header += bytes(-len(header) % alignment)
    ROOT.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix('.partial')
    with open(partial, 'wb') as dest:
      dest.write(header)
      for _, _, _, ranges, at in kept:
        dest.write(bytes(len(header) + at - dest.tell()))
        for begin, n in ranges:
          file.seek(begin)
          while n:
            chunk = file.read(min(n, 64 << 20))
            dest.write(chunk)
            n -= len(chunk)
    partial.rename(out)
  return out
