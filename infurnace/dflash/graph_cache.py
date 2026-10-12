"""On-disk cache of DFlash's captured graphs.

Capturing DFlash's graphs means scheduling and lowering every kernel of the
16-row target forward, the draft and the one-row step: tens of seconds per
process. A captured graph pickles to its compiled program plus the buffers it
touches. Buffers that belong to the live model, draft, verifier or RNG are
saved as names (their attribute path) and re-bound to the current process's
buffers on load, so weights and caches are never written; the rest (scratch
and graph-private constants) are saved with their contents.

The key covers everything that changes the captured programs: tinygrad and
infurnace sources, both GGUF files, context length, block size, fused rounds,
the device and compilation flags. Any mismatch or failure is a cache miss.
"""
from __future__ import annotations
import hashlib, os, pickle, sys
from pathlib import Path
from tinygrad import Device, Tensor
from tinygrad.device import Buffer
from tinygrad.engine.jit import CapturedJit, _TinyJit
from tinygrad.uop.ops import Ops
from infurnace import disk_cache

FORMAT = 1
_FLAGS = ('BEAM', 'JITBEAM', 'NOOPT', 'PROFILE', 'DEBUG', 'JIT', 'IMAGE', 'HCQ_CACHE_THRESH', 'VALIDATE_WITH_CPU')


def cache_dir() -> Path:
  return disk_cache.root() / 'dflash-graphs'


def key(generator, block: int) -> str:
  digest = hashlib.sha256(f'format {FORMAT} python {sys.version_info[:2]} {disk_cache.sources()}'.encode())
  for path in (generator.model._path, generator.draft.path): digest.update(disk_cache.file_identity(path).encode())
  device = Device[generator.model.device]
  digest.update(f'{generator.model.config.context_length} {block} {generator.fused} {device.device} {getattr(device, "arch", "")}'.encode())
  digest.update(repr([(flag, os.environ.get(flag)) for flag in _FLAGS]).encode())
  return digest.hexdigest()[:32]


def named_buffers(generator) -> dict[str, Buffer]:
  """Every realized buffer reachable from the generator and tinygrad's RNG state,
  named by its smallest attribute path (independent of traversal order)."""
  names: dict[Buffer, str] = {}
  def note(buffer: Buffer, path: str):
    if buffer not in names or path < names[buffer]: names[buffer] = path
  def visit(obj, path: str, depth: dict[int, str]):
    if isinstance(obj, (_TinyJit, CapturedJit)): return
    if isinstance(obj, Tensor):
      # Buffer identity, allocated or not: `Tensor.empty().realize()` owns a buffer before its first write.
      if (base := obj.uop.base).op is Ops.BUFFER:
        note(base.buffer, path)
        if base.buffer._base is not None: note(base.buffer._base, path + '#base')
      return
    if id(obj) in depth and depth[id(obj)] <= path: return
    depth[id(obj)] = path
    if isinstance(obj, (list, tuple)): items = [(str(i), v) for i, v in enumerate(obj)]
    elif isinstance(obj, dict): items = [(str(k), v) for k, v in obj.items()]
    elif hasattr(obj, '__dict__') and type(obj).__module__.startswith(('infurnace', 'tinygrad.llm', 'tinygrad.nn')):
      items = list(vars(obj).items())
    else: return
    for name, value in items: visit(value, f'{path}.{name}', depth)
  depth: dict[int, str] = {}
  visit(generator, 'generator', depth)
  visit(Tensor._device_seeds, 'rng.seeds', depth)
  visit(Tensor._device_rng_counters, 'rng.counters', depth)
  return {name: buffer for buffer, name in names.items()}


class _Pickler(pickle.Pickler):
  def __init__(self, file, by_buffer: dict[Buffer, str]):
    super().__init__(file, protocol=5)
    self.by_buffer = by_buffer
  def persistent_id(self, obj):
    return self.by_buffer.get(obj) if isinstance(obj, Buffer) else None


class _Unpickler(pickle.Unpickler):
  def __init__(self, file, by_name: dict[str, Buffer]):
    super().__init__(file)
    self.by_name = by_name
  def persistent_load(self, name):
    return self.by_name[name].ensure_allocated()  # KeyError: a buffer this process does not have, so a cache miss


def save(path: Path, graphs: dict[str, _TinyJit], buffers: dict[str, Buffer]) -> None:
  captured = {name: jit.captured for name, jit in graphs.items() if jit.captured is not None}
  path.parent.mkdir(parents=True, exist_ok=True)
  partial = path.with_suffix('.tmp')
  with partial.open('wb') as f: _Pickler(f, {buffer: name for name, buffer in buffers.items()}).dump({'format': FORMAT, 'graphs': captured})
  partial.replace(path)
  disk_cache.prune(path.parent, keep=4)  # each entry holds only graph-private buffers


def load(path: Path, buffers: dict[str, Buffer]) -> dict[str, CapturedJit] | None:
  try:
    with path.open('rb') as f: payload = _Unpickler(f, buffers).load()
  except (OSError, EOFError, KeyError, pickle.UnpicklingError, AttributeError, ImportError, TypeError, ValueError):
    return None
  return payload['graphs'] if isinstance(payload, dict) and payload.get('format') == FORMAT else None
