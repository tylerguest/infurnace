"""The models infurnace knows by name, and their downloads.

A name (e.g. qwen3.5:4b) stands for a GGUF download, pinned to a revision, and optionally a DFlash draft that
NVIDIA GPUs use automatically. Downloads go into user storage ($XDG_DATA_HOME/infurnace/models) on first use; a
copy already there, or in an editable checkout's models/ directory, is used as is. Any other GGUF of a supported
family (infurnace/models) runs from its path.
"""
from __future__ import annotations
import datetime
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from tinygrad.helpers import fetch

@dataclass(frozen=True)
class Entry:
  name: str
  url: str                   # the model GGUF
  architecture: str          # its general.architecture: the family that runs it
  draft: str | None = None   # a DFlash draft GGUF for it

  @property
  def urls(self) -> list[str]: return [url for url in (self.url, self.draft) if url]

ENTRIES = [
  Entry("qwen3.5:4b", "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf",
        "qwen35", "https://huggingface.co/Anbeeld/Qwen3.5-4B-DFlash-GGUF/resolve/a79bafa7f383f8627f925250f547fe1ebbc43ff6/qwen35-4b-dflash-Q4_K_M.gguf"),
]
CATALOG = {entry.name: entry for entry in ENTRIES}
DEFAULT = ENTRIES[0].name

def names() -> str: return ", ".join(CATALOG)

def entry(name: str) -> Entry:
  if name not in CATALOG: raise ValueError(f"unknown model {name!r}; available: {names()}")
  return CATALOG[name]

def _file_name(url: str) -> str: return url.rsplit("/", 1)[1]

def _user_models() -> Path:
  return Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))).expanduser() / "infurnace/models"

def local(url: str) -> Path | None:
  """An existing copy of a download: user storage first, then an editable checkout's models/."""
  name, checkout = _file_name(url), Path(__file__).resolve().parent.parent / "models"
  for path in (_user_models() / name, *(checkout / sub / name for sub in ("", "qwen35-validation", "dflash-qwen35"))):
    if path.is_file(): return path
  return None

def fetch_model(model: str | Path) -> Path:
  """A catalog name or download URL as a local file, downloaded on first use; a path is returned as is."""
  url = CATALOG[model].url if model in CATALOG else str(model)
  if not url.startswith("https://"): return Path(url).expanduser()
  return local(url) or fetch(url, name=_user_models() / _file_name(url))

def details(name: str, path: str | Path | None = None) -> dict:
  """What /api/tags, /api/show and `infurnace list` say about a model: family, size label and quantization."""
  known = CATALOG.get(name)
  file_name = _file_name(known.url) if known else Path(path or name).name
  quant = re.search(r"(Q\d_K(?:_[SML])?|Q\d_\d|F16|BF16)", file_name, re.IGNORECASE)
  return {'format': 'gguf', 'family': known.architecture if known else '', 'parameter_size': name.split(':')[-1].upper() if known else '',
          'quantization_level': quant.group(1).upper() if quant else ''}

def listing() -> list[dict]:
  """Every catalog model, downloaded or not, as /api/tags lists them (sizes include the DFlash draft)."""
  out = []
  for e in ENTRIES:
    path, draft = local(e.url), local(e.draft) if e.draft else None
    size = (path.stat().st_size if path else 0) + (draft.stat().st_size if draft else 0)
    modified = datetime.datetime.fromtimestamp(path.stat().st_mtime, datetime.timezone.utc).isoformat() if path else None
    out.append({'name': e.name, 'model': e.name, 'size': size, 'modified_at': modified, 'digest': '', 'details': details(e.name),
                'path': str(path) if path else None, 'draft': str(draft) if draft else None, 'has_draft': e.draft is not None,
                'downloaded': path is not None})
  return out

def installed() -> list[dict]:
  """Downloaded catalog models (/api/tags)."""
  return [m for m in listing() if m['downloaded']]

def pull(name: str) -> list[Path]:
  """Download a catalog model and its DFlash draft (each only if missing)."""
  return [fetch_model(url) for url in entry(name).urls]

def remove(name: str) -> list[Path]:
  """Delete a catalog model's downloads from user storage (an editable checkout's models/ is left alone)."""
  removed = []
  for url in entry(name).urls:
    if (path := _user_models() / _file_name(url)).is_file():
      path.unlink()
      removed.append(path)
  return removed

def draft_for(model: str, architecture: Callable[[], str | None]) -> str | Path | None:
  """The DFlash draft for a model: INFURNACE_DFLASH_DRAFT, its catalog entry's (a download URL), or for a local GGUF
  an already downloaded draft of a catalog model of the same architecture (asked of `architecture()` only then)."""
  if path := os.environ.get("INFURNACE_DFLASH_DRAFT"): return Path(path).expanduser()
  if model in CATALOG: return CATALOG[model].draft
  arch = architecture()
  return next((found for e in ENTRIES if e.architecture == arch and e.draft and (found := local(e.draft))), None)
