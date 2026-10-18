"""Choosing and loading Qwen3.5-4B and its optional DFlash draft.

`--model` takes a name from MODELS or a path to a local Qwen3.5 GGUF file. A named model and
its DFlash draft are downloaded into user storage ($XDG_DATA_HOME/infurnace/models) on first
use; a copy already there, or in an editable checkout's models/ directory, is used as is.
"""
from __future__ import annotations
import datetime
import os
import re
import sys
from pathlib import Path
from tinygrad.helpers import fetch
from infurnace.device import fast_path_problem, nvidia, select_device, supports_nv_quant
from infurnace.gguf import read_gguf_metadata
from infurnace.kernels.quant_gemm import nv_prefill_supported
from infurnace.models.qwen35 import Qwen35Model
from infurnace.tokenizer import Qwen35Tokenizer

# name: (model URL, DFlash draft URL). The first entry is the default.
MODELS = {
  "qwen3.5:4b": ("https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf",
                 "https://huggingface.co/Anbeeld/Qwen3.5-4B-DFlash-GGUF/resolve/a79bafa7f383f8627f925250f547fe1ebbc43ff6/qwen35-4b-dflash-Q4_K_M.gguf"),
}

def _user_models() -> Path:
  return Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))).expanduser() / "infurnace/models"

def _local(url: str) -> Path | None:
  """An existing copy of a download: user storage first, then an editable checkout's models/."""
  name, checkout = url.rsplit("/", 1)[1], Path(__file__).resolve().parent.parent / "models"
  for path in (_user_models() / name, *(checkout / sub / name for sub in ("", "qwen35-validation", "dflash-qwen35"))):
    if path.is_file(): return path
  return None

def fetch_model(model: str | Path) -> Path:
  """A MODELS name or download URL as a local file, downloaded on first use; a path is returned as is."""
  url = MODELS[model][0] if model in MODELS else str(model)
  if not url.startswith("https://"): return Path(url).expanduser()
  return _local(url) or fetch(url, name=_user_models() / url.rsplit("/", 1)[1])

def _details(name: str, url: str) -> dict:
  quant = re.search(r"(Q\d_K(?:_[SML])?|Q\d_\d|F16|BF16)", url.rsplit("/", 1)[1], re.IGNORECASE)
  return {'format': 'gguf', 'family': 'qwen35', 'parameter_size': name.split(':')[-1].upper(),
          'quantization_level': quant.group(1).upper() if quant else ''}

def installed() -> list[dict]:
  """Downloaded models, as /api/tags lists them (sizes include the DFlash draft)."""
  out = []
  for name, (url, draft_url) in MODELS.items():
    if (path := _local(url)) is None: continue
    draft = _local(draft_url) if draft_url else None
    modified = datetime.datetime.fromtimestamp(path.stat().st_mtime, datetime.timezone.utc).isoformat()
    out.append({'name': name, 'model': name, 'size': path.stat().st_size + (draft.stat().st_size if draft else 0), 'modified_at': modified,
                'digest': '', 'details': _details(name, url), 'path': str(path), 'draft': str(draft) if draft else None})
  return out

def pull(name: str) -> list[Path]:
  """Download a named model and its DFlash draft (each only if missing)."""
  if name not in MODELS: raise ValueError(f"unknown model {name!r}; available: {', '.join(MODELS)}")
  return [fetch_model(url) for url in MODELS[name] if url]

def remove(name: str) -> list[Path]:
  """Delete a named model's downloads from user storage (an editable checkout's models/ is left alone)."""
  if name not in MODELS: raise ValueError(f"unknown model {name!r}; available: {', '.join(MODELS)}")
  removed = []
  for url in MODELS[name]:
    if url and (path := _user_models() / url.rsplit("/", 1)[1]).is_file():
      path.unlink()
      removed.append(path)
  return removed

def model_info(name: str, path: Path, dflash: bool) -> dict:
  """What the server reports for the loaded model (/api/ps, /api/show)."""
  url = MODELS[name][0] if name in MODELS else str(path)
  return {'size': Path(path).stat().st_size, 'dflash': dflash, 'details': _details(name, url)}

# Beyond the weights and the caches: a fixed runtime (the CUDA context ~0.47 GB, tinygrad's ~0.39 GB) and graph
# scratch and working buffers, which grow with the model; measured together as ~1.45 GB for Qwen3.5-4B with its
# DFlash draft (3.1 GB of weights) on an RTX 5060. The margin leaves room for what the desktop does meanwhile.
RUNTIME, SCRATCH_PER_WEIGHT_BYTE, MARGIN, CONTEXT_STEP = 0.86e9, 0.19, 0.25e9, 8192

def cache_bytes_per_token(metadata: dict) -> int:
  """KV cache bytes per context token: full-attention layers of the model (Qwen3.5: every `full_attention_interval`
  block) or of a DFlash draft (its non-sliding layers; sliding ones use fixed ring caches), FP16 keys and values."""
  arch = metadata.get("general.architecture", "")
  prefix = "dflash" if "dflash.block_count" in metadata else arch
  blocks = metadata.get(f"{prefix}.block_count", 0)
  if (pattern := metadata.get(f"{prefix}.attention.sliding_window_pattern")) is not None: layers = sum(not p for p in pattern)
  elif (interval := metadata.get(f"{prefix}.full_attention_interval")): layers = blocks // interval
  else: layers = blocks
  heads = metadata.get(f"{prefix}.attention.head_count_kv", 0)
  key = metadata.get(f"{prefix}.attention.key_length", 0)
  return layers * heads * (key + metadata.get(f"{prefix}.attention.value_length", key)) * 2

def fit_context(model_path: str | Path, draft_path: str | Path | None, free: int) -> int:
  """The largest context (a multiple of 8,192, up to the model's own limit) whose weights, caches and buffers fit
  in `free` bytes of GPU memory; 4,096 when none does."""
  paths = [Path(model_path), *([Path(draft_path)] if draft_path else [])]
  metadata = [read_gguf_metadata(path) for path in paths]
  per_token = sum(cache_bytes_per_token(m) for m in metadata)
  arch = metadata[0].get("general.architecture", "")
  limit = metadata[0].get(f"{arch}.context_length", 32768)
  weights = sum(path.stat().st_size for path in paths)
  room = free - MARGIN - RUNTIME - weights * (1 + SCRATCH_PER_WEIGHT_BYTE)
  steps = int(room // (per_token * CONTEXT_STEP)) if per_token > 0 and room > 0 else 0
  if not steps and room < per_token * 4096:
    print(f"Warning: the model may not fit in {free / 1e9:.1f} GB of free GPU memory; free some (close other GPU programs) "
          "or use --max-context.", file=sys.stderr)
  return min(limit, steps * CONTEXT_STEP) if steps else 4096

def dflash_draft(model: str) -> str | Path | None:
  """The draft override, named model's download URL, or an installed draft for a local Qwen3.5 file."""
  if path := os.environ.get("INFURNACE_DFLASH_DRAFT"): return Path(path).expanduser()
  if model in MODELS: return MODELS[model][1]
  return _local(MODELS["qwen3.5:4b"][1]) if is_qwen35(model) else None

def is_qwen35(path: str | Path) -> bool:
  return Path(path).is_file() and read_gguf_metadata(path).get("general.architecture") == "qwen35"

def load_model(path: str | Path, max_context: int, device: str | None = None, nv_quant: bool | None = None, *,
               nv_prefill: bool | None = None) -> tuple[Qwen35Model, Qwen35Tokenizer]:
  """Load a dense Qwen3.5 GGUF and tokenizer; NV packed acceleration is automatic when supported.

  nv_quant/nv_prefill overrides exist for benchmarks comparing packed and generic execution.
  Tinygrad owns attention and recurrence; no other model architecture is supported.
  """
  if Path(path).is_dir(): raise ValueError("--model must name a local GGUF file")
  metadata = read_gguf_metadata(path)
  if metadata.get("general.architecture") != "qwen35": raise ValueError("only dense Qwen3.5 GGUF models are supported")
  device = select_device(device)
  # On NVIDIA, the generic path is far too slow to use: say what is missing instead of falling back to it.
  if nv_quant is None and (problem := fast_path_problem(device)): raise ValueError(problem)
  packed = supports_nv_quant(device) if nv_quant is None else nv_quant
  if packed and not supports_nv_quant(device): raise ValueError("packed acceleration requires an NVIDIA device (NV or CUDA) with the CUDA C renderer")
  if nv_prefill and not packed: raise ValueError("NV prefill requires NV quantized acceleration")
  prefill = packed and nv_prefill_supported(device) if nv_prefill is None else nv_prefill
  if prefill and not nv_prefill_supported(device): raise ValueError("NV prefill requires CUDA-C FP16 tensor cores")
  tokenizer = Qwen35Tokenizer.from_gguf(metadata)
  model = Qwen35Model.from_gguf(path, max_context=max_context, device=device, nv_quant=packed, nv_prefill=prefill)
  # Prompt snapshots go just before the last turn header (see Qwen35Model.save_snapshot).
  if (backend := getattr(tokenizer, "_backend", None)) is not None:
    model.turn_token, model.tool_marker = backend._special_tokens.get("<|im_start|>"), backend._special_tokens.get("<tool_response>")
    if model.turn_token is not None: model.user_turn = (model.turn_token, *tokenizer.encode("user\n"))
  return model, tokenizer

def load_dflash(model, draft_path: str | Path):
  """Parallel multirow verification on NV, else the sequential CPU reference."""
  from infurnace.dflash import DFlashDraft, DFlashGenerator
  verifier = 'parallel' if nvidia(model.device) else 'sequential'
  return DFlashGenerator(model, DFlashDraft.from_gguf(draft_path, model), verifier=verifier)
