"""Choosing and loading Qwen3.5-4B and its optional DFlash draft.

`--model` takes a name from MODELS or a path to a local Qwen3.5 GGUF file. A named model and
its DFlash draft are downloaded into user storage ($XDG_DATA_HOME/infurnace/models) on first
use; a copy already there, or in an editable checkout's models/ directory, is used as is.
"""
from __future__ import annotations
import os
from pathlib import Path
from tinygrad.helpers import fetch
from infurnace.device import select_device, supports_nv_quant
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
  packed = supports_nv_quant(device) if nv_quant is None else nv_quant
  if packed and not supports_nv_quant(device): raise ValueError("NV quantized acceleration requires NV with the CUDA C renderer")
  if nv_prefill and not packed: raise ValueError("NV prefill requires NV quantized acceleration")
  prefill = packed and nv_prefill_supported(device) if nv_prefill is None else nv_prefill
  if prefill and not nv_prefill_supported(device): raise ValueError("NV prefill requires CUDA-C FP16 tensor cores")
  tokenizer = Qwen35Tokenizer.from_gguf(metadata)
  model = Qwen35Model.from_gguf(path, max_context=max_context, device=device, nv_quant=packed, nv_prefill=prefill)
  # Prompt snapshots go just before the last turn header (see Qwen35Model.save_snapshot).
  if (backend := getattr(tokenizer, "_backend", None)) is not None: model.turn_token = backend._special_tokens.get("<|im_start|>")
  return model, tokenizer

def load_dflash(model, draft_path: str | Path, temperature: float = 0.0):
  """Parallel multirow verification on NV, else the sequential CPU reference."""
  from infurnace.dflash import DFlashDraft, DFlashGenerator
  verifier = 'parallel' if model.device.split(':')[0] == 'NV' else 'sequential'
  return DFlashGenerator(model, DFlashDraft.from_gguf(draft_path, model), verifier=verifier, temperature=temperature)
