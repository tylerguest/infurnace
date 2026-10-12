"""Choosing and loading models: the target GGUF, its tokenizer, and optional speculative drafts.

`--model` takes a name from MODELS or a path to a local GGUF file. A named model and its DFlash
draft are downloaded into user storage ($XDG_DATA_HOME/infurnace/models) on first use; a copy
already there, or in an editable checkout's models/ directory, is used as is.
"""
from __future__ import annotations
import os
from pathlib import Path
from tinygrad.helpers import fetch
from infurnace.device import select_device, supports_nv_quant
from infurnace.kernels.quant_gemm import nv_prefill_supported
from infurnace.models.qwen3 import Qwen3Model
from infurnace.tokenizer import Tokenizer

# name: (model URL, DFlash draft URL or None). The first entry is the default.
MODELS = {
  "qwen3.5:4b": ("https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf",
                 "https://huggingface.co/Anbeeld/Qwen3.5-4B-DFlash-GGUF/resolve/a79bafa7f383f8627f925250f547fe1ebbc43ff6/qwen35-4b-dflash-Q4_K_M.gguf"),
  "qwen3:4b": ("https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf", None),
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
  """The DFlash draft for `model`: INFURNACE_DFLASH_DRAFT, a named model's draft (downloaded when
  used), or for a local Qwen3.5 file the Qwen3.5-4B draft if it is already installed."""
  if path := os.environ.get("INFURNACE_DFLASH_DRAFT"): return Path(path).expanduser()
  if model in MODELS: return MODELS[model][1]
  return _local(MODELS["qwen3.5:4b"][1]) if is_qwen35(model) else None

def is_qwen35(path: str | Path) -> bool:
  from infurnace.gguf import read_gguf_metadata
  return Path(path).is_file() and read_gguf_metadata(path).get("general.architecture") == "qwen35"

def load_model(path: str, max_context: int, device: str | None = None, nv_quant: bool | None = None, *,
               nv_prefill: bool | None = None, nv_attention: bool = False, nv_fused_attention: bool | None = None,
               nv_prepared_weights: bool | None = None) -> tuple[Qwen3Model, Tokenizer]:
  """Load a Qwen3 or Qwen3.5 GGUF and its tokenizer.

  NV acceleration is chosen automatically; the `nv_*` options force a choice and exist for
  benchmarks and tests that compare variants (the CLI does not expose them)."""
  device = select_device(device)
  if Path(path).is_dir(): raise ValueError("--model must name a local GGUF file")
  if is_qwen35(path):
    from infurnace.gguf import read_gguf_metadata
    from infurnace.models.qwen35 import Qwen35Model
    from infurnace.tokenizer import Qwen35Tokenizer
    if any((nv_attention, nv_fused_attention, nv_prepared_weights)):
      raise ValueError("Qwen3.5 supports only nv_quant/nv_prefill; upstream tinygrad owns attention and recurrence")
    packed = supports_nv_quant(device) if nv_quant is None else nv_quant
    if packed and not supports_nv_quant(device): raise ValueError("NV quantized acceleration requires NV with the CUDA C renderer")
    if nv_prefill and not packed: raise ValueError("NV prefill requires NV quantized acceleration")
    tokenizer = Qwen35Tokenizer.from_gguf(read_gguf_metadata(path))
    return Qwen35Model.from_gguf(path, max_context=max_context, device=device, nv_quant=packed, nv_prefill=nv_prefill), tokenizer

  enabled = supports_nv_quant(device) if nv_quant is None else nv_quant
  if enabled and not supports_nv_quant(device): raise ValueError("NV quantized acceleration requires NV with the CUDA C renderer")
  options: dict = {"max_context": max_context, "device": device}
  if enabled: options["nv_quant"] = True
  prepared = enabled if nv_prepared_weights is None else nv_prepared_weights
  if prepared and not enabled: raise ValueError("prepared weights require NV quantized acceleration")
  if enabled: options["nv_prepared_weights"] = prepared
  prefill = enabled and nv_prefill_supported(device) if nv_prefill is None else nv_prefill
  if prefill:
    if not enabled: raise ValueError("NV prefill requires NV quantized acceleration")
    if not nv_prefill_supported(device): raise ValueError("NV prefill requires CUDA-C FP16 tensor cores")
    options["nv_prefill"] = True
  fused = enabled and not nv_attention if nv_fused_attention is None else nv_fused_attention
  if nv_attention or fused:
    if not enabled: raise ValueError("NV attention requires NV quantized acceleration")
    if nv_attention: options["nv_attention"] = True
    if fused: options["nv_fused_attention"] = True
  model = Qwen3Model.from_gguf(path, **options)
  return model, Tokenizer.from_gguf(model.metadata)

def load_dflash(model, draft_path: str | Path):
  """DFlash generator for a Qwen3.5 target: parallel multirow verification on NV, else sequential."""
  from infurnace.dflash import DFlashDraft, DFlashGenerator
  verifier = 'parallel' if model.device.split(':')[0] == 'NV' else 'sequential'
  return DFlashGenerator(model, DFlashDraft.from_gguf(draft_path, model), verifier=verifier)
