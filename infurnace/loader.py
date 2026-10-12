"""Finding and loading models: the target GGUF, its tokenizer, and optional speculative drafts.

Model files are looked up in `INFURNACE_MODEL` / `INFURNACE_DFLASH_DRAFT`, then user storage
($XDG_DATA_HOME/infurnace/models), then an editable checkout's models/ directory. The default
target is Qwen3.5-4B (with its DFlash draft), else Qwen3-4B.
"""
from __future__ import annotations
import os
from pathlib import Path
from infurnace.device import select_device, supports_nv_quant
from infurnace.kernels.quant_gemm import nv_prefill_supported
from infurnace.models.qwen3 import Qwen3Model
from infurnace.tokenizer import Tokenizer

# (file name, subdirectories of a checkout's models/ to look in)
TARGETS = (("Qwen3.5-4B-Q4_K_M.gguf", ("", "qwen35-validation")), ("Qwen3-4B-Q4_K_M.gguf", ("",)))
DFLASH_DRAFT = ("qwen35-4b-dflash-Q4_K_M.gguf", ("", "dflash-qwen35"))


def _user_models() -> Path:
  return Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))).expanduser() / "infurnace/models"


def _find(candidates: tuple[tuple[str, tuple[str, ...]], ...]) -> Path | None:
  """User storage first (every candidate), then an editable checkout's models/."""
  checkout = Path(__file__).resolve().parent.parent / "models"
  for path in [_user_models() / name for name, _ in candidates] + [checkout / sub / name for name, subs in candidates for sub in subs]:
    if path.is_file(): return path
  return None


def default_model() -> str:
  if path := os.environ.get("INFURNACE_MODEL"): return str(Path(path).expanduser())
  if found := _find(TARGETS): return str(found)
  return str(_user_models() / TARGETS[-1][0])


def default_dflash_draft() -> Path | None:
  if path := os.environ.get("INFURNACE_DFLASH_DRAFT"): return Path(path).expanduser()
  return _find((DFLASH_DRAFT,))


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


def load_eagle(model, checkpoint: Path):
  """EAGLE-3 generator for a Qwen3 target on NV (experimental)."""
  from infurnace.eagle import EagleDraft
  from infurnace.eagle.generate import EagleGenerator
  from infurnace.eagle.verify import BatchedVerifier
  BatchedVerifier(model)  # fail eligibility checks before loading draft weights
  return EagleGenerator(model, EagleDraft.from_checkpoint(checkpoint, model), fast=True)
