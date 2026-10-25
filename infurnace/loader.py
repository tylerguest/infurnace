"""Loading a model of a supported family (infurnace/models) and its optional DFlash draft, and sizing its context.

Names and downloads are in catalog.py.
"""
from __future__ import annotations
import sys
from pathlib import Path
from infurnace.catalog import details, draft_for
from infurnace.device import fast_path_problem, nvidia, select_device, supports_nv_quant
from infurnace.gguf import Q2_0, parse_gguf, read_gguf_metadata
from infurnace.kernels.quant_gemm import nv_prefill_supported
from infurnace.engine import Model
from infurnace.models import family_for

def model_info(name: str, path: Path, dflash: bool) -> dict:
  """What the server reports for the loaded model (/api/ps, /api/show)."""
  return {'size': Path(path).stat().st_size, 'dflash': dflash, 'details': details(name, path)}

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

def dflash_draft(model: str | Path) -> str | Path | None:
  """The draft override, or the draft of a catalog model (by name, or by the path of its file)."""
  return draft_for(str(model))

def load_model(path: str | Path, max_context: int, device: str | None = None, nv_quant: bool | None = None, *,
               nv_prefill: bool | None = None) -> tuple[Model, object]:
  """Load a GGUF model of a supported family and its tokenizer; NV packed acceleration is automatic when supported.

  nv_quant/nv_prefill overrides exist for benchmarks comparing packed and generic execution.
  """
  if Path(path).is_dir(): raise ValueError("--model must name a local GGUF file")
  metadata, entries = parse_gguf(path)
  family = family_for(metadata.get("general.architecture"))
  device = select_device(device)
  # On NVIDIA, the generic path is far too slow to use: say what is missing instead of falling back to it.
  if nv_quant is None and (problem := fast_path_problem(device)): raise ValueError(problem)
  packed = supports_nv_quant(device) if nv_quant is None else nv_quant
  if packed and not supports_nv_quant(device): raise ValueError("packed acceleration requires an NVIDIA device (NV or CUDA) with the CUDA C renderer")
  # Ternary (Q2_0) weights and Hadamard-folded inputs exist only in infurnace's packed path: tinygrad's generic
  # model can't read Q2_0, and without the rotations it would compute a different model.
  if not packed and (metadata.get("prism.hadamard.version") is not None or any(e.ggml_type == Q2_0 for e in entries.values())):
    raise ValueError(f"{Path(path).name} has ternary (Q2_0) weights with Hadamard-rotated inputs, which need an NVIDIA GPU "
                     "with infurnace's kernels (the NV or CUDA device and NVIDIA's CUDA compiler)")
  if nv_prefill and not packed: raise ValueError("NV prefill requires NV quantized acceleration")
  prefill = packed and nv_prefill_supported(device) if nv_prefill is None else nv_prefill
  if prefill and not nv_prefill_supported(device): raise ValueError("NV prefill requires CUDA-C FP16 tensor cores")
  tokenizer = family.tokenizer(metadata)
  model = Model.from_gguf(path, max_context=max_context, device=device, nv_quant=packed, nv_prefill=prefill, family=family)
  # Prompt snapshots go just before the last turn header (see Model.save_snapshot).
  model.turn_token, model.user_turn, model.tool_marker = family.turn_markers(tokenizer)
  return model, tokenizer

def load_dflash(model, draft_path: str | Path):
  """Parallel multirow verification on NV, else the sequential CPU reference."""
  from infurnace.dflash import DFlashDraft, DFlashGenerator
  verifier = 'parallel' if nvidia(model.device) else 'sequential'
  return DFlashGenerator(model, DFlashDraft.from_gguf(draft_path, model), verifier=verifier)
