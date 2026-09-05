from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any
from tinygrad import Tensor, dtypes
from tinygrad.llm.gguf import gguf_load
from infurnace.models import Qwen3Config, qwen3_config_from_gguf
from infurnace.models.manifest import CheckpointManifest, verified_artifact

class WeightMappingError(ValueError):
  """Loaded tensors do not satisfy the exact model contract."""

class WeightPolicy(str, Enum):
  LAZY_FP16 = "lazy-fp16"
  REALIZED_FP16 = "realized-fp16"

@dataclass(frozen=True, slots=True)
class Qwen3Weights:
  config: Qwen3Config
  tensors: Mapping[str, Tensor]
  policy: WeightPolicy

@dataclass(frozen=True, slots=True)
class Qwen3Checkpoint:
  """Parsed GGUF checkpoint: validated weights plus raw GGUF metadata.

  The metadata drives the tokenizer (``SimpleTokenizer.from_gguf_kv``), so the
  checkpoint is parsed once and both products are returned together.
  """
  weights: Qwen3Weights
  metadata: Mapping[str, Any]

_PINNED_IDENTITY = ("qwen3-4b-q4_k_m", 2497280256, "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5", "GGUF", "Q4_K_M",)

def _policy(value: WeightPolicy | str) -> WeightPolicy:
  try: return WeightPolicy(value)
  except ValueError as error: raise WeightMappingError(f"unsupported weight policy: {value}") from error

def _validate_tensors(config: Qwen3Config, tensors: Mapping[str, Tensor]) -> None:
  if not isinstance(tensors, Mapping): raise WeightMappingError("GGUF tensors must be a mapping")
  if any(type(name) is not str for name in tensors): raise WeightMappingError("GGUF tensor names must be strings")
  expected = {spec.name: spec for spec in config.tensors}
  missing, unexpected = expected.keys() - tensors.keys(), tensors.keys() - expected.keys()
  if missing: raise WeightMappingError(f"missing tensors: {', '.join(sorted(missing))}")
  if unexpected: raise WeightMappingError(f"unexpected tensors: {', '.join(sorted(unexpected))}")
  for name, spec in expected.items():
    tensor = tensors[name]
    try: shape, dtype = tuple(tensor.shape), tensor.dtype.name
    except (AttributeError, TypeError) as error: raise WeightMappingError(f"tensor {name} has no valid shape or dtype") from error
    if shape != spec.shape: raise WeightMappingError(f"tensor {name} shape mismatch: expected {spec.shape}, got {shape}")
    if dtype != spec.logical_dtype:
      raise WeightMappingError(f"tensor {name} dtype mismatch: expected {spec.logical_dtype}, got {dtype}")

def map_qwen3_weights(metadata: Mapping[str, Any], tensors: Mapping[str, Tensor],
                      policy: WeightPolicy | str = WeightPolicy.LAZY_FP16) -> Qwen3Weights:
  """Validate and transform tensors parsed from an identity-verified GGUF."""
  selected_policy = _policy(policy)
  config = qwen3_config_from_gguf(metadata)
  _validate_tensors(config, tensors)

  mapped = {spec.name: tensors[spec.name].cast(dtypes.float16) for spec in config.tensors}
  if selected_policy is WeightPolicy.REALIZED_FP16:
    mapped = {name: tensor.contiguous() for name, tensor in mapped.items()}
    Tensor.realize(*mapped.values())
  mapped["output.weight"] = mapped["token_embd.weight"]
  return Qwen3Weights(config, MappingProxyType(mapped), selected_policy)

def load_qwen3_checkpoint(path: str | Path, manifest: CheckpointManifest,
                           policy: WeightPolicy | str = WeightPolicy.LAZY_FP16) -> Qwen3Checkpoint:
  """Verify, parse, validate, and transform the pinned Qwen3 checkpoint once.

  Returns both the validated ``Qwen3Weights`` and the raw GGUF ``metadata``
  (used to build the tokenizer) so the GGUF is only read and parsed a single time.
  """
  identity = manifest.id, manifest.size_bytes, manifest.sha256, manifest.format, manifest.quantization
  if identity != _PINNED_IDENTITY: raise WeightMappingError("manifest is not the pinned Qwen3-4B Q4_K_M checkpoint")
  with verified_artifact(path, manifest) as artifact:
    gguf_tensor = Tensor.empty(artifact.size_bytes, dtype=dtypes.uint8, device=f"disk:{artifact.path}")
    metadata, tensors = gguf_load(gguf_tensor)
    weights = map_qwen3_weights(metadata, tensors, policy)
  return Qwen3Checkpoint(weights, metadata)

def load_qwen3_weights(path: str | Path, manifest: CheckpointManifest,
                        policy: WeightPolicy | str = WeightPolicy.LAZY_FP16) -> Qwen3Weights:
  """Verify, parse, validate, and transform the pinned Qwen3 checkpoint.

  Returns the validated weights; use ``load_qwen3_checkpoint`` when the GGUF
  metadata (for the tokenizer) is also needed.
  """
  return load_qwen3_checkpoint(path, manifest, policy).weights
