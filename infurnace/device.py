"""Prefer a usable accelerator; hardware preference is not a performance benchmark."""
import os

from tinygrad import Device, Tensor
from tinygrad.renderer.cstyle import CUDARenderer


def _probe(device: str) -> None:
  # Check actual execution, not just that a driver module can be imported.
  (Tensor([1.0], device=device) + 1).tolist()


def select_device(requested: str | None = None) -> str:
  if requested is not None: return requested
  # Respect tinygrad's explicit environment configuration, including renderer selection.
  if os.environ.get("DEV"): return Device.DEFAULT
  failures = []
  for backend in ("NV", "AMD", "METAL", "CUDA", "CL", "CPU"):
    try:
      _probe(backend)
      return backend
    except (RuntimeError, OSError, ImportError, AssertionError, MemoryError) as exc:
      failures.append(f"{backend}: {exc}")
  raise RuntimeError("no usable compute device: " + "; ".join(failures))


def supports_nv_quant(device: str) -> bool:
  return device.split(":")[0].upper() == "NV" and isinstance(Device[device].renderer, CUDARenderer)
