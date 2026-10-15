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

def nvidia(device: str) -> bool:
  """An NVIDIA backend: tinygrad's NV (its own driver interface, Linux) or CUDA (the CUDA driver API, also on
  Windows). infurnace's kernels are CUDA C and run on either."""
  return isinstance(device, str) and device.split(":")[0].upper() in ("NV", "CUDA")

def supports_nv_quant(device: str) -> bool:
  # The kernels are CUDA C: tinygrad's PTX or NVCC renderers can't compile their intrinsics.
  return nvidia(device) and isinstance(Device[device].renderer, CUDARenderer)
