"""Prefer a usable accelerator; hardware preference is not a performance benchmark."""
import os
import shutil
import subprocess

from tinygrad import Device, Tensor
from tinygrad.renderer.cstyle import CUDARenderer

# infurnace's kernels are CUDA C, compiled by NVRTC (tinygrad's CUDA C renderer). NVRTC and the cuda_fp16.h
# header ship with the CUDA toolkit, not the driver; without them tinygrad picks its PTX renderer and infurnace
# would run the generic path, far too slow to use. How to provide them is the user's choice.
NVRTC_HELP = ("infurnace's fast kernels need NVIDIA's NVRTC compiler and CUDA headers. Install the CUDA toolkit, "
              "or point tinygrad at an existing copy: NVRTC_PATH=/path/to/libnvrtc.so CUDA_PATH=/dir/containing/include "
              "(for example from the nvidia-cuda-nvrtc-cu12 and nvidia-cuda-runtime-cu12 pip packages)")

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

def fast_path_problem(device: str) -> str | None:
  """Why infurnace's kernels can't run on this NVIDIA device, or None. Compiles one FP16 kernel, which needs
  both NVRTC and cuda_fp16.h."""
  if not nvidia(device): return None
  if not supports_nv_quant(device): return f"{device} has no CUDA C compiler (NVRTC not found). {NVRTC_HELP}."
  try: (Tensor([1.0], device=device).half() * 2).float().tolist()
  except Exception as exc:  # NVRTC reports a missing header as a compile error
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    detail = next((line for line in lines if "error" in line.lower() and ":" in line), lines[0] if lines else type(exc).__name__)
    return f"compiling a kernel on {device} failed ({detail[:200]}). {NVRTC_HELP}."
  return None

def free_vram(device: str) -> int | None:
  """Free memory of the NVIDIA GPU in bytes (nvidia-smi, before infurnace allocates anything), or None."""
  if not nvidia(device): return None
  if (smi := shutil.which("nvidia-smi")) is None: return None
  index = device.split(":")[1] if ":" in device else "0"
  try: out = subprocess.run([smi, "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", index],
                            capture_output=True, text=True, timeout=10).stdout.strip()
  except (OSError, subprocess.SubprocessError): return None
  return int(out.splitlines()[0]) * 2**20 if out[:1].isdigit() else None
