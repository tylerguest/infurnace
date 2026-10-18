"""Prefer a usable accelerator; hardware preference is not a performance benchmark."""
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tinygrad import Device, Tensor
from tinygrad.renderer.cstyle import CUDARenderer

# infurnace's kernels are CUDA C, compiled by NVRTC (tinygrad's CUDA C renderer). NVRTC and the cuda_fp16.h
# header ship with the CUDA toolkit, not the driver; without them tinygrad picks its PTX renderer and infurnace
# would run the generic path, far too slow to use. How to provide them is the user's choice: the toolkit, or the
# pip packages nvidia-cuda-nvrtc-cu12 and nvidia-cuda-runtime-cu12, which tinygrad finds through NVRTC_PATH and
# CUDA_PATH. use_pip_compiler sets those when only the pip packages are there; it runs on import, before tinygrad
# loads its CUDA compiler (tinygrad reads CUDA_PATH once, then).
NVRTC_HELP = ("infurnace's fast kernels need NVIDIA's CUDA compiler (NVRTC) and headers. Install the CUDA toolkit, "
              "or just those two pieces: python3 -m pip install nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12")

def _toolkit() -> bool:
  """A toolkit tinygrad finds by itself: NVRTC where it looks for it, and cuda_fp16.h in its include paths."""
  if sys.platform == "win32": return False  # tinygrad finds NVRTC on Windows through PATH only
  libraries = [Path(p) for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p]
  libraries += [Path(prefix) / "cuda/targets" / f"{os.uname().machine}-linux" / "lib" for prefix in ("/opt", "/usr/local")]
  libraries += [Path(p) for p in ("/usr/lib64", "/usr/lib", "/usr/local/lib", "/lib", "/lib64", f"/lib/{os.uname().machine}-linux-gnu",
                                  f"/usr/lib/{os.uname().machine}-linux-gnu")]
  nvrtc = any(any(d.glob("libnvrtc.so*")) for d in libraries if d.is_dir())
  headers = any((Path(d) / "cuda_fp16.h").is_file() for d in ("/usr/local/cuda/include", "/usr/include", "/opt/cuda/include"))
  return nvrtc and headers

def _package(name: str) -> Path | None:
  try: spec = importlib.util.find_spec(f"nvidia.{name}")
  except (ImportError, ValueError): return None
  locations = list(spec.submodule_search_locations or []) if spec else []
  return Path(locations[0]) if locations else None

def use_pip_compiler() -> bool:
  """Set NVRTC_PATH and CUDA_PATH to NVIDIA's pip packages when they are installed and nothing else is set up.
  True when it did."""
  if os.environ.get("NVRTC_PATH") or os.environ.get("CUDA_PATH") or _toolkit(): return False
  nvrtc, runtime = _package("cuda_nvrtc"), _package("cuda_runtime")
  if nvrtc is None or runtime is None or not (runtime / "include" / "cuda_fp16.h").is_file(): return False
  # The regular library: the package also ships libnvrtc-builtins (which NVRTC loads itself) and .alt variants.
  pattern = "bin/nvrtc64_*.dll" if sys.platform == "win32" else "lib/libnvrtc.so*"
  library = next((p for p in sorted(nvrtc.glob(pattern)) if "builtins" not in p.name and ".alt" not in p.name), None)
  if library is None: return False
  os.environ["NVRTC_PATH"], os.environ["CUDA_PATH"] = str(library), str(runtime)
  return True

use_pip_compiler()

def _probe(device: str) -> None:
  # Check actual execution, not just that a driver module can be imported.
  (Tensor([1.0], device=device) + 1).tolist()

def nvidia_gpu_present() -> bool:
  """An NVIDIA GPU the driver lists (nvidia-smi -L), whether or not tinygrad can open it."""
  if (smi := shutil.which("nvidia-smi")) is None: return False
  try: return subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=10).stdout.lstrip().startswith("GPU ")
  except (OSError, subprocess.SubprocessError): return False

NVIDIA_RETRY_SECONDS = 10.0

class DeviceError(RuntimeError): pass

def select_device(requested: str | None = None) -> str:
  if requested is not None: return requested
  # Respect tinygrad's explicit environment configuration, including renderer selection.
  if os.environ.get("DEV"): return Device.DEFAULT
  failures = []
  for backend in ("NV", "AMD", "METAL", "CUDA", "CL", "CPU"):
    try:
      _probe(backend)
      return backend
    except Exception as exc:  # any failure (tinygrad raises ExceptionGroups for missing interfaces) means: try the next
      failures.append(f"{backend}: {exc}")
    if backend == "CUDA" and nvidia_gpu_present():
      # With an NVIDIA GPU, the CPU would be minutes per reply: wait for the GPU instead (a process that just
      # exited may still be releasing it), then say what failed.
      deadline = time.monotonic() + NVIDIA_RETRY_SECONDS
      while time.monotonic() < deadline:
        time.sleep(1.0)
        for nvidia_backend in ("NV", "CUDA"):
          try:
            _probe(nvidia_backend)
            return nvidia_backend
          except Exception: pass
      raise DeviceError("the NVIDIA GPU is present but tinygrad could not open it ("
                         + "; ".join(f for f in failures if f.startswith(("NV", "CUDA"))) + "). If another program just released it, "
                         "try again; pass --device CPU to run on the CPU anyway (very slow).")
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
