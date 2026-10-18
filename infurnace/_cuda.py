"""Point tinygrad at NVIDIA's pip-installed CUDA compiler when that is how the user provided it.

infurnace's kernels are CUDA C compiled by NVRTC, which also needs the CUDA headers (cuda_fp16.h). They come with
the CUDA toolkit, or as the pip packages nvidia-cuda-nvrtc-cu12 and nvidia-cuda-runtime-cu12. tinygrad finds a
toolkit by itself; for the pip packages it needs NVRTC_PATH and CUDA_PATH, which this sets when they are installed
and neither variable nor a toolkit is there. It must run before tinygrad loads its CUDA compiler: tinygrad reads
CUDA_PATH once, on import.
"""
import importlib.util
import os
import sys
from pathlib import Path

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
