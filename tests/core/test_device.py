import pytest

from infurnace import device
from infurnace import loader
from infurnace.models import Family

def test_prefer_gpu(monkeypatch):
  monkeypatch.delenv("DEV", raising=False)
  attempts = []
  def probe(name):
    attempts.append(name)
    if name == "NV": raise RuntimeError("unavailable")
  monkeypatch.setattr(device, "_probe", probe)
  assert device.select_device() == "AMD"
  assert attempts == ["NV", "AMD"]

def test_cpu_fallback(monkeypatch):
  monkeypatch.delenv("DEV", raising=False)
  def probe(name):
    if name != "CPU": raise OSError("no driver")
  monkeypatch.setattr(device, "_probe", probe)
  assert device.select_device() == "CPU"

def test_explicit_override(monkeypatch):
  def probe(name): pytest.fail("explicit device must not trigger auto probing")
  monkeypatch.setattr(device, "_probe", probe)
  assert device.select_device("CPU") == "CPU"

def test_auto_acceleration(monkeypatch):
  monkeypatch.setattr(loader, "select_device", lambda requested: "NV")
  monkeypatch.setattr(loader, "supports_nv_quant", lambda device: True)
  monkeypatch.setattr(loader, "fast_path_problem", lambda device: None)
  monkeypatch.setattr(loader, "nv_prefill_supported", lambda device: True)
  calls = []
  class Model:
    metadata = {}
  def build(path, family, **kwargs):
    calls.append(kwargs)
    return Model()
  monkeypatch.setattr(loader, 'read_gguf_metadata', lambda path: {'general.architecture': 'qwen35'})
  monkeypatch.setattr(loader.Model, "from_gguf", build)
  monkeypatch.setattr(loader, "family_for", lambda arch: Family('qwen35', ('qwen35',), tokenizer=lambda metadata: None))
  loader.load_model("model", 16, None)
  assert calls[-1] == {"max_context": 16, "device": "NV", "nv_quant": True, "nv_prefill": True}
  loader.load_model("model", 16, None, False)
  assert calls[-1] == {"max_context": 16, "device": "NV", "nv_quant": False, "nv_prefill": False}

def test_fast_path_problem_names_what_is_missing(monkeypatch):
  assert device.fast_path_problem("CPU") is None
  monkeypatch.setattr(device, "supports_nv_quant", lambda name: False)
  problem = device.fast_path_problem("NV")
  assert "NVRTC not found" in problem and "nvidia-cuda-nvrtc-cu12" in problem and "CUDA toolkit" in problem
  def failing_compile(*args, **kwargs):
    raise RuntimeError('Nvrtc Error 6, NVRTC_ERROR_COMPILATION\n<null>(5): catastrophic error: cannot open source file "cuda_fp16.h"\n')
  monkeypatch.setattr(device, "supports_nv_quant", lambda name: True)
  monkeypatch.setattr(device, "Tensor", failing_compile)
  problem = device.fast_path_problem("NV")
  assert 'cannot open source file "cuda_fp16.h"' in problem and "nvidia-cuda-runtime-cu12" in problem

def test_probe_failures_of_any_kind_move_on(monkeypatch):
  # tinygrad reports a missing interface as an ExceptionGroup; probing must try the next backend, not crash.
  monkeypatch.delenv("DEV", raising=False)
  def probe(name):
    if name == "NV": raise RuntimeError("no compiler")
    if name == "AMD": raise ExceptionGroup("No interface for AMD:0 is available", [FileNotFoundError("/dev/kfd")])
  monkeypatch.setattr(device, "_probe", probe)
  assert device.select_device() == "METAL"

@pytest.fixture
def pip_compiler(tmp_path, monkeypatch):
  """Fake nvidia.cuda_nvrtc and nvidia.cuda_runtime pip packages, no toolkit, no variables set."""
  nvrtc, runtime = tmp_path / "cuda_nvrtc", tmp_path / "cuda_runtime"
  (nvrtc / "lib").mkdir(parents=True)
  for name in ("libnvrtc-builtins.so.12.9", "libnvrtc-builtins.alt.so.12.9", "libnvrtc.alt.so.12"): (nvrtc / "lib" / name).touch()
  (nvrtc / "lib" / "libnvrtc.so.12").touch()
  (runtime / "include").mkdir(parents=True)
  (runtime / "include" / "cuda_fp16.h").touch()
  monkeypatch.setattr(device, "_package", {"cuda_nvrtc": nvrtc, "cuda_runtime": runtime}.get)
  monkeypatch.setattr(device, "_toolkit", lambda: False)
  monkeypatch.setattr(device.sys, "platform", "linux")
  for name in ("NVRTC_PATH", "CUDA_PATH"):  # recorded, so what use_pip_compiler sets is undone after the test
    monkeypatch.setenv(name, "")
    monkeypatch.delenv(name)
  return nvrtc, runtime

def test_points_tinygrad_at_the_pip_packages(pip_compiler, monkeypatch):
  nvrtc, runtime = pip_compiler
  assert device.use_pip_compiler()
  assert device.os.environ["NVRTC_PATH"] == str(nvrtc / "lib" / "libnvrtc.so.12")  # not the builtins or .alt libraries
  assert device.os.environ["CUDA_PATH"] == str(runtime)

@pytest.mark.parametrize("setup", ["toolkit", "variable", "no_runtime", "no_header"])
def test_leaves_other_setups_alone(pip_compiler, monkeypatch, setup):
  nvrtc, runtime = pip_compiler
  if setup == "toolkit": monkeypatch.setattr(device, "_toolkit", lambda: True)
  if setup == "variable": monkeypatch.setenv("CUDA_PATH", "/opt/cuda")
  if setup == "no_runtime": monkeypatch.setattr(device, "_package", {"cuda_nvrtc": nvrtc}.get)
  if setup == "no_header": (runtime / "include" / "cuda_fp16.h").unlink()
  assert not device.use_pip_compiler()
  assert "NVRTC_PATH" not in device.os.environ
