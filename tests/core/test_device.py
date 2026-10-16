import pytest

from infurnace import device
from infurnace import loader

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
  def build(path, **kwargs):
    calls.append(kwargs)
    return Model()
  monkeypatch.setattr(loader, 'read_gguf_metadata', lambda path: {'general.architecture': 'qwen35'})
  monkeypatch.setattr(loader.Qwen35Model, "from_gguf", build)
  monkeypatch.setattr(loader.Qwen35Tokenizer, "from_gguf", lambda metadata: None)
  loader.load_model("model", 16, None)
  assert calls[-1] == {"max_context": 16, "device": "NV", "nv_quant": True, "nv_prefill": True}
  loader.load_model("model", 16, None, False)
  assert calls[-1] == {"max_context": 16, "device": "NV", "nv_quant": False, "nv_prefill": False}

def test_fast_path_problem_names_what_is_missing(monkeypatch):
  assert device.fast_path_problem("CPU") is None
  monkeypatch.setattr(device, "supports_nv_quant", lambda name: False)
  problem = device.fast_path_problem("NV")
  assert "NVRTC not found" in problem and "NVRTC_PATH" in problem and "CUDA toolkit" in problem
  def failing_compile(*args, **kwargs):
    raise RuntimeError('Nvrtc Error 6, NVRTC_ERROR_COMPILATION\n<null>(5): catastrophic error: cannot open source file "cuda_fp16.h"\n')
  monkeypatch.setattr(device, "supports_nv_quant", lambda name: True)
  monkeypatch.setattr(device, "Tensor", failing_compile)
  problem = device.fast_path_problem("NV")
  assert 'cannot open source file "cuda_fp16.h"' in problem and "CUDA_PATH" in problem
