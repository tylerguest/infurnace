import pytest

from infurnace import device
from infurnace import cli


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
  monkeypatch.setattr(cli, "select_device", lambda requested: "NV")
  monkeypatch.setattr(cli, "supports_nv_quant", lambda device: True)
  calls = []
  class Model:
    metadata = {}
  def build(path, **kwargs):
    calls.append(kwargs)
    return Model()
  monkeypatch.setattr(cli.Qwen3Model, "from_gguf", build)
  monkeypatch.setattr(cli.Tokenizer, "from_gguf", lambda metadata: None)
  cli._build("model", 16, None)
  assert calls[-1] == {"max_context": 16, "device": "NV", "nv_quant": True}
  cli._build("model", 16, None, False)
  assert "nv_quant" not in calls[-1]
