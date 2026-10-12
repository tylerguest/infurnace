from types import SimpleNamespace

import pytest

from infurnace import loader


def test_default_model_environment(monkeypatch):
  monkeypatch.setenv("INFURNACE_MODEL", "~/custom.gguf")
  from pathlib import Path
  assert loader.default_model() == str(Path.home() / "custom.gguf")


def test_default_model_user_storage(monkeypatch, tmp_path):
  monkeypatch.delenv("INFURNACE_MODEL", raising=False)
  monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
  path = tmp_path / "infurnace/models/Qwen3-4B-Q4_K_M.gguf"
  path.parent.mkdir(parents=True)
  path.touch()
  monkeypatch.chdir(tmp_path)
  assert loader.default_model() == str(path)


def test_build_uses_loaded_metadata(monkeypatch):
  metadata = {"test": True}
  model = SimpleNamespace(metadata=metadata)
  calls = []
  def from_file(path, **kwargs):
    calls.append((path, kwargs))
    return model
  monkeypatch.setattr(loader.Qwen3Model, "from_gguf", from_file)
  monkeypatch.setattr(loader.Tokenizer, "from_gguf", lambda md: md)
  result, tokenizer = loader.load_model("model", 16, "CPU")
  assert result is model and tokenizer is metadata
  assert calls == [("model", {"max_context": 16, "device": "CPU"})]


def test_build_prefill_forwarding_and_backend_gate(monkeypatch):
  monkeypatch.setattr(loader, "select_device", lambda device: device)
  monkeypatch.setattr(loader, "supports_nv_quant", lambda device: device == "NV")
  monkeypatch.setattr(loader, "nv_prefill_supported", lambda device: device == "NV")
  calls = []
  model = SimpleNamespace(metadata={})
  def from_file(path, **kwargs):
    calls.append(kwargs)
    return model
  monkeypatch.setattr(loader.Qwen3Model, "from_gguf", from_file)
  monkeypatch.setattr(loader.Tokenizer, "from_gguf", lambda md: md)
  assert loader.load_model("model", 64, "NV", nv_quant=True, nv_prefill=True)[0] is model
  assert calls == [{"max_context": 64, "device": "NV", "nv_quant": True, "nv_prefill": True,
                    "nv_prepared_weights": True, "nv_fused_attention": True}]
  with pytest.raises(ValueError, match="requires NV quantized"):
    loader.load_model("model", 64, "CPU", nv_quant=False, nv_prefill=True)
  assert len(calls) == 1


@pytest.mark.parametrize("quant_supported,prefill_supported,quant_override,prefill_override,expected", [
  (True, True, None, None, {"nv_quant": True, "nv_prefill": True, "nv_prepared_weights": True, "nv_fused_attention": True}),
  (True, True, None, False, {"nv_quant": True, "nv_prepared_weights": True, "nv_fused_attention": True}),
  (True, False, None, None, {"nv_quant": True, "nv_prepared_weights": True, "nv_fused_attention": True}),
  (False, False, None, None, {}),
  (True, True, False, None, {}),
])
def test_build_automatic_prefill(monkeypatch, quant_supported, prefill_supported, quant_override, prefill_override, expected):
  monkeypatch.setattr(loader, "select_device", lambda device: device)
  monkeypatch.setattr(loader, "supports_nv_quant", lambda device: quant_supported)
  monkeypatch.setattr(loader, "nv_prefill_supported", lambda device: prefill_supported)
  calls = []
  def from_file(path, **kwargs):
    calls.append(kwargs)
    return SimpleNamespace(metadata={})
  monkeypatch.setattr(loader.Qwen3Model, "from_gguf", from_file)
  monkeypatch.setattr(loader.Tokenizer, "from_gguf", lambda md: md)
  loader.load_model("model", 64, "test-device", quant_override, nv_prefill=prefill_override)
  assert calls == [{"max_context": 64, "device": "test-device", **expected}]


@pytest.mark.parametrize("fused", [False, True])
def test_build_attention_forwarding_and_gate(monkeypatch, fused):
  monkeypatch.setattr(loader, "select_device", lambda device: device)
  monkeypatch.setattr(loader, "supports_nv_quant", lambda device: device == "NV")
  monkeypatch.setattr(loader, "nv_prefill_supported", lambda device: False)
  calls = []
  def from_file(path, **kwargs):
    calls.append(kwargs)
    return SimpleNamespace(metadata={})
  monkeypatch.setattr(loader.Qwen3Model, "from_gguf", from_file)
  monkeypatch.setattr(loader.Tokenizer, "from_gguf", lambda md: md)
  option = {"nv_fused_attention": True} if fused else {"nv_attention": True}
  loader.load_model("model", 64, "NV", **option)
  assert calls == [{"max_context": 64, "device": "NV", "nv_quant": True, "nv_prepared_weights": True, **option}]
  with pytest.raises(ValueError, match="requires NV quantized"):
    loader.load_model("model", 64, "CPU", **option)
  assert len(calls) == 1


def test_build_explicit_prefill_rejects_unsupported_tensor_cores(monkeypatch):
  monkeypatch.setattr(loader, "select_device", lambda device: device)
  monkeypatch.setattr(loader, "supports_nv_quant", lambda device: True)
  monkeypatch.setattr(loader, "nv_prefill_supported", lambda device: False)
  with pytest.raises(ValueError, match="requires CUDA-C FP16 tensor cores"):
    loader.load_model("model", 64, "NV", nv_prefill=True)
