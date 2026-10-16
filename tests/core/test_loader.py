from pathlib import Path
from types import SimpleNamespace

import pytest

from infurnace import loader

@pytest.fixture
def tiny_models(monkeypatch, tmp_path):
  monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
  monkeypatch.setattr(loader, "MODELS", {"tiny:1b": ("https://example.com/m/tiny.gguf", "https://example.com/m/tiny-draft.gguf")})
  fetched = []
  monkeypatch.setattr(loader, "fetch", lambda url, name: fetched.append((url, name)) or name)
  return tmp_path / "infurnace/models", fetched

def test_name_downloads_into_user_storage(tiny_models):
  storage, fetched = tiny_models
  assert loader.fetch_model("tiny:1b") == storage / "tiny.gguf"
  assert fetched == [("https://example.com/m/tiny.gguf", storage / "tiny.gguf")]

def test_name_reuses_existing_download(tiny_models):
  storage, fetched = tiny_models
  storage.mkdir(parents=True)
  (storage / "tiny.gguf").touch()
  assert loader.fetch_model("tiny:1b") == storage / "tiny.gguf" and not fetched

def test_path_is_used_as_is(tiny_models):
  _, fetched = tiny_models
  assert loader.fetch_model("~/custom.gguf") == Path.home() / "custom.gguf" and not fetched

def test_dflash_draft_for_names_and_paths(tiny_models, monkeypatch, tmp_path):
  monkeypatch.delenv("INFURNACE_DFLASH_DRAFT")
  assert loader.dflash_draft("tiny:1b") == "https://example.com/m/tiny-draft.gguf"
  assert loader.dflash_draft(str(tmp_path / "missing.gguf")) is None
  monkeypatch.setenv("INFURNACE_DFLASH_DRAFT", "~/draft.gguf")
  assert loader.dflash_draft("tiny:1b") == Path.home() / "draft.gguf"

def test_only_supported_preset():
  assert list(loader.MODELS) == ['qwen3.5:4b']
  assert loader.MODELS['qwen3.5:4b'][1].endswith('qwen35-4b-dflash-Q4_K_M.gguf')

@pytest.fixture
def model_loader(monkeypatch):
  metadata = {'general.architecture': 'qwen35'}
  monkeypatch.setattr(loader, 'read_gguf_metadata', lambda path: metadata)
  monkeypatch.setattr(loader, 'select_device', lambda device: device)
  monkeypatch.setattr(loader, 'supports_nv_quant', lambda device: device == 'NV')
  monkeypatch.setattr(loader, 'nv_prefill_supported', lambda device: device == 'NV')
  monkeypatch.setattr(loader, 'fast_path_problem', lambda device: None)
  calls = []
  model = SimpleNamespace(metadata=metadata)
  def load(path, **kwargs):
    calls.append((path, kwargs))
    return model
  monkeypatch.setattr(loader.Qwen35Model, 'from_gguf', load)
  monkeypatch.setattr(loader.Qwen35Tokenizer, 'from_gguf', lambda md: md)
  return metadata, model, calls

def test_build_uses_validated_metadata(model_loader):
  metadata, model, calls = model_loader
  result, tokenizer = loader.load_model('model', 16, 'CPU')
  assert result is model and tokenizer is metadata
  assert calls == [('model', {'max_context': 16, 'device': 'CPU', 'nv_quant': False, 'nv_prefill': False})]

@pytest.mark.parametrize('architecture', ['qwen3', 'llama', 'qwen35moe', None])
def test_rejects_other_architectures_before_loading(model_loader, architecture):
  metadata, _, calls = model_loader
  metadata['general.architecture'] = architecture
  with pytest.raises(ValueError, match='only dense Qwen3.5'):
    loader.load_model('model', 16, 'CPU')
  assert not calls

@pytest.mark.parametrize('quant_supported,prefill_supported,quant_override,prefill_override,packed,prefill', [
  (True, True, None, None, True, True),
  (True, True, None, False, True, False),
  (True, False, None, None, True, False),
  (False, False, None, None, False, False),
  (True, True, False, None, False, False),
])
def test_automatic_acceleration(model_loader, monkeypatch, quant_supported, prefill_supported, quant_override,
                                prefill_override, packed, prefill):
  _, _, calls = model_loader
  monkeypatch.setattr(loader, 'supports_nv_quant', lambda device: quant_supported)
  monkeypatch.setattr(loader, 'nv_prefill_supported', lambda device: prefill_supported)
  loader.load_model('model', 64, 'test-device', quant_override, nv_prefill=prefill_override)
  assert calls == [('model', {'max_context': 64, 'device': 'test-device', 'nv_quant': packed, 'nv_prefill': prefill})]

def test_prefill_forwarding_and_backend_gate(model_loader):
  _, model, calls = model_loader
  assert loader.load_model('model', 64, 'NV', nv_quant=True, nv_prefill=True)[0] is model
  assert calls == [('model', {'max_context': 64, 'device': 'NV', 'nv_quant': True, 'nv_prefill': True})]
  with pytest.raises(ValueError, match='requires NV quantized'):
    loader.load_model('model', 64, 'CPU', nv_quant=False, nv_prefill=True)
  with pytest.raises(ValueError, match='requires an NVIDIA device'):
    loader.load_model('model', 64, 'CPU', nv_quant=True)
  assert len(calls) == 1

def test_explicit_prefill_rejects_unsupported_tensor_cores(model_loader, monkeypatch):
  _, _, calls = model_loader
  monkeypatch.setattr(loader, 'nv_prefill_supported', lambda device: False)
  with pytest.raises(ValueError, match='requires CUDA-C FP16 tensor cores'):
    loader.load_model('model', 64, 'NV', nv_prefill=True)
  assert not calls

def test_missing_compiler_stops_instead_of_running_the_slow_path(model_loader, monkeypatch):
  _, model, calls = model_loader
  monkeypatch.setattr(loader, 'fast_path_problem', lambda device: 'NV has no CUDA C compiler')
  with pytest.raises(ValueError, match='no CUDA C compiler'):
    loader.load_model('model', 64, 'NV')
  assert not calls
  assert loader.load_model('model', 64, 'NV', nv_quant=False)[0] is model  # an explicit request still runs it
