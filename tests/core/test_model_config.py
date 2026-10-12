import os
from pathlib import Path

import pytest
from tinygrad import Tensor

import infurnace.models.qwen3 as model
from infurnace.gguf import parse_gguf


@pytest.fixture
def metadata():
  return {
    "general.architecture": "qwen3", "qwen3.block_count": 2,
    "qwen3.embedding_length": 8, "qwen3.feed_forward_length": 16,
    "qwen3.attention.head_count": 4, "qwen3.attention.head_count_kv": 2,
    "qwen3.attention.key_length": 4, "qwen3.attention.value_length": 4,
    "qwen3.context_length": 32, "qwen3.rope.freq_base": 1000000.0,
    "qwen3.attention.layer_norm_rms_epsilon": 1e-6,
    "tokenizer.ggml.tokens": [f"token{i}" for i in range(24)],
  }


def test_explicit_head_size(metadata):
  config = model.config_from_gguf(metadata, max_context=16)
  assert config.key_length == 4  # Not embedding_length / heads (2).
  assert config.rope_dimension_count == 4
  assert config.context_length == 16
  assert config.kv_cache_bytes == 2 * 16 * 2 * (4 + 4) * 2
  shapes = model.expected_weight_shapes(config)
  assert shapes["blk.0.attn_q.weight"] == (16, 8)
  assert shapes["blk.0.attn_output.weight"] == (8, 16)
  assert shapes["blk.0.attn_k.weight"] == (8, 8)
  assert shapes["blk.1.attn_q_norm.weight"] == (4,)


def test_context_cap(metadata):
  assert model.config_from_gguf(metadata).context_length == 32
  assert model.config_from_gguf(metadata, 64).context_length == 32


def test_head_size_fallback(metadata):
  del metadata["qwen3.attention.key_length"]
  del metadata["qwen3.attention.value_length"]
  config = model.config_from_gguf(metadata)
  assert config.key_length == config.value_length == config.rope_dimension_count == 2
  metadata["qwen3.embedding_length"] = 9
  with pytest.raises(ValueError, match="integral"): model.config_from_gguf(metadata)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "16"])
def test_invalid_context_cap(metadata, value):
  with pytest.raises(ValueError, match="max_context"): model.config_from_gguf(metadata, value)


@pytest.mark.parametrize("key,value,error", [
  ("general.architecture", "llama", "dense Qwen3"),
  ("qwen3.block_count", 0, "block_count"),
  ("qwen3.embedding_length", True, "embedding_length"),
  ("qwen3.attention.key_length", None, "key_length"),
  ("qwen3.attention.head_count_kv", 3, "divisible"),
  ("qwen3.attention.value_length", 2, "equal K/V"),
  ("qwen3.rope.dimension_count", 3, "RoPE dimension"),
  ("qwen3.rope.dimension_count", 6, "RoPE dimension"),
  ("qwen3.rope.freq_base", float("inf"), "finite"),
  ("qwen3.rope.freq_base", True, "finite"),
  ("qwen3.attention.layer_norm_rms_epsilon", float("nan"), "finite"),
  ("qwen3.attention.layer_norm_rms_epsilon", -1, "finite"),
  ("tokenizer.ggml.tokens", [], "vocabulary"),
  ("tokenizer.ggml.tokens", [1], "vocabulary"),
  ("qwen3.expert_count", 4, "unsupported"),
  ("qwen3.nextn_predict_layers", 1, "unsupported"),
  ("qwen3.attention.sliding_window", 16, "unsupported"),
  ("qwen3.rope.scaling.type", "linear", "scaling"),
  ("qwen3.rope.scaling.factor", 2.0, "scaling"),
])
def test_invalid_metadata(metadata, key, value, error):
  metadata[key] = value
  with pytest.raises(ValueError, match=error): model.config_from_gguf(metadata)


@pytest.mark.parametrize("key", ["qwen3.block_count", "qwen3.feed_forward_length", "qwen3.rope.freq_base", "tokenizer.ggml.tokens"])
def test_missing_metadata(metadata, key):
  del metadata[key]
  with pytest.raises(ValueError): model.config_from_gguf(metadata)


def weights_for(config):
  return {name: Tensor.empty(*shape, device="CPU") for name, shape in model.expected_weight_shapes(config).items()}


def test_tied_weights_do_not_mutate_input(metadata):
  config = model.config_from_gguf(metadata)
  weights = weights_for(config)
  del weights["output.weight"]
  prepared = model.prepare_weights(config, weights)
  assert "output.weight" not in weights
  assert prepared["output.weight"] is prepared["token_embd.weight"]


def test_separate_output_preserved(metadata):
  config = model.config_from_gguf(metadata)
  weights = weights_for(config)
  assert model.prepare_weights(config, weights)["output.weight"] is weights["output.weight"]


@pytest.mark.parametrize("mode,error", [("missing", "missing required"), ("shape", "expected"), ("bias", "unsupported")])
def test_invalid_weights(metadata, mode, error):
  config = model.config_from_gguf(metadata)
  weights = weights_for(config)
  if mode == "missing": del weights["blk.1.attn_q_norm.weight"]
  elif mode == "shape": weights["blk.0.attn_q.weight"] = Tensor.empty(8, 8, device="CPU")
  else: weights["blk.0.attn_q.bias"] = Tensor.empty(16, device="CPU")
  with pytest.raises(ValueError, match=error): model.prepare_weights(config, weights)


def test_model_factory_caps_context_and_shares_weights(metadata, monkeypatch):
  config = model.config_from_gguf(metadata)
  class Entry:
    def __init__(self, tensor): self.tensor, self.shape = tensor, tensor.shape
    def to(self): return self
    def decode(self): return self.tensor
  weights = weights_for(config)
  del weights["output.weight"]
  entries = {name: Entry(tensor) for name, tensor in weights.items()}
  monkeypatch.setattr(model, "parse_gguf", lambda path: (metadata, entries))
  class Constructed(model.Qwen3Model):
    def __init__(self, config, weights): self.config, self.weights = config, weights
  result = Constructed.from_gguf("unused", max_context=8)
  assert result.config.context_length == 8
  assert result.weights["output.weight"] is result.weights["token_embd.weight"]


def test_factory_validates_before_staging(metadata, monkeypatch):
  class InvalidEntry:
    shape = (1,)
    def to(self): pytest.fail("staged bytes before validating shapes")
  monkeypatch.setattr(model, "parse_gguf", lambda path: (metadata, {"token_embd.weight": InvalidEntry()}))
  with pytest.raises(ValueError, match="expected"): model.Qwen3Model.from_gguf("unused")


@pytest.mark.skipif(not os.environ.get("INFURNACE_TEST_MODEL"), reason="set INFURNACE_TEST_MODEL to a local GGUF path")
def test_local_qwen3_contract():
  metadata, entries = parse_gguf(Path(os.environ["INFURNACE_TEST_MODEL"]))
  config = model.config_from_gguf(metadata, max_context=2048)
  prepared = model.prepare_weights(config, entries)
  assert config.key_length == 128
  assert config.attention_head_count * config.key_length == 4096
  assert config.kv_cache_bytes == 288 * 1024**2
  assert prepared["output.weight"] is prepared["token_embd.weight"]
