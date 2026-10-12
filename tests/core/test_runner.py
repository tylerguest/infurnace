from dataclasses import replace
import random
import os

import pytest
from tinygrad import Context, Tensor, dtypes
from tinygrad.llm.model import Transformer
from tinygrad.nn.state import load_state_dict

from infurnace.models.qwen3 import Qwen3Config, Qwen3Model, expected_weight_shapes, transformer_config


def tiny_model():
  config = Qwen3Config(1, 8, 16, 2, 1, 4, 4, 4, 10000.0, 1e-6, 32, 16)
  rng = random.Random(4)
  weights = {}
  for name, shape in expected_weight_shapes(config).items():
    n = 1
    for dim in shape: n *= dim
    weights[name] = Tensor([rng.uniform(-0.2, 0.2) for _ in range(n)], device="CPU").reshape(shape)
  weights["output.weight"] = weights["token_embd.weight"]
  return Qwen3Model(config, weights), weights


def test_reference_generation_and_tied_storage():
  runner, weights = tiny_model()
  with Context(DEV="CPU"):
    reference = Transformer(transformer_config(runner.config))
    load_state_dict(reference, {k: v.half() for k, v in weights.items()}, verbose=False, realize=False)
    stream = reference.generate([1, 2, 3], chunk_size=4)
    expected = [next(stream) for _ in range(4)]
    stream.close()
  prompt = [1, 2, 3]
  assert list(runner.generate(prompt, 4, chunk_size=4)) == expected
  assert prompt == [1, 2, 3]
  assert runner.transformer.output.weight.uop is runner.transformer.token_embd.weight.uop


def test_chunks_reset_and_divergence():
  runner, _ = tiny_model()
  prompt = [1, 2, 3, 4, 5]
  expected = list(runner.generate(prompt, 3, chunk_size=4))
  assert list(runner.generate(prompt, 3, chunk_size=4, reuse_cache=False)) == expected
  changed = [6, 7, 8]
  actual = list(runner.generate(changed, 3, chunk_size=4))
  assert list(runner.generate(changed, 3, chunk_size=4, reuse_cache=False)) == actual


def test_warmup_then_generation():
  runner, _ = tiny_model()
  runner.warmup()
  assert runner.transformer._cached_tokens == []
  assert not runner._generating
  result = list(runner.generate([1, 2, 3], 3))
  assert list(runner.generate([1, 2, 3], 3, reuse_cache=False)) == result


def test_limits_and_abandoned_stream():
  runner, _ = tiny_model()
  assert list(runner.generate([1], 0)) == []
  stream = runner.generate([1, 2], 3, chunk_size=4)
  first = next(stream)
  with pytest.raises(RuntimeError): next(runner.generate([1], 1))
  with pytest.raises(RuntimeError): runner.reset()
  stream.close()
  prompt = [1, 2, first, 4]
  resumed = list(runner.generate(prompt, 2, chunk_size=4))
  assert list(runner.generate(prompt, 2, chunk_size=4, reuse_cache=False)) == resumed
  assert len(list(runner.generate([1]*15, 4, chunk_size=4, reuse_cache=False))) == 1


@pytest.mark.parametrize("tokens,count,kwargs", [([], 1, {}), ([32], 1, {}), ([True], 1, {}), ([1]*16, 1, {}),
                                                   ([1], -1, {}), ([1], 1, {"chunk_size": 0}),
                                                   ([1], 1, {"temperature": float("nan")})])
def test_invalid_generation(tokens, count, kwargs):
  runner, _ = tiny_model()
  with pytest.raises(ValueError): list(runner.generate(tokens, count, **kwargs))


def test_mixed_devices_rejected():
  runner, weights = tiny_model()
  weights["output_norm.weight"] = Tensor.empty(8, device="PYTHON")
  with pytest.raises(ValueError, match="same compute device"): Qwen3Model(runner.config, weights)


def test_config_mapping():
  runner, _ = tiny_model()
  upstream = transformer_config(replace(runner.config, context_length=8))
  assert upstream.max_context == 8
  assert upstream.qk_norm == 4
  assert upstream.n_kv_heads == 1
  assert runner.transformer.token_embd.weight.dtype == dtypes.float16


@pytest.mark.skipif(not (os.environ.get("INFURNACE_TEST_MODEL") and os.environ.get("INFURNACE_TEST_GENERATION")),
                    reason="set INFURNACE_TEST_MODEL and INFURNACE_TEST_GENERATION=1 for real-model generation")
def test_local_generation():
  from infurnace.tokenizer import Tokenizer
  runner = Qwen3Model.from_gguf(os.environ["INFURNACE_TEST_MODEL"], max_context=2048,
                               device=os.environ.get("INFURNACE_GENERATION_DEVICE", "NV"),
                               nv_quant=bool(os.environ.get("INFURNACE_TEST_NV_QUANT")),
                               nv_prefill=bool(os.environ.get("INFURNACE_TEST_NV_PREFILL")),
                               nv_attention=bool(os.environ.get("INFURNACE_TEST_NV_ATTENTION")),
                               nv_fused_attention=bool(os.environ.get("INFURNACE_TEST_NV_FUSED_ATTENTION")))
  tokenizer = Tokenizer.from_gguf(runner.metadata)
  prompt = tokenizer.encode(tokenizer.render_chat([{"role": "user", "content": "Reply with just the word hello."}], enable_thinking=False))
  ids = []
  stream = runner.generate(prompt, 32)
  try:
    for token in stream:
      if tokenizer.is_end(token): break
      ids.append(token)
  finally: stream.close()
  assert tokenizer.decode(ids).strip().lower() == "hello"
