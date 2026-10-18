import os

import pytest
from tinygrad import Context, Tensor, dtypes
from tinygrad.llm.model import Transformer
from infurnace.engine import Model
from test_qwen35 import checkpoint

@pytest.fixture
def runner(tmp_path):
  with Context(DEV='CPU'):
    return Model.from_gguf(checkpoint(tmp_path), device='CPU')

def test_reference_generation_and_tied_storage(runner, tmp_path):
  with Context(DEV='CPU'):
    reference, _ = Transformer.from_gguf(tmp_path / 'model.gguf', realize=False)
    Tensor.manual_seed(123)
    stream = reference.generate([1, 2, 3], chunk_size=4)
    try: expected = [next(stream) for _ in range(4)]
    finally: stream.close()
  prompt = [1, 2, 3]
  Tensor.manual_seed(123)
  assert list(runner.generate(prompt, 4, chunk_size=4)) == expected
  assert prompt == [1, 2, 3]
  assert runner.transformer.output.weight.uop is runner.transformer.token_embd.weight.uop

def test_chunks_reset_and_divergence(runner):
  prompt = [1, 2, 3, 4, 5]
  expected = list(runner.generate(prompt, 3, chunk_size=4))
  assert list(runner.generate(prompt, 3, chunk_size=4, reuse_cache=False)) == expected
  changed = [6, 7, 8]
  actual = list(runner.generate(changed, 3, chunk_size=4))
  assert list(runner.generate(changed, 3, chunk_size=4, reuse_cache=False)) == actual

def test_warmup_then_generation(runner):
  runner.warmup()
  assert runner.transformer._cached_tokens == []
  assert not runner._generating
  result = list(runner.generate([1, 2, 3], 3))
  assert list(runner.generate([1, 2, 3], 3, reuse_cache=False)) == result

def test_limits_and_abandoned_stream(runner):
  assert list(runner.generate([1], 0)) == []
  stream = runner.generate([1, 2], 3, chunk_size=4)
  first = next(stream)
  with pytest.raises(RuntimeError): next(runner.generate([1], 1))
  with pytest.raises(RuntimeError): runner.reset()
  stream.close()
  assert not runner._generating
  prompt = [1, 2, first, 4]
  resumed = list(runner.generate(prompt, 2, chunk_size=4))
  assert list(runner.generate(prompt, 2, chunk_size=4, reuse_cache=False)) == resumed
  assert len(list(runner.generate([1]*15, 4, chunk_size=4, reuse_cache=False))) == 1

@pytest.mark.parametrize('tokens,count,kwargs', [([], 1, {}), ([16], 1, {}), ([True], 1, {}), ([1]*16, 1, {}),
                                               ([1], -1, {}), ([1], 1, {'chunk_size': 0}),
                                               ([1], 1, {'temperature': float('nan')})])
def test_invalid_generation(runner, tokens, count, kwargs):
  with pytest.raises(ValueError): list(runner.generate(tokens, count, **kwargs))

def test_loaded_config(runner):
  assert runner.config.context_length == 16
  assert runner.config.block_count == 2
  assert runner.config.attention_head_count_kv == 1
  assert runner.transformer.token_embd.weight.dtype == dtypes.float16

@pytest.mark.skipif(not (os.environ.get('INFURNACE_TEST_MODEL') and os.environ.get('INFURNACE_TEST_GENERATION')),
                    reason='set INFURNACE_TEST_MODEL and INFURNACE_TEST_GENERATION=1 for real-model generation')
def test_local_generation():
  from infurnace.tokenizer import Qwen35Tokenizer
  model = Model.from_gguf(os.environ['INFURNACE_TEST_MODEL'], max_context=2048,
                              device=os.environ.get('INFURNACE_GENERATION_DEVICE', 'NV'),
                              nv_quant=bool(os.environ.get('INFURNACE_TEST_NV_QUANT')),
                              nv_prefill=bool(os.environ.get('INFURNACE_TEST_NV_PREFILL')))
  tokenizer = Qwen35Tokenizer.from_gguf(model.metadata)
  from infurnace.chat import user_prompt
  prompt = tokenizer.encode(user_prompt(model, tokenizer, 'Reply with just the word hello.'))
  ids = []
  stream = model.generate(prompt, 32)
  try:
    for token in stream:
      if tokenizer.is_end(token): break
      ids.append(token)
  finally: stream.close()
  assert tokenizer.decode(ids).strip().lower() == 'hello'
