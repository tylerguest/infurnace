import ast
import os
from pathlib import Path
import subprocess
import unicodedata

import pytest

from infurnace.gguf import read_gguf_metadata
from infurnace.tokenizer import Qwen35Tokenizer

@pytest.fixture
def metadata():
  # GPT-2's reversible mapping from every byte to a vocabulary character.
  direct = [*range(33, 127), *range(161, 173), *range(174, 256)]
  mapping = {b: chr(b) for b in direct}
  mapping.update({b: chr(256+i) for i, b in enumerate(b for b in range(256) if b not in direct)})
  return {
    'tokenizer.ggml.model': 'gpt2', 'tokenizer.ggml.pre': 'qwen35', 'tokenizer.ggml.merges': [],
    'tokenizer.ggml.tokens': [mapping[b] for b in range(256)] + ['<|im_start|>', '<|im_end|>', '<|endoftext|>'],
    'tokenizer.ggml.token_type': [1]*256 + [3]*3,
    'tokenizer.ggml.bos_token_id': 258, 'tokenizer.ggml.eos_token_id': 258,
    'tokenizer.ggml.add_bos_token': False,
  }

TEXT_CASES = ['', 'Hello 123456!', '你好 café é 👩🏽‍💻', ' \t\r\n', '<|im_start|>user\nhello<|im_end|>']

@pytest.mark.parametrize('text', TEXT_CASES)
def test_roundtrip_and_streaming(metadata, text):
  tokenizer = Qwen35Tokenizer.from_gguf(metadata)
  ids = tokenizer.encode(text)
  expected = unicodedata.normalize('NFC', text)
  assert tokenizer.decode(ids) == expected
  decoder = tokenizer.stream_decoder()
  assert ''.join(decoder(i) for i in ids) + decoder(None) == expected

def test_streams_are_independent_and_flush(metadata):
  tokenizer = Qwen35Tokenizer.from_gguf(metadata)
  a, b = tokenizer.stream_decoder(), tokenizer.stream_decoder()
  emoji = list('🙂'.encode())
  assert a(emoji[0]) == ''
  assert b(ord('x')) == 'x'
  assert ''.join(a(i) for i in emoji[1:]) == '🙂'
  assert b(None) == ''
  incomplete = tokenizer.stream_decoder()
  assert incomplete(emoji[0]) == ''
  assert incomplete(None) == '�'

def test_bos_and_stop_policy(metadata):
  tokenizer = Qwen35Tokenizer.from_gguf(metadata)
  assert tokenizer.bos_id is None
  assert tokenizer.encode('a') == [97]
  assert tokenizer.is_end(257)
  assert tokenizer.is_end(258)
  assert not tokenizer.is_end(97)
  metadata['tokenizer.ggml.add_bos_token'] = True
  tokenizer = Qwen35Tokenizer.from_gguf(metadata)
  assert tokenizer.bos_id is None  # Qwen3.5 deliberately suppresses BOS even for converted metadata.
  assert tokenizer.encode('a') == [97]

@pytest.mark.parametrize('key,value,error', [
  ('tokenizer.ggml.model', 'llama', 'GPT-2'),
  ('tokenizer.ggml.pre', 'qwen2', 'Qwen3.5'),
  ('tokenizer.ggml.tokens', [], 'vocabulary'),
  ('tokenizer.ggml.tokens', ['a', 'a'], 'duplicate'),
  ('tokenizer.ggml.token_type', [1], 'token types'),
  ('tokenizer.ggml.eos_token_id', -1, 'eos_token_id'),
  ('tokenizer.ggml.bos_token_id', True, 'bos_token_id'),
  ('tokenizer.ggml.eot_token_id', 999, 'eot_token_id'),
  ('tokenizer.ggml.add_bos_token', 1, 'boolean'),
  ('tokenizer.ggml.merges', None, 'BPE merges'),
  ('tokenizer.ggml.merges', ['bad'], 'BPE merges'),
])
def test_invalid_metadata(metadata, key, value, error):
  metadata[key] = value
  with pytest.raises(ValueError, match=error): Qwen35Tokenizer.from_gguf(metadata)

def test_missing_eos(metadata):
  del metadata['tokenizer.ggml.eos_token_id']
  with pytest.raises(ValueError, match='eos_token_id'): Qwen35Tokenizer.from_gguf(metadata)

def test_chat_single_turn(metadata):
  tokenizer = Qwen35Tokenizer.from_gguf(metadata)
  messages = [{'role': 'system', 'content': 'Be concise.'}, {'role': 'user', 'content': 'Hello'}]
  expected = '<|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n'
  assert tokenizer.render_chat(messages) == expected + '<|im_start|>assistant\n<think>\n'
  assert tokenizer.render_chat(messages, enable_thinking=False) == expected + '<|im_start|>assistant\n<think>\n\n</think>\n\n'
  assert tokenizer.render_chat(messages, add_generation_prompt=False) == expected
  assert messages[0]['content'] == 'Be concise.'

def test_chat_historical_reasoning(metadata):
  tokenizer = Qwen35Tokenizer.from_gguf(metadata)
  messages = [{'role': 'user', 'content': 'one'},
              {'role': 'assistant', 'content': '<think>old secret</think>\n\nanswer'},
              {'role': 'user', 'content': 'two'}]
  rendered = tokenizer.render_chat(messages)
  assert 'old secret' not in rendered
  assert '<|im_start|>assistant\nanswer<|im_end|>' in rendered
  messages.append({'role': 'assistant', 'content': '\nnew answer', 'reasoning_content': '\nnew thought\n'})
  assert '<think>\nnew thought\n</think>\n\nnew answer<|im_end|>' in tokenizer.render_chat(messages)

@pytest.mark.parametrize('messages', [[], [None], [{'role': 'tool', 'content': 'x'}],
                                     [{'role': 'user', 'content': ['x']}], [{'role': 'user', 'content': None}],
                                     [{'role': 'assistant', 'content': 'x', 'tool_calls': [{'name': 'foo'}]}],
                                     [{'role': 'assistant', 'content': 'x', 'reasoning_content': 1}],
                                     [{'role': 'user', 'content': '<tool_response>x</tool_response>'}]])
def test_invalid_chat(metadata, messages):
  with pytest.raises(ValueError): Qwen35Tokenizer.from_gguf(metadata).render_chat(messages)

def test_invalid_chat_flags(metadata):
  with pytest.raises(ValueError, match='boolean'):
    Qwen35Tokenizer.from_gguf(metadata).render_chat([{'role': 'user', 'content': 'x'}], enable_thinking=0)

@pytest.mark.skipif(not os.environ.get('INFURNACE_TEST_MODEL'), reason='set INFURNACE_TEST_MODEL for Qwen3.5 template parity')
def test_bundled_chat_template_parity():
  from infurnace.serve import ChatTemplate
  model_metadata = read_gguf_metadata(os.environ['INFURNACE_TEST_MODEL'])
  tokenizer = Qwen35Tokenizer.from_gguf(model_metadata)
  cases = [
    [{'role': 'user', 'content': 'Hello'}],
    [{'role': 'system', 'content': 'Be concise.'}, {'role': 'user', 'content': '你好'}],
    [{'role': 'user', 'content': 'one'}, {'role': 'assistant', 'content': '<think>old</think>\n\nanswer'},
     {'role': 'user', 'content': 'two'}],
  ]
  for messages in cases:
    for thinking in (False, True):
      for prefix in (False, True):
        template = ChatTemplate(model_metadata, tokenizer, enable_thinking=thinking)
        expected = template.render(messages=messages, tools=[], add_generation_prompt=prefix)
        actual = tokenizer.render_chat(messages, enable_thinking=thinking, add_generation_prompt=prefix)
        assert actual == expected
        assert tokenizer.encode(actual) == tokenizer.encode(expected)

@pytest.mark.skipif(not (os.environ.get('INFURNACE_TEST_MODEL') and os.environ.get('INFURNACE_LLAMA_TOKENIZE')),
                    reason='set INFURNACE_TEST_MODEL and INFURNACE_LLAMA_TOKENIZE for live reference parity')
def test_live_llama_parity():
  path = Path(os.environ['INFURNACE_TEST_MODEL']).resolve()
  tokenizer = Qwen35Tokenizer.from_gguf(read_gguf_metadata(path))
  for text in TEXT_CASES:
    result = subprocess.run([os.environ['INFURNACE_LLAMA_TOKENIZE'], '-m', str(path), '--stdin', '--ids', '--no-bos', '--log-disable'],
                            input=text, text=True, capture_output=True, check=True, timeout=30)
    assert tokenizer.encode(text) == ast.literal_eval(result.stdout.strip()), repr(text)
