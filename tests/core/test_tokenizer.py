import ast
import os
from pathlib import Path
import subprocess
import unicodedata

import pytest

from infurnace.gguf import read_gguf_metadata
from infurnace.tokenizer import PRESETS, BPETokenizer

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
  tokenizer = BPETokenizer.from_gguf(metadata)
  ids = tokenizer.encode(text)
  expected = unicodedata.normalize('NFC', text)
  assert tokenizer.decode(ids) == expected
  decoder = tokenizer.stream_decoder()
  assert ''.join(decoder(i) for i in ids) + decoder(None) == expected

def test_streams_are_independent_and_flush(metadata):
  tokenizer = BPETokenizer.from_gguf(metadata)
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
  tokenizer = BPETokenizer.from_gguf(metadata)
  assert tokenizer.bos_id is None
  assert tokenizer.encode('a') == [97]
  assert tokenizer.is_end(257)
  assert tokenizer.is_end(258)
  assert not tokenizer.is_end(97)
  metadata['tokenizer.ggml.add_bos_token'] = True
  tokenizer = BPETokenizer.from_gguf(metadata)
  assert tokenizer.bos_id is None  # Qwen deliberately suppresses BOS even for converted metadata.
  assert tokenizer.encode('a') == [97]
  metadata['tokenizer.ggml.pre'] = 'llama-bpe'
  assert BPETokenizer.from_gguf(metadata).bos_id == 258  # Llama 3 takes the BOS its metadata asks for
  metadata['tokenizer.ggml.add_bos_token'] = False
  assert BPETokenizer.from_gguf(metadata).bos_id is None

@pytest.mark.parametrize('preset,digits', [('qwen35', ['1', '2', '3', '4']), ('qwen2', ['1', '2', '3', '4']), ('llama-bpe', ['123', '4'])])
def test_presets_split_digits_their_way(metadata, preset, digits):
  # No merges: every pretokenized piece is its own sequence of byte tokens, so the pieces show in the decode.
  metadata['tokenizer.ggml.pre'] = preset
  tokenizer = BPETokenizer.from_gguf(metadata)
  pieces = tokenizer._hf.pre_tokenizer.pre_tokenize_str('1234')
  assert [piece for piece, _ in pieces] == digits
  assert tokenizer.decode(tokenizer.encode('1234')) == '1234'

@pytest.mark.parametrize('key,value,error', [
  ('tokenizer.ggml.model', 'llama', 'GPT-2'),
  ('tokenizer.ggml.pre', 'gpt-4o', 'unsupported tokenizer preset'),
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
  with pytest.raises(ValueError, match=error): BPETokenizer.from_gguf(metadata)

def test_missing_eos(metadata):
  del metadata['tokenizer.ggml.eos_token_id']
  with pytest.raises(ValueError, match='eos_token_id'): BPETokenizer.from_gguf(metadata)

@pytest.mark.skipif(not (os.environ.get('INFURNACE_TEST_MODEL') and os.environ.get('INFURNACE_LLAMA_TOKENIZE')),
                    reason='set INFURNACE_TEST_MODEL and INFURNACE_LLAMA_TOKENIZE for live reference parity')
def test_live_llama_parity():
  path = Path(os.environ['INFURNACE_TEST_MODEL']).resolve()
  metadata = read_gguf_metadata(path)
  tokenizer = BPETokenizer.from_gguf(metadata)
  for text in TEXT_CASES:
    # Qwen's tokenizer.json normalizes to NFC and llama.cpp doesn't: compare on normalized text.
    if PRESETS[metadata['tokenizer.ggml.pre']][1]: text = unicodedata.normalize('NFC', text)
    result = subprocess.run([os.environ['INFURNACE_LLAMA_TOKENIZE'], '-m', str(path), '--stdin', '--ids', '--no-bos', '--log-disable'],
                            input=text, text=True, capture_output=True, check=True, timeout=30)
    assert tokenizer.encode(text) == ast.literal_eval(result.stdout.strip()), repr(text)
