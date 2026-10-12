import pytest

from infurnace.benchmark import benchmark_prompt_ids, validate_benchmark_context

class Tokenizer:
  bos_id = None
  def encode(self, text): return [ord(char) for char in text]

@pytest.mark.parametrize("length", [1, 8, 31, 32, 33, 512, 2048])
@pytest.mark.parametrize("bos", [None, 0, 7])
def test_exact_synthetic_length_and_determinism(length, bos):
  tokenizer = Tokenizer()
  tokenizer.bos_id = bos
  ids = benchmark_prompt_ids(tokenizer, length)
  assert len(ids) == length
  assert ids == benchmark_prompt_ids(tokenizer, length)
  if bos is not None: assert ids[0] == bos

def test_raw_prompt_never_truncated():
  tokenizer = Tokenizer()
  tokenizer.bos_id = 0
  text = "A real prompt with more tokens than the synthetic length."
  assert benchmark_prompt_ids(tokenizer, 2, text) == [0]+tokenizer.encode(text)

@pytest.mark.parametrize("length", [0, -1, True])
def test_invalid_prompt_length(length):
  with pytest.raises(ValueError): benchmark_prompt_ids(Tokenizer(), length)

def test_empty_file_prompt_rejected():
  with pytest.raises(ValueError, match="contain tokens"): benchmark_prompt_ids(Tokenizer(), prompt_text="")

def test_context_includes_requested_answer():
  validate_benchmark_context(512, 32, 544)
  with pytest.raises(ValueError, match="544"): validate_benchmark_context(512, 32, 543)
  with pytest.raises(ValueError): validate_benchmark_context(0, 32, 512)
