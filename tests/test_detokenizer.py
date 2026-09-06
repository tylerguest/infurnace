import os
import random

import pytest
from infurnace.tokenizer import StreamingDetokenizer, GGUFTokenizer

_ARTIFACT = "artifacts/models/Qwen3-4B-Q4_K_M.gguf"


@pytest.mark.model
@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(_ARTIFACT), reason="GGUF artifact not present")
def test_incremental_matches_full_decode_real_tokenizer():
  tok = GGUFTokenizer.from_artifact(_ARTIFACT)
  random.seed(1)
  for _ in range(50):
    seq = [random.randint(0, 151935) for _ in range(random.randint(1, 40))]
    detok = StreamingDetokenizer(tok)
    out = "".join(detok.update(seq[:i]) for i in range(1, len(seq) + 1))
    assert out == tok.decode(seq)


@pytest.mark.model
@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(_ARTIFACT), reason="GGUF artifact not present")
def test_roundtrip_real_tokenizer():
  tok = GGUFTokenizer.from_artifact("artifacts/models/Qwen3-4B-Q4_K_M.gguf")
  text = "Hello, world!"
  ids = tok.encode(text)
  assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
  assert tok.decode(ids) == text

