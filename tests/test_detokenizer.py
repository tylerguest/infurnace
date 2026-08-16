import os
import random
import unittest

import pytest
from infurnace.tokenizer import detokenize_incrementally, StreamingDetokenizer, GGUFTokenizer
from infurnace.tokenizer.base import Tokenizer

_ARTIFACT = "artifacts/models/Qwen3-0.6B-Q8_0.gguf"


class FakeTokenizer(Tokenizer):
    def encode(self, text): return [ord(c) for c in text]
    def decode(self, ids): return "".join(chr(i) for i in ids)
    def is_end(self, token_id): return False


class TestDetokenizeIncrementally(unittest.TestCase):
    def test_invariant_over_random_sequences(self):
        random.seed(0)
        tok = FakeTokenizer()
        for _ in range(200):
            seq = [random.randint(0, 1000) for _ in range(random.randint(1, 30))]
            detok = StreamingDetokenizer(tok)
            out = "".join(detok.update(seq[:i]) for i in range(1, len(seq) + 1))
            self.assertEqual(out, tok.decode(seq))

    def test_prompt_not_echoed(self):
        detok = StreamingDetokenizer(FakeTokenizer(), skip_prompt_len=2)
        self.assertEqual(detok.update([10, 11, 12, 13]), FakeTokenizer().decode([12, 13]))
        self.assertEqual(detok.text, FakeTokenizer().decode([12, 13]))

    def test_multibyte_safe(self):
        seq = [0xE4, 0xE5, 0x61]  # ä å a
        detok = StreamingDetokenizer(FakeTokenizer())
        out = "".join(detok.update(seq[:i]) for i in range(1, len(seq) + 1))
        self.assertEqual(out, FakeTokenizer().decode(seq))


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
