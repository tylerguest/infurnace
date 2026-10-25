"""DFlash2 (block convolutions and the candidate selector; llama.cpp's src/models/dflash.cpp) against plain-Python
references on small random weights, and on the real Bonsai-4B draft: identical output to plain decoding."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from tinygrad import Tensor, dtypes, nn

from infurnace.dflash.draft import DFlashConfig, DFlashDraft

pytestmark = pytest.mark.skipif(not os.environ.get("INFURNACE_TEST_NV"), reason="set INFURNACE_TEST_NV=1 on an NV GPU")

def config(**extra):
  return DFlashConfig(layers=1, dim=256, hidden=512, heads=2, kv_heads=1, head_dim=128, epsilon=1e-6, theta=1e6, block_size=4,
                      mask_token=0, feature_layers=(1,), sliding=(False,), window=0, **extra)

def test_block_convolution_follows_llama_cpp():
  c = config(conv_kernel=2, conv_group=16)
  draft = SimpleNamespace(config=c)
  rows, groups = 4, c.dim // c.conv_group
  x, coefficients, base = Tensor.randn(1, rows, c.dim), Tensor.randn(1, rows, groups * c.conv_kernel * 2), Tensor.randn(2, c.conv_kernel, c.dim)
  xs, cs, bs = x.tolist()[0], coefficients.tolist()[0], base.tolist()
  for side in (0, 1):
    got = DFlashDraft._conv(draft, x, coefficients, base, side).tolist()[0]
    for p in range(rows):
      want = [sum((cs[p][(side * c.conv_kernel + tap) * groups + h // c.conv_group] + bs[side][tap][h]) * xs[p - tap][h]
                  for tap in range(c.conv_kernel) if p - tap >= 0) for h in range(c.dim)]
      assert got[p] == pytest.approx(want, rel=1e-4, abs=1e-4)

def test_selector_walks_greedily_from_the_anchor():
  c = config(selector_rank=128, selector_top_k=4)
  vocab, rows = 512, 3
  weights = SimpleNamespace(selector_hidden=nn.Linear(c.dim, c.selector_rank, bias=False),
                            selector_predecessor=nn.Embedding(vocab, c.selector_rank), selector_successor=nn.Embedding(vocab, c.selector_rank))
  for t in (weights.selector_predecessor, weights.selector_successor): t.weight = t.weight.half().realize()
  draft = SimpleNamespace(config=c, weights=weights, anchor=Tensor([17], dtype=dtypes.int32).realize())
  logits, hidden = Tensor.randn(rows, vocab).realize(), Tensor.randn(rows, c.dim).realize()
  got = DFlashDraft._select(draft, logits, hidden).tolist()[0]
  ls, gates = logits.tolist(), (hidden @ weights.selector_hidden.weight.T).tolist()
  prev, succ = weights.selector_predecessor.weight.float().tolist(), weights.selector_successor.weight.float().tolist()
  previous, want = 17, []
  for i in range(rows):
    candidates = sorted(range(vocab), key=lambda v: (-ls[i][v], v))[:c.selector_top_k]
    scores = [sum(succ[v][r] * prev[previous][r] * gates[i][r] for r in range(c.selector_rank)) + ls[i][v] for v in candidates]
    previous = candidates[max(range(len(scores)), key=lambda k: (scores[k], -k))]
    want.append(previous)
  assert got == want

MODELS = Path.home() / '.local/share/infurnace/models'
TARGET, DRAFT = MODELS / 'Ternary-Bonsai-2-4B-Q2_0.gguf', MODELS / 'Ternary-Bonsai-2-4B-DFlash2-BF16.gguf'

@pytest.mark.skipif(not (TARGET.is_file() and DRAFT.is_file()), reason='needs Bonsai-4B and its DFlash2 draft (infurnace pull bonsai:4b)')
def test_bonsai_dflash2_matches_plain_decoding():
  from infurnace import loader
  from infurnace.chat import user_prompt
  model, tokenizer = loader.load_model(str(TARGET), 1024, 'NV')
  dflash = loader.load_dflash(model, DRAFT)
  assert dflash.draft.config.selector_top_k == 16 and dflash.draft.config.conv_kernel == 2
  ids = tokenizer.encode(user_prompt(model, tokenizer, 'Write a Python function that checks whether a number is prime.'))
  plain = list(model.generate(ids, 64))
  eos = tuple(i for i in (tokenizer.eos_id, tokenizer.eot_id) if i is not None)
  drafted = list(dflash.stream(ids, max_new_tokens=64, block_size=16, eos_token_ids=eos))
  assert drafted == plain[:len(drafted)] and len(drafted) == 64
  assert dflash.stats.accepted / dflash.stats.rounds > 5  # on code the draft is right ~10 tokens a round
