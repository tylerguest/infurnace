"""DFlash with a temperature samples from the target's distribution, like plain decoding.

The draft proposes greedily (a one-hot distribution), so accepting a proposal exactly when the
target's sample at that row equals it is speculative sampling. Statistical checks on the tiny CPU
model, seeded so every run draws the same samples.
"""
import collections
import math
import pytest
from tinygrad import Context, Tensor
from infurnace.dflash import DFlashGenerator
from infurnace.models.qwen35 import Qwen35Model
from test_dflash import RecordedDraft
from test_qwen35 import checkpoint

PROMPT = [1, 2, 3]

def model_and_generator(tmp_path, verifier, seed=0):
  Tensor.manual_seed(seed)
  model = Qwen35Model.from_gguf(checkpoint(tmp_path), device='CPU')
  greedy = list(model.generate(list(PROMPT), 8, reuse_cache=False))
  # Proposes the greedy continuation whatever was sampled: a deterministic draft is all the rule needs.
  generator = DFlashGenerator(model, RecordedDraft(model, PROMPT, greedy), verifier=verifier)
  generator.min_tau, generator.plain_cost = 0.0, float('inf')  # always draft
  return model, generator

def chi_square_pvalue(counts: collections.Counter, probs: list[float], draws: int) -> float:
  cells = [i for i, p in enumerate(probs) if draws * p >= 5]
  x, k = sum((counts[i] - draws * probs[i]) ** 2 / (draws * probs[i]) for i in cells), len(cells) - 1
  z = ((x / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))  # Wilson-Hilferty
  return 0.5 * math.erfc(z / math.sqrt(2))

@pytest.mark.parametrize('verifier', ['parallel', 'sequential'])
def test_first_token_follows_the_target_probabilities(tmp_path, verifier):
  temperature, draws = 0.05, 800
  with Context(DEV='CPU'):
    model, generator = model_and_generator(tmp_path, verifier, seed=3)  # a model with several likely tokens
    t, x = model.transformer, None
    for position, token in enumerate(PROMPT):
      x = t.token_embd(Tensor([[token]], dtype='int32')).float()
      for block in t.blk: x = block(x, position)
      x = x.realize()
    probs = (t.output(t.output_norm(x[:, -1:]))[0, -1].float() / temperature).softmax().tolist()
    assert sum(p > 0.05 for p in probs) >= 2
    counts = collections.Counter(list(generator.stream(list(PROMPT), max_new_tokens=1, block_size=4, temperature=temperature))[0]
                                 for _ in range(draws))
    assert chi_square_pvalue(counts, probs, draws) > 0.001

@pytest.mark.parametrize('verifier', ['parallel', 'sequential'])
def test_accepted_proposals_keep_the_sequence_distribution(tmp_path, verifier):
  # Peaked: speculation accepts almost every proposal, so the accepting path decides the output.
  temperature, draws = 0.02, 300
  with Context(DEV='CPU'):
    model, generator = model_and_generator(tmp_path, verifier)
    plain, spec, accepted, proposed = collections.Counter(), collections.Counter(), 0, 0
    for _ in range(draws):
      plain[tuple(model.generate(list(PROMPT), 3, temperature=temperature, reuse_cache=False))] += 1
      spec[tuple(generator.stream(list(PROMPT), max_new_tokens=3, block_size=4, temperature=temperature))] += 1
      accepted, proposed = accepted + generator.stats.accepted, proposed + generator.stats.proposed
    assert accepted >= 0.9 * proposed
    assert 0.5 * sum(abs(plain[k] - spec[k]) for k in set(plain) | set(spec)) / draws < 0.1

def test_one_generator_serves_every_temperature(tmp_path):
  # The temperature is a buffer the captured graphs read: requests switch it freely, and greedy
  # requests between sampled ones stay exactly greedy.
  with Context(DEV='CPU'):
    model, generator = model_and_generator(tmp_path, 'parallel')
    greedy = list(generator.stream(list(PROMPT), max_new_tokens=4, block_size=4))
    list(generator.stream(list(PROMPT), max_new_tokens=4, block_size=4, temperature=1.5))
    assert list(generator.stream(list(PROMPT), max_new_tokens=4, block_size=4, temperature=0.0)) == greedy
    assert greedy == list(model.generate(list(PROMPT), 4, reuse_cache=False))
    with pytest.raises(ValueError, match='temperature'): list(generator.stream(list(PROMPT), max_new_tokens=2, temperature=-1))
