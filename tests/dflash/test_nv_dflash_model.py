"""Real-model checks for fused DFlash rounds (Qwen3.5-4B Q4_K_M + Q4_K_M draft, NV).

Marked nv (INFURNACE_TEST_NV=1); needs the model files below; loads ~3 GB once.
"""
import os
from pathlib import Path
import pytest

TARGET = Path('models/qwen35-validation/Qwen3.5-4B-Q4_K_M.gguf')
DRAFT = Path('models/dflash-qwen35/qwen35-4b-dflash-Q4_K_M.gguf')
pytestmark = [pytest.mark.nv, pytest.mark.skipif(not (TARGET.is_file() and DRAFT.is_file()),
                                                  reason='needs the Qwen3.5-4B target and DFlash draft under models/')]


@pytest.fixture(scope='module')
def loaded():
  # One generator for the module, as in real use: each generator owns its own
  # captured graphs over the same model state.
  from infurnace.cli import _build
  from infurnace.dflash import DFlashDraft, DFlashGenerator
  model, tokenizer = _build(str(TARGET), 1024, 'NV')
  draft = DFlashDraft.from_gguf(DRAFT, model)
  return model, tokenizer, DFlashGenerator(model, draft, verifier='parallel')


def configure(generator, *, fused=True, plain_cost=float('inf')):
  """`fused` only takes effect when fused rounds are enabled for the session."""
  fused = fused and os.environ.get('INFURNACE_DFLASH_FUSED') == '1'
  generator.min_tau, generator.plain_cost, generator.fused = 0.0, plain_cost, fused  # inf: always draft
  return generator


def prompt(tokenizer, text, thinking=False):
  return tokenizer.encode(tokenizer.render_chat([{'role': 'user', 'content': text}], enable_thinking=thinking))


fused_only = pytest.mark.skipif(os.environ.get('INFURNACE_DFLASH_FUSED') != '1',
  reason='fused rounds are opt-in: set INFURNACE_DFLASH_FUSED=1')


@fused_only
def test_fused_rounds_match_unfused_rounds(loaded):
  model, tokenizer, generator = loaded
  ids = prompt(tokenizer, 'Write a Python function that returns the n-th Fibonacci number.')
  expected = list(configure(generator, fused=False).stream(list(ids), max_new_tokens=48, block_size=16))
  rounds = (generator.stats.rounds, generator.stats.accepted)
  configure(generator)
  for _ in range(3):  # eager, capture, replay
    assert list(generator.stream(list(ids), max_new_tokens=48, block_size=16)) == expected
    assert (generator.stats.rounds, generator.stats.accepted) == rounds
  assert generator._parallel_verifier(16)._round_jit is not None


@fused_only
def test_fused_device_anchor_and_replay_without_deadlock(loaded):
  # This flow deterministically deadlocked NV graph replay when commit, draft
  # and verify were captured together in one (or two) graphs.
  from infurnace.dflash.parallel import ParallelVerifier
  model, tokenizer, generator = loaded
  configure(generator)
  checked, original = [], ParallelVerifier.fused_round
  def tracked(self, draft, count, anchor):
    result = original(self, draft, count, anchor)
    checked.append(self.last_device_anchor == anchor)  # read back with the round, no extra device read
    return result
  ParallelVerifier.fused_round = tracked
  try:
    for text in ('What are the pros and cons of remote work?', 'Explain in three sentences why the sky is blue.'):
      ids = prompt(tokenizer, text)
      for _ in range(2): assert len(list(generator.stream(list(ids), max_new_tokens=128, block_size=16))) == 128
  finally: ParallelVerifier.fused_round = original
  assert len(checked) > 40 and all(checked)
  assert not model._generating


def test_default_path_matches_plain_decoding_prefix(loaded):
  model, tokenizer, generator = loaded
  ids = prompt(tokenizer, 'Write a Python function that returns the n-th Fibonacci number.')
  plain = list(model.generate(list(ids), 48, reuse_cache=False))
  for _ in range(3):
    assert list(configure(generator, fused=False).stream(list(ids), max_new_tokens=48, block_size=16)) == plain


def test_timing_fallback_matches_plain_decoding_prefix(loaded):
  # Losing rounds flush the held block and decode plain rows; probes re-enter
  # the parallel path. Multirow and one-row kernels reduce in different orders,
  # so compare with plain decoding over a prefix before any near-tie.
  model, tokenizer, generator = loaded
  ids = prompt(tokenizer, 'What are the pros and cons of remote work?')
  plain = list(model.generate(list(ids), 32, reuse_cache=False))
  got = list(configure(generator, fused=generator.fused, plain_cost=0.0).stream(list(ids), max_new_tokens=32, block_size=16))
  assert got == plain
  assert generator.stats.proposed < generator.stats.rounds * 15  # most rounds were plain steps
