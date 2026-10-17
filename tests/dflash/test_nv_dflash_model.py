"""Real-model checks for fused DFlash rounds (Qwen3.5-4B Q4_K_M + Q4_K_M draft, NV).

Marked nv (INFURNACE_TEST_NV=1); needs the model files below; loads ~3 GB once.
"""
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
  from infurnace.loader import load_model as _build
  from infurnace.dflash import DFlashDraft, DFlashGenerator
  model, tokenizer = _build(str(TARGET), 1024, 'NV')
  draft = DFlashDraft.from_gguf(DRAFT, model)
  return model, tokenizer, DFlashGenerator(model, draft, verifier='parallel')

def configure(generator, *, fused=True, plain_cost=float('inf')):
  generator.min_tau, generator.plain_cost, generator.fused = 0.0, plain_cost, fused  # inf: always draft
  return generator

def prompt(tokenizer, text, thinking=False):
  return tokenizer.encode(tokenizer.render_chat([{'role': 'user', 'content': text}], enable_thinking=thinking))

def test_cold_first_request_matches_plain_decoding_prefix(loaded):
  # Must run first in this module: DFlash is the model's first work, as for a
  # bare `infurnace --prompt` run. Upstream target state starts lazy; before it
  # was realized up front, this request diverged within a few tokens.
  from infurnace.cli import encode as _prompt_ids
  model, tokenizer, generator = loaded
  ids = _prompt_ids(tokenizer, 'Write a Python function that returns the n-th Fibonacci number.')
  generator.min_tau, generator.plain_cost, generator.fused = 3.5, None, True  # CLI defaults: adaptive, so plain steps run too
  got = list(generator.stream(list(ids), max_new_tokens=32, block_size=16))
  assert got == list(model.generate(list(ids), 32, reuse_cache=False))

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

def test_unfused_path_matches_plain_decoding_prefix(loaded):
  model, tokenizer, generator = loaded
  ids = prompt(tokenizer, 'Write a Python function that returns the n-th Fibonacci number.')
  plain = list(model.generate(list(ids), 48, reuse_cache=False))
  runs = [list(configure(generator, fused=False).stream(list(ids), max_new_tokens=48, block_size=16)) for _ in range(3)]
  assert runs[0] == runs[1] == runs[2]  # repeatable
  # Multirow and one-row kernels reduce in different orders: at token 28 plain decoding's own
  # logits put "\n\n" and "\n" 0.08 apart, so compare the prefix before that near-tie.
  assert runs[0][:28] == plain[:28]

def test_timing_fallback_matches_plain_decoding_prefix(loaded):
  # Losing rounds flush the held block and decode plain rows; probes re-enter
  # the parallel path. Multirow and one-row kernels reduce in different orders,
  # so compare with plain decoding over a prefix before any near-tie: at token 17 this prompt
  # has " new" vs " unique" 0.05 logits apart in plain decoding's own logits.
  model, tokenizer, generator = loaded
  ids = prompt(tokenizer, 'What are the pros and cons of remote work?')
  plain = list(model.generate(list(ids), 32, reuse_cache=False))
  got = list(configure(generator, plain_cost=0.0).stream(list(ids), max_new_tokens=32, block_size=16))
  assert got[:16] == plain[:16]
  assert generator.stats.proposed < generator.stats.rounds * 15  # most rounds were plain steps

def test_chunked_prompt_matches_one_row_prompt(loaded):
  # After warmup, prompts read through the model's 64-row prompt chunks, which also hand the draft its feature
  # taps. Returning only the taps once left the blocks after the last tap (30 and 31) out of the captured graph:
  # their caches and states never saw the prompt. Easy prompts still decode the same tokens without them, so
  # every layer's state is compared with one-row prompt reading.
  from tinygrad import Tensor
  from infurnace.models.qwen35 import PROMPT_ROWS
  model, tokenizer, generator = loaded
  configure(generator)
  generator.warmup(16)
  assert model._prompt_jit.captured is not None
  notes = ' '.join(f'Step {i}: read the file, find the function, and check its callers.' for i in range(10))
  ids = prompt(tokenizer, notes + '\n\nWrite a Python function that returns the n-th Fibonacci number.')
  assert len(ids) > 2 * PROMPT_ROWS
  def states():
    out = []
    for block in model.transformer.blk:
      if hasattr(block, 'cache_kv'): out.append(Tensor(block.cache_kv[:, :, :, :len(ids)].float().tolist()))
      else: out.append(Tensor(block.recurrent_state.float().tolist()))
    return out
  got = list(generator.stream(list(ids), max_new_tokens=24, block_size=16))
  model.reset()
  list(generator.stream(list(ids), max_new_tokens=1, block_size=16))   # the prompt alone, then its first decision
  chunked = states()
  model._multirow_prefill = False
  try:
    expected = list(model.generate(list(ids), 24, reuse_cache=False))
    model.reset()
    list(model.generate(list(ids), 1, reuse_cache=False))
    one_row = states()
  finally: model._multirow_prefill = True
  for i, (a, b) in enumerate(zip(chunked, one_row)):
    # FP16 GEMM vs Q8 GEMV rounding compounds through the layers to 0.4-9% (mean); a layer left out is ~100%.
    assert ((a - b).abs().mean() / b.abs().mean()).item() < 0.25, i
  assert got == expected
