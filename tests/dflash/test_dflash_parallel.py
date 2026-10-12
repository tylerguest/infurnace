import pytest
from tinygrad import Context, Tensor
from infurnace.dflash import DFlashGenerator
from infurnace.dflash.parallel import _decays, _unit_lower_inverse
from infurnace.models.qwen35 import Qwen35Model
from test_dflash import RecordedDraft, tiny_draft
from test_qwen35 import checkpoint

def test_closed_form_gated_delta_matches_recurrence_and_every_prefix_state():
  B, H, T, K, V = 1, 3, 16, 8, 6
  with Context(DEV='CPU'):
    Tensor.manual_seed(0)
    k, q = Tensor.randn(B, H, T, K).normalize(dim=-1), Tensor.randn(B, H, T, K) * K**-0.5
    v, beta, log_alpha = Tensor.randn(B, H, T, V), Tensor.rand(B, H, T), -Tensor.rand(B, H, T) * 0.5
    S0 = Tensor.randn(B, H, V, K)
    # Reference: the per-row gated delta rule used by one-row decoding.
    state, outs, states = S0, [], []
    for t in range(T):
      s1 = state * log_alpha[:, :, t].exp().reshape(B, H, 1, 1)
      delta = (v[:, :, t].unsqueeze(-1) - (s1 * k[:, :, t].unsqueeze(-2)).sum(-1, keepdim=True)) * beta[:, :, t].reshape(B, H, 1, 1)
      state = s1 + delta * k[:, :, t].unsqueeze(-2)
      outs.append((state * q[:, :, t].unsqueeze(-2)).sum(-1))
      states.append(state)
    g = log_alpha.cumsum(-1)
    strict, inclusive = _decays(g, T)
    A = beta.unsqueeze(-1) * strict * (k @ k.transpose(-1, -2))
    u = _unit_lower_inverse(A, T) @ (beta.unsqueeze(-1) * (v - g.exp().unsqueeze(-1) * (k @ S0.transpose(-1, -2))))
    out = g.exp().unsqueeze(-1) * (q @ S0.transpose(-1, -2)) + (inclusive * (q @ k.transpose(-1, -2))) @ u
    assert (out - Tensor.stack(*outs, dim=2)).abs().max().item() < 1e-5
    # Commit formula: the state after any accepted prefix of m rows.
    index = Tensor.arange(T)
    for m in range(1, T+1):
      last = (g * (index == m-1)).sum(-1, keepdim=True)
      weight = (index < m).where(last - g, float('-inf')).exp()
      committed = last.exp().unsqueeze(-1) * S0 + (u * weight.unsqueeze(-1)).transpose(-1, -2) @ k
      assert (committed - states[m-1]).abs().max().item() < 1e-5, m

@pytest.mark.parametrize('kind', ['random', 'perfect', 'wrong'])
@pytest.mark.parametrize('block', [2, 4])
def test_parallel_verifier_reproduces_greedy_decoding(tmp_path, kind, block):
  # Exercises multirow prefill, held/flush of verified blocks and the
  # closed-form commit on CPU; drafting is forced on (no timing fallback).
  with Context(DEV='CPU'):
    model = Qwen35Model.from_gguf(checkpoint(tmp_path), device='CPU')
    prompt = [1, 2, 3]
    expected = list(model.generate(list(prompt), 9, reuse_cache=False))
    draft = tiny_draft(model) if kind == 'random' else RecordedDraft(model, prompt, expected, wrong=kind == 'wrong')
    generator = DFlashGenerator(model, draft, verifier='parallel')
    generator.min_tau, generator.plain_cost = 0.0, float('inf')
    for _ in range(2):
      assert list(generator.stream(list(prompt), max_new_tokens=9, block_size=block)) == expected
      assert not model._generating
      if kind == 'perfect': assert generator.stats.accepted == generator.stats.proposed > 0
      if kind == 'wrong': assert generator.stats.accepted < generator.stats.proposed

def test_timing_fallback_decodes_plain_rows_when_drafting_loses(tmp_path):
  with Context(DEV='CPU'):
    model = Qwen35Model.from_gguf(checkpoint(tmp_path), device='CPU')
    prompt = [1, 2, 3]
    expected = list(model.generate(list(prompt), 9, reuse_cache=False))
    generator = DFlashGenerator(model, RecordedDraft(model, prompt, expected, wrong=True), verifier='parallel')
    generator.plain_cost = 0.0  # any drafting round is slower than a measured plain step
    assert list(generator.stream(list(prompt), max_new_tokens=9, block_size=4)) == expected
    # Once enough timed rounds show drafting losing, the generator decodes ordinary rows.
    assert generator.stats.proposed <= generator.min_timed_rounds * 3
