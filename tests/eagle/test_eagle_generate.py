import pytest

from tinygrad import Context
from tinygrad.helpers import DEV

from infurnace.eagle.generate import EagleGenerator, generate_eagle
from infurnace.eagle.reference import SequentialVerifier
from test_speculative import tiny_model

class RecordedDraft:
  def __init__(self, model, expected, *, wrong=False):
    self.device, self.capacity = model.device, model.config.context_length
    self.expected, self.wrong = expected, wrong
    self.observations = []
    self.resets = 0

  def reset(self):
    self.resets += 1
    self.length = 0

  def observe(self, features, paired_ids):
    assert features.shape[1] == len(paired_ids)
    self.observations.append((self.length, list(paired_ids), features.tolist()))
    self.length += len(paired_ids)

  def propose(self, count, *, fast_loop):
    assert fast_loop
    return [31]*count if self.wrong else self.expected[self.length+1:self.length+1+count]

  def observe_propose(self, features, paired_ids, count):
    self.observe(features, paired_ids)
    return self.propose(count, fast_loop=True)

  def observe_propose_scored(self, features, paired_ids, count):
    return self.observe_propose(features, paired_ids, count), [2.0]*count

@pytest.mark.parametrize('wrong', [False, True])
@pytest.mark.parametrize('fused', [False, True])
def test_verified_generation_alignment_and_budget(wrong, fused):
  model, _ = tiny_model()
  prompt = [1, 2, 3]
  with SequentialVerifier(model) as verifier:
    prefill = verifier.prefill(prompt)
    steps, tokens = [], []
    for _ in range(6):
      result = verifier.verify_chain([])
      steps.extend(result.steps)
      tokens.extend(result.tokens)
  draft = RecordedDraft(model, prompt+tokens, wrong=wrong)
  rounds = list(generate_eagle(model, draft, prompt, draft_tokens=2, max_new_tokens=6, fused_draft=fused))
  assert [token for result in rounds for token in result.tokens] == tokens
  assert rounds[-1].stop_reason == 'limit'
  assert draft.resets == 2 and draft.length == 0
  first = draft.observations[0]
  assert first[1] == prompt[1:]+tokens[:1]
  assert first[2] == prefill[0].features.tolist()
  for position, paired, features in draft.observations[1:]:
    start = position-len(prompt)
    assert paired == tokens[start+1:start+len(paired)+1]
    expected = steps[start].features.cat(*(step.features for step in steps[start+1:start+len(paired)]), dim=1)
    assert features == expected.tolist()
  assert not model._generating and model.transformer._cached_tokens == []

@pytest.mark.parametrize('margin', [0.0, 3.0])
def test_confidence_gating_is_exact_and_can_skip_proposals(margin):
  model, _ = tiny_model()
  prompt = [1, 2, 3]
  with SequentialVerifier(model) as verifier:
    verifier.prefill(prompt)
    tokens = [verifier.verify_chain([]).tokens[0] for _ in range(6)]
  draft = RecordedDraft(model, prompt+tokens)
  rounds = list(generate_eagle(model, draft, prompt, draft_tokens=1, max_new_tokens=6,
                               fused_draft=True, draft_confidence=margin))
  assert [token for result in rounds for token in result.tokens] == tokens
  if margin > 2: assert all(result.proposed == 0 for result in rounds)
  assert draft.resets == 2 and not model._generating

@pytest.mark.parametrize('fused', [False, True])
def test_cancellation_releases_target_and_draft(fused):
  model, _ = tiny_model()
  draft = RecordedDraft(model, [], wrong=True)
  stream = generate_eagle(model, draft, [1, 2, 3], draft_tokens=2, fused_draft=fused)
  next(stream)
  assert model._generating
  stream.close()
  assert not model._generating and draft.length == 0 and draft.resets == 2

def test_token_stream_close_mid_accepted_round_and_device_context():
  model, _ = tiny_model()
  prompt = [1, 2, 3]
  with SequentialVerifier(model) as verifier:
    verifier.prefill(prompt)
    expected = [verifier.verify_chain([]).tokens[0] for _ in range(4)]
  draft = RecordedDraft(model, prompt+expected)
  engine = EagleGenerator(model, draft)
  with Context(DEV='NULL'):
    outside_device = DEV.value
    stream = engine.stream(prompt, draft_tokens=2, max_new_tokens=4)
    assert next(stream) == expected[0]
    assert DEV.value == outside_device and model._generating
    stream.close()
    assert DEV.value == outside_device and not model._generating
  assert draft.resets == 2 and draft.length == 0

@pytest.mark.parametrize('flag', [True, 1])
def test_cta_local_candidate_requires_explicit_batched_boolean_flag(flag):
  model, _ = tiny_model()
  draft = RecordedDraft(model, [], wrong=True)
  with pytest.raises(ValueError, match='local_ffn_down requires'):
    list(generate_eagle(model, draft, [1, 2, 3], local_ffn_down=flag))
  assert not model._generating and draft.resets == 0

@pytest.mark.parametrize('eos', [False, True])
@pytest.mark.parametrize('fused', [False, True])
def test_last_pending_root_needs_no_drafting_or_verification(monkeypatch, eos, fused):
  model, _ = tiny_model()
  prompt = [1, 2, 3]
  with SequentialVerifier(model) as verifier:
    root = verifier.prefill(prompt)[-1].next_token
  draft = RecordedDraft(model, [], wrong=True)
  monkeypatch.setattr(draft, 'observe', lambda *a: pytest.fail('unnecessary final-root observation'))
  monkeypatch.setattr(SequentialVerifier, 'verify_chain', lambda *a, **kw: pytest.fail('unnecessary final-root pass'))
  rounds = list(generate_eagle(model, draft, prompt, max_new_tokens=1, eos_token_ids=(root,) if eos else (), fused_draft=fused))
  assert rounds[0].tokens == (root,) and rounds[0].stop_reason == ('eos' if eos else 'limit')
  assert draft.resets == 2 and not model._generating

def test_eos_and_context_boundaries():
  model, _ = tiny_model()
  prompt = [1, 2, 3]
  with SequentialVerifier(model) as verifier:
    root = verifier.prefill(prompt)[-1].next_token
  draft = RecordedDraft(model, [], wrong=True)
  rounds = list(generate_eagle(model, draft, prompt, eos_token_ids=(root,)))
  assert rounds[0].tokens == (root,) and rounds[0].stop_reason == 'eos' and len(rounds) == 1
  rounds = list(generate_eagle(model, draft, list(range(1, 16))))
  assert len(rounds) == 1 and len(rounds[0].tokens) == 1 and rounds[0].stop_reason == 'context'
