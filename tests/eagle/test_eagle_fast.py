from types import SimpleNamespace

import pytest
from tinygrad import Context, Tensor, dtypes

from infurnace.eagle.generate import EagleGenerator
from infurnace.eagle.io import mapped_tensor, upload_tokens

def test_fast_profile_and_explicit_overrides(monkeypatch):
  import infurnace.eagle.generate as generation
  layer = SimpleNamespace(in_features=2560)
  model = SimpleNamespace(device='NV', transformer=SimpleNamespace(output=layer, blk=[]))
  enabled = []
  draft = SimpleNamespace(device='NV', enable_fast_io=lambda: enabled.append(True))
  monkeypatch.setattr(generation, 'generate_eagle', lambda model, draft, tokens, **kw: kw)
  engine = EagleGenerator(model, draft, fast=True)
  options = engine.generate([1])
  assert enabled == [True]
  assert options['draft_tokens'] == 1
  for flag in ('batched', 'fused_draft', 'loop_projections', 'fused_projections', 'zero_copy_features', 'fast_io', 'parallel_sampling'):
    assert options[flag] is True
  assert engine.generate([1], draft_tokens=3)['draft_tokens'] == 3
  # Unvalidated chunk counts retain ordinary reductions instead of crashing.
  layer.in_features = 6144
  assert EagleGenerator(model, draft, fast=True).generate([1])['fused_projections'] is False
  assert EagleGenerator(model, draft).generate([1]) == {'kernel_cache': {}}

def test_fast_profile_rejects_non_nv_and_non_boolean():
  model, draft = SimpleNamespace(device='CPU'), SimpleNamespace(device='CPU')
  with pytest.raises(ValueError, match='NV'): EagleGenerator(model, draft, fast=True)
  with pytest.raises(ValueError, match='boolean'): EagleGenerator(model, draft, fast=1)
  with pytest.raises(ValueError, match='NV'): mapped_tensor((1, 2), dtypes.int32, 'CPU')

def test_token_scratch_rejects_mismatched_size():
  with Context(DEV='CPU'):
    tensor = Tensor.empty(1, 2, dtype=dtypes.int32, device='CPU').realize()
    with pytest.raises(ValueError, match='shape/dtype'): upload_tokens(tensor, [1])

@pytest.mark.nv
@pytest.mark.parametrize('scored', [False, True])
def test_fast_draft_io_matches_reference_reset_and_replay(scored):
  from test_eagle import make_draft
  with Context(DEV='NV'):
    reference, fast = make_draft('NV'), make_draft('NV')
    fast.enable_fast_io()
    features = Tensor.randn(1, 2, 192, device='NV').realize()
    method = 'observe_propose_scored' if scored else 'observe_propose'
    for count in (1, 2, 4):
      for _ in range(4):
        reference.reset()
        fast.reset()
        assert getattr(fast, method)(features, [2, 4], count) == getattr(reference, method)(features, [2, 4], count)
        assert fast.length == reference.length == 2
        assert fast.last_hidden.tolist() == reference.last_hidden.tolist()
        assert fast.cache[:, :, :, :2].tolist() == reference.cache[:, :, :, :2].tolist()
    with pytest.raises(ValueError, match='before first use'): reference.enable_fast_io()
    assert all(t.uop.buffer.get_storage().host is not None for t in fast._token_inputs.values())
    assert all(t.uop.buffer.get_storage().host is not None for t in fast._proposal_outputs.values())

@pytest.mark.nv
def test_verifier_cache_invalidates_when_io_and_sampling_mode_changes():
  from infurnace.eagle.verify import BatchedVerifier
  from test_speculative import nv_model
  with Context(DEV='NV'):
    model, cache, records = nv_model(), {}, []
    for fast in (False, True, False):
      Tensor.manual_seed(91)
      with BatchedVerifier(model, feature_layers=(0, 1, 3), kernel_cache=cache,
                           fast_io=fast, parallel_sampling=fast) as verifier:
        verifier.prefill([1, 2, 3])
        assert not verifier._batch_jits and not verifier._sampling_jits
        rows = []
        for _ in range(4):
          result = verifier.verify_chain([6, 7])
          rows.append((result.tokens, result.next_token, result.features.tolist()))
        records.append(rows)
    assert records[0] == records[1] == records[2]
