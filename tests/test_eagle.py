"""Draft cache/alignment and whole-chain JIT regression checks."""
import os

import pytest
from tinygrad import Context, Tensor, TinyJit, UOp, dtypes

from infurnace.eagle import EagleDraft


def draft_fixture(device='CPU'):
  config = dict(hidden_size=64, intermediate_size=128, num_attention_heads=2,
                num_key_value_heads=1, head_dim=64, vocab_size=32, draft_vocab_size=16,
                rms_norm_eps=1e-6, rope_theta=1e6, num_hidden_layers=1,
                hidden_act='silu', max_position_embeddings=32)
  shapes = {'fc.weight': (64, 192), 'lm_head.weight': (16, 64),
            'midlayer.self_attn.q_proj.weight': (128, 128),
            'midlayer.self_attn.k_proj.weight': (64, 128),
            'midlayer.self_attn.v_proj.weight': (64, 128),
            'midlayer.self_attn.o_proj.weight': (64, 128),
            'midlayer.mlp.gate_proj.weight': (128, 64),
            'midlayer.mlp.up_proj.weight': (128, 64),
            'midlayer.mlp.down_proj.weight': (64, 128)}
  with Context(DEV=device):
    Tensor.manual_seed(81)
    weights = {k: (Tensor.randn(*s, device=device)*0.04).bfloat16().realize() for k, s in shapes.items()}
    for name in ('midlayer.hidden_norm', 'midlayer.input_layernorm', 'midlayer.post_attention_layernorm', 'norm'):
      weights[name+'.weight'] = Tensor.ones(64, device=device).bfloat16().realize()
    # Sparse target IDs deliberately distinguish offsets from absolute IDs.
    weights['d2t'] = Tensor(list(range(16)), dtype=dtypes.int64, device=device)
    weights['t2d'] = Tensor([i%2 == 0 for i in range(32)], dtype=dtypes.bool, device=device)
    embedding = (Tensor.randn(32, 64, device=device)*0.1).bfloat16().realize()
  return config, weights, lambda ids: embedding[ids]


def make_draft(device='CPU'):
  config, weights, embedding = draft_fixture(device)
  return EagleDraft(config, weights, embedding, capacity=32, device=device)


def test_vocabulary_and_strict_loading():
  config, weights, embedding = draft_fixture()
  draft = EagleDraft(config, weights, embedding, capacity=32, device='CPU')
  assert draft.mapping.tolist() == list(range(0, 32, 2))
  with pytest.raises(ValueError, match='keys'):
    EagleDraft(config, weights | {'unexpected': weights['fc.weight']}, embedding, capacity=32, device='CPU')
  with pytest.raises(ValueError, match='tensor'):
    EagleDraft(config, weights | {'fc.weight': weights['fc.weight'].float()}, embedding, capacity=32, device='CPU')
  with pytest.raises(ValueError, match='vocabulary'):
    EagleDraft(config, weights | {'d2t': Tensor.zeros(16, dtype=dtypes.int64)}, embedding, capacity=32, device='CPU')


@pytest.mark.parametrize('device', ['CPU', 'NV'])
def test_chain_replay_matches_reference_and_preserves_prefix(device):
  if device == 'NV' and not os.environ.get('INFURNACE_TEST_NV'): pytest.skip('set INFURNACE_TEST_NV=1')
  with Context(DEV=device):
    draft = make_draft(device)
    with pytest.raises(RuntimeError, match='observe'): draft.propose(1)
    features = (Tensor.randn(1, 3, 192, device=device)*0.1).realize()
    draft.observe(features, [2, 4, 6])
    for round_index in range(4):
      prefix = draft.cache[:, :, :, :draft.length].contiguous().clone().realize()
      hidden = draft.last_hidden.clone().realize()
      for count in (1, 2, 4):
        expected = draft.propose(count, fast_loop=False)
        # Exercise warmup, capture and replay; each result is an independent list.
        for _ in range(3): assert draft.propose(count, fast_loop=True) == expected
        assert (draft.last_hidden-hidden).float().abs().max().item() == 0
        assert (draft.cache[:, :, :, :draft.length]-prefix).float().abs().max().item() == 0
      assert draft.length == 3+round_index
      draft.observe(features[:, :1].contiguous(), [8+2*round_index])
    draft.reset()
    assert draft.length == 0 and draft.last_hidden is None
    draft.observe(features, [2, 4, 6])
    assert draft.propose(4, fast_loop=True) == draft.propose(4, fast_loop=False)


@pytest.mark.parametrize('device', ['CPU', 'NV'])
@pytest.mark.parametrize('count', [1, 2, 4, 8])
def test_last_query_observation_matches_full_rows_and_kv_on_replay(device, count):
  if device == 'NV' and not os.environ.get('INFURNACE_TEST_NV'): pytest.skip('set INFURNACE_TEST_NV=1')
  with Context(DEV=device):
    draft = make_draft(device)
    full, last = TinyJit(draft._forward), TinyJit(draft._observe)
    for repeat in range(4):
      features = (Tensor.randn(1, count, 192, device=device)*0.1).realize()
      ids = Tensor([[2+2*i for i in range(count)]], device=device, dtype=dtypes.int32).realize()
      position = UOp.variable('observe_test_position', 0, 24).bind(repeat)
      expected = full(features, ids, position)[:, -1:].contiguous().clone().realize()
      cache = draft.cache.clone().realize()
      actual = last(features, ids, position).clone().realize()
      assert (actual-expected).float().abs().max().item() == 0
      assert (draft.cache-cache).float().abs().max().item() == 0


@pytest.mark.parametrize('device', ['CPU', 'NV'])
@pytest.mark.parametrize('count', [1, 2, 4, 7])
def test_combined_observation_proposal_matches_separate_captures(device, count):
  if device == 'NV' and not os.environ.get('INFURNACE_TEST_NV'): pytest.skip('set INFURNACE_TEST_NV=1')
  with Context(DEV=device):
    reference, fused = make_draft(device), make_draft(device)
    for repeat in range(4):
      features = (Tensor.randn(1, 5, 192, device=device)*0.1).realize()[:, :3].contiguous()
      ids = [2+repeat, 4+repeat, 6+repeat]
      reference.observe(features, ids)
      expected = reference.propose(count, fast_loop=True)
      assert fused.observe_propose(features, ids, count) == expected
      assert fused.length == reference.length
      assert (fused.last_hidden-reference.last_hidden).float().abs().max().item() == 0
      assert (fused.cache[:, :, :, :fused.length]-reference.cache[:, :, :, :reference.length]).float().abs().max().item() == 0
    assert fused.propose(count, fast_loop=True) == reference.propose(count, fast_loop=True)
    fused.reset()
    for invalid in (0, True, 8):
      with pytest.raises(ValueError): fused.observe_propose(features, ids, invalid)
    assert fused.length == 0


@pytest.mark.parametrize('device', ['CPU', 'NV'])
@pytest.mark.parametrize('count', [1, 2])
def test_scored_capture_preserves_proposals_and_replay(device, count):
  if device == 'NV' and not os.environ.get('INFURNACE_TEST_NV'): pytest.skip('set INFURNACE_TEST_NV=1')
  with Context(DEV=device):
    reference, scored = make_draft(device), make_draft(device)
    for repeat in range(4):
      features = Tensor.randn(1, 2, 192, device=device).realize()
      ids = [repeat+1, repeat+2]
      expected = reference.observe_propose(features, ids, count)
      actual, margins = scored.observe_propose_scored(features, ids, count)
      assert actual == expected and len(margins) == count and all(m >= 0 for m in margins)
      assert (scored.last_hidden-reference.last_hidden).float().abs().max().item() == 0
      assert (scored.cache[:, :, :, :scored.length]-reference.cache[:, :, :, :reference.length]).float().abs().max().item() == 0


def test_observe_and_proposal_validation():
  draft = make_draft()
  f = Tensor.zeros(1, 1, 192, device='CPU').realize()
  with pytest.raises(ValueError): draft.observe(f, [])
  with pytest.raises(ValueError): draft.observe(f, [32])
  with pytest.raises(ValueError): draft.observe(f[:, :, :64], [2])
  draft.observe(f, [2])
  assert draft.propose(0) == []
  for count in (-1, True, 33):
    with pytest.raises(ValueError): draft.propose(count)
