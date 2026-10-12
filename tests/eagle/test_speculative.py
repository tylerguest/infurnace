import os
import random
import struct

import pytest
from tinygrad import Context, Tensor, UOp, dtypes

from infurnace.models.qwen3 import Qwen3Config, Qwen3Model, expected_weight_shapes
from infurnace.eagle.reference import SequentialVerifier

def tiny_model():
  config = Qwen3Config(8, 8, 16, 2, 1, 4, 4, 4, 10000.0, 1e-6, 32, 16)
  rng, weights = random.Random(12), {}
  for name, shape in expected_weight_shapes(config).items():
    size = 1
    for dim in shape: size *= dim
    values = [1.0]*size if 'norm' in name else [rng.uniform(-0.2, 0.2) for _ in range(size)]
    weights[name] = Tensor(values, device='CPU').reshape(shape)
  weights['output.weight'] = weights['token_embd.weight']
  return Qwen3Model(config, weights), weights

def reference_step(model, tokens, position, layers, consumed=1):
  with Context(DEV=model.device):
    x, features = model.transformer.token_embd(tokens).float(), []
    for index, block in enumerate(model.transformer.blk):
      if index in layers:
        x = x.contiguous().realize()
        features.append(Tensor(x.uop.base).flatten()[:consumed*model.config.embedding_length]
                        .reshape(1, consumed, model.config.embedding_length).clone().realize())
      x = block(x, position)
    logits = model.transformer.output(model.transformer.output_norm(x[:, -1:]))[:, -1, :]
    return logits.tolist(), features[0].cat(*features[1:], dim=-1).tolist()

def test_prefill_taps_match_target_and_snapshots_survive_replay():
  model, weights = tiny_model()
  reference = Qwen3Model(model.config, weights)
  prompt, saved = [1, 2, 3, 4, 5], []
  with SequentialVerifier(model, chunk_size=3) as verifier:
    steps = verifier.prefill(prompt)
    assert verifier.feature_layers == (2, 4, 5)
    for index, step in enumerate(steps):
      position, size = index*3, min(3, len(prompt)-index*3)
      sp, nt = UOp.variable('start_pos', 0, 15).bind(position), UOp.variable('toks', 1, 3).bind(size)
      t = Tensor(prompt+[0]*(16-len(prompt)), dtype=dtypes.int32, device='CPU').reshape(1, 16)
      logits, features = reference_step(reference, t[:, sp:sp+nt], sp, verifier.feature_layers, size)
      assert step.logits.tolist() == logits
      assert step.features.tolist() == features
      saved.append((step, step.logits.tolist(), step.features.tolist()))
    for _ in range(5):
      result = verifier.verify_chain([])
      token, step = result.tokens[0], result.steps[0]
      position = UOp.variable('start_pos', 0, 15).bind(verifier.committed_length-1)
      logits, features = reference_step(reference, Tensor([[token]], device='CPU', dtype=dtypes.int32), position,
                                        verifier.feature_layers)
      assert step.logits.tolist() == logits
      assert step.features.tolist() == features
      saved.append((step, step.logits.tolist(), step.features.tolist()))
    for step, logits, features in saved:
      assert step.logits.tolist() == logits
      assert step.features.tolist() == features
  assert prompt == [1, 2, 3, 4, 5]
  assert model.transformer.output.weight.uop is model.transformer.token_embd.weight.uop

def test_upstream_zero_temperature_decisions_including_exact_ties():
  model, weights = tiny_model()
  weights['token_embd.weight'] = Tensor.zeros(32, 8, device='CPU')
  weights['output.weight'] = weights['token_embd.weight']
  reference, candidate = Qwen3Model(model.config, weights), Qwen3Model(model.config, weights)
  Tensor.manual_seed(271)
  expected = list(reference.generate([1, 2, 3], 5, chunk_size=3, reuse_cache=False))
  Tensor.manual_seed(271)
  with SequentialVerifier(candidate, chunk_size=3) as verifier:
    steps = verifier.prefill([1, 2, 3])
    assert steps[-1].logits.abs().max().item() == 0
    actual = [verifier.verify_chain([]).tokens[0] for _ in range(5)]
  assert actual == expected

def continuation(model, prompt, count):
  with SequentialVerifier(model, chunk_size=3) as verifier:
    verifier.prefill(prompt)
    return [verifier.verify_chain([]).tokens[0] for _ in range(count)]

def test_chain_acceptance_mismatch_and_pending_root():
  model, _ = tiny_model()
  prompt = [1, 2, 3]
  expected = continuation(model, prompt, 6)
  with SequentialVerifier(model, chunk_size=3) as verifier:
    verifier.prefill(prompt)
    result = verifier.verify_chain(expected[1:4])
    assert result.tokens == tuple(expected[:4])
    assert result.accepted_drafts == 3
    assert result.next_token == expected[4]
    assert result.stop_reason == 'proposals'
    assert verifier.committed_tokens == tuple(prompt+expected[:4])
    bad = (expected[5]+1)%model.config.vocab_size
    result = verifier.verify_chain([bad])
    assert result.tokens == (expected[4],)
    assert result.accepted_drafts == 0
    assert result.next_token == expected[5]
    assert result.stop_reason == 'mismatch'
    # The mismatch was not written. Reusing its slot processes the correct root.
    assert verifier.verify_chain([]).tokens == (expected[5],)

def test_eos_context_and_round_budget():
  model, _ = tiny_model()
  prompt = [1, 2, 3]
  expected = continuation(model, prompt, 4)
  with SequentialVerifier(model, chunk_size=3, eos_token_ids=(expected[0],)) as verifier:
    verifier.prefill(prompt)
    result = verifier.verify_chain(expected[1:])
    assert result.tokens == (expected[0],)
    assert result.next_token is None and result.stop_reason == 'eos'
    with pytest.raises(RuntimeError, match='finished'): verifier.verify_chain([])
  with SequentialVerifier(model, chunk_size=3) as verifier:
    verifier.prefill(prompt)
    result = verifier.verify_chain(expected[1:], max_tokens=2)
    assert result.tokens == tuple(expected[:2])
    assert result.next_token == expected[2] and result.stop_reason == 'limit'
  with SequentialVerifier(model, chunk_size=3) as verifier:
    verifier.prefill([1]*15)
    result = verifier.verify_chain([2, 3, 4])
    assert len(result.tokens) == 1
    assert verifier.committed_length == 16
    assert result.next_token is None and result.stop_reason == 'context'

def test_exclusive_ownership_close_and_normal_generation_resume():
  model, _ = tiny_model()
  expected = list(model.generate([1, 2, 3], 4, chunk_size=3, reuse_cache=False))
  verifier = SequentialVerifier(model, chunk_size=3)
  with verifier:
    assert model.transformer._cached_tokens == []
    verifier.prefill([4, 5, 6])
    verifier.verify_chain([])
    with pytest.raises(RuntimeError): next(model.generate([1], 1))
    with pytest.raises(RuntimeError): model.reset()
    with pytest.raises(RuntimeError):
      with SequentialVerifier(model): pass
  assert not model._generating and model.transformer._cached_tokens == []
  assert list(model.generate([1, 2, 3], 4, chunk_size=3)) == expected
  with pytest.raises(RuntimeError, match='single-use'):
    with verifier: pass
  with pytest.raises(RuntimeError, match='context manager'): verifier.verify_chain([])

def test_exception_invalidates_partial_kv_and_releases_owner(monkeypatch):
  model, _ = tiny_model()
  with SequentialVerifier(model, chunk_size=3) as verifier:
    verifier.prefill([1, 2, 3])
    def fail(*args): raise RuntimeError('simulated device failure')
    monkeypatch.setattr(verifier, '_decode_jit', fail)
    with pytest.raises(RuntimeError, match='simulated device failure'): verifier.verify_chain([])
    assert not model._generating and model.transformer._cached_tokens == []
    with pytest.raises(RuntimeError, match='context manager'): verifier.verify_chain([])
  assert len(list(model.generate([1, 2], 1, chunk_size=3))) == 1

@pytest.mark.parametrize('layers', [(2, 2, 3), (3, 2, 4), (0, 1, 8), (False, 1, 3), (0, 1)])
def test_invalid_feature_layers(layers):
  model, _ = tiny_model()
  with pytest.raises(ValueError, match='feature_layers'): SequentialVerifier(model, feature_layers=layers)

def test_invalid_inputs_and_lifecycle():
  model, _ = tiny_model()
  verifier = SequentialVerifier(model)
  with pytest.raises(RuntimeError, match='context manager'): verifier.prefill([1])
  with verifier:
    with pytest.raises(RuntimeError, match='prefill'): verifier.verify_chain([])
    for prompt in ([], [True], [32], [1]*16):
      with pytest.raises(ValueError): verifier.prefill(prompt)
    verifier.prefill([1, 2])
    with pytest.raises(RuntimeError, match='already'): verifier.prefill([3])
    for draft in ([True], [-1], [32]):
      with pytest.raises(ValueError): verifier.verify_chain(draft)
    for budget in (0, -1, True, 1.5):
      with pytest.raises(ValueError): verifier.verify_chain([], max_tokens=budget)

def nv_model():
  from infurnace.gguf import PackedTensor
  from infurnace.models.qwen3 import enable_decode_attention
  config = Qwen3Config(4, 256, 512, 32, 8, 128, 128, 128, 1000000.0, 1e-6, 32, 64)
  rng, weights, entries = random.Random(29), {}, {}
  for name, shape in expected_weight_shapes(config).items():
    if 'norm' in name:
      weights[name] = Tensor.ones(*shape, device='NV')
      continue
    rows, width = shape
    raw = bytearray(rng.getrandbits(8) for _ in range(rows*width//256*144))
    for i in range(0, len(raw), 144): raw[i:i+4] = struct.pack('<ee', 0.001, 0.0002)
    tensor = Tensor(bytes(raw), dtype=dtypes.uint8, device='NV').realize()
    entries[name] = PackedTensor(tensor, shape, 12)
    weights[name] = entries[name].decode()
  weights['output.weight'], entries['output.weight'] = weights['token_embd.weight'], entries['token_embd.weight']
  model = Qwen3Model(config, weights)
  model._enable_nv_quant(entries, nv_prefill=True, nv_prepared_weights=True)
  assert enable_decode_attention(model.transformer.blk, fused=True) == 4
  return model

@pytest.mark.nv
def test_nv_packed_target_features_jit_and_stale_tail():
  model, reference = nv_model(), nv_model()
  layers, prompt = (0, 1, 3), [1, 2, 3, 4, 5]
  with SequentialVerifier(model, chunk_size=32, feature_layers=layers) as verifier:
    steps = verifier.prefill(prompt)
    with Context(DEV='NV'):
      t = Tensor([prompt+[0]*59], dtype=dtypes.int32, device='NV')
      for index, step in enumerate(steps):
        position, size = index*32, min(32, len(prompt)-index*32)
        sp, nt = UOp.variable('start_pos', 0, 63).bind(position), UOp.variable('toks', 1, 32).bind(size)
        logits, features = reference_step(reference, t[:, sp:sp+nt], sp, layers, size)
        assert step.logits.tolist() == logits
        assert step.features.tolist() == features
      # Poison inactive tail slots. Prefix length, not physical contents, must
      # determine which KV entries a future token can see.
      for block in model.transformer.blk:
        block.cache_kv[:, :, :, len(prompt):, :].assign(123.0).realize()
      saved = []
      for _ in range(5):
        result = verifier.verify_chain([])
        step, token = result.steps[0], result.tokens[0]
        position = UOp.variable('start_pos', 0, 63).bind(verifier.committed_length-1)
        logits, features = reference_step(reference, Tensor([[token]], dtype=dtypes.int32, device='NV'), position, layers)
        assert step.logits.tolist() == logits
        assert step.features.tolist() == features
        saved.append((step, step.logits.tolist(), step.features.tolist()))
      for step, logits, features in saved:
        assert step.logits.tolist() == logits
        assert step.features.tolist() == features
  assert model.transformer.output.weight.uop is model.transformer.token_embd.weight.uop

@pytest.mark.skipif(not (os.environ.get('INFURNACE_TEST_NV') and os.environ.get('INFURNACE_TEST_MODEL')),
                    reason='set INFURNACE_TEST_NV=1 and INFURNACE_TEST_MODEL for real NV target')
def test_local_nv_target_reference_features_and_logits():
  model = Qwen3Model.from_gguf(os.environ['INFURNACE_TEST_MODEL'], max_context=1024, device='NV',
                              nv_quant=True, nv_prefill=True, nv_fused_attention=True)
  prompt = list(range(1, 36))
  with SequentialVerifier(model) as verifier:
    steps = verifier.prefill(prompt)
    assert verifier.feature_layers == (2, 18, 33)
    with Context(DEV='NV'):
      t = Tensor([prompt+[0]*(1024-len(prompt))], dtype=dtypes.int32, device='NV')
      for index, step in enumerate(steps):
        position, size = index*32, min(32, len(prompt)-index*32)
        sp, nt = UOp.variable('start_pos', 0, 1023).bind(position), UOp.variable('toks', 1, 32).bind(size)
        logits, features = reference_step(model, t[:, sp:sp+nt], sp, verifier.feature_layers, size)
        assert step.logits.tolist() == logits
        assert step.features.tolist() == features
      for _ in range(4):
        result = verifier.verify_chain([])
        step, token = result.steps[0], result.tokens[0]
        position = UOp.variable('start_pos', 0, 1023).bind(verifier.committed_length-1)
        logits, features = reference_step(model, Tensor([[token]], dtype=dtypes.int32, device='NV'), position,
                                          verifier.feature_layers)
        assert step.logits.tolist() == logits
        assert step.features.tolist() == features
  assert model.transformer._cached_tokens == [] and not model._generating
