import os

import pytest
from tinygrad import Context, Tensor, TinyJit, UOp

from infurnace.eagle.reference import SequentialVerifier

from infurnace.eagle.verify import BatchedVerifier
from infurnace.kernels.quant_gemv import quant_gemv
from test_nv_shared_q8 import packed
from test_speculative import nv_model, reference_step


pytestmark = pytest.mark.skipif(not os.environ.get('INFURNACE_TEST_NV'), reason='set INFURNACE_TEST_NV=1')


@pytest.mark.parametrize('typ', [12, 14])
@pytest.mark.parametrize('count', [2, 3, 4])
@pytest.mark.parametrize('reuse', [False, True])
def test_interleaved_q8_projection_matches_single_rows(typ, count, reuse):
  weight = packed(typ, 16, 2560, 'NV')
  jit = TinyJit(lambda x: quant_gemv(x, weight, out_features=16, in_features=2560, ggml_type=typ,
                                    interleave_tokens=True, exact_rows=True, reuse_tokens=reuse))
  for _ in range(4):
    x = Tensor.randn(1, count, 2560, device='NV').realize()
    actual = jit(x).clone().realize()
    expected = Tensor.cat(*(quant_gemv(x[:, i:i+1], weight, out_features=16, in_features=2560, ggml_type=typ)
                            for i in range(count)), dim=1)
    assert (actual-expected).abs().max().item() == 0


@pytest.mark.parametrize('compact', [False, True])
def test_batched_target_logits_features_and_rejected_tail(compact):
  model, reference = nv_model(), nv_model()
  layers, prompt = (0, 1, 3), [1, 2, 3, 4, 5]
  with BatchedVerifier(model, feature_layers=layers, compact_verify=compact) as verifier:
    verifier.prefill(prompt)
    with Context(DEV='NV'):
      # Build the reference prefix with the same symbolic prefill convention.
      with SequentialVerifier(reference, feature_layers=layers) as baseline:
        baseline.prefill(prompt)
        for _ in range(4):
          inputs = Tensor([[verifier._next_token, 6, 7, 8]], device='NV').int().realize()
          pos = verifier.committed_length
          logits, features = verifier._forward_many(inputs, UOp.variable('verify_position', 0, 60).bind(pos))
          Tensor.realize(logits, features)
          for index, token in enumerate(inputs.tolist()[0]):
            expected_logits, expected_features = reference_step(reference, Tensor([[token]], device='NV').int(),
              UOp.variable('start_pos', 0, 63).bind(pos+index), layers)
            assert logits[:, index, :].tolist() == expected_logits
            assert features[:, index:index+1, :].tolist() == expected_features
          # Rewrites these speculative slots; committed prefix advances only
          # along matching proposals, and future stale slots are masked.
          result = verifier.verify_chain([6, 7, 8])
          for index, token in enumerate(result.tokens):
            expected_logits, expected_features = reference_step(reference, Tensor([[token]], device='NV').int(),
              UOp.variable('start_pos', 0, 63).bind(pos+index), layers)
            assert result.features[:, index:index+1, :].tolist() == expected_features
          assert len(result.tokens) >= 1


@pytest.mark.parametrize('size', [1, 2, 3])
def test_sampling_selects_live_row_on_replay(size):
  jit = TinyJit(BatchedVerifier._sample)
  index = UOp.variable('verify_row', 0, size-1)
  temperature = Tensor([0.0], device='NV').realize()
  for repeat in range(4):
    logits = Tensor([[[10.0 if j == (i+repeat)%8 else 0.0 for j in range(8)]
                      for i in range(size)]], device='NV').realize()
    for row in range(size):
      assert jit(logits, temperature, index.bind(row)).item() == (row+repeat)%8


@pytest.mark.parametrize('transient', [False, True])
@pytest.mark.parametrize('fused', [False, True])
@pytest.mark.parametrize('zero_copy', [False, True])
@pytest.mark.parametrize('compact', [False, True])
def test_accepted_chain_continuation_matches_sequential(transient, fused, zero_copy, compact):
  model = nv_model()
  prompt, layers = [1, 2, 3, 4, 5], (0, 1, 3)
  with SequentialVerifier(model, feature_layers=layers) as reference:
    reference.prefill(prompt)
    tokens, steps = [], []
    for _ in range(12):
      result = reference.verify_chain([])
      tokens.extend(result.tokens)
      steps.extend(result.steps)
  with BatchedVerifier(model, feature_layers=layers, fuse_root_sampling=fused,
                       zero_copy_features=zero_copy, compact_verify=compact) as verifier:
    verifier.prefill(prompt)
    for i in range(0, 12, 3):
      result = verifier.verify_chain(tokens[i+1:i+3], transient_features=transient)
      assert result.tokens == tuple(tokens[i:i+3])
      assert result.accepted_drafts == 2
      if i+3 < 12: assert result.next_token == tokens[i+3]
      expected = steps[i].features.cat(*(s.features for s in steps[i+1:i+3]), dim=1)
      assert result.features.tolist() == expected.tolist()
      capture = verifier._batch_jits[3].captured
      if capture is not None:
        shared = result.features.uop.buffer.base is capture.ret[1].uop.buffer.base
        assert shared == (transient and zero_copy)


def test_failed_verification_invalidates_cache_and_releases_ownership():
  model, cache = nv_model(), {}
  with BatchedVerifier(model, feature_layers=(0, 1, 3), kernel_cache=cache) as verifier:
    verifier.prefill([1, 2, 3, 4, 5])
    assert cache
    def fail(*args): raise RuntimeError('injected verification failure')
    verifier._forward_many = fail
    with pytest.raises(RuntimeError, match='injected'):
      verifier.verify_chain([6, 7])
    assert cache == {} and not model._generating
  assert model.transformer._cached_tokens == []


@pytest.mark.parametrize('compact,fast', [(False, False), (True, False), (False, True)])
def test_kernel_cache_reuses_captures_without_reusing_session_state(compact, fast):
  model, cache = nv_model(), {}
  layers, prompt = (0, 1, 3), [1, 2, 3, 4, 5]
  saved = []
  for _ in range(2):
    with BatchedVerifier(model, feature_layers=layers, kernel_cache=cache, compact_verify=compact,
                         fast_io=fast, parallel_sampling=fast, loop_projections=fast, fused_projections=fast) as verifier:
      verifier.prefill(prompt)
      records = []
      for _ in range(4):
        result = verifier.verify_chain([6, 7], transient_features=True)
        records.append((result.tokens, result.next_token, result.features.tolist()))
      saved.append(records)
    assert not model._generating
    first_jit = cache['_batch_jits'][3]
    if len(saved) == 1: retained = first_jit
    else: assert first_jit is retained
  assert saved[0] == saved[1]
  # Different feature taps must not inherit captured feature layout.
  with BatchedVerifier(model, feature_layers=(0, 2, 3), kernel_cache=cache) as verifier:
    verifier.prefill(prompt)
    assert cache['_batch_jits'] == {}


@pytest.mark.skipif(not os.environ.get('INFURNACE_TEST_MODEL'), reason='set INFURNACE_TEST_MODEL')
def test_local_nv_batched_logits_and_features_exact():
  from infurnace.models.qwen3 import Qwen3Model
  model = Qwen3Model.from_gguf(os.environ['INFURNACE_TEST_MODEL'], max_context=512, device='NV',
                              nv_quant=True, nv_prefill=True, nv_fused_attention=True)
  prompt = list(range(1, 36))
  with SequentialVerifier(model) as reference:
    reference.prefill(prompt)
    tokens, steps = [], []
    for _ in range(12):
      result = reference.verify_chain([])
      tokens.extend(result.tokens)
      steps.extend(result.steps)
  # Reuse one resident target: compiled function caches can retain weights
  # beyond a parametrized test's lifetime, making a second full load unsafe.
  for fused in (False, True):
    for count in (2, 3, 4):
      with BatchedVerifier(model, local_ffn_down=True, loop_projections=True, batch_residuals=True,
                           batch_qk_norm=True, fused_projections=fused) as verifier, Context(DEV='NV'):
        verifier.prefill(prompt)
        jit = TinyJit(verifier._forward_many)
        for i in range(0, 12, count):
          inputs = Tensor([tokens[i:i+count]], device='NV').int().realize()
          logits, features = jit(inputs, UOp.variable('verify_position', 0, 512-count).bind(len(prompt)+i))
          Tensor.realize(logits, features)
          for row in range(count):
            assert (logits[:, row, :]-steps[i+row].logits).abs().max().item() == 0, (fused, count, i, row)
            assert (features[:, row:row+1, :]-steps[i+row].features).abs().max().item() == 0, (fused, count, i, row)
