import pytest
from tinygrad import Context, Tensor, UOp
from tinygrad.nn.state import get_state_dict
from infurnace.dflash import DFlashDraft, DFlashGenerator
from infurnace.engine import Model
from test_dflash import tiny_draft
from test_qwen35 import checkpoint

pytestmark = pytest.mark.nv

@pytest.mark.parametrize('block', [1, 4])
def test_nv_verified_output_rng_and_replay(tmp_path, block, verifier='sequential'):
  with Context(DEV='CPU'): checkpoint(tmp_path)
  model = Model.from_gguf(tmp_path / 'model.gguf', device='NV')
  with Context(DEV='NV'):
    draft = tiny_draft(model)
    Tensor.manual_seed(123)
    expected = list(model.generate([1, 2, 3], 7, reuse_cache=False))
    following = Tensor.rand(8).tolist()
    Tensor.manual_seed(123)
    generator = DFlashGenerator(model, draft, verifier=verifier)
    assert list(generator.stream([1, 2, 3], max_new_tokens=7, block_size=block)) == expected
    assert Tensor.rand(8).tolist() == following
    for _ in range(3):
      assert list(generator.stream([1, 2, 3], max_new_tokens=7, block_size=block)) == expected
      assert generator.stats.target_steps == 9
      assert not model._generating and draft.length == 0

def test_nv_denoise_matches_cpu_hidden_and_transient_cache(tmp_path):
  with Context(DEV='CPU'):
    checkpoint(tmp_path)
    cpu = Model.from_gguf(tmp_path / 'model.gguf', device='CPU')
    reference = tiny_draft(cpu)
    weights = get_state_dict(reference.weights)
    inputs = [(Tensor.randn(1, n, 8).realize(), Tensor.randn(1, 4, 8).realize(), pos) for n, pos in [(3, 3), (2, 5), (1, 6)]]
  gpu = Model.from_gguf(tmp_path / 'model.gguf', device='NV')
  with Context(DEV='NV'): draft = DFlashDraft(reference.config, gpu, weights)
  for features, noise, pos in inputs:
    position = UOp.variable('cross_position', features.shape[1], 12).bind(pos)
    with Context(DEV='CPU'): expected = reference._observe(features, noise, position).tolist()[0]
    with Context(DEV='NV'): actual = draft._observe(features.to('NV'), noise.to('NV'), position).tolist()[0]
    for got, wanted in zip(actual, expected): assert got == pytest.approx(wanted, abs=5e-5, rel=2e-4)
