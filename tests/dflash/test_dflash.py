from types import SimpleNamespace
import math
import struct
import pytest
from tinygrad import Context, Tensor, UOp, dtypes
from tinygrad.nn.state import get_state_dict
from infurnace.dflash import DFlashDraft, DFlashGenerator
from infurnace.dflash.draft import DFlashConfig, DraftWeights
from infurnace.dflash.target import TargetObserver
from infurnace.engine import Model
from test_qwen35 import checkpoint

def tiny_target(path):
  checkpoint(path)
  return Model.from_gguf(path / 'model.gguf', device='CPU')

def tiny_draft(model):
  config = DFlashConfig(2, 8, 12, 2, 1, 4, 1e-6, 10000000, 4, 15, (1,), (True, False), 8)
  state = {k: Tensor.randn(*v.shape) * 0.05 for k, v in get_state_dict(DraftWeights(config)).items()}
  return DFlashDraft(config, model, state)

class RecordedDraft:
  def __init__(self, model, prompt, expected, wrong=False):
    self.target, self.device = model, model.device
    self.config = SimpleNamespace(feature_layers=(1,), block_size=4)
    self.prompt, self.expected, self.wrong = prompt, expected, wrong
    self.observations, self.rows = [], []
    self.resets = 0

  def reset(self):
    self.resets += 1
    self.rows = []

  def synced(self, position): self.rows = []

  def observe(self, features):
    self.rows += features.tolist()[0]  # read now: step features alias capture scratch

  def propose(self, anchor, position, count):
    self.observations.append((position, anchor, [self.rows]))
    self.rows = []
    index = position - len(self.prompt)
    values = self.expected[index+1:index+1+count]
    if self.wrong: values = [(values[0]+1) % self.target.config.vocab_size] + values[1:]
    return values

@pytest.mark.parametrize('wrong', [False, True])
@pytest.mark.parametrize('budget', [1, 2, 7, 13])
def test_exact_output_rng_and_terminal_row(tmp_path, wrong, budget, verifier='sequential'):
  with Context(DEV='CPU'):
    model = tiny_target(tmp_path)
    prompt = [1, 2, 3]
    Tensor.manual_seed(123)
    expected = list(model.generate(prompt, budget, reuse_cache=False))
    following = Tensor.rand(8).tolist()
    Tensor.manual_seed(123)
    draft = RecordedDraft(model, prompt, expected, wrong)
    generator = DFlashGenerator(model, draft, verifier=verifier)
    actual = list(generator.stream(prompt, max_new_tokens=budget))
    assert actual == expected
    assert Tensor.rand(8).tolist() == following
    assert generator.stats.target_steps == len(prompt) + len(expected) - 1
    assert not model._generating and not model.transformer._cached_tokens
    assert draft.resets == 2
    if wrong: assert generator.stats.accepted == 0
    elif budget > 1: assert generator.stats.accepted > 0

def test_observation_alignment_and_snapshot_lifetime(tmp_path, verifier='sequential'):
  with Context(DEV='CPU'):
    model = tiny_target(tmp_path)
    prompt = [1, 2, 3]
    expected = list(model.generate(prompt, 8, reuse_cache=False))
    draft = RecordedDraft(model, prompt, expected)
    generator = DFlashGenerator(model, draft, verifier=verifier)
    assert list(generator.stream(prompt, max_new_tokens=8)) == expected
    with TargetObserver(model, (1,)) as observer:
      snapshots = []
      for token in prompt + expected[:-1]:
        _, features = observer.step(token)
        snapshots.append(features)
      for index, (position, anchor, observed) in enumerate(draft.observations):
        start = 0 if index == 0 else draft.observations[index-1][0]
        wanted = snapshots[start].cat(*snapshots[start+1:position], dim=1).tolist()
        assert observed == wanted
        assert anchor == expected[position-len(prompt)]
      assert snapshots[0].tolist() != snapshots[-1].tolist()

def test_cancellation_eos_and_replay(tmp_path, verifier='sequential'):
  with Context(DEV='CPU'):
    model = tiny_target(tmp_path)
    prompt = [1, 2]
    expected = list(model.generate(prompt, 7, reuse_cache=False))
    draft = RecordedDraft(model, prompt, expected)
    generator = DFlashGenerator(model, draft, verifier=verifier)
    stream = generator.stream(prompt, max_new_tokens=7)
    assert next(stream) == expected[0]
    with pytest.raises(RuntimeError): model.reset()
    with pytest.raises(RuntimeError): list(generator.stream(prompt, max_new_tokens=2))
    stream.close()
    assert not model._generating and draft.resets == 2
    for _ in range(3): assert list(generator.stream(prompt, max_new_tokens=7)) == expected
    assert list(generator.stream(prompt, max_new_tokens=7, eos_token_ids=(expected[0],))) == expected[:1]
    assert generator.stats.target_steps == len(prompt)

def test_real_draft_determinism_and_transient_cache(tmp_path):
  with Context(DEV='CPU'):
    model = tiny_target(tmp_path)
    draft = tiny_draft(model)
    features = Tensor.randn(1, 3, 8).realize()
    more = Tensor.randn(1, 2, 8).realize()
    results = []
    for _ in range(3):
      draft.reset()
      draft.observe(features)
      results.append(draft.propose(2, 3, 3))
      assert draft.length == 3
      draft.observe(more)
      assert len(draft.propose(3, 5, 2)) == 2
    assert results[0] == results[1] == results[2]
    with pytest.raises(ValueError): draft.propose(2, 3, 3)  # position 3 is no longer the observed prefix
    draft.reset()
    draft.observe(features)
    with pytest.raises(ValueError): draft.propose(2, 2, 3)
    with pytest.raises(ValueError): draft.propose(2, 3, 4)  # count must stay below the block size

def test_observing_rows_in_pieces_matches_one_observation(tmp_path):
  # Staging splits rows at chunk boundaries; any split must absorb identically.
  with Context(DEV='CPU'):
    model = tiny_target(tmp_path)
    draft = tiny_draft(model)
    rows = Tensor.randn(1, 11, 8).realize()  # crosses two 4-row chunks
    draft.observe(rows)
    whole = draft.propose(5, 11, 3)
    for pieces in ([1] * 11, [3, 1, 6, 1], [4, 4, 3], [7, 4]):
      draft.reset()
      start = 0
      for n in pieces:
        draft.observe(rows[:, start:start + n].contiguous().realize())
        start += n
      assert draft.observed == 11 and draft.propose(5, 11, 3) == whole

def test_real_tiny_draft_verified_output(tmp_path, verifier='sequential'):
  with Context(DEV='CPU'):
    model = tiny_target(tmp_path)
    expected = list(model.generate([1, 2, 3], 7, reuse_cache=False))
    draft = tiny_draft(model)
    generator = DFlashGenerator(model, draft, verifier=verifier)
    for _ in range(3): assert list(generator.stream([1, 2, 3], max_new_tokens=7)) == expected

def test_zero_budget_and_invalid_options(tmp_path):
  with Context(DEV='CPU'):
    model = tiny_target(tmp_path)
    draft = tiny_draft(model)
    generator = DFlashGenerator(model, draft)
    assert list(generator.stream([1], max_new_tokens=0)) == []
    for options in ({'temperature': -0.1}, {'block_size': 0}, {'block_size': 5}, {'eos_token_ids': (16,)}):
      with pytest.raises(ValueError): list(generator.stream([1], max_new_tokens=2, **options))
    assert not model._generating

@pytest.mark.parametrize('causal', [None, False, True])
@pytest.mark.parametrize('sliding', [False, True])
def test_attention_mask_against_uniform_scalar_oracle(tmp_path, causal, sliding):
  with Context(DEV='CPU'):
    model = tiny_target(tmp_path)
    config = DFlashConfig(1, 8, 12, 2, 1, 4, 1e-6, 10000000, 4, 15, (1,), (sliding,), 2, causal)
    shapes = get_state_dict(DraftWeights(config))
    state = {k: Tensor.ones(*v.shape) if k.endswith('norm.weight') else Tensor.zeros(*v.shape) for k, v in shapes.items()}
    state['blk.0.attn_v.weight'] = Tensor([[int(j == i) for j in range(8)] for i in range(4)])
    state['blk.0.attn_output.weight'] = Tensor.eye(8)
    draft = DFlashDraft(config, model, state)
    noise = [[1., 2., 3., 4., 5., 6., 7., 8.], [2., -1., 4., 3., 0., -2., 1., 5.], [0., 3., 1., 2., 4., 5., -1., 6.]]
    normalized = [[v / math.sqrt(sum(x*x for x in row)/8 + config.epsilon) for v in row] for row in noise]
    values = [[0.]*4]*2 + [[struct.unpack('<e', struct.pack('<e', v))[0] for v in row[:4]] for row in normalized]
    expected = []
    expected_causal = sliding if causal is None else causal
    for q in (3, 4):
      visible = [k for k in range(5) if (not expected_causal or k <= q) and (not sliding or abs(q-k) < 2)]
      average = [sum(values[k][j] for k in visible)/len(visible) for j in range(4)] * 2
      residual = [x+y for x, y in zip(noise[q-2], average)]
      expected.append([v / math.sqrt(sum(x*x for x in residual)/8 + config.epsilon) for v in residual])
    features = Tensor.zeros(1, 2, 8)
    ids = Tensor(noise).reshape(1, 3, 8)
    position = UOp.variable('oracle_position', 2, 12).bind(2)
    for future_value in (0., 100.):
      draft.cache[0].assign(Tensor.full(draft.cache[0].shape, future_value, dtype=dtypes.float16)).realize()
      actual = draft._observe(features, ids, position).tolist()[0]
      for got, wanted in zip(actual, expected): assert got == pytest.approx(wanted, abs=2e-5)

def test_gguf_metadata_not_stale_sidecar():
  model = SimpleNamespace(config=SimpleNamespace(embedding_length=8, block_count=32, vocab_size=16),
                          metadata={'tokenizer.ggml.tokens': [str(i) for i in range(16)]})
  metadata = {'general.architecture': 'dflash', 'dflash.block_count': 6, 'dflash.embedding_length': 8,
    'dflash.feed_forward_length': 12, 'dflash.attention.head_count': 2, 'dflash.attention.head_count_kv': 1,
    'dflash.attention.key_length': 4, 'dflash.attention.layer_norm_rms_epsilon': 1e-6, 'dflash.rope.freq_base': 10000000,
    'dflash.block_size': 16, 'dflash.target_layers': [2, 6, 10, 14, 18, 22, 26, 30],
    'dflash.attention.sliding_window': 4096, 'dflash.attention.sliding_window_pattern': [True]*5+[False],
    'tokenizer.ggml.mask_token_id': 15, 'tokenizer.ggml.tokens': model.metadata['tokenizer.ggml.tokens']}
  c = DFlashConfig.from_gguf(metadata, model)
  assert c.layers == 6 and len(c.feature_layers) == 8 and c.mask_token == 15
  assert c.sliding == (True,)*5 + (False,) and c.causal is None
  assert [c.layer_causal(i) for i in range(6)] == [True]*5 + [False]
  for flag in (False, True):
    explicit = DFlashConfig.from_gguf(dict(metadata, **{'dflash.attention.causal': flag}), model)
    assert [explicit.layer_causal(i) for i in range(6)] == [flag]*6
  for key, value in [('tokenizer.ggml.tokens', []), ('tokenizer.ggml.mask_token_id', 16),
                     ('dflash.target_layers', [2, 2]), ('dflash.attention.causal', 1), ('dflash.selector_rank', 16)]:
    invalid = dict(metadata)
    invalid[key] = value
    with pytest.raises(ValueError): DFlashConfig.from_gguf(invalid, model)

def full_cache_draft(model, source):
  """`source`'s weights with every layer on a full-length cache, the reference for its ring."""
  draft = DFlashDraft(source.config, model, {k: v.float() for k, v in get_state_dict(source.weights).items()})
  draft.ring = [False] * draft.config.layers
  draft.cache = [Tensor.zeros(2, 1, draft.config.kv_heads, draft.cache_length, draft.config.head_dim, dtype=dtypes.float16)
                 .contiguous().realize() for _ in draft.ring]
  return draft

def test_ring_cache_matches_a_full_cache(tmp_path):
  # The sliding layer keeps window + 2 blocks = 16 slots; 60 positions wrap it several times.
  with Context(DEV='CPU'):
    model = Model.from_gguf(checkpoint(tmp_path, context=64), device='CPU')
    ring = tiny_draft(model)
    full = full_cache_draft(model, ring)
    assert ring.ring == [True, False] and ring.cache[0].shape[3] == ring.ring_length == 16
    rows, noise = Tensor.randn(1, 60, 8).realize(), Tensor.randn(1, 4, 8).realize()
    done = 0
    for n in (3, 5, 1, 7, 4, 9, 2, 6, 8, 5, 3):
      for draft in (ring, full):
        draft.observe(rows[:, done:done + n].contiguous().realize())
        draft.propose(1, done + n, 3)  # absorbs the staged rows and writes a noise block past them
      done += n
      hidden = [draft._denoise(noise, done).realize() for draft in (ring, full)]
      assert (hidden[0] - hidden[1]).abs().max().item() < 1e-3, done

def test_restoring_past_a_wrapped_ring_hides_overwritten_slots(tmp_path):
  with Context(DEV='CPU'):
    model = Model.from_gguf(checkpoint(tmp_path, context=64), device='CPU')
    draft = tiny_draft(model)
    rows = Tensor.randn(1, 50, 8).realize()
    draft.observe(rows[:, :10].contiguous().realize())
    early = draft.save_state()
    draft.observe(rows[:, 10:12].contiguous().realize())
    draft.reset()
    draft.load_state(early)            # nothing wrapped yet: every slot is still position k's
    assert draft._floor.tolist() == [0]
    draft.observe(rows[:, 10:50].contiguous().realize())
    draft.reset()
    draft.load_state(early)            # 40 more rows lapped the 16-slot ring
    assert draft._floor.tolist() == [10] and draft.observed == 10
    draft.reset()
    assert draft._floor.tolist() == [0]  # reading from scratch rewrites every slot it reads

def test_shorter_proposals_reuse_the_full_block_graph(tmp_path):
  with Context(DEV='CPU'):
    model = Model.from_gguf(checkpoint(tmp_path, context=64), device='CPU')
    draft, rows = tiny_draft(model), Tensor.randn(1, 40, 8).realize()
    full, done = draft.config.block_size - 1, 0
    for count in (full, 1, 2, full):
      draft.observe(rows[:, done:done + 5].contiguous().realize())
      done += 5
      assert len(draft.propose(1, done, count)) == count
    assert list(draft._jits) == [full]  # each new row count would capture its own graph (seconds on a GPU)
