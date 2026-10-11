import itertools
import pytest
from tinygrad import Context, Tensor
from infurnace.dflash import DFlashGenerator
from infurnace.dflash.target import TargetObserver
from infurnace.dflash.verify import BlockVerifier, _store
from infurnace.qwen35 import Qwen35Model
from test_qwen35 import checkpoint
from test_dflash import RecordedDraft


def states(model):
  return [(b.conv_state.tolist(), b.recurrent_state.tolist()) for b in model.transformer.blk if hasattr(b, 'conv_state')]


def kv(model, position):
  return [b.cache_kv[:, :, :, :position].tolist() for b in model.transformer.blk if hasattr(b, 'cache_kv')]


def test_prefix_state_and_repeated_jit(tmp_path):
  with Context(DEV='CPU'):
    file = checkpoint(tmp_path)
    model = Qwen35Model.from_gguf(file, device='CPU')
    control = Qwen35Model.from_gguf(file, device='CPU')
    with TargetObserver(control, (1,)) as reference:
      for token in [1, 2, 3]: root, _ = reference.step(token)
      frames, decisions, features = [], [], []
      for _ in range(7):
        root, hidden = reference.step(root)
        frames.append((states(control), kv(control, reference.position)))
        features.append(hidden.tolist())
        decisions.append(root)
    with TargetObserver(model, (1,)) as observer:
      for token in [1, 2, 3]: root, _ = observer.step(token)
      verifier = BlockVerifier(observer)
      verifier.prepare(4)
      for row in range(7):
        before = states(model)
        with verifier.verify([root, (root+1) % 16, root, root]) as packet:
          assert states(model) == before, f'zero restore {row}'
          with pytest.raises(RuntimeError): observer.step(root)
          with pytest.raises(RuntimeError): verifier.verify([root])
          packet.commit(1)
          assert (states(model), kv(model, observer.position)) == frames[row], f'prefix restore {row}'
          assert packet.feature(0).tolist() == features[row]
          root = packet.decisions[0]
          assert root == decisions[row], f'decision {row}'
          with pytest.raises(ValueError): packet.commit(2)
        with pytest.raises(RuntimeError): packet.commit(1)
      assert verifier._block_jits[4].cnt >= 3


@pytest.mark.parametrize('count', [0, 1, 2, 3, 4])
def test_every_prefix_checkpoint_rng_and_following_row(tmp_path, count):
  with Context(DEV='CPU'):
    file = checkpoint(tmp_path)
    control, model = Qwen35Model.from_gguf(file), Qwen35Model.from_gguf(file)
    Tensor.manual_seed(999)
    with TargetObserver(control, (1,)) as reference:
      for token in [1, 2, 3]: anchor, _ = reference.step(token)
      initial = (states(control), kv(control, reference.position), Tensor._device_rng_counters['CPU'].tolist())
      decisions, frames, features = [], [initial], []
      root = anchor
      for _ in range(5):
        root, hidden = reference.step(root)
        decisions.append(root)
        features.append(hidden.tolist())
        frames.append((states(control), kv(control, reference.position), Tensor._device_rng_counters['CPU'].tolist()))
    Tensor.manual_seed(999)
    with TargetObserver(model, (1,)) as observer:
      for token in [1, 2, 3]: root, _ = observer.step(token)
      verifier = BlockVerifier(observer)
      verifier.prepare(4)
      with verifier.verify([root, *decisions[:3]]) as packet:
        assert packet.decisions == tuple(decisions[:4])
        packet.commit(count)
        assert observer.position == 3+count
        assert (states(model), kv(model, observer.position), Tensor._device_rng_counters['CPU'].tolist()) == frames[count]
        for row in range(count): assert packet.feature(row).tolist() == features[row]
        with pytest.raises(ValueError): packet.feature(count)
      # Stale speculative suffix K/V must not affect subsequent ordinary decode.
      following, hidden = observer.step(observer.next_decision)
      assert following == decisions[count]
      assert hidden.tolist() == features[count]
      assert (states(model), kv(model, observer.position), Tensor._device_rng_counters['CPU'].tolist()) == frames[count+1]


@pytest.mark.parametrize('stop', [2, 3, 4, 5, 8])
def test_cancellation_inside_accepted_block_matches_rng(tmp_path, stop):
  with Context(DEV='CPU'):
    model = Qwen35Model.from_gguf(checkpoint(tmp_path))
    prompt = [1, 2, 3]
    expected = list(model.generate(prompt, 13, reuse_cache=False))
    Tensor.manual_seed(999)
    with TargetObserver(model, (1,)) as reference:
      for token in prompt: root, _ = reference.step(token)
      for _ in range(stop-1): root, _ = reference.step(root)
    following = Tensor.rand(8).tolist()
    Tensor.manual_seed(999)
    engine = DFlashGenerator(model, RecordedDraft(model, prompt, expected), verifier='block')
    stream = engine.stream(prompt, max_new_tokens=13)
    assert list(itertools.islice(stream, stop)) == expected[:stop]
    stream.close()
    assert Tensor.rand(8).tolist() == following
    assert engine.stats.target_steps == len(prompt)+stop-1
    assert not model._generating and not engine.observer._block_active
    assert engine.block_verifier._active is None


def test_eos_inside_accepted_block_and_counter_reset_identity(tmp_path):
  with Context(DEV='CPU'):
    # An explicit permuted GGUF head makes greedy output change between rows,
    # so the EOS is inside the accepted block rather than the initial anchor.
    model = Qwen35Model.from_gguf(checkpoint(tmp_path, permuted_head=True))
    prompt = [1, 2, 3]
    Tensor.manual_seed(999)
    expected = list(model.generate(prompt, 3, reuse_cache=False))
    assert expected[2] not in expected[:2]
    following = Tensor.rand(8).tolist()
    Tensor.manual_seed(999)
    Tensor._next_counter('CPU', 0)
    counter = Tensor._device_rng_counters['CPU']
    counter.realize()
    buffer = counter.uop.buf_uop
    engine = DFlashGenerator(model, RecordedDraft(model, prompt, expected+[0]*8), verifier='block')
    assert list(engine.stream(prompt, max_new_tokens=8, eos_token_ids=(expected[2],))) == expected
    assert Tensor.rand(8).tolist() == following
    assert engine.stats.target_steps == len(prompt)+2 and engine.stats.speculative_rows == 2
    # In-place detachment must retain both caller-held Tensor and storage.
    assert counter is Tensor._device_rng_counters['CPU']
    counter.assign(Tensor([0, 0], dtype=counter.dtype)).realize()
    assert counter.uop.buf_uop is buffer and counter.tolist() == [0, 0]


def test_failure_restores_initial_state_rng_and_releases_owner(tmp_path, monkeypatch):
  with Context(DEV='CPU'):
    model = Qwen35Model.from_gguf(checkpoint(tmp_path))
    with TargetObserver(model, (1,)) as observer:
      for token in [1, 2, 3]: root, _ = observer.step(token)
      verifier = BlockVerifier(observer)
      verifier.prepare(4)
      before, counter = states(model), Tensor._device_rng_counters['CPU'].tolist()
      def fail(*args, **kw):
        _store(verifier._histories[0][0], Tensor.full(verifier._histories[0][0].shape, 99.))
        _store(verifier._counter, Tensor.full(verifier._counter.shape, 99, dtype=verifier._counter.dtype))
        raise RuntimeError('injected verification failure')
      monkeypatch.setattr(verifier, '_forward', fail)
      with pytest.raises(RuntimeError, match='injected'): verifier.verify([root]*4)
      assert states(model) == before and Tensor._device_rng_counters['CPU'].tolist() == counter
      assert not model._generating and not observer.active and not observer._block_active
      assert not model.transformer._cached_tokens
