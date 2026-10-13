"""Prompt snapshots: the next chat turn resumes from the state saved before the last turn header."""
import pytest
from tinygrad import Context
from infurnace.dflash import DFlashGenerator
from infurnace.models.qwen35 import Qwen35Model
from test_dflash import tiny_draft
from test_qwen35 import checkpoint

TURN = 7  # stands in for <|im_start|>
FIRST = [1, 2, 3, 4, TURN, 5, 6]                    # snapshot point: index 4, before the last turn header
SECOND = FIRST[:4] + [TURN, 11, 12, TURN, 13, 2]    # the next turn: same prefix, then a re-rendered reply
OTHER = [1, 9, 3, 4, TURN, 5, 6]                    # diverges before the snapshot point

def model_with_spy(tmp_path):
  model = Qwen35Model.from_gguf(checkpoint(tmp_path), device='CPU')
  model.turn_token, resumed = TURN, []
  restore = model.restore_snapshot
  def spy(*args, **kwargs):
    resumed.append(restore(*args, **kwargs))
    return resumed[-1]
  model.restore_snapshot = spy
  return model, resumed

def test_plain_decoding_resumes_from_the_snapshot(tmp_path):
  with Context(DEV='CPU'):
    model, resumed = model_with_spy(tmp_path)
    list(model.generate(list(FIRST), 4))
    restored = list(model.generate(list(SECOND), 6))
    assert resumed == [0, 4]                         # the second prompt resumed at the snapshot point
    assert restored == list(model.generate(list(SECOND), 6, reuse_cache=False))  # same as from scratch

def test_diverging_prompt_drops_the_snapshot(tmp_path):
  with Context(DEV='CPU'):
    model, resumed = model_with_spy(tmp_path)
    list(model.generate(list(FIRST), 2))
    assert model._snapshot is not None and model._snapshot[0] == FIRST[:4]
    list(model.generate(list(OTHER), 2))
    assert resumed[-1] == 0 and model._snapshot[0] == OTHER[:4]  # read from scratch, then a new snapshot

@pytest.mark.parametrize('verifier,block', [('sequential', 4), ('parallel', 4)])
def test_dflash_resumes_with_its_draft(tmp_path, verifier, block):
  with Context(DEV='CPU'):
    model, resumed = model_with_spy(tmp_path)
    generator = DFlashGenerator(model, tiny_draft(model), verifier=verifier)
    generator.min_tau, generator.plain_cost = 0.0, float('inf')  # always draft
    list(generator.stream(list(FIRST), max_new_tokens=4, block_size=block))
    restored = list(generator.stream(list(SECOND), max_new_tokens=6, block_size=block))
    assert resumed == [0, 4] and model._snapshot[1] is generator.draft
    model.reset()                                    # drops the snapshot: read the same prompt from scratch
    assert restored == list(generator.stream(list(SECOND), max_new_tokens=6, block_size=block))

def test_plain_snapshot_is_not_used_by_dflash(tmp_path):
  # A snapshot saved by plain decoding has no draft state, so DFlash reads the prompt again.
  with Context(DEV='CPU'):
    model, resumed = model_with_spy(tmp_path)
    generator = DFlashGenerator(model, tiny_draft(model), verifier='sequential')
    list(model.generate(list(FIRST), 2))
    list(generator.stream(list(SECOND), max_new_tokens=2, block_size=4))
    assert resumed[-1] == 0
