"""Prompt snapshots: later prompts resume from the newest state saved before a turn header that they still match."""
import pytest
from tinygrad import Context
from infurnace.dflash import DFlashGenerator
from infurnace.models import qwen35
from infurnace.models.qwen35 import SNAPSHOTS, Qwen35Model
from test_dflash import tiny_draft
from test_qwen35 import checkpoint

TURN, USER, TOOL, ASSISTANT = 7, 8, 9, 10  # stand in for <|im_start|>, user, <tool_response>, assistant
FIRST = [1, 2, 3, 4, TURN, 5, 6]                    # snapshot point: index 4, before the last turn header
SECOND = FIRST[:4] + [TURN, 11, 12, TURN, 13, 2]    # the next turn: same prefix, then a re-rendered reply
OTHER = [1, 9, 3, 4, TURN, 5, 6]                    # diverges before the snapshot point

def model_with_spy(tmp_path, context=16):
  model = Qwen35Model.from_gguf(checkpoint(tmp_path, context=context), device='CPU')
  model.turn_token, model.user_turn, model.tool_marker, resumed = TURN, (TURN, USER), TOOL, []
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
    assert [s.prefix for s in model._snapshots] == [FIRST[:4]]
    assert all(state.device == 'CPU' for state in model._snapshots[0].states)  # kept in host memory
    live = {state.uop.buffer for state in model._recurrent_states()}
    assert not live & {state.uop.buffer for state in model._snapshots[0].states}  # a copy, even on the CPU
    list(model.generate(list(OTHER), 2))
    assert resumed[-1] == 0 and [s.prefix for s in model._snapshots] == [OTHER[:4]]  # read from scratch, then a new snapshot

@pytest.mark.parametrize('verifier,block', [('sequential', 4), ('parallel', 4)])
def test_dflash_resumes_with_its_draft(tmp_path, verifier, block):
  with Context(DEV='CPU'):
    model, resumed = model_with_spy(tmp_path)
    generator = DFlashGenerator(model, tiny_draft(model), verifier=verifier)
    generator.min_tau, generator.plain_cost = 0.0, float('inf')  # always draft
    list(generator.stream(list(FIRST), max_new_tokens=4, block_size=block))
    restored = list(generator.stream(list(SECOND), max_new_tokens=6, block_size=block))
    assert resumed == [0, 4] and model._snapshots[-1].owner is generator.draft
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

SYSTEM = [1, 2]
def user(x): return [TURN, USER, x]
def tool(x): return [TURN, USER, TOOL, x]
def reply(x, thinking): return [TURN, ASSISTANT] + ([11] if thinking else []) + [x]
GENERATE = [TURN, ASSISTANT]

def agent_turn(steps: int, thinking: bool) -> list[int]:
  """A user message and a tool loop after it, as an agent's prompts render it: the replies keep their
  thinking while the loop runs and lose it once the next user message arrives."""
  return user(3) + [t for i in range(steps) for t in reply(4 + i % 2, thinking) + tool(5 + i % 3)]

def test_new_user_message_resumes_after_a_long_tool_loop(tmp_path):
  steps = SNAPSHOTS + 2  # more tool calls than snapshots are kept
  with Context(DEV='CPU'):
    model, resumed = model_with_spy(tmp_path, context=128)
    for i in range(steps + 1):
      list(model.generate(SYSTEM + agent_turn(i, True) + GENERATE, 1))
    assert len(model._snapshots) == SNAPSHOTS and model._snapshots[0].user
    nxt = SYSTEM + agent_turn(steps, False) + user(6) + GENERATE
    out = list(model.generate(list(nxt), 4))
    assert resumed[-1] == len(SYSTEM + user(3))      # the end of the previous user message
    assert [s.prefix for s in model._snapshots] == [SYSTEM + user(3), nxt[:-2]]  # the stale tool loop is gone
    assert out == list(model.generate(list(nxt), 4, reuse_cache=False))

def test_retrying_an_earlier_turn_resumes_from_its_snapshot(tmp_path):
  with Context(DEV='CPU'):
    model, resumed = model_with_spy(tmp_path, context=64)
    prompts = [SYSTEM + agent_turn(i, True) + GENERATE for i in range(3)]
    for prompt in prompts: list(model.generate(list(prompt), 1))
    out = list(model.generate(list(prompts[1]), 4))  # regenerate the second step
    assert resumed[-1] == len(prompts[1]) - 2
    assert out == list(model.generate(list(prompts[1]), 4, reuse_cache=False))

@pytest.mark.parametrize('verifier', ['sequential', 'parallel'])
def test_dflash_new_user_message_resumes_after_a_tool_loop(tmp_path, monkeypatch, verifier):
  monkeypatch.setattr(qwen35, 'SNAPSHOTS', 3)
  with Context(DEV='CPU'):
    model, resumed = model_with_spy(tmp_path, context=96)
    generator = DFlashGenerator(model, tiny_draft(model), verifier=verifier)
    generator.min_tau, generator.plain_cost = 0.0, float('inf')  # always draft
    for i in range(5):
      list(generator.stream(SYSTEM + agent_turn(i, True) + GENERATE, max_new_tokens=1, block_size=4))
    nxt = SYSTEM + agent_turn(4, False) + user(6) + GENERATE
    out = list(generator.stream(list(nxt), max_new_tokens=6, block_size=4))
    assert resumed[-1] == len(SYSTEM + user(3))
    model.reset()
    assert out == list(generator.stream(list(nxt), max_new_tokens=6, block_size=4))
