from types import SimpleNamespace

import pytest

from infurnace import cli


class Tokenizer:
  bos_id = None
  eos_id, eot_id = 9, 10
  def encode(self, text): return [1, 2]
  def render_chat(self, messages, **kwargs): return 'chat'
  def is_end(self, token): return token in (9, 10)
  def stream_decoder(self): return lambda token: '!' if token is None else str(token)


class Engine:
  def __init__(self):
    self.closed, self.calls = False, []
  def stream(self, ids, **kwargs):
    self.calls.append((ids, kwargs))
    try:
      yield from (3, 9, 4)
    finally:
      self.closed = True


def test_eagle_stream_eos_budget_and_cancellation(capsys):
  model = SimpleNamespace(config=SimpleNamespace(context_length=16))
  engine = Engine()
  assert cli._stream(model, Tokenizer(), [1, 2], None, eagle=engine, draft_tokens=3) == '3!'
  assert engine.closed
  assert engine.calls == [([1, 2], dict(max_new_tokens=14, draft_tokens=3, eos_token_ids=(9, 10), batched=True))]
  assert capsys.readouterr().out == '3!\n'


@pytest.mark.parametrize('args', [
  ['--eagle-draft-tokens', '2'],
  ['--no-eagle-fast'],
  ['--eagle3', 'draft', '--eagle-draft-tokens', '-1'],
  ['--eagle3', 'draft', '--eagle-draft-tokens', '8'],
  ['--eagle3', 'draft', '--benchmark', '8'],
  ['--eagle3', 'draft', '--device', 'CPU'],
  ['--eagle3', 'draft', '--no-nv-quant'],
  ['--eagle3', 'draft', '--no-nv-fused-attention'],
  ['--eagle3', 'draft', '--nv-attention'],
  ['--eagle3', 'missing-checkpoint'],
])
def test_invalid_eagle_flags_before_model_loading(monkeypatch, args):
  monkeypatch.setattr('sys.argv', ['infurnace', *args])
  monkeypatch.setattr(cli, '_build', lambda *a, **kw: pytest.fail('loaded target for invalid flags'))
  with pytest.raises(SystemExit) as exc: cli.main()
  assert exc.value.code == 2


@pytest.mark.parametrize('message', ['NV signal wait timed out', 'PYTHON hang detected', 'unrelated bug'])
def test_cli_device_timeout_diagnostic_does_not_swallow_other_errors(monkeypatch, capsys, message):
  model = SimpleNamespace(config=SimpleNamespace(context_length=16))
  def fail(*args): raise RuntimeError(message)
  model.generate = fail
  monkeypatch.setattr('sys.argv', ['infurnace', '--prompt', 'hello'])
  monkeypatch.setattr(cli, '_build', lambda *a, **kw: (model, Tokenizer()))
  if message == 'unrelated bug':
    with pytest.raises(RuntimeError, match=message): cli.main()
  else:
    assert cli.main() == 1
    assert 'No GPU reset was attempted' in capsys.readouterr().err


@pytest.mark.parametrize('reference', [False, True])
def test_cli_builds_reusable_eagle_engine(monkeypatch, tmp_path, capsys, reference):
  from infurnace.eagle.verify import BatchedVerifier
  from infurnace.eagle import EagleDraft
  from infurnace.eagle import generate as eagle_generate
  (tmp_path/'config.json').write_text('{}')
  (tmp_path/'model.safetensors').touch()
  model = SimpleNamespace(config=SimpleNamespace(context_length=16))
  engine, draft = Engine(), object()
  monkeypatch.setattr('sys.argv', ['infurnace', '--eagle3', str(tmp_path), '--prompt', 'hello', '--max-tokens', '5']
                      + (['--no-eagle-fast'] if reference else []))
  monkeypatch.setattr(cli, '_build', lambda *a, **kw: (model, Tokenizer()))
  monkeypatch.setattr(BatchedVerifier, '__init__', lambda self, target: None)
  monkeypatch.setattr(EagleDraft, 'from_checkpoint', lambda path, target: draft)
  def build(target, loaded, *, fast):
    assert fast is not reference
    assert target is model and loaded is draft
    return engine
  monkeypatch.setattr(eagle_generate, 'EagleGenerator', build)
  assert cli.main() == 0
  assert engine.calls[0][1]['max_new_tokens'] == 5
  assert engine.calls[0][1]['draft_tokens'] == 1
  assert engine.closed and 'experimental' in capsys.readouterr().err
