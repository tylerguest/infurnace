from types import SimpleNamespace

import pytest

from infurnace import cli, loader

class Tokenizer:
  bos_id = None
  eos_id, eot_id = 9, 10
  def encode(self, text): return [1, 2]
  def render_chat(self, messages, **kwargs): return 'chat'
  def is_end(self, token): return token in (9, 10)
  def stream_decoder(self): return lambda token: '!' if token is None else str(token)

class Engine:
  def __init__(self, verifier='parallel', block_size=16):
    self.closed, self.calls = False, []
    self.verifier, self.draft = verifier, SimpleNamespace(config=SimpleNamespace(block_size=block_size))  # DFlash-shaped
  def stream(self, ids, **kwargs):
    self.calls.append((ids, kwargs))
    try:
      yield from (3, 9, 4)
    finally:
      self.closed = True

def test_eagle_stream_eos_budget_and_cancellation(capsys):
  model = SimpleNamespace(config=SimpleNamespace(context_length=16))
  engine = Engine()
  assert cli.reply(cli.generator(model, Tokenizer(), eagle=engine), Tokenizer(), [1, 2], None, 16) == '3!'
  assert engine.closed
  assert engine.calls == [([1, 2], dict(max_new_tokens=14, draft_tokens=1, eos_token_ids=(9, 10), batched=True))]
  assert capsys.readouterr().out == '3!\n'

@pytest.mark.parametrize('args', [['--eagle3', 'draft', '--benchmark', '8'], ['--eagle3', 'draft', '--device', 'CPU'],
                                  ['--eagle3', 'missing-checkpoint'], ['--eagle3', 'draft', '--serve']])
def test_invalid_eagle_flags_before_model_loading(monkeypatch, args):
  monkeypatch.setattr(loader, 'load_model', lambda *a, **kw: pytest.fail('loaded target for invalid flags'))
  with pytest.raises(SystemExit) as exc: cli.main(args)
  assert exc.value.code == 2

@pytest.mark.parametrize('message', ['NV signal wait timed out', 'PYTHON hang detected', 'unrelated bug'])
def test_cli_device_timeout_diagnostic_does_not_swallow_other_errors(monkeypatch, capsys, message):
  model = SimpleNamespace(config=SimpleNamespace(context_length=16), device='NV')
  def fail(*args): raise RuntimeError(message)
  model.generate = fail
  monkeypatch.setattr(loader, 'load_model', lambda *a, **kw: (model, Tokenizer()))
  if message == 'unrelated bug':
    with pytest.raises(RuntimeError, match=message): cli.main(['--prompt', 'hello', '--no-dflash'])
  else:
    assert cli.main(['--prompt', 'hello', '--no-dflash']) == 1
    assert 'the GPU stopped responding' in capsys.readouterr().err

def test_cli_builds_eagle_engine(monkeypatch, tmp_path, capsys):
  (tmp_path / 'config.json').write_text('{}')
  (tmp_path / 'model.safetensors').touch()
  model, engine = SimpleNamespace(config=SimpleNamespace(context_length=16), device='NV'), Engine()
  monkeypatch.setattr(loader, 'load_model', lambda *a, **kw: (model, Tokenizer()))
  monkeypatch.setattr(loader, 'load_eagle', lambda target, checkpoint: engine if (target, checkpoint) == (model, tmp_path) else None)
  assert cli.main(['--eagle3', str(tmp_path), '--prompt', 'hello', '--max-tokens', '5']) == 0
  assert engine.calls[0][1]['max_new_tokens'] == 5 and engine.calls[0][1]['draft_tokens'] == 1
  assert engine.closed and 'experimental' in capsys.readouterr().err
