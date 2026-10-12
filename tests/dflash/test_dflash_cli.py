from types import SimpleNamespace
import pytest
from infurnace import cli, loader
from infurnace.chat import Decoder, stream_reply
from test_eagle_cli import Engine, Tokenizer


@pytest.mark.parametrize('verifier,block', [('parallel', 16), ('sequential', 4)])
def test_dflash_stream_block_eos_and_close(capsys, verifier, block):
  model = SimpleNamespace(config=SimpleNamespace(context_length=16))
  engine = Engine(verifier=verifier)
  assert stream_reply(Decoder(model, Tokenizer(), dflash=engine), Tokenizer(), [1, 2], None) == '3!'
  assert engine.closed
  assert engine.calls == [([1, 2], {'max_new_tokens': 14, 'block_size': block, 'eos_token_ids': (9, 10)})]
  assert capsys.readouterr().out == '3!\n'


@pytest.mark.parametrize('extra', [['--dflash', 'missing.gguf'], ['--dflash', 'draft.gguf', '--benchmark', '8'],
                                   ['--dflash', 'draft.gguf', '--eagle3', 'eagle'], ['--dflash', 'draft.gguf', '--no-dflash'],
                                   ['--dflash-block-size', '4']])
def test_invalid_flags_before_loading(monkeypatch, extra):
  monkeypatch.setattr(loader, 'load_model', lambda *a, **kw: pytest.fail('loaded model for invalid flags'))
  with pytest.raises(SystemExit) as exc: cli.main(extra)
  assert exc.value.code == 2


@pytest.mark.parametrize('device', ['CPU', 'NV'])
def test_explicit_draft_is_used_on_any_device(monkeypatch, tmp_path, capsys, device):
  path = tmp_path / 'draft.gguf'
  path.touch()
  model, engine = SimpleNamespace(config=SimpleNamespace(context_length=16), device=device), Engine()
  monkeypatch.setattr(loader, 'load_model', lambda *a, **kw: (model, Tokenizer()))
  monkeypatch.setattr(loader, 'load_dflash', lambda target, draft: engine if (target, draft) == (model, path) else None)
  assert cli.main(['--dflash', str(path), '--prompt', 'hello', '--max-tokens', '5']) == 0
  assert engine.calls[0][1]['max_new_tokens'] == 5 and engine.closed and 'DFlash' in capsys.readouterr().err


@pytest.mark.parametrize('device,flags,enabled', [('NV', [], True), ('CPU', [], False), ('NV', ['--no-dflash'], False),
                                                  ('NV', ['--serve'], False)])
def test_installed_draft_is_used_automatically(monkeypatch, tmp_path, device, flags, enabled):
  target, draft = tmp_path / 'target.gguf', tmp_path / 'draft.gguf'
  target.touch()
  draft.touch()
  monkeypatch.setenv('INFURNACE_DFLASH_DRAFT', str(draft))
  monkeypatch.setattr(loader, 'is_qwen35', lambda path: True)
  model = SimpleNamespace(config=SimpleNamespace(context_length=16), device=device)
  monkeypatch.setattr(loader, 'load_model', lambda *a, **kw: (model, Tokenizer()))
  loaded, decoders = [], []
  monkeypatch.setattr(loader, 'load_dflash', lambda target_model, path: loaded.append(path) or Engine())
  monkeypatch.setattr(cli, 'run_prompt', lambda decoder, *a: decoders.append(decoder) or 0)
  monkeypatch.setattr(Decoder, 'warmup', lambda self: None)
  monkeypatch.setattr('infurnace.serve.serve', lambda model, *a, **kw: decoders.append(None))
  run = ['--prompt', 'hello'] if '--serve' not in flags else []
  assert cli.main(['--model', str(target), *run, *flags]) == 0
  assert loaded == ([draft] if enabled else [])
  if '--serve' not in flags: assert (decoders[0].dflash is not None) == enabled
