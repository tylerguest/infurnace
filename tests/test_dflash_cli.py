from types import SimpleNamespace
import pytest
from infurnace import cli
from test_eagle_cli import Engine, Tokenizer


def test_dflash_stream_eos_and_close(capsys):
  model = SimpleNamespace(config=SimpleNamespace(context_length=16))
  engine = Engine()
  assert cli._stream(model, Tokenizer(), [1, 2], None, dflash=engine, dflash_block_size=8) == '3!'
  assert engine.closed
  assert engine.calls == [([1, 2], {'max_new_tokens': 14, 'block_size': 8, 'eos_token_ids': (9, 10)})]
  assert capsys.readouterr().out == '3!\n'


@pytest.mark.parametrize('extra', [
  ['--dflash-block-size', '4'], ['--dflash', 'missing.gguf'], ['--dflash-verifier', 'block'], ['--dflash-legacy-noncausal'],
  ['--dflash', 'draft.gguf', '--dflash-block-size', '0'],
  ['--dflash', 'draft.gguf', '--benchmark', '8'], ['--dflash', 'draft.gguf', '--eagle3', 'eagle']])
def test_invalid_flags_before_loading(monkeypatch, extra):
  monkeypatch.setattr('sys.argv', ['infurnace', *extra])
  monkeypatch.setattr(cli, '_build', lambda *a, **kw: pytest.fail('loaded model for invalid flags'))
  with pytest.raises(SystemExit) as exc: cli.main()
  assert exc.value.code == 2


@pytest.mark.parametrize('block', [None, 8])
@pytest.mark.parametrize('verifier', [None, 'block'])
@pytest.mark.parametrize('legacy', [False, True])
def test_cli_builds_reference_engine(monkeypatch, tmp_path, capsys, block, verifier, legacy):
  import infurnace.dflash as dflash
  path = tmp_path / 'draft.gguf'
  path.touch()
  args = ['infurnace', '--dflash', str(path), '--prompt', 'hello', '--max-tokens', '5']
  if block is not None: args += ['--dflash-block-size', str(block)]
  if verifier is not None: args += ['--dflash-verifier', verifier]
  if legacy: args += ['--dflash-legacy-noncausal']
  monkeypatch.setattr('sys.argv', args)
  model = SimpleNamespace(config=SimpleNamespace(context_length=16))
  draft = SimpleNamespace(config=SimpleNamespace(block_size=16))
  engine = Engine()
  monkeypatch.setattr(cli, '_build', lambda *a, **kw: (model, Tokenizer()))
  def load(checkpoint, target, **options):
    assert options == ({'legacy_noncausal': True} if legacy else {})
    return draft
  monkeypatch.setattr(dflash.DFlashDraft, 'from_gguf', load)
  def build(target, loaded, **options):
    assert target is model and loaded is draft
    assert options == ({} if verifier is None else {'verifier': verifier})
    return engine
  monkeypatch.setattr(dflash, 'DFlashGenerator', build)
  assert cli.main() == 0
  assert engine.calls[0][1]['max_new_tokens'] == 5
  assert engine.calls[0][1]['block_size'] == (4 if block is None else block)
  assert engine.closed and 'not a speedup claim' in capsys.readouterr().err
