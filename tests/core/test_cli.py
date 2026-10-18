from types import SimpleNamespace
import pytest

from infurnace import cli, loader

@pytest.mark.parametrize("args", [["--prompt", "hi", "--chat"], ["--max-tokens", "0"], ["--benchmark", "-1"],
                                  ["--max-context", "0"], ["--prompt", "hi", "--no-thinking"], ["--serve", "0"],
                                  ["--serve", "--max-tokens", "5"],
                                  ["--dflash", "missing.gguf"], ["--nv-prefill"]])
def test_invalid_flags_before_loading(monkeypatch, args):
  monkeypatch.setattr(loader, "load_model", lambda *a, **kw: pytest.fail("loaded a model for invalid flags"))
  with pytest.raises(SystemExit) as error: cli.main(args)
  assert error.value.code == 2

@pytest.mark.parametrize("value,expected", [("32", (512, 32)), ("1000,50", (1000, 50))])
def test_benchmark_spec(value, expected):
  assert cli.parse_args(["--benchmark", value]).benchmark == expected
  assert cli.parse_args(["--benchmark"]).benchmark == (512, 20)

@pytest.mark.parametrize('message', ['NV signal wait timed out', 'PYTHON hang detected', 'unrelated bug'])
def test_cli_device_timeout_diagnostic_does_not_swallow_other_errors(monkeypatch, capsys, message):
  model = SimpleNamespace(config=SimpleNamespace(context_length=16), device='NV')
  def fail(*args, **kwargs): raise RuntimeError(message)
  model.generate = fail
  tokenizer = SimpleNamespace(bos_id=None, encode=lambda text: [1], stream_decoder=lambda: lambda token: '')
  monkeypatch.setattr(loader, 'load_model', lambda *a, **kw: (model, tokenizer))
  if message == 'unrelated bug':
    with pytest.raises(RuntimeError, match=message): cli.main(['--prompt', 'hello', '--no-dflash'])
  else:
    assert cli.main(['--prompt', 'hello', '--no-dflash']) == 1
    assert 'the GPU stopped responding' in capsys.readouterr().err

@pytest.mark.parametrize('value,model', [('qwen3.5:4b', True), ('~/m/x.gguf', True), ('other:7b', True), ('Explain RAII', False),
                                         ('https://example.com/notes.txt', False), ('note:', False)])
def test_run_tells_models_from_prompts(value, model):
  assert cli._looks_like_model(value) is model

def test_unknown_models_are_rejected_before_any_server(monkeypatch):
  monkeypatch.setattr('infurnace.client.run', lambda *args: pytest.fail('must not contact a server'))
  with pytest.raises(ValueError, match="unknown model 'qwen3:4b'"): cli.command(['run', 'qwen3:4b', 'hi'])
  cli._check_model('qwen3.5:4b')
  cli._check_model('~/m/x.gguf')
