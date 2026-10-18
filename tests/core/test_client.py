import io
import json
import threading

import pytest

from infurnace import client, loader
from infurnace.serve import ChatTemplate, Server, ServerModel
from test_serve import Model, Tokenizer

@pytest.mark.parametrize('value,expected', [(None, '127.0.0.1:8000'), ('0.0.0.0:9000', '0.0.0.0:9000'),
                                            ('http://example.lan:1234/', 'example.lan:1234'), ('example.lan', 'example.lan:8000')])
def test_host(monkeypatch, value, expected):
  if value is None: monkeypatch.delenv('INFURNACE_HOST', raising=False)
  else: monkeypatch.setenv('INFURNACE_HOST', value)
  assert client.host() == expected

def test_configure_pi_keeps_user_entries(tmp_path, monkeypatch):
  monkeypatch.setenv('INFURNACE_HOST', '127.0.0.1:8123')
  agent = tmp_path / '.pi' / 'agent'
  agent.mkdir(parents=True)
  mine = {'id': 'infurnace-qwen35', 'contextWindow': 32768}
  (agent / 'models.json').write_text(json.dumps({'providers': {'infurnace': {'baseUrl': 'http://127.0.0.1:8000/v1', 'models': [mine]},
                                                               'other': {'models': []}}}))
  (agent / 'settings.json').write_text(json.dumps({'theme': 'dark', 'defaultModel': 'x'}))
  for context in (8192, 32768):  # a second launch replaces its own entry, never duplicates it
    client.configure_pi('qwen3.5:4b', context, home=tmp_path)
  config = json.loads((agent / 'models.json').read_text())
  models = config['providers']['infurnace']['models']
  assert models[0] == mine and len(models) == 2
  assert models[1]['id'] == 'qwen3.5:4b' and models[1]['_launch'] and models[1]['contextWindow'] == 32768
  assert config['providers']['infurnace']['baseUrl'] == 'http://127.0.0.1:8123/v1'  # pointed at the server just configured
  assert 'other' in config['providers']
  settings = json.loads((agent / 'settings.json').read_text())
  assert settings == {'theme': 'dark', 'defaultModel': 'qwen3.5:4b', 'defaultProvider': 'infurnace'}
  # The backups hold the files as they were before the first launch, not the first launch's edits.
  assert json.loads((agent / 'models.json.infurnace.bak').read_text())['providers']['infurnace']['models'] == [mine]
  assert json.loads((agent / 'settings.json.infurnace.bak').read_text()) == {'theme': 'dark', 'defaultModel': 'x'}

def test_list_pull_and_remove_use_user_storage(tmp_path, monkeypatch):
  monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path))
  monkeypatch.setattr(loader, 'MODELS', {'tiny:1b': ('https://example.com/m/tiny-Q4_K_M.gguf', 'https://example.com/m/tiny-draft.gguf')})
  monkeypatch.setattr(loader, '_local', lambda url: p if (p := tmp_path / 'infurnace/models' / url.rsplit('/', 1)[1]).is_file() else None)
  def fake_fetch(url, name):
    name.parent.mkdir(parents=True, exist_ok=True)
    name.write_bytes(b'x' * 10)
    return name
  monkeypatch.setattr(loader, 'fetch', fake_fetch)
  assert loader.installed() == []
  assert len(loader.pull('tiny:1b')) == 2
  [entry] = loader.installed()
  assert entry['name'] == 'tiny:1b' and entry['size'] == 20 and entry['details']['quantization_level'] == 'Q4_K_M'
  assert len(loader.remove('tiny:1b')) == 2 and loader.installed() == []
  with pytest.raises(ValueError, match='unknown model'): loader.pull('nope')

def test_run_one_prompt_through_a_server(monkeypatch, capsys):
  model, tok = Model(), Tokenizer()
  model.output = 'hello'
  template = ChatTemplate({'tokenizer.chat_template': '{% for m in messages %}{{ m.content }}{% endfor %}'}, tok)
  server = Server(('127.0.0.1', 0), ServerModel(model), 'qwen3.5:4b', tok, template)
  thread = threading.Thread(target=server.serve_forever, daemon=True)
  thread.start()
  monkeypatch.setenv('INFURNACE_HOST', '%s:%d' % server.server_address)
  monkeypatch.setattr('sys.stdin', io.StringIO(''))  # not a terminal, nothing piped
  try:
    assert client.run(None, 'say hi', None, {}, []) == 0
    assert 'hello' in capsys.readouterr().out
    assert tok.rendered == 'say hi'
    with pytest.raises(client.ServerError, match='serving qwen3.5:4b'): client.run('other:1b', 'x', None, {}, [])
    assert client.connect('/models/qwen3.5:4b.gguf', [])['model'] == 'qwen3.5:4b'  # a path matches the stem it is served as
    assert client.ps() == 0 and 'qwen3.5:4b' in capsys.readouterr().out
    assert client.stop() == 0
  finally:
    thread.join(5)
    server.server_close()

def test_interactive_chat_commands(monkeypatch, capsys):
  model, tok = Model(), Tokenizer()
  model.output = 'hi there'
  template = ChatTemplate({'tokenizer.chat_template': '{% for m in messages %}[{{ m.role }}:{{ m.content }}]{% endfor %}'}, tok)
  server = Server(('127.0.0.1', 0), ServerModel(model), 'qwen3.5:4b', tok, template)
  thread = threading.Thread(target=server.serve_forever, daemon=True)
  thread.start()
  monkeypatch.setenv('INFURNACE_HOST', '%s:%d' % server.server_address)
  lines = iter(['hello', '/set nothink', '"""first', 'second"""', '/clear', 'again', '/bye'])
  monkeypatch.setattr('builtins.input', lambda prompt='': next(lines))
  try:
    assert client.chat('qwen3.5:4b', None, {}) == 0
    assert tok.rendered == '[user:again]'  # /clear forgot the earlier turns
    out = capsys.readouterr().out
    assert out.count('hi there') == 3 and 'Cleared the conversation.' in out
  finally:
    server.shutdown()
    thread.join(5)
    server.server_close()

def test_chat_survives_server_errors(monkeypatch, capsys):
  # A message the server rejects (here: too long even without history) is reported; the chat goes on.
  replies = iter([client.ServerError('prompt has 9000 tokens, but the model context is 8192'), 'fine'])
  def reply(*args, **kwargs):
    value = next(replies)
    if isinstance(value, Exception): raise value
    return value
  monkeypatch.setattr(client, '_reply', reply)
  monkeypatch.setattr(client, 'wait_ready', lambda: None)
  lines = iter(['huge message', 'small message', '/bye'])
  monkeypatch.setattr('builtins.input', lambda prompt='': next(lines))
  assert client.chat('qwen3.5:4b', None, {}) == 0
  assert 'prompt has 9000 tokens' in capsys.readouterr().err

def test_empty_piped_input_starts_no_server(monkeypatch):
  monkeypatch.setattr('sys.stdin', io.StringIO(''))
  monkeypatch.setattr(client, 'connect', lambda *args: pytest.fail('must not connect or start a server'))
  assert client.run(None, None, None, {}, []) == 0

