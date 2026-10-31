import io
import pytest

from infurnace import cli, menu
from infurnace.menu import BACK, DOWN, ENTER, UP

@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
  """A home of its own: a ready server is added to pi's model list there, not in yours."""
  monkeypatch.setenv('HOME', str(tmp_path))
  return tmp_path

def keys(*sequence):
  it = iter(sequence)
  return lambda timeout=None: next(it)

@pytest.mark.parametrize('sequence,picked', [((ENTER,), 0), ((DOWN, DOWN, ENTER), 2), ((UP, ENTER), 2), ((DOWN, BACK), None),
                                             (('x', '3'), 2), (('9', '2'), 1)])
def test_choose_moves_and_picks(sequence, picked):
  assert menu.choose('title', ['a', 'b', 'c'], read_key=keys(*sequence), write=io.StringIO().write) == picked

def test_choose_reads_numbers_without_raw_keys(monkeypatch, capsys):
  answers = iter(['7', 'two', '2'])
  monkeypatch.setattr('builtins.input', lambda prompt='': next(answers))
  assert menu.choose('title', ['a', 'b']) == 1
  monkeypatch.setattr('builtins.input', lambda prompt='': '')
  assert menu.choose('title', ['a', 'b']) is None

MODELS = [{'name': 'qwen3.5:4b', 'details': {'quantization_level': 'Q4_K_M'}, 'size': 3.4e9, 'downloaded': True, 'has_draft': True},
          {'name': 'bonsai:4b', 'details': {'quantization_level': 'Q2_0'}, 'size': 0, 'downloaded': False, 'has_draft': True}]

def run_menu(monkeypatch, sequence, status=None, shutdowns=None):
  """The subcommands the menu ran; `status` is the server's (a function: called each time), `shutdowns` collects
  the shutdown requests the menu sends."""
  ran = []
  monkeypatch.setattr('infurnace.catalog.listing', lambda: MODELS)
  monkeypatch.setattr('infurnace.client.status', status if callable(status) else lambda: status)
  monkeypatch.setattr('infurnace.client.request', lambda method, path, body=None, timeout=None:
                      (shutdowns if shutdowns is not None else []).append(path) or {})
  monkeypatch.setattr('builtins.input', lambda prompt='': '')
  monkeypatch.setattr('sys.stdout', io.StringIO())
  assert menu.Menu(lambda argv: ran.append(argv) or 0, keys(*sequence)).main() == 0
  return ran

def test_menu_runs_the_subcommands(monkeypatch):
  # chat with the second model; the server's status; quit
  assert run_menu(monkeypatch, [DOWN, ENTER, DOWN, ENTER, '3', BACK]) == [['run', 'bonsai:4b'], ['ps']]

class Starting:
  """A server process still loading its model."""
  def __init__(self): self.terminated = False
  def poll(self): return None
  def terminate(self): self.terminated = True

def test_menu_starts_the_server_in_its_header(monkeypatch):
  # Start the server (the second model); it warms up, then is ready, and the header says where to connect
  states = iter([None, None, None, {'status': 'warming up', 'model': 'bonsai:4b'}])
  spawned, drawn = [], io.StringIO()
  monkeypatch.setattr('infurnace.client.spawn_server', lambda args: spawned.append(args) or Starting())
  monkeypatch.setattr('infurnace.client.host', lambda: '127.0.0.1:8124')
  monkeypatch.setattr('time.monotonic', iter(range(10, 10_000)).__next__)  # every poll asks the server
  sequence = [ENTER, DOWN, ENTER, '', '', BACK]
  def key(timeout=None): return sequence.pop(0)
  monkeypatch.setattr('infurnace.catalog.listing', lambda: MODELS)
  monkeypatch.setattr('infurnace.client.status', lambda: next(states, {'status': 'ready', 'model': 'bonsai:4b'}))
  monkeypatch.setattr('infurnace.client.request', lambda *a, **k: {})
  monkeypatch.setattr('sys.stdout', drawn)
  assert menu.Menu(lambda argv: 0, key).main() == 0
  assert spawned == [['--model', 'bonsai:4b']]
  out = drawn.getvalue()
  assert out.index('starting bonsai:4b') < out.index('warming up · bonsai:4b') < out.index('ready · bonsai:4b · http://127.0.0.1:8124/v1')

def test_stopping_a_server_still_loading_ends_its_process(monkeypatch):
  process = Starting()
  monkeypatch.setattr('infurnace.client.spawn_server', lambda args: process)
  run_menu(monkeypatch, [ENTER, ENTER, ENTER, BACK])  # Start, the first model, Stop
  assert process.terminated

def test_menu_chats_with_the_servers_model_and_stops_it(monkeypatch):
  # the server's model comes first in the picker: Enter chats with it; Stop asks the server to shut down
  shutdowns = []
  assert run_menu(monkeypatch, ['2', ENTER, ENTER, BACK], {'status': 'ready', 'model': 'bonsai:4b'}, shutdowns) == [['run', 'bonsai:4b']]
  assert shutdowns == ['/api/shutdown']

def test_closing_the_menu_stops_the_server_it_started(monkeypatch):
  calls = iter([None])  # not running when the menu opens; running after the chat started it
  shutdowns = []
  run_menu(monkeypatch, ['2', ENTER, BACK], lambda: next(calls, {'status': 'ready', 'model': 'qwen3.5:4b'}), shutdowns)
  assert shutdowns == ['/api/shutdown']

def test_closing_the_menu_keeps_a_server_that_was_already_running(monkeypatch, capsys):
  shutdowns = []
  run_menu(monkeypatch, [BACK], {'status': 'ready', 'model': 'qwen3.5:4b'}, shutdowns)
  assert shutdowns == [] and 'keeps running' in capsys.readouterr().err

def test_a_server_stopped_and_started_again_from_the_menu_goes_with_it(monkeypatch):
  shutdowns = []
  run_menu(monkeypatch, [ENTER, BACK], {'status': 'ready', 'model': 'qwen3.5:4b'}, shutdowns)  # Stop (faked: still answers)
  assert shutdowns == ['/api/shutdown']

def test_ctrl_c_in_the_menu_still_stops_its_server(monkeypatch):
  calls, shutdowns = iter([None]), []
  def interrupt(timeout=None): raise KeyboardInterrupt
  monkeypatch.setattr('infurnace.catalog.listing', lambda: MODELS)
  monkeypatch.setattr('infurnace.client.status', lambda: next(calls, {'status': 'ready'}))
  monkeypatch.setattr('infurnace.client.request', lambda method, path, body=None, timeout=None: shutdowns.append(path) or {})
  monkeypatch.setattr('sys.stdout', io.StringIO())
  with pytest.raises(KeyboardInterrupt): menu.Menu(lambda argv: 0, interrupt).main()
  assert shutdowns == ['/api/shutdown']

def test_menu_keeps_going_after_a_failed_command(monkeypatch, capsys):
  from infurnace.client import ServerError
  def fail(argv): raise ServerError('the server went away')
  monkeypatch.setattr('infurnace.client.status', lambda: None)
  monkeypatch.setattr('builtins.input', lambda prompt='': '')
  assert menu.Menu(fail, keys('3', '4')).main() == 0
  assert 'the server went away' in capsys.readouterr().err

@pytest.mark.parametrize('tty', [True, False])
def test_bare_infurnace_opens_the_menu_only_in_a_terminal(monkeypatch, tty):
  calls = []
  monkeypatch.setattr('sys.stdin.isatty', lambda: tty, raising=False)
  monkeypatch.setattr('sys.stdout.isatty', lambda: tty, raising=False)
  monkeypatch.setattr(menu, 'main', lambda command: calls.append('menu') or 0)
  monkeypatch.setattr(cli, 'command', lambda argv: calls.append(argv) or 0)
  assert cli.main([]) == 0
  assert calls == (['menu'] if tty else [['run']])

def test_logo_is_four_lines_of_five_cells(monkeypatch):
  monkeypatch.delenv('NO_COLOR', raising=False)
  monkeypatch.delenv('TERM_PROGRAM', raising=False)
  lines = menu.logo_lines()
  assert len(lines) == 4 and all(menu._width(line) == 5 for line in lines)
  monkeypatch.setenv('NO_COLOR', '1')
  assert menu.logo_lines() == []
  monkeypatch.delenv('NO_COLOR')
  monkeypatch.setenv('TERM_PROGRAM', 'Apple_Terminal')
  assert menu.logo_lines() == []

@pytest.mark.parametrize('columns,shown', [(80, True), (40, False)])
def test_logo_beside_the_title_only_when_it_fits(monkeypatch, columns, shown):
  import os
  monkeypatch.delenv('NO_COLOR', raising=False)
  monkeypatch.delenv('TERM_PROGRAM', raising=False)
  monkeypatch.setattr('shutil.get_terminal_size', lambda fallback=None: os.terminal_size((columns, 24)))
  art, lines = menu.logo_lines(), ['infurnace', 'Server: not running']
  drawn = menu._beside(art, lines)
  if not shown: assert drawn == lines
  else:  # the two lines centred on the logo's four
    assert len(drawn) == 4 and drawn[1].endswith('infurnace') and drawn[2].endswith('Server: not running')
    assert all(d.startswith(a.rstrip()) for d, a in zip(drawn, art))

def test_header_shows_the_servers_average_speeds(monkeypatch):
  status = {'status': 'ready', 'model': 'bonsai:4b', 'speeds': {'requests': 3, 'prefill_tokens_per_second': 6061, 'decode_tokens_per_second': 198,
                                                                  'peak_prefill_tokens_per_second': 6209, 'peak_decode_tokens_per_second': 213}}
  monkeypatch.setattr('infurnace.client.status', lambda: status)
  notes, _, wait = menu.Menu(lambda argv: 0, None).header()
  assert notes[1:] == ['Average: prefill 6,061 tok/s · decode 198 tok/s (3 requests)', 'Peak:    prefill 6,209 tok/s · decode 213 tok/s']
  assert wait == 1.0

def test_a_ready_server_is_listed_in_pi_without_changing_its_default(monkeypatch, home):
  import json
  agent = home / '.pi' / 'agent'
  agent.mkdir(parents=True)
  (agent / 'settings.json').write_text('{"defaultProvider": "opencode-go", "defaultModel": "kimi"}')
  monkeypatch.setattr('infurnace.client.status', lambda: {'status': 'ready', 'model': 'bonsai:4b'})
  monkeypatch.setattr('infurnace.client.request', lambda method, path, body=None, timeout=None:
                      {'models': [{'context_length': 32768}]} if path == '/api/ps' else {})
  m = menu.Menu(lambda argv: 0, None)
  m.header()
  m.header()  # once per model
  models = json.loads((agent / 'models.json').read_text())['providers']['infurnace']['models']
  assert [(e['id'], e['contextWindow']) for e in models] == [('bonsai:4b', 32768)]
  assert json.loads((agent / 'settings.json').read_text())['defaultModel'] == 'kimi'

def test_nothing_is_written_for_pi_when_pi_is_not_installed(monkeypatch, home):
  monkeypatch.setattr('infurnace.client.status', lambda: {'status': 'ready', 'model': 'bonsai:4b'})
  menu.Menu(lambda argv: 0, None).header()
  assert not (home / '.pi').exists()
