import pytest
from types import SimpleNamespace

from infurnace import cli

class Tokenizer:
  bos_id = eos_id = eot_id = None
  def encode(self, text): return [1, 2]
  def render_chat(self, messages, **kwargs): return "chat"
  def is_end(self, token): return token == 9
  def stream_decoder(self): return lambda token: "!" if token is None else str(token)

class Model:
  config = SimpleNamespace(context_length=16)
  device = "CPU"
  metadata = {}
  closed = False
  resets = 0
  def reset(self): self.resets += 1
  def generate(self, ids, count):
    try:
      for token in [3, 9, 4][:count]: yield token
    finally: self.closed = True

def test_stream_eos_and_flush(capsys):
  model = Model()
  assert cli.reply(model.generate, Tokenizer(), [1], 3, model.config.context_length) == "3!"
  assert capsys.readouterr().out == "3!\n"
  assert model.closed

def test_stream_limit_flush(capsys):
  assert cli.reply(Model().generate, Tokenizer(), [1, 2], 1, 16) == "3!"
  output = capsys.readouterr()
  assert output.out == "3!\n"
  assert "1-token limit" in output.err
  assert "includes thinking" in output.err

def test_chat_default_uses_remaining_context(capsys):
  model = Model()
  seen = []
  original = model.generate
  def generate(ids, count):
    seen.append(count)
    return original(ids, count)
  model.generate = generate
  cli.reply(model.generate, Tokenizer(), [1, 2], None, model.config.context_length)
  assert seen == [14]

def test_chat_rolls_old_turns(monkeypatch, capsys):
  class CountingTokenizer(Tokenizer):
    def render_chat(self, messages, **kwargs): return "x " * (len(messages)*3)
    def encode(self, text): return [1]*len(text.split())
  values = iter(["one", "two", "three"])
  def input_line(prompt):
    try: return next(values)
    except StopIteration: raise EOFError
  monkeypatch.setattr("builtins.input", input_line)
  assert cli.chat(Model().generate, CountingTokenizer(), 16, None) == 0
  assert capsys.readouterr().err.count("discarded 1 oldest turn") == 2

def test_context_generation_stop(capsys):
  model = Model()
  model.config = SimpleNamespace(context_length=2)
  cli.reply(model.generate, Tokenizer(), [1], 1, model.config.context_length)
  assert "context limit reached" in capsys.readouterr().err

def test_chat_eof(monkeypatch, capsys):
  values = iter(["hello"])
  def input_line(prompt):
    try: return next(values)
    except StopIteration: raise EOFError
  monkeypatch.setattr("builtins.input", input_line)
  assert cli.chat(Model().generate, Tokenizer(), 16, 3) == 0
  assert capsys.readouterr().out == "3!\n"

def test_context_error_keeps_history(monkeypatch, capsys):
  model = Model()
  model.config = SimpleNamespace(context_length=2)
  values = iter(["hello"])
  def input_line(prompt):
    try: return next(values)
    except StopIteration: raise EOFError
  monkeypatch.setattr("builtins.input", input_line)
  cli.chat(model.generate, Tokenizer(), 2, 1)
  assert "History was not truncated" in capsys.readouterr().err
  assert not model.closed

def test_warmup_status(capsys):
  model = Model()
  calls = []
  model.warmup = lambda prompt_graph=True: calls.append(prompt_graph)
  cli.warmup(model)
  assert calls == [True]
  output = capsys.readouterr()
  assert not output.out
  assert "Warming up CPU" in output.err and "Ready" in output.err
def test_chat_waits_for_warmup_before_each_reply(monkeypatch, capsys):
  events = []
  model = Model()
  original = model.generate
  def generate(ids, count):
    events.append("generate")
    return original(ids, count)
  values = iter(["hello"])
  def input_line(prompt):
    events.append("input")
    try: return next(values)
    except StopIteration: raise EOFError
  monkeypatch.setattr("builtins.input", input_line)
  assert cli.chat(generate, Tokenizer(), 16, 3, ready=lambda: events.append("ready")) == 0
  assert events == ["input", "ready", "generate", "input"]

def test_background_warmup_runs_off_the_main_thread_and_reraises(capsys):
  import threading
  started, release = threading.Event(), threading.Event()
  model = Model()
  def slow_failure(prompt_graph=True):
    started.set()
    assert threading.current_thread() is not threading.main_thread()
    release.wait(5)
    raise RuntimeError("warmup failed")
  model.warmup = slow_failure
  background = cli.BackgroundWarmup(model)
  assert started.wait(5)  # running while the main thread is free (e.g. reading the keyboard)
  release.set()
  with pytest.raises(RuntimeError, match="warmup failed"): background.wait()
  background.wait()  # the error is reported once
