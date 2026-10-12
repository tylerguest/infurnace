from types import SimpleNamespace

from infurnace import benchmark
from infurnace.chat import Decoder, run_chat, run_prompt, stream_reply


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
  assert stream_reply(Decoder(model, Tokenizer()), Tokenizer(), [1], 3) == "3!"
  assert capsys.readouterr().out == "3!\n"
  assert model.closed


def test_stream_limit_flush(capsys):
  assert run_prompt(Decoder(Model(), Tokenizer()), Tokenizer(), "hello", 1) == 0
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
  stream_reply(Decoder(model, Tokenizer()), Tokenizer(), [1, 2], None)
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
  assert run_chat(Decoder(Model(), CountingTokenizer()), CountingTokenizer(), None, 16) == 0
  assert capsys.readouterr().err.count("discarded 1 oldest turn") == 2


def test_context_generation_stop(capsys):
  model = Model()
  model.config = SimpleNamespace(context_length=2)
  stream_reply(Decoder(model, Tokenizer()), Tokenizer(), [1], 1)
  assert "context limit reached" in capsys.readouterr().err


def test_chat_eof(monkeypatch, capsys):
  values = iter(["hello"])
  def input_line(prompt):
    try: return next(values)
    except StopIteration: raise EOFError
  monkeypatch.setattr("builtins.input", input_line)
  assert run_chat(Decoder(Model(), Tokenizer()), Tokenizer(), 3, 16) == 0
  assert capsys.readouterr().out == "3!\n"


def test_context_error_keeps_history(monkeypatch, capsys):
  model = Model()
  model.config = SimpleNamespace(context_length=2)
  values = iter(["hello"])
  def input_line(prompt):
    try: return next(values)
    except StopIteration: raise EOFError
  monkeypatch.setattr("builtins.input", input_line)
  run_chat(Decoder(model, Tokenizer()), Tokenizer(), 1, 2)
  assert "History was not truncated" in capsys.readouterr().err
  assert not model.closed


def test_warmup_status(capsys):
  model = Model()
  calls = []
  model.warmup = lambda: calls.append(True)
  Decoder(model, Tokenizer()).warmup()
  assert calls == [True]
  output = capsys.readouterr()
  assert not output.out
  assert "Warming up CPU" in output.err and "Ready" in output.err


def test_benchmark_counts_and_reset():
  model = Model()
  result = benchmark._measure(model, [1, 2], 3)
  assert model.resets == 1
  assert result["generated_tokens"] == 3  # Includes EOS intentionally.
  assert result["prompt_tokens"] == 2
  assert result["ttft_seconds"] >= 0
  assert result["decode_tokens_per_second"] > 0
