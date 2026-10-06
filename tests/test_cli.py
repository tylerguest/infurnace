from types import SimpleNamespace

import pytest

from infurnace import cli


class Tokenizer:
  bos_id = None
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
  assert cli._stream(model, Tokenizer(), [1], 3) == "3!"
  assert capsys.readouterr().out == "3!\n"
  assert model.closed


def test_stream_limit_flush(capsys):
  assert cli._run_single_shot(Model(), Tokenizer(), "hello", 1) == 0
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
  cli._stream(model, Tokenizer(), [1, 2], None)
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
  assert cli._run_chat(Model(), CountingTokenizer(), None, 16) == 0
  assert capsys.readouterr().err.count("discarded 1 oldest turn") == 2


def test_context_generation_stop(capsys):
  model = Model()
  model.config = SimpleNamespace(context_length=2)
  cli._stream(model, Tokenizer(), [1], 1)
  assert "context limit reached" in capsys.readouterr().err


def test_chat_eof(monkeypatch, capsys):
  values = iter(["hello"])
  def input_line(prompt):
    try: return next(values)
    except StopIteration: raise EOFError
  monkeypatch.setattr("builtins.input", input_line)
  assert cli._run_chat(Model(), Tokenizer(), 3, 16) == 0
  assert capsys.readouterr().out == "3!\n"


def test_context_error_keeps_history(monkeypatch, capsys):
  model = Model()
  model.config = SimpleNamespace(context_length=2)
  values = iter(["hello"])
  def input_line(prompt):
    try: return next(values)
    except StopIteration: raise EOFError
  monkeypatch.setattr("builtins.input", input_line)
  cli._run_chat(model, Tokenizer(), 1, 2)
  assert "History was not truncated" in capsys.readouterr().err
  assert not model.closed


@pytest.mark.parametrize("args", [["--prompt", "hi", "--chat"], ["--max-tokens", "0"], ["--benchmark", "-1"],
                                  ["--max-context", "0"], ["--prompt", "hi", "--no-thinking"]])
def test_invalid_flags_before_loading(monkeypatch, args):
  monkeypatch.setattr("sys.argv", ["infurnace", *args])
  def unexpected(*args): pytest.fail("loaded model for invalid flags")
  monkeypatch.setattr(cli, "_build", unexpected)
  with pytest.raises(SystemExit) as error: cli.main()
  assert error.value.code == 2


def test_warmup_status(capsys):
  model = Model()
  calls = []
  model.warmup = lambda: calls.append(True)
  cli._warmup(model)
  assert calls == [True]
  output = capsys.readouterr()
  assert not output.out
  assert "Warming up CPU" in output.err and "Ready" in output.err


def test_benchmark_counts_and_reset():
  model = Model()
  result = cli._measure(model, [1, 2], 3)
  assert model.resets == 1
  assert result["generated_tokens"] == 3  # Includes EOS intentionally.
  assert result["prompt_tokens"] == 2
  assert result["ttft_seconds"] >= 0
  assert result["decode_tokens_per_second"] > 0


def test_default_model_environment(monkeypatch):
  monkeypatch.setenv("INFURNACE_MODEL", "~/custom.gguf")
  from pathlib import Path
  assert cli._default_model() == str(Path.home() / "custom.gguf")


def test_default_model_user_storage(monkeypatch, tmp_path):
  monkeypatch.delenv("INFURNACE_MODEL", raising=False)
  monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
  path = tmp_path / "infurnace/models/Qwen3-4B-Q4_K_M.gguf"
  path.parent.mkdir(parents=True)
  path.touch()
  monkeypatch.chdir(tmp_path)
  assert cli._default_model() == str(path)


def test_build_uses_loaded_metadata(monkeypatch):
  metadata = {"test": True}
  model = SimpleNamespace(metadata=metadata)
  calls = []
  def from_file(path, **kwargs):
    calls.append((path, kwargs))
    return model
  monkeypatch.setattr(cli.Qwen3Model, "from_gguf", from_file)
  monkeypatch.setattr(cli.Tokenizer, "from_gguf", lambda md: md)
  result, tokenizer = cli._build("model", 16, "CPU")
  assert result is model and tokenizer is metadata
  assert calls == [("model", {"max_context": 16, "device": "CPU"})]
