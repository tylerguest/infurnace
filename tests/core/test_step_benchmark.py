import argparse
from types import SimpleNamespace

import pytest

from infurnace import cli


@pytest.mark.parametrize("value,expected", [("32", (None, 32)), ("1000,50", (1000, 50))])
def test_benchmark_spec(value, expected):
  assert cli._benchmark_spec(value) == expected


@pytest.mark.parametrize("value", ["0", "-1", "0,5", "5,-1", "1,2,3", "x", "5,"])
def test_invalid_benchmark_spec(value):
  with pytest.raises(argparse.ArgumentTypeError): cli._benchmark_spec(value)


def test_per_step_output_and_cleanup(monkeypatch, capsys):
  class Model:
    config = SimpleNamespace(context_length=32)
    device = "CPU"
    closed = False
    def reset(self): pass
    def generate(self, ids, count):
      try:
        yield from [3, 4, 5][:count]
      finally: self.closed = True
  class Tokenizer:
    bos_id = None
    def encode(self, text): return [1, 2]
    def stream_decoder(self): return lambda token: str(token)
  class Timer:
    et = 10_000_000
    def __init__(self, **kwargs): pass
    def __enter__(self): pass
    def __exit__(self, *args): pass
  monkeypatch.setattr(cli, "Timing", Timer)
  monkeypatch.setattr(cli, "profile_marker", lambda marker: None)
  model = Model()
  cli._run_step_benchmark(model, Tokenizer(), 3, prompt_tokens=2, prompt_text="hi")
  text = capsys.readouterr().out
  assert "prefill     2:" in text
  assert "decode      0:" in text and "decode      1:" in text
  assert "decode aggregate: 100.00 tok/s (2 steps" in text
  assert "est GB/s" in text and "active=4" in text
  assert model.closed
