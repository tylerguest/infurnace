import argparse
from types import SimpleNamespace

import pytest

from infurnace import cli

@pytest.mark.parametrize("value,expected", [("32", (512, 32)), ("1000,50", (1000, 50))])
def test_benchmark_spec(value, expected):
  assert cli._benchmark_spec(value) == expected

@pytest.mark.parametrize("value", ["0", "-1", "0,5", "5,-1", "1,2,3", "x", "5,"])
def test_invalid_benchmark_spec(value):
  with pytest.raises(argparse.ArgumentTypeError): cli._benchmark_spec(value)

def test_benchmark_raises_context_to_fit():
  assert cli.parse_args(["--benchmark", "4000,100"]).max_context >= 4100

def test_per_step_output_and_cleanup(monkeypatch, capsys):
  class Model:
    config = SimpleNamespace(context_length=32)
    closed = warmed = False
    def reset(self): pass
    def warmup(self): self.warmed = True
    def generate(self, ids, count):
      assert ids == [0, 1]
      try:
        yield from [3, 4, 5][:count]
      finally: self.closed = True
  class Tokenizer:
    def encode(self, text): return [int(word) for word in text.split()]
    def stream_decoder(self): return lambda token: str(token)
  class Timer:
    def __init__(self, prefix="", on_exit=None, enabled=True): self.prefix, self.on_exit, self.enabled = prefix, on_exit, enabled
    def __enter__(self): pass
    def __exit__(self, *args):
      if self.enabled: print(self.prefix + "10.00 ms" + self.on_exit(10_000_000))
  monkeypatch.setattr(cli, "Timing", Timer)
  monkeypatch.setattr(cli, "profile_marker", lambda marker: None)
  model = Model()
  assert cli.benchmark(model, Tokenizer(), 2, 3) == 0
  text = capsys.readouterr().out
  assert "prefill     2: 10.00 ms,   200.00 tok/s" in text
  assert "decode      0: 10.00 ms,   100.00 tok/s" in text and "decode      1:" in text
  assert text.splitlines()[-1].endswith("--  345")  # decoded text through this step
  assert model.warmed and model.closed
