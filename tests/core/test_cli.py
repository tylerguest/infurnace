import pytest

from infurnace import cli, loader

@pytest.mark.parametrize("args", [["--prompt", "hi", "--chat"], ["--max-tokens", "0"], ["--benchmark", "-1"],
                                  ["--max-context", "0"], ["--prompt", "hi", "--no-thinking"], ["--serve", "0"],
                                  ["--serve", "--max-tokens", "5"], ["--no-dflash", "--eagle3", "x"],
                                  ["--dflash", "missing.gguf"], ["--eagle3", "missing-dir"], ["--nv-prefill"]])
def test_invalid_flags_before_loading(monkeypatch, args):
  monkeypatch.setattr(loader, "load_model", lambda *a, **kw: pytest.fail("loaded a model for invalid flags"))
  with pytest.raises(SystemExit) as error: cli.main(args)
  assert error.value.code == 2

@pytest.mark.parametrize("value,expected", [("32", (None, 32)), ("1000,50", (1000, 50))])
def test_benchmark_spec(value, expected):
  assert cli.parse_args(["--benchmark", value]).benchmark == expected
  assert cli.parse_args(["--benchmark"]).benchmark == (None, 20)
