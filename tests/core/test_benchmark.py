import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("compare_benchmark", Path(__file__).resolve().parents[2]/"benchmarks/compare.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def step(seconds):
  return {"seconds": seconds, "estimated_bytes": 100, "estimated_ops": 200, "kernel_count": 2}


def test_summary_uses_aggregate_rate():
  result = benchmark.summarize([step(0.01), step(0.03)])
  assert result["tokens_per_second"] == pytest.approx(50)
  assert result["latency_ms_median"] == pytest.approx(20)
  assert result["latency_ms_p95"] == pytest.approx(30)
  assert result["estimated_bytes_per_token"] == 100


@pytest.mark.parametrize("steps", [[], [step(0)], [step(-1)]])
def test_invalid_summary(steps):
  with pytest.raises(ValueError): benchmark.summarize(steps)


def test_measure_closes_stream_and_excludes_first_token():
  closed = []
  def stream():
    try:
      yield from [1, 2, 3]
    finally:
      closed.append(True)
  result = benchmark.measure(stream(), 3, "CPU")
  assert closed == [True]
  assert result["all_steps"]["tokens"] == 3
  assert result["decode"]["tokens"] == 2
  assert [s["token_id"] for s in result["steps"]] == [1, 2, 3]


def test_measure_closes_on_error():
  closed = []
  def stream():
    try:
      yield 1
      raise RuntimeError("failure")
    finally:
      closed.append(True)
  with pytest.raises(RuntimeError, match="failure"): benchmark.measure(stream(), 3, "CPU")
  assert closed == [True]


@pytest.mark.parametrize("flags,expected", [([], None), (["--nv-prefill"], True), (["--no-nv-prefill"], False)])
def test_prefill_selection_forwarded_to_children(monkeypatch, flags, expected):
  from types import SimpleNamespace
  args = benchmark.parser().parse_args(flags)
  assert args.nv_prefill is expected
  commands = []
  def run(command, **kwargs):
    commands.append(command)
    return SimpleNamespace(returncode=0, stdout="{}", stderr="")
  monkeypatch.setattr(benchmark.subprocess, "run", run)
  benchmark.run_child(args, "infurnace")
  assert ("--nv-prefill" in commands[0]) is (expected is True)
  assert ("--no-nv-prefill" in commands[0]) is (expected is False)


@pytest.mark.parametrize("flag", ["--nv-attention", "--nv-fused-attention"])
def test_attention_forwarded_to_child(monkeypatch, flag):
  from types import SimpleNamespace
  args = benchmark.parser().parse_args([flag])
  commands = []
  def run(command, **kwargs):
    commands.append(command)
    return SimpleNamespace(returncode=0, stdout="{}", stderr="")
  monkeypatch.setattr(benchmark.subprocess, "run", run)
  benchmark.run_child(args, "infurnace")
  assert flag in commands[0]
