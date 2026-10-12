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

@pytest.mark.parametrize('worker', ['infurnace', 'tinygrad'])
def test_comparison_worker_runs_qwen35_fixture(tmp_path, monkeypatch, worker):
  from types import SimpleNamespace
  from tinygrad import Context
  from infurnace.tokenizer import Qwen35Tokenizer
  from test_qwen35 import checkpoint
  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'benchmarks'))
  with Context(DEV='CPU'):
    path = checkpoint(tmp_path)
  tokenizer = SimpleNamespace(bos_id=None, encode=lambda text: [1, 2])
  monkeypatch.setattr(Qwen35Tokenizer, 'from_gguf', lambda metadata: tokenizer)
  args = benchmark.parser().parse_args(['--worker', worker, '--model', str(path), '--device', 'CPU', '--context', '16',
                                       '--tokens', '3', '--runs', '1', '--prompt-tokens', '2', '--no-nv-quant'])
  report = benchmark.worker(args)
  assert report['runner'] == worker
  assert report['prompt_ids'] == [1, 2]
  assert report['context_capacity'] == 16
  assert report['warm_runs'][0]['all_steps']['tokens'] == 3
  assert report['nv_quant_linears'] == 0


def test_comparison_defaults_to_supported_model():
  assert benchmark.parser().parse_args([]).model == 'qwen3.5:4b'
