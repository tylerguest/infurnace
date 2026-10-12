import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('eagle_benchmark', Path(__file__).resolve().parents[2]/'benchmarks/eagle.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)

@pytest.mark.parametrize('phase', ['target', 'draft'])
def test_loading_failure_is_checkpointed(monkeypatch, tmp_path, phase):
  def fail(*args): raise RuntimeError('injected loading failure')
  monkeypatch.setattr(benchmark, '_build', fail if phase == 'target' else lambda *args: (None, None))
  monkeypatch.setattr(benchmark.EagleDraft, 'from_checkpoint', fail)
  path = tmp_path/'failed.json'
  monkeypatch.setattr('sys.argv', ['eagle.py', 'model', 'draft', '--json', str(path)])
  with pytest.raises(RuntimeError, match='injected loading'): benchmark.main()
  report = json.loads(path.read_text())
  assert report['status'] == 'aborted' and report['stage'] == 'loading'
  assert report['error'] and not report['records'] and not report['comparisons']
  assert report['runtime']['rtalloc_pool_bytes'] > 0

@pytest.mark.parametrize('failure', [None, 'mismatch', 'missing_bracket', 'late_failure'])
def test_matched_report_rejects_mismatch_and_incomplete_bracket(monkeypatch, tmp_path, failure):
  calls, syncs = [], []
  class Model:
    device = 'NV'
    def generate(self, prompt, count, **kwargs):
      assert kwargs == dict(temperature=0.0, reuse_cache=False)
      calls.append('baseline')
      if (failure == 'missing_bracket' and len(calls) == 3) or (failure == 'late_failure' and len(calls) == 4):
        raise RuntimeError('injected timeout')
      yield from (3, 4, 5)
  class Engine:
    def __init__(self, *args): pass
    def generate(self, *args, **kwargs):
      yield SimpleNamespace(tokens=(3, 4), accepted=1, proposed=2)
      yield SimpleNamespace(tokens=(6 if failure == 'mismatch' else 5,), accepted=0, proposed=0)
  tokenizer = SimpleNamespace(encode=lambda _: [1, 2], render_chat=lambda _: 'chat')
  monkeypatch.setattr(benchmark, '_build', lambda *args: (Model(), tokenizer))
  monkeypatch.setattr(benchmark.EagleDraft, 'from_checkpoint', lambda *args: None)
  monkeypatch.setattr(benchmark, 'EagleGenerator', Engine)
  monkeypatch.setattr(benchmark, 'Device', {'NV': SimpleNamespace(synchronize=lambda: syncs.append(True))})
  path = tmp_path/'result.json'
  monkeypatch.setattr('sys.argv', ['eagle.py', 'model', 'draft', '--context', '8', '--output', '3',
                                  '--warmup-pairs', '1', '--repeats', '2' if failure == 'late_failure' else '1', '--json', str(path)])
  if failure:
    with pytest.raises(RuntimeError): benchmark.main()
  else:
    benchmark.main()
  report = json.loads(path.read_text())
  assert report['status'] == ('aborted' if failure else 'complete')
  if failure:
    assert report['error'] and report['comparisons'] == []
  else:
    assert [r['mode'] for r in report['records']] == ['baseline', 'eagle', 'baseline']
    eagle = report['records'][1]
    assert eagle['accepted'] == 1 and eagle['proposed'] == 2 and eagle['acceptance'] == 0.5
    assert eagle['committed_per_round'] == 1.5
    assert eagle['ttft_seconds'] >= 0 and eagle['decode_tok_s'] > 0
    assert len(report['comparisons']) == 1 and report['comparisons'][0]['speedup'] > 0
    assert len(syncs) == 10
