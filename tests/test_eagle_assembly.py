import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def inspection():
  path = Path(__file__).resolve().parents[1]/'benchmarks/experiments/eagle3/inspect_verification.py'
  spec = importlib.util.spec_from_file_location('eagle_assembly_inspection', path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_assembly_checkpoint_marks_diagnostic_and_failure(tmp_path, inspection):
  args = SimpleNamespace(output_dir=tmp_path/'assembly')
  inspection.checkpoint(args, 'aborted', [{'exact_error': 0}], RuntimeError('failed'))
  report = json.loads((args.output_dir/'report.json').read_text())
  assert report == {'status': 'aborted', 'diagnostic_only': True,
                    'records': [{'exact_error': 0}], 'error': 'RuntimeError: failed'}


@pytest.mark.parametrize('failure', [None, RuntimeError('compile failed')])
def test_assembly_main_records_completion_or_abort(tmp_path, monkeypatch, inspection, failure):
  monkeypatch.setattr('sys.argv', ['inspect_verification.py', '--output-dir', str(tmp_path)])
  def run(args, records):
    records.append({'exact_error': 0})
    if failure is not None: raise failure
  monkeypatch.setattr(inspection, 'inspect', run)
  if failure is None: inspection.main()
  else:
    with pytest.raises(RuntimeError, match='compile failed'): inspection.main()
  report = json.loads((tmp_path/'report.json').read_text())
  assert report['status'] == ('complete' if failure is None else 'aborted')
  assert report['diagnostic_only'] and report['records'] == [{'exact_error': 0}]


@pytest.mark.parametrize('materialize_scores,parallel_scores', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('mapped_draft_output,fused_root', [(False, False), (True, False), (True, True)])
def test_mapped_io_scope_preserves_baseline_and_restores_descriptors(materialize_scores, parallel_scores, mapped_draft_output, fused_root):
  from tinygrad import Tensor, dtypes
  from tinygrad.device import Buffer
  from infurnace.eagle.verify import BatchedVerifier
  import infurnace.eagle.sampling as chain_sampling
  from infurnace.eagle import EagleDraft
  old_draft_chain = EagleDraft.__dict__['_observe_and_chain']
  old_init = BatchedVerifier.__dict__['__init__']
  path = Path(__file__).resolve().parents[1]/'benchmarks/experiments/eagle3/mapped_io_benchmark.py'
  spec = importlib.util.spec_from_file_location('eagle_mapped_io', path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  original = (Tensor.__dict__.get('empty'), Buffer.__dict__['copy_from'], BatchedVerifier.__dict__['_sample'],
              BatchedVerifier.__dict__['_commit_forward'], chain_sampling.sample_chain, chain_sampling.score_argmax)
  with pytest.raises(RuntimeError, match='cancel'):
    with module.mapped_io(materialize_scores=materialize_scores, parallel_scores=parallel_scores,
                          mapped_draft_output=mapped_draft_output, fused_root=fused_root):
      # An ordinary caller must not get host-mapped NV storage.
      tensor = Tensor.empty(1, 2, device='CPU', dtype=dtypes.int32)
      assert tensor.device == 'CPU' and tensor.shape == (1, 2)
      raise RuntimeError('cancel')
  assert (Tensor.__dict__.get('empty'), Buffer.__dict__['copy_from'], BatchedVerifier.__dict__['_sample'],
          BatchedVerifier.__dict__['_commit_forward'], chain_sampling.sample_chain, chain_sampling.score_argmax) == original
  assert EagleDraft.__dict__['_observe_and_chain'] is old_draft_chain
  assert BatchedVerifier.__dict__['__init__'] is old_init


def test_mapped_io_does_not_bypass_unowned_buffer_copies(monkeypatch):
  from tinygrad.device import Buffer
  path = Path(__file__).resolve().parents[1]/'benchmarks/experiments/eagle3/mapped_io_benchmark.py'
  spec = importlib.util.spec_from_file_location('eagle_mapped_copy_scope', path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  copies = []
  def ordinary_copy(dest, src):
    copies.append((dest, src))
    return dest
  monkeypatch.setattr(Buffer, 'copy_from', ordinary_copy)
  dest, src = object(), object()
  with module.mapped_io(): assert Buffer.copy_from(dest, src) is dest
  assert copies == [(dest, src)]


def test_sampling_beam_scope_and_exception_restoration(monkeypatch):
  from tinygrad.helpers import BEAM
  from tinygrad.engine.jit import _TinyJit
  from benchmarks.experiments.eagle3 import sampling_beam_benchmark as module
  seen = []
  def fake_call(self): seen.append(BEAM.value)
  monkeypatch.setattr(_TinyJit, '__call__', fake_call)
  monkeypatch.setattr('sys.argv', ['sampling_beam_benchmark.py'])
  def sample(): pass
  sample.__module__ = 'benchmarks.experiments.eagle3.mapped_io_benchmark'
  def ordinary(): pass
  def benchmark():
    _TinyJit.__call__(SimpleNamespace(fxn=sample))
    _TinyJit.__call__(SimpleNamespace(fxn=ordinary))
    raise RuntimeError('cancelled')
  monkeypatch.setattr(module, 'benchmark', benchmark)
  original_beam = BEAM.value
  with pytest.raises(RuntimeError, match='cancelled'): module.main()
  assert seen == [2, original_beam]
  assert _TinyJit.__call__ is fake_call
  assert BEAM.value == original_beam
