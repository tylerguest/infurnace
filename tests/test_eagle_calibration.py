"""CPU guards for isolated calibration tools; no training or model download."""
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_save

TOOLS = Path(__file__).resolve().parents[1]/'benchmarks'/'experiments'/'eagle3'


def load_tool(name):
  spec = importlib.util.spec_from_file_location(name, TOOLS/f'{name}.py')
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_calibration_hashes_stream_file_bytes(tmp_path):
  path = tmp_path/'payload'
  path.write_bytes(b'original target bytes')
  expected = hashlib.sha256(path.read_bytes()).hexdigest()
  assert load_tool('calibrate_head').file_hash(path) == expected
  assert load_tool('calibrated_benchmark').file_hash(path) == expected


@pytest.mark.parametrize('shape,dtype', [((3, 4), dtypes.bfloat16), ((2, 4), dtypes.float32)])
def test_calibrated_head_rejects_wrong_shape_or_dtype(tmp_path, shape, dtype):
  head = tmp_path/'head.safetensors'
  safe_save({'lm_head.weight': Tensor.zeros(shape, device='CPU', dtype=dtype).realize()}, head)
  original = Tensor.zeros((2, 4), device='CPU', dtype=dtypes.bfloat16).realize()
  draft = SimpleNamespace(device='CPU', draft_vocab=2, dim=4, weights={'lm_head.weight': original})
  with pytest.raises(ValueError, match='shape/dtype'): load_tool('calibrated_benchmark').load_head(draft, head)
  assert draft.weights['lm_head.weight'] is original


def test_calibrated_head_only_replaces_draft_weight(tmp_path):
  head = tmp_path/'head.safetensors'
  safe_save({'lm_head.weight': Tensor.full((2, 4), 2., device='CPU', dtype=dtypes.bfloat16).realize()}, head)
  original = Tensor.zeros((2, 4), device='CPU', dtype=dtypes.bfloat16).realize()
  draft = SimpleNamespace(device='CPU', draft_vocab=2, dim=4, weights={'lm_head.weight': original, 'other': original})
  load_tool('calibrated_benchmark').load_head(draft, head)
  assert draft.weights['lm_head.weight'].float().tolist() == [[2.]*4]*2
  assert draft.weights['other'] is original
  assert original.float().tolist() == [[0.]*4]*2


def test_calibrated_head_rejects_extra_tensors(tmp_path):
  head = tmp_path/'head.safetensors'
  safe_save({'extra': Tensor.zeros(1, device='CPU').realize()}, head)
  with pytest.raises(ValueError, match='only lm_head.weight'):
    load_tool('calibrated_benchmark').load_head(SimpleNamespace(), head)


def test_calibration_cli_cannot_replace_original_checkpoint(tmp_path, monkeypatch):
  module = load_tool('calibrate_head')
  checkpoint = tmp_path/'checkpoint'
  monkeypatch.setattr(sys, 'argv', ['calibrate_head.py', 'fit', '--checkpoint', str(checkpoint),
                                  '--dataset', str(tmp_path/'data.npz'), '--head', str(checkpoint/'model.safetensors')])
  with pytest.raises(SystemExit) as error: module.main()
  assert error.value.code == 2
