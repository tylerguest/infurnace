"""Explicit experimental head override around the exact-output comparison protocol.

Example (from the repository root):
  PYTHONPATH=. python benchmarks/experiments/eagle3/calibrated_benchmark.py \\
    --head models/eagle3-head-calibration-q8.safetensors \\
    models/Qwen3-4B-Q4_K_M.gguf models/eagle3-qwen3-4b \\
    --context 512 --output 256 --draft-tokens 1 --loop-projections --fused-draft \\
    --warmup-pairs 2 --repeats 2 --json /tmp/calibrated-eagle.json

The target and original checkpoint are never modified. Only a fresh draft object's
head is overridden, with metadata hashes checked against both source files.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from tinygrad import dtypes
from tinygrad.nn.state import safe_load


def file_hash(path):
  with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_head(draft, head):
  weights = safe_load(head)
  if set(weights) != {'lm_head.weight'}: raise ValueError('head file must contain only lm_head.weight')
  weight = weights['lm_head.weight']
  if weight.shape != (draft.draft_vocab, draft.dim) or weight.dtype != dtypes.bfloat16:
    raise ValueError('head shape/dtype does not match draft')
  draft.weights['lm_head.weight'] = weight.to(draft.device).contiguous().realize()


def main():
  parser = argparse.ArgumentParser(description=__doc__, add_help=False)
  parser.add_argument('--head', type=Path, required=True)
  parser.add_argument('--json', type=Path, required=True)
  args, remaining = parser.parse_known_args()
  metadata = json.loads(args.head.with_suffix('.json').read_text())
  if not isinstance(metadata, dict) or not {'target_sha256', 'checkpoint_sha256'} <= metadata.keys():
    raise ValueError('head metadata must identify target and original checkpoint')
  spec = importlib.util.spec_from_file_location('eagle_comparison', Path(__file__).resolve().parents[2]/'eagle.py')
  benchmark = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(benchmark)
  build = benchmark._build
  original = benchmark.EagleDraft.from_checkpoint.__func__

  def checked_build(path, *values, **options):
    if file_hash(path) != metadata['target_sha256']: raise ValueError('head/target identity mismatch')
    return build(path, *values, **options)

  def calibrated(cls, checkpoint, model):
    if file_hash(checkpoint/'model.safetensors') != metadata['checkpoint_sha256']:
      raise ValueError('head/original checkpoint identity mismatch')
    draft = original(cls, checkpoint, model)
    load_head(draft, args.head)
    print('loaded experimental draft head; target/checkpoint hashes matched', flush=True)
    return draft

  benchmark._build = checked_build
  benchmark.EagleDraft.from_checkpoint = classmethod(calibrated)
  saved_argv = sys.argv
  sys.argv = [str(spec.origin), *remaining, '--json', str(args.json)]
  try:
    benchmark.main()
  finally:
    sys.argv = saved_argv
    benchmark.EagleDraft.from_checkpoint = classmethod(original)
    if args.json.exists():
      report = json.loads(args.json.read_text())
      report.update(experimental_draft_head=str(args.head), calibration=metadata)
      args.json.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__': main()
