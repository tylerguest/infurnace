"""Inspect exact Gumbel-score/argmax scheduling, excluding RNG generation.

VIZ=2 saves rewrite/profile traces; this tool explicitly disables privileged
PMA counters. Use a dedicated TMPDIR to isolate the traces.
These synchronized timings are diagnostics, not application speedup evidence.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from tinygrad import Context, Device, Tensor, TinyJit, UOp
from tinygrad.engine.realize import lower_and_compile, run_linear, time_call
from tinygrad.codegen import to_program_cache
from tinygrad.helpers import VIZ
from tinygrad.uop.ops import Ops

from infurnace.eagle.sampling import score_argmax
from benchmarks.experiments.eagle3.inspect_verification import checkpoint, dump_program


def inspect(args, records):
  Tensor.manual_seed(101)
  for size in args.tokens:
    with Context(DEBUG=0, BEAM=0):
      logits = Tensor.randn(1, size, args.vocabulary, device='NV').realize()
      random = Tensor.rand(1, args.vocabulary, device='NV').realize()
      temperature = Tensor([0.0], device='NV').realize()
    for variant in (('original', 'materialized', 'parallel') if args.parallel_argmax else ('original', 'materialized')):
      materialized = variant != 'original'
      name = f'{size}_{variant}'
      def sample(logits, random, index):
        row = logits[:, index:index+1, :].reshape(1, args.vocabulary).contiguous()
        if variant == 'parallel':
          from benchmarks.experiments.eagle3.parallel_argmax import parallel_score_argmax
          return parallel_score_argmax(row, temperature, random)
        return score_argmax(row, temperature, random, materialize_scores=materialized)
      index = UOp.variable('sampling_row', 0, size-1).bind(0)
      with Context(DEBUG=0):
        output = sample(logits, random, index)
        linear, values = Tensor.linear_with_vars(output)
        compiled = lower_and_compile(linear)
        run_linear(compiled, values)
        Device['NV'].synchronize()
        decision = Tensor(output.uop.buf_uop).reshape(output.shape).item()
        with Context(BEAM=0):
          reference = score_argmax(logits[:, :1, :].reshape(1, args.vocabulary), temperature, random).item()
        if decision != reference: raise AssertionError((name, decision, reference))
      kernels = []
      for i, call in enumerate(compiled.src):
        stem = f'{name}_{i:02d}'
        details = dump_program(call.body, args.output_dir, stem, args.debug)
        with Context(DEBUG=0):
          timer = time_call(call, values)
          samples = [next(timer)*1e6 for _ in range(args.trials+2)][2:]
        kernels.append(dict(stem=stem,median_us=statistics.median(samples),**details))
      jit = TinyJit(sample)
      with Context(DEBUG=0):
        for _ in range(3):
          result = jit(logits, random, index)
          Device['NV'].synchronize()
          if result.item() != reference: raise AssertionError((name, 'JIT warmup', result.item(), reference))
        samples = []
        for _ in range(args.trials):
          start = time.perf_counter()
          result = jit(logits, random, index)
          Device['NV'].synchronize()
          samples.append((time.perf_counter()-start)*1e6)
          if result.item() != reference: raise AssertionError((name, 'JIT replay', result.item(), reference))
      jit_programs = []
      # Inspect selected capture programs, not the compiler cache: BEAM's
      # cache also contains rejected search candidates and CPU submissions.
      selected_keys = set()
      for node in jit.captured._linear.toposort():
        metadata = getattr(node.arg, 'aux', None)
        for kernel in getattr(metadata, 'kernels', ()):
          if kernel[4] is not None: selected_keys.add(kernel[4])
      for program in dict.fromkeys(to_program_cache.values()):
        if program.key not in selected_keys: continue
        if not any(u.op is Ops.SOURCE and '__global__' in u.arg for u in program.src): continue
        stem = f'{name}_jit_{len(jit_programs):02d}'
        jit_programs.append(dict(stem=stem, **dump_program(program, args.output_dir, stem, args.debug)))
      if not selected_keys or len(jit_programs) != len(selected_keys):
        raise AssertionError(('incomplete selected JIT shader dump', len(jit_programs), len(selected_keys)))
      record = dict(tokens=size, materialized_scores=materialized, variant=variant, exact_decision=True, jit_programs=jit_programs,
                    jit_median_us=statistics.median(samples), kernels=kernels, viz=VIZ.value, beam=args.beam,
                    decision=decision, reference_decision=reference)
      print(f'OUTPUT {name}: decision={decision} reference={reference}', flush=True)
      records.append(record)
      checkpoint(args, 'running', records)
      print('RESULT', json.dumps(record), flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--tokens', type=int, nargs='+', default=[1, 2, 3])
  parser.add_argument('--vocabulary', type=int, default=151936)
  parser.add_argument('--trials', type=int, default=8)
  parser.add_argument('--debug', type=int, default=6)
  parser.add_argument('--beam', type=int, default=0, help='tinygrad equivalent-kernel search width')
  parser.add_argument('--parallel-argmax', action='store_true', help='also inspect experimental two-pass reduction')
  args = parser.parse_args()
  if args.beam < 0 or args.vocabulary < 2 or args.trials < 3 or any(not 1 <= n <= 8 for n in args.tokens): parser.error('invalid sizes')
  records = []
  checkpoint(args, 'running', records)
  try:
    # PMA's default is bool on this pin: getenv casts the string '0' to True,
    # so PMA=0 in the environment cannot disable it when VIZ=2. Set the
    # context value before NVDevice is initialized; do not change the driver.
    from tinygrad.runtime.ops_nv import PMA  # noqa: F401
    with Context(PMA=0, BEAM=args.beam): inspect(args, records)
  except BaseException as error:
    checkpoint(args, 'aborted', records, error)
    raise
  checkpoint(args, 'complete', records)


if __name__ == '__main__': main()
