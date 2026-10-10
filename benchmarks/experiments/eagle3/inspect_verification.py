"""Dump verification projection CUDA/PTX/SASS and cold-cache kernel diagnostics.

DEBUG=6 prints CUDA/IR on the pinned tinygrad; SASS auto-printing requires DEBUG=7.
This tool explicitly saves both PTX and the actual runtime cubin's disassembly.
Timings run under DEBUG=0, never constitute whole-request speedup evidence.
"""
import argparse
import json
import random
import re
import statistics
import struct
import subprocess
from pathlib import Path

from tinygrad import Context, Device, Tensor, dtypes
from tinygrad.engine.realize import lower_and_compile, time_call
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import Ops

from infurnace.kernels.quant_gemv import prepare_q4, prepare_q6, q8_quantize, quant_gemv

SHAPES = {'gate': (12, 9728, 2560), 'attention_out': (12, 2560, 4096),
          'down': (14, 2560, 9728), 'head': (14, 151936, 2560)}
VARIANTS = {'normal': {}, 'loop': {'loop_chunks': True},
            'local': {'loop_chunks': True, 'local_tokens': True},
            'reuse': {'loop_chunks': True, 'reuse_tokens': True},
            'fused': {'loop_chunks': True, 'fuse_chunks': True}}


def weights(typ, rows, width):
  size = 144 if typ == 12 else 210
  raw = bytearray(random.Random(53).randbytes(rows*width//256*size))
  header = struct.pack('<ee', .001, .0002) if typ == 12 else struct.pack('<e', .001)
  offset = 0 if typ == 12 else 208
  for block in range(0, len(raw), size): raw[block+offset:block+offset+len(header)] = header
  tensor = Tensor(bytes(raw), device='NV', dtype=dtypes.uint8).realize()
  return (prepare_q4(tensor, in_features=width) if typ == 12
          else prepare_q6(tensor, in_features=width, swizzled=True))


def dump_program(program, directory, stem, debug=0):
  directory.mkdir(parents=True, exist_ok=True)
  source = next(u.arg for u in program.src if u.op is Ops.SOURCE)
  binary = next(u.arg for u in program.src if u.op is Ops.BINARY)
  (directory/f'{stem}.cu').write_text(source)
  cubin = directory/f'{stem}.cubin'
  cubin.write_bytes(binary)
  arch = Device['NV'].renderer.compiler.arch
  ptx = NVRTCCompiler(arch, ptx=True).compile_cached(source).rstrip(b'\x00')
  (directory/f'{stem}.ptx').write_bytes(ptx)
  resources = subprocess.run(['cuobjdump', '--dump-resource-usage', str(cubin)], check=True,
                             capture_output=True, text=True).stdout
  sass = subprocess.run(['nvdisasm', str(cubin)], check=True, capture_output=True, text=True).stdout
  (directory/f'{stem}.resources.txt').write_text(resources)
  (directory/f'{stem}.sass').write_text(sass)
  if debug >= 6:
    print(source, flush=True)
    print(ptx.decode(), flush=True)
  if debug >= 7: print(sass, flush=True)
  instructions = re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)', sass)
  counts = {name: instructions.count(name) for name in sorted(set(instructions))}
  return {'resources': resources.strip(), 'sass_instruction_counts': counts,
          'ptx_local_ops': len(re.findall(r'\b(?:ld|st)\.local\b', ptx.decode()))}


def checkpoint(args, status, records, error=None):
  args.output_dir.mkdir(parents=True, exist_ok=True)
  payload = dict(status=status, diagnostic_only=True, records=records)
  if error is not None: payload['error'] = f'{type(error).__name__}: {error}'
  (args.output_dir/'report.json').write_text(json.dumps(payload, indent=2)+'\n')


def inspect(args, records):
  Tensor.manual_seed(101)
  for shape in args.shapes:
    typ, rows, width = SHAPES[shape]
    with Context(DEBUG=0): weight = weights(typ, rows, width)
    for count in args.tokens:
      with Context(DEBUG=0):
        x = Tensor.randn(1, count, width, device='NV').realize()
        q8 = q8_quantize(x, count, width)
        Tensor.realize(*q8)
      settings = dict(out_features=rows, in_features=width, ggml_type=typ,
                      q4_swizzled=typ == 12, q6_aligned=typ == 14, q6_swizzled=typ == 14,
                      interleave_tokens=True, return_parts=True, q8=q8)
      with Context(DEBUG=0): reference = quant_gemv(x, weight, **settings).realize()
      for variant in args.variants:
        stem = f'{shape}_{count}_{variant}'
        print(f'INSPECT {stem}', flush=True)
        with Context(DEBUG=args.debug):
          out = quant_gemv(x, weight, **settings, **VARIANTS[variant])
          linear = out.schedule_linear()
          compiled = lower_and_compile(linear)
        programs = [u for u in compiled.toposort() if u.op is Ops.PROGRAM]
        if len(programs) != 1: raise RuntimeError(f'expected single projection, got {len(programs)}')
        details = dump_program(programs[0], args.output_dir, stem, args.debug)
        with Context(DEBUG=0):
          timer = time_call(compiled.src[0], clear_l2=True)
          samples = [next(timer)*1e6 for _ in range(args.trials+3)][3:]
          actual = Tensor(out.uop.buf_uop).reshape(out.shape)
          expected = reference.sum(-1, keepdim=True) if variant == 'fused' else reference
          error = (actual-expected).abs().max().item()
        if error: raise AssertionError((stem, error))
        record = dict(shape=shape, tokens=count, variant=variant, rows=rows, width=width,
                      median_us=statistics.median(samples), exact_error=error, **details)
        records.append(record)
        checkpoint(args, 'running', records)
        print('RESULT', json.dumps(record), flush=True)
    del weight


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--shapes', nargs='+', choices=SHAPES, default=list(SHAPES))
  parser.add_argument('--tokens', type=int, nargs='+', default=[1, 2, 3])
  parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=['normal', 'loop', 'local', 'reuse'])
  parser.add_argument('--trials', type=int, default=12)
  parser.add_argument('--debug', type=int, default=6)
  args = parser.parse_args()
  if args.trials < 3 or any(not 1 <= count <= 8 for count in args.tokens): parser.error('invalid counts')
  records = []
  checkpoint(args, 'running', records)
  try:
    inspect(args, records)
  except BaseException as error:
    checkpoint(args, 'aborted', records, error)
    raise
  checkpoint(args, 'complete', records)


if __name__ == '__main__': main()
