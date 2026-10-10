"""Diagnostic BEAM=2 search for mapped verifier sampling only; baseline unchanged.

This scoped experiment did not outperform the prior BEAM=0 application result.
Use the same arguments as mapped_io_benchmark.py. Not an inference default.
"""
import json
from pathlib import Path
import sys

from tinygrad import Context
from tinygrad.engine.jit import _TinyJit
from benchmarks.experiments.eagle3.mapped_io_benchmark import main as benchmark


def main():
  original = _TinyJit.__call__
  def call(self, *args, **kwargs):
    function = getattr(self, 'fxn', None)
    if (getattr(function, '__module__', '') == 'benchmarks.experiments.eagle3.mapped_io_benchmark'
        and getattr(function, '__name__', '') == 'sample'):
      with Context(BEAM=2): return original(self, *args, **kwargs)
    return original(self, *args, **kwargs)
  _TinyJit.__call__ = call
  try: benchmark()
  finally:
    _TinyJit.__call__ = original
    if '--json' in sys.argv:
      path = Path(sys.argv[sys.argv.index('--json')+1])
      if path.exists():
        report = json.loads(path.read_text())
        report['experimental_sampling_beam'] = 2
        path.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__': main()
