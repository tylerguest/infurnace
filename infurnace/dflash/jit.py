"""Captured graphs for DFlash."""
from tinygrad import TinyJit


def captured(fn) -> TinyJit:
  """A TinyJit that captures on its first call.

  TinyJit normally runs a function once eagerly and captures on the second call,
  scheduling every kernel twice; the 16-row target forward alone takes seconds to
  schedule. All state DFlash reads is realized before its first call
  (TargetObserver), so the eager pass has nothing left to materialize.
  """
  jit = TinyJit(fn)
  jit.cnt = 1
  return jit
