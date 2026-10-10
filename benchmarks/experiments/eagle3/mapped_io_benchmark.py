"""Scratch NV host-mapped EAGLE I/O experiment; not an ordinary inference option.

Only draft/verifier token buffers and verifier decision/compact-packet outputs are replaced.
Target arithmetic, sampling expression, RNG advancement, and baseline decode
are unchanged. Host reads/writes wait for prior GPU readers/writers. An earlier
protocol aborted with an NV synchronization error; a retry completed with only
a modest gain. This is not a reliability fix.
"""
from __future__ import annotations

from contextlib import contextmanager
import functools
import json
from pathlib import Path
import sys

from tinygrad import Device, Tensor, UOp
from tinygrad.device import Buffer, BufferSpec
from infurnace.eagle.verify import BatchedVerifier
from infurnace.eagle import EagleDraft
import infurnace.eagle.sampling as chain_sampling


def host_tensor(shape, dtype):
  size = 1
  for dim in shape: size *= dim
  storage = Buffer('NV', size, dtype, options=BufferSpec(host=True, cpu_access=True), preallocate=True)
  return Tensor(UOp.from_buffer(storage, 'NV')).reshape(shape)


@contextmanager
def mapped_io(*, materialize_scores=False, parallel_scores=False, mapped_draft_output=False, fused_root=False):
  if type(materialize_scores) is not bool: raise ValueError('materialize_scores must be boolean')
  if type(parallel_scores) is not bool or (parallel_scores and not materialize_scores):
    raise ValueError('parallel_scores requires materialize_scores and a boolean flag')
  if type(mapped_draft_output) is not bool: raise ValueError('mapped_draft_output must be boolean')
  if type(fused_root) is not bool: raise ValueError('fused_root must be boolean')
  old_init = BatchedVerifier.__dict__['__init__']
  old_draft_chain = EagleDraft.__dict__['_observe_and_chain']
  old_score = chain_sampling.score_argmax
  old_chain = chain_sampling.sample_chain
  old_empty, old_copy, old_sample = Tensor.empty, Buffer.copy_from, BatchedVerifier._sample
  old_commit = BatchedVerifier._commit_forward
  empty_descriptor = Tensor.__dict__.get('empty')
  descriptors = Buffer.__dict__['copy_from'], BatchedVerifier.__dict__['_sample'], BatchedVerifier.__dict__['_commit_forward']
  token_callers = {BatchedVerifier.verify_chain.__code__, EagleDraft._observation_inputs.__code__}
  outputs, token_storage = {}, set()
  def empty(*shape, **kwargs):
    if sys._getframe(1).f_code in token_callers and kwargs.get('device') == 'NV':
      result = host_tensor(shape, kwargs['dtype'])
      token_storage.add(result.uop.buffer)
      return result
    return old_empty(*shape, **kwargs)
  def copy(dest, src):
    if dest in token_storage and src.device == 'PYTHON':
      assert dest.nbytes == src.nbytes
      Device['NV'].synchronize()
      dest.host[:] = src.host[:]
      return dest
    return old_copy(dest, src)
  def sample(logits, temperature, index):
    if materialize_scores:
      row = logits[:, index:index+1, :].reshape(1, logits.shape[-1]).contiguous()
      decision = chain_sampling.score_argmax(row, temperature, Tensor.rand_like(row), materialize_scores=True)
    else: decision = old_sample(logits, temperature, index)
    key = logits.shape[1]
    if key not in outputs: outputs[key] = host_tensor(decision.shape, decision.dtype)
    result = Tensor(outputs[key].uop.buf_uop).reshape(decision.shape)
    return result.assign(decision).realize()
  def commit(self, tokens, position, temperature, key, counter):
    packet, features, logits = old_commit(self, tokens, position, temperature, key, counter)
    identity = ('packet', tokens.shape[1])
    if identity not in outputs: outputs[identity] = host_tensor(packet.shape, packet.dtype)
    result = Tensor(outputs[identity].uop.buf_uop).reshape(packet.shape)
    return result.assign(packet).realize(), features, logits
  def draft_chain(self, features, ids, position, *, count, scored=False):
    hidden, tokens = old_draft_chain(self, features, ids, position, count=count, scored=scored)
    identity = ('draft', self, ids.shape[1], count, scored)
    if identity not in outputs: outputs[identity] = host_tensor(tokens.shape, tokens.dtype)
    result = Tensor(outputs[identity].uop.buf_uop).reshape(tokens.shape)
    return hidden, result.assign(tokens).realize()
  Tensor.empty, Buffer.copy_from, BatchedVerifier._sample = staticmethod(empty), copy, staticmethod(sample)
  if mapped_draft_output: EagleDraft._observe_and_chain = draft_chain
  if fused_root:
    def verifier_init(self, *args, **kwargs):
      if kwargs.get('compact_verify'): raise ValueError('fused root experiment excludes compact verification')
      kwargs['fuse_root_sampling'] = True
      old_init(self, *args, **kwargs)
    BatchedVerifier.__init__ = verifier_init
  BatchedVerifier._commit_forward = commit
  if materialize_scores:
    chain_sampling.sample_chain = functools.partial(old_chain, materialize_scores=True)
  if parallel_scores:
    from benchmarks.experiments.eagle3.parallel_argmax import parallel_score_argmax
    chain_sampling.score_argmax = parallel_score_argmax
  try: yield
  finally:
    if empty_descriptor is None: delattr(Tensor, 'empty')
    else: Tensor.empty = empty_descriptor
    Buffer.copy_from, BatchedVerifier._sample, BatchedVerifier._commit_forward = descriptors
    chain_sampling.sample_chain = old_chain
    chain_sampling.score_argmax = old_score
    EagleDraft._observe_and_chain = old_draft_chain
    BatchedVerifier.__init__ = old_init


def main():
  if '--head' in sys.argv:
    from benchmarks.experiments.eagle3.calibrated_benchmark import main as benchmark
  else:
    from benchmarks.eagle import main as benchmark
  if Device.DEFAULT != 'NV': raise ValueError('mapped I/O experiment requires DEV=NV')
  materialize_scores = '--materialize-scores' in sys.argv
  if materialize_scores: sys.argv.remove('--materialize-scores')
  parallel_scores = '--parallel-argmax' in sys.argv
  if parallel_scores: sys.argv.remove('--parallel-argmax')
  mapped_draft_output = '--mapped-draft-output' in sys.argv
  if mapped_draft_output: sys.argv.remove('--mapped-draft-output')
  fused_root = '--fused-root' in sys.argv
  if fused_root: sys.argv.remove('--fused-root')
  try:
    with mapped_io(materialize_scores=materialize_scores, parallel_scores=parallel_scores,
                   mapped_draft_output=mapped_draft_output, fused_root=fused_root): benchmark()
  finally:
    if '--json' in sys.argv:
      path = Path(sys.argv[sys.argv.index('--json')+1])
      if path.exists():
        report = json.loads(path.read_text())
        report['experimental_mapped_io'] = True
        report['experimental_materialized_scores'] = materialize_scores
        report['experimental_parallel_argmax'] = parallel_scores
        report['experimental_mapped_draft_output'] = mapped_draft_output
        report['experimental_fused_root'] = fused_root
        report['mapped_io_scope'] = ['draft token inputs', 'verifier token inputs', 'verifier decision/compact-packet outputs']
        if mapped_draft_output: report['mapped_io_scope'].append('fused draft proposal outputs')
        path.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__': main()
