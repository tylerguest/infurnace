"""Transactional, layer-major block verification using unchanged upstream blocks.

Each block still receives one row at a time: no GEMM/recurrence arithmetic change.
A single captured graph processes several candidate rows, checkpointing recurrent,
convolution and RNG state. Only an authorized prefix is committed. Full-attention
suffix K/V may remain in storage but is never authorized/read by causal decoding.
This is an explicit experimental schedule, not a native multirow target kernel.
"""
from functools import partial
from tinygrad import Context, Device, Tensor, TinyJit, UOp, dtypes
from tinygrad.llm.model import GatedDeltaNetBlock
from infurnace.dflash.target import TargetObserver


def _raw(tensor):
  return Tensor(tensor.uop.buf_uop).reshape(tensor.shape)


def _store(destination, source):
  # Do not replace native state Tensor objects or RNG buffer identities.
  Tensor(destination.uop.after(destination.uop.store(source.uop))).realize()


class BlockTransaction:
  def __init__(self, verifier, inputs, decisions, features, position):
    self.verifier, self.inputs, self.decisions, self.features = verifier, tuple(inputs), tuple(decisions), features
    self.position, self.committed, self.closed = position, 0, False
    self.initial_decision = verifier.observer.next_decision
    self.authorized = 1
    while self.authorized < len(inputs) and inputs[self.authorized] == decisions[self.authorized - 1]: self.authorized += 1

  def __enter__(self): return self
  def __exit__(self, *exc): self.close()

  def commit(self, count):
    if self.closed or not self.verifier.observer.active: raise RuntimeError('verification transaction is closed')
    if type(count) is not int or not self.committed <= count <= self.authorized: raise ValueError('invalid authorized prefix')
    with Context(DEV=self.verifier.observer.model.device):
      self.verifier._commit_jit(self.verifier._count.bind(count))
    self.committed = count
    self.verifier.observer.position = self.position + count
    self.verifier.observer.next_decision = self.decisions[count-1] if count else self.initial_decision

  def feature(self, row):
    if self.closed: raise RuntimeError('verification transaction is closed')
    if type(row) is not int or not 0 <= row < self.committed: raise ValueError('feature row has not been committed')
    with Context(DEV=self.verifier.observer.model.device):
      return _raw(self.features)[:, row:row+1].contiguous().clone().realize()

  def close(self):
    if not self.closed:
      self.closed = True
      self.verifier.observer._block_active = False
      self.verifier._active = None


class BlockVerifier:
  def __init__(self, observer: TargetObserver):
    if observer.model.device.split(':')[0] not in ('CPU', 'NV'): raise ValueError('block verification currently requires CPU/NV')
    self.observer = observer
    self.capacity, self._active = 0, None
    self._histories, self._block_jits = (), {}
    self._save_jit = self._commit_jit = self._rng_history = None

  def prepare(self, capacity):
    if not self.observer.active or not self.observer.position: raise RuntimeError('prefill the target before preparing verification')
    if self._active or self.observer._block_active: raise RuntimeError('target has an active verification transaction')
    if type(capacity) is not int or not 1 <= capacity <= 16: raise ValueError('block verifier supports at most 16 rows')
    if capacity == self.capacity: return
    device = self.observer.model.device
    with Context(DEV=device):
      # All sizes share one history allocation; retain no per-size state copies.
      # Synchronize before releasing old captured graphs on a capacity change.
      if self.capacity: Device[device].synchronize()
      self._block_jits.clear()
      self._save_jit = self._commit_jit = self._rng_history = None
      self._histories = ()
      states = []
      for block in self.observer.model.transformer.blk:
        if isinstance(block, GatedDeltaNetBlock):
          states.extend((block.conv_state, block.recurrent_state))
      self._histories = tuple((_raw(s), Tensor.empty(capacity+1, *s.shape, dtype=s.dtype, device=device).realize()) for s in states)
      self._counter = _raw(Tensor._device_rng_counters[device])
      self._counter_buffer = self._counter.uop.buf_uop
      self._rng_history = Tensor.empty(capacity+1, *self._counter.shape, dtype=self._counter.dtype, device=device).realize()
      self._temperature = Tensor([0.0], device=device).realize()
      dim = self.observer.model.config.embedding_length
      # Owned, already-allocated inputs keep the pinned planner from aliasing
      # a not-yet-executed embedding/concatenation buffer across opaque calls.
      self._work = tuple(Tensor.empty(1, capacity, dim, device=device).realize() for _ in range(2))
      self._features = Tensor.empty(1, capacity, dim * len(self.observer.layers), device=device).realize()
      self._count = UOp.variable('dflash_commit_count', 0, capacity)
      # Each row uses the SAME bounds as ordinary one-row inference. Narrower
      # positional bounds can change attention reduction padding/arithmetic.
      self._positions = tuple(UOp.variable(f'dflash_row_{i}', 0, self.observer.model.config.context_length-1) for i in range(capacity))
      self._save_jit, self._commit_jit = TinyJit(self._save), TinyJit(self._commit)
      self.capacity = capacity

  def _save(self):
    for state, history in self._histories: _store(history[0:1], state.unsqueeze(0))
    _store(self._rng_history[0:1], self._counter.unsqueeze(0))
    return tuple(h for _, h in self._histories), self._rng_history

  def _commit(self, count):
    for state, history in self._histories: _store(state, history[count:count+1].reshape(state.shape))
    _store(self._counter, self._rng_history[count:count+1].reshape(self._counter.shape))
    return tuple(s for s, _ in self._histories), self._counter

  def _forward(self, tokens, temperature, *positions, rows):
    target = self.observer.model.transformer
    x = self._work[0][:, :rows]
    _store(x, target.token_embd(tokens).float())
    state_index = feature_index = 0
    dim = self.observer.model.config.embedding_length
    for index, block in enumerate(target.blk):
      if index in self.observer.layers:
        _store(self._features[:, :rows, feature_index*dim:(feature_index+1)*dim], x)
        feature_index += 1
      next_x = self._work[(index+1) % 2][:, :rows]
      for row in range(rows):
        result = block(x[:, row:row+1].contiguous(), positions[row]).contiguous().realize()
        _store(next_x[:, row:row+1], result)
        if isinstance(block, GatedDeltaNetBlock):
          for state, history in self._histories[state_index:state_index+2]: _store(history[row+1:row+2], state.unsqueeze(0))
      if isinstance(block, GatedDeltaNetBlock): state_index += 2
      x = next_x
    decisions = []
    for row in range(rows):
      logits = target.output(target.output_norm(x[:, row:row+1]))[:, -1, :]
      decision = (logits / temperature.maximum(1e-12)
        - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True).realize()
      decisions.append(decision.clone().realize())
      _store(self._rng_history[row+1:row+2], self._counter.unsqueeze(0))
    decisions = decisions[0].cat(*decisions[1:], dim=1).contiguous()
    # Return the full owned buffer. A smaller contiguous prefix can still be a
    # view into it; detaching that view would discard its storage extent/offset.
    decisions.realize()
    return decisions, self._features

  def verify(self, inputs):
    observer = self.observer
    if not observer.active: raise RuntimeError('use verifier inside the target observer')
    if self._active or observer._block_active: raise RuntimeError('target has an active verification transaction')
    if not isinstance(inputs, list) or not inputs: raise ValueError('empty verification block')
    if not self.capacity or len(inputs) > self.capacity: raise ValueError('prepare sufficient block capacity first')
    if any(type(t) is not int or not 0 <= t < observer.model.config.vocab_size for t in inputs): raise ValueError('invalid target token')
    if inputs[0] != observer.next_decision: raise ValueError('verification must start with the known target anchor')
    if observer.position + len(inputs) > observer.model.config.context_length: raise ValueError('target context exhausted')
    device, saved = observer.model.device, False
    if Tensor._device_rng_counters[device].uop.buf_uop is not self._counter_buffer:
      raise RuntimeError('RNG buffer changed; construct a new block verifier')
    observer._block_active = True
    try:
      with Context(DEV=device):
        self._save_jit()
        saved = True
        rows = len(inputs)
        if rows not in self._block_jits: self._block_jits[rows] = TinyJit(partial(self._forward, rows=rows))
        ids = Tensor([inputs], dtype=dtypes.int32, device=device).contiguous()
        decisions, features = self._block_jits[rows](ids, self._temperature,
          *(self._positions[i].bind(observer.position+i) for i in range(rows)))
        values = [int(v) for v in decisions.tolist()[0]]
        # Sampling updated the original counter Tensor's IR during capture.
        # Detach that executed graph IN PLACE, preserving its Python object and
        # storage identity; later counter resets must not walk the target again.
        Tensor._device_rng_counters[device].replace(self._counter)
        # Restore zero before returning to Python/user code. Commit is incremental
        # across yields, so cancellation consumes only the baseline RNG prefix.
        self._commit_jit(self._count.bind(0))
        packet = BlockTransaction(self, inputs, values, _raw(features), observer.position)
        self._active = packet
        return packet
    except BaseException:
      if saved:
        try:
          Tensor._device_rng_counters[device].replace(self._counter)
          with Context(DEV=device): self._commit_jit(self._count.bind(0))
        except BaseException: pass  # preserve the original device/cancellation failure
      observer._block_active = False
      observer.close()
      raise
