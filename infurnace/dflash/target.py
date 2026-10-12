"""One-row tinygrad target observation and exclusive request ownership."""
from tinygrad import Context, Tensor, TinyJit, UOp, dtypes
from tinygrad.uop.ops import Ops
from infurnace.qwen35 import Qwen35Model


class TargetObserver:
  def __init__(self, model: Qwen35Model, feature_layers: tuple[int, ...]):
    if not isinstance(model, Qwen35Model): raise ValueError('DFlash requires Qwen3.5 target support')
    if (not feature_layers or tuple(sorted(set(feature_layers))) != feature_layers
        or any(type(i) is not int or not 0 < i < model.config.block_count for i in feature_layers)):
      raise ValueError('invalid DFlash target feature taps')
    self.model, self.layers = model, feature_layers
    self.position = 0
    self.active = False
    self._block_active = False
    self.next_decision = None
    self._position = UOp.variable('start_pos', 0, model.config.context_length - 1)
    self._jit = TinyJit(self._forward)
    self._ids = self._temperature = None

  def __enter__(self):
    if self.active or self.model._generating: raise RuntimeError('only one generation may run at a time')
    self.model.transformer._cached_tokens = []
    self.model._generating = self.active = True
    self.position = 0
    self.next_decision = None
    return self

  def __exit__(self, *exc):
    self.close()

  def close(self):
    if self.active:
      # No partially executed request authorizes normal-decode prefix reuse.
      self.model.transformer._cached_tokens = []
      self.model._generating = self.active = False

  def _forward(self, tokens, position, temperature):
    target = self.model.transformer
    x, features = target.token_embd(tokens).float(), []
    for index, block in enumerate(target.blk):
      # GGUF tap N means HF hidden_states[N], i.e. after target block N-1.
      if index in self.layers: features.append(x)
      x = block(x, position)
    logits = target.output(target.output_norm(x[:, -1:]))[:, -1, :]
    # Identical to ordinary tinygrad target sampling, including greedy RNG
    # consumption and its tie behavior. Draft argmax never consumes this RNG.
    decision = (logits / temperature.maximum(1e-12)
      - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)
    return decision, features[0].cat(*features[1:], dim=-1).contiguous()

  def step(self, token: int):
    if not self.active: raise RuntimeError('use TargetObserver inside its context manager')
    if self._block_active: raise RuntimeError('target has an active verification transaction')
    if type(token) is not int or not 0 <= token < self.model.config.vocab_size: raise ValueError('invalid target token')
    if self.position >= self.model.config.context_length: raise ValueError('target context exhausted')
    try:
      with Context(DEV=self.model.device):
        # Persistent input buffers: the captured graph always sees the same addresses.
        if self._ids is None:
          self._ids = Tensor.empty(1, 1, dtype=dtypes.int32, device=self.model.device).realize()
          self._temperature = Tensor([0.0], device=self.model.device).realize()
        token_value = Tensor([[token]], dtype=dtypes.int32, device=self.model.device)
        Tensor(self._ids.uop.after(self._ids.uop.store(token_value.uop))).realize()
        decision, features = self._jit(self._ids, self._position.bind(self.position), self._temperature)
        Tensor.realize(decision, features)
        # JIT outputs alias capture scratch. Copy only executed storage before
        # another target row overwrites it; detach executed RNG/target graphs.
        features = Tensor(features.uop.buf_uop).reshape(features.shape).clone().realize()
        decision = Tensor(decision.uop.buf_uop).reshape(decision.shape)
        selected = int(decision.item())
        # Detach the captured RNG update graph in place (buffer identity kept);
        # live graphs are walked on every realize.
        counter = Tensor._device_rng_counters[self.model.device]
        if counter.uop.op is not Ops.BUFFER: counter.replace(Tensor(counter.uop.buf_uop).reshape(counter.shape))
      self.position += 1
      self.next_decision = selected
      return selected, features
    except BaseException:
      self.close()
      raise
