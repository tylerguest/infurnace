"""Opt-in EAGLE verification reference; not a speculative throughput engine.

The target weights, block math and zero-temperature sampling expression remain
upstream. The verifier owns the existing KV buffers exclusively while active.
Rejected proposals are never executed by this sequential correctness scaffold.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tinygrad import Context, Tensor, TinyJit, UOp, dtypes

if TYPE_CHECKING:
  from infurnace.models.qwen3 import Qwen3Model

@dataclass(frozen=True)
class TargetStep:
  next_token: int
  logits: Tensor  # [1, vocabulary]; independent snapshot, not a JIT output alias
  features: Tensor  # [1, consumed tokens, 3*dim], before the selected blocks

@dataclass(frozen=True)
class VerifiedChain:
  tokens: tuple[int, ...]  # target-selected root, followed by accepted proposals
  accepted_drafts: int
  steps: tuple[TargetStep, ...]  # matching target features for every committed token
  next_token: int | None  # target-selected next root; not yet written into KV
  stop_reason: str  # proposals, mismatch, limit, eos or context

class SequentialVerifier:
  """Context-managed target reference for greedy EAGLE chains.

  Prompt prefill mirrors upstream symbolic chunking. Decode uses the existing
  one-token block path (including NV Q8/fused attention when enabled), not FP16
  multi-token prefill as a substitute verifier. No drafter or acceleration yet.
  Prefix reuse is deliberately invalidated on entry/exit; normal generation may
  resume safely afterward, but cannot run concurrently. Instances are single-use.

  Zero-temperature decisions use upstream's Gumbel-max expression, not a new
  argmax sampler. Near ties therefore retain its RNG semantics; an independent
  run with a different RNG state is not promised to emit identical tokens.
  """
  def __init__(self, model: "Qwen3Model", *, feature_layers: tuple[int, int, int] | None = None,
               eos_token_ids: tuple[int, ...] = (), chunk_size: int = 32):
    count = len(model.transformer.blk)
    layers = (2, count//2, count-3) if feature_layers is None else feature_layers
    if (len(layers) != 3 or any(type(i) is not int or not 0 <= i < count for i in layers)
        or tuple(sorted(set(layers))) != tuple(layers)):
      raise ValueError('feature_layers must contain three distinct increasing valid block indices')
    if type(chunk_size) is not int or chunk_size <= 0: raise ValueError('chunk_size must be positive')
    self.model, self.feature_layers, self.chunk_size = model, tuple(layers), chunk_size
    self.eos_token_ids = frozenset(self._validate_tokens(eos_token_ids))
    self._active = self._used = False
    self._tokens: list[int] = []
    self._next_token: int | None = None
    self._prefilled = False
    self._position = UOp.variable('start_pos', 0, model.config.context_length-1)
    self._chunk = UOp.variable('toks', 1, chunk_size)
    self._prefill_jit, self._decode_jit = TinyJit(self._forward), TinyJit(self._forward)

  @property
  def committed_tokens(self) -> tuple[int, ...]: return tuple(self._tokens)

  @property
  def committed_length(self) -> int: return len(self._tokens)

  @property
  def pending_token(self) -> int | None:
    """Already target-selected next token, not yet processed into KV."""
    return self._next_token

  def _validate_tokens(self, tokens) -> tuple[int, ...]:
    values = tuple(tokens)
    if any(type(t) is not int or not 0 <= t < self.model.config.vocab_size for t in values):
      raise ValueError('invalid token ID')
    return values

  def __enter__(self):
    if self._used: raise RuntimeError('verifier instances are single-use')
    if self.model._generating: raise RuntimeError('only one generation or verifier may run at a time')
    self.model.transformer._cached_tokens = []
    self.model._generating = self._active = self._used = True
    return self

  def __exit__(self, exc_type, exc_value, traceback): self.close()

  def close(self):
    if self._active:
      self.model.transformer._cached_tokens = []
      self.model._generating = self._active = False

  def _require_active(self):
    if not self._active: raise RuntimeError('use the verifier inside its context manager')

  def _forward(self, tokens: Tensor, position: UOp, temperature: Tensor):
    target = self.model.transformer
    x, features = target.token_embd(tokens).float(), []
    for index, block in enumerate(target.blk):
      if index in self.feature_layers: features.append(x)
      x = block(x, position)
    logits = target.output(target.output_norm(x[:, -1:]))[:, -1, :]
    # Keep the pinned upstream Transformer.forward sampling expression intact.
    decision = (logits/temperature.maximum(1e-12)
                - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)
    return decision, logits, features[0].cat(*features[1:], dim=-1).contiguous()

  def _execute(self, tokens: Tensor, position: int, *, prefill: bool, consumed: int = 1) -> TargetStep:
    try:
      with Context(DEV=self.model.device):
        decision, logits, features = (self._prefill_jit if prefill else self._decode_jit)(
          tokens.contiguous(), self._position.bind(position), Tensor([0.0], device=self.model.device))
        # Realize all outputs before copying. JIT output buffers are reused on
        # replay; callers may retain these snapshots across arbitrarily many steps.
        Tensor.realize(decision, logits, features)
        # Symbolic prefill results retain a max-sized flat backing buffer. Copy
        # only the known live rows into a concrete-shaped independent snapshot.
        live = consumed*3*self.model.config.embedding_length
        # Rewrap executed storage before snapshot copies. Keeping AFTER/CALL
        # dependencies here needlessly revisits the full target graph.
        features = Tensor(features.uop.buf_uop).flatten()[:live].reshape(
          1, consumed, 3*self.model.config.embedding_length).clone().realize()
        logits = Tensor(logits.uop.buf_uop).reshape(logits.shape).clone().realize()
        decision = Tensor(decision.uop.buf_uop).reshape(decision.shape)
        return TargetStep(int(decision.item()), logits, features)
    except BaseException:
      self.close()  # Partial KV writes must never authorize prefix reuse.
      raise

  def prefill(self, tokens: list[int]) -> tuple[TargetStep, ...]:
    self._require_active()
    if self._prefilled: raise RuntimeError('prompt has already been prefilled')
    values = self._validate_tokens(tokens)
    if not values or len(values) >= self.model.config.context_length:
      raise ValueError('prompt must be nonempty and leave at least one context slot')
    with Context(DEV=self.model.device):
      padded = Tensor(list(values)+[0]*(self.model.config.context_length-len(values)),
                      dtype=dtypes.int32, device=self.model.device).reshape(1, self.model.config.context_length)
    steps = []
    for position in range(0, len(values), self.chunk_size):
      size = min(self.chunk_size, len(values)-position)
      sp, nt = self._position.bind(position), self._chunk.bind(size)
      step = self._execute(padded[:, sp:sp+nt], position, prefill=True, consumed=size)
      steps.append(step)
      self._tokens.extend(values[position:position+size])
    self._next_token, self._prefilled = steps[-1].next_token, True
    return tuple(steps)

  def verify_chain(self, proposals: list[int], *, max_tokens: int | None = None) -> VerifiedChain:
    """Commit the known root and longest matching draft prefix.

    Proposals are descendants AFTER the known target-selected root. The next
    target decision is returned as a pending root, not silently emitted/processed.
    max_tokens bounds this round's committed output, not the whole response.
    """
    self._require_active()
    if not self._prefilled: raise RuntimeError('prefill a prompt first')
    if self._next_token is None: raise RuntimeError('verification session has finished')
    values = self._validate_tokens(proposals)
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens <= 0):
      raise ValueError('max_tokens must be positive')
    remaining = self.model.config.context_length-len(self._tokens)
    limit = min(remaining, len(values)+1, max_tokens if max_tokens is not None else len(values)+1)
    committed, steps, reason = [], [], 'proposals'
    for index in range(limit):
      token = self._next_token
      if index and values[index-1] != token:
        reason = 'mismatch'
        break
      with Context(DEV=self.model.device):
        tensor = Tensor([[token]], dtype=dtypes.int32, device=self.model.device)
      step = self._execute(tensor, len(self._tokens), prefill=False)
      self._tokens.append(token)
      committed.append(token)
      steps.append(step)
      self._next_token = step.next_token
      if token in self.eos_token_ids:
        self._next_token, reason = None, 'eos'
        break
      if len(self._tokens) == self.model.config.context_length:
        self._next_token, reason = None, 'context'
        break
    else:
      if max_tokens is not None and limit == max_tokens: reason = 'limit'
    return VerifiedChain(tuple(committed), max(0, len(committed)-1), tuple(steps), self._next_token, reason)
