"""Greedy DFlash generation with sequential or parallel (multirow) verification.

Sequential verification steps the unchanged upstream target one row at a time;
parallel verification runs the whole draft block through the target at once
(see parallel.py).
"""
import collections
import math
from dataclasses import dataclass
import os
import time
from infurnace.dflash.target import TargetObserver

@dataclass
class DFlashStats:
  rounds: int = 0
  proposed: int = 0
  accepted: int = 0
  target_steps: int = 0

def _raw_features(verifier):
  from infurnace.dflash.parallel import _raw
  return _raw(verifier._features)

def _accepted(proposals: list[int], decisions: list[int]) -> int:
  accepted = 0
  while accepted < len(proposals) and proposals[accepted] == decisions[accepted]: accepted += 1
  return accepted

class _DraftPolicy:
  """Adaptive drafting: draft only while a round costs less wall time per emitted token than an
  ordinary one-row step. Otherwise decode ordinary rows (still collecting features) for a cooldown,
  backing off exponentially while probes keep losing. Until a one-row step has been timed, a
  tokens-per-round threshold decides the first fallback. The cost is a ratio of running averages
  (round time / emitted tokens), so a single round with nothing accepted cannot outweigh good rounds.

  min_tau and plain_cost are read from the generator on every decision: warmup overrides them,
  and plain steps update plain_cost mid-request."""
  def __init__(self, generator, block: int):
    self.generator, self.block = generator, block
    self.tau, self.cooldown, self.backoff, self.probing = float(block), 0, 1, False
    self.spec_time = self.spec_tokens = None
    self.timed_rounds = 0  # timed rounds since the last probe
    self.started = 0.0

  def start(self, count: int) -> int:
    """Proposals for the next round (0 while cooling down); starts the round's timer."""
    if self.cooldown:
      self.cooldown, count = self.cooldown - 1, 0
      self.probing = not self.cooldown
    self.started = time.perf_counter()
    return count

  def record(self, accepted: int):
    g, elapsed = self.generator, time.perf_counter() - self.started
    if self.spec_time is None or self.probing: self.spec_time, self.spec_tokens, self.timed_rounds = elapsed, accepted + 1, 0
    else:
      m = g.spec_memory
      self.spec_time, self.spec_tokens = m * self.spec_time + (1 - m) * elapsed, m * self.spec_tokens + (1 - m) * (accepted + 1)
    self.tau, self.probing, self.timed_rounds = 0.5 * self.tau + 0.5 * (accepted + 1), False, self.timed_rounds + 1
    if g.plain_cost is not None: losing = self.timed_rounds >= g.min_timed_rounds and self.spec_time / self.spec_tokens > g.plain_cost
    else: losing = self.tau < g.min_tau
    if losing: self._back_off()
    elif self.timed_rounds >= g.min_timed_rounds: self.backoff = 1

  def record_untimed(self, accepted: int):
    """A round whose wall time is not representative (a capture, or the unfused probe that
    re-enters fused rounds) only updates the tokens-per-round estimate; `probing` stays set
    for the next timed round."""
    self.tau = 0.5 * self.tau + 0.5 * (accepted + 1)
    if self.tau < self.generator.min_tau: self._back_off()

  def _back_off(self):
    g = self.generator
    self.tau, self.cooldown, self.backoff = float(self.block), g.cooldown_steps * self.backoff, min(self.backoff * 2, g.max_backoff)

class DFlashGenerator:
  def __init__(self, model, draft, *, verifier: str = 'sequential', temperature: float = 0.0):
    if model.device.split(':')[0] not in ('CPU', 'NV'):
      raise ValueError('DFlash reference currently requires CPU/NV one-row upstream prefill')
    if draft.target is not model or draft.device != model.device: raise ValueError('DFlash draft belongs to a different target/device')
    if verifier not in ('sequential', 'parallel'): raise ValueError('invalid DFlash verifier')
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
      raise ValueError('temperature must be finite and nonnegative')
    self.model, self.draft, self.verifier = model, draft, verifier
    # Fixed per generator: sampling changes the captured verify graph (argmax vs Gumbel-max).
    self.temperature = float(temperature)
    self.observer = TargetObserver(model, draft.config.feature_layers)
    self.observer.temperature = self.temperature
    self._parallel = {}
    # Break-even tokens/round for parallel verification vs one-row decode on
    # the 5060 (one round ~= 3.2 ordinary steps), and the fallback length.
    self.min_tau, self.cooldown_steps, self.max_backoff = 3.5, 8, 16
    # Decide on a long horizon (~20 rounds) and only after 4 timed rounds since a probe: thinking text dips
    # below break-even for a few rounds while averaging well above it, and plain prose stays below it.
    self.spec_memory = 0.95  # weight of history in the round time/token averages
    self.min_timed_rounds = 4  # timed rounds after a probe before deciding
    self.plain_cost = None  # measured seconds per one-row target step (kept across requests)
    # Timing samples seen per kind ('plain', 'fused'): the first calls of a
    # captured graph run eagerly and then capture, so they are not timed.
    self._samples = collections.Counter()
    # Device-side fused rounds (NV, block 16) unless INFURNACE_DFLASH_FUSED=0.
    self.fused = os.environ.get('INFURNACE_DFLASH_FUSED', '1') != '0'
    self.stats = DFlashStats()

  def stream(self, tokens: list[int], *, max_new_tokens: int, block_size: int | None = None,
             eos_token_ids: tuple[int, ...] = (), temperature: float | None = None):
    if temperature is not None and temperature != self.temperature:
      raise ValueError(f'this DFlash generator samples at temperature {self.temperature}; create one for {temperature}')
    if not isinstance(tokens, list) or not tokens: raise ValueError('prompt must be a nonempty token list')
    if any(type(t) is not int or not 0 <= t < self.model.config.vocab_size for t in tokens): raise ValueError('invalid prompt token')
    if type(max_new_tokens) is not int or max_new_tokens < 0: raise ValueError('invalid output budget')
    if len(tokens) >= self.model.config.context_length: raise ValueError('prompt must leave at least one context slot')
    if any(type(t) is not int or not 0 <= t < self.model.config.vocab_size for t in eos_token_ids): raise ValueError('invalid EOS token')
    block = self.draft.config.block_size if block_size is None else block_size
    if type(block) is not int or not 1 <= block <= self.draft.config.block_size: raise ValueError('invalid DFlash block size')
    limit = min(max_new_tokens, self.model.config.context_length - len(tokens))
    if not limit: return
    with self.observer as observer:
      self.stats = DFlashStats()
      self.draft.reset()
      # A parallel verify is held uncommitted: the next fused round commits it
      # on device; any other path flushes it (commit + keep its feature rows).
      self._held = self._held_decisions = None
      try:
        anchor, policy, emitted = self._prefill(tokens, block), _DraftPolicy(self, block), 0
        while True:
          emitted += 1
          yield anchor
          if anchor in eos_token_ids or emitted == limit: return
          count = policy.start(min(block - 1, limit - emitted, self.model.config.context_length - observer.position - 1))
          # A round yields its accepted tokens and returns the next anchor. Stopping here leaves it
          # suspended, so a sequential round never steps the target past an EOS or the limit.
          drafted = self._round(block, anchor, count, policy)
          while True:
            try: token = next(drafted)
            except StopIteration as stop:
              anchor = stop.value
              break
            emitted += 1
            yield token
            if token in eos_token_ids or emitted == limit: return
      finally:
        self.draft.reset()

  def _step(self, token: int) -> int:
    # Every committed target row's features go to the draft as they are
    # produced (draft.observe copies them into fixed-shape staging).
    decision, _ = self.observer.step(token, snapshot=False, consume=self.draft.observe)
    self.stats.target_steps += 1
    return decision

  def _prefill(self, tokens: list[int], block: int) -> int:
    """Run the prompt through the target; returns its decision for the first output. Resumes from
    the model's prompt snapshot when the prompt still matches it (restoring the draft with it), and
    saves a new one before the prompt's last turn header (see Qwen35Model.save_snapshot)."""
    done = self.model.restore_snapshot(tokens, 0, draft=self.draft)
    self.observer.position = done
    point = self.model.snapshot_point(tokens, done)
    for stop in ([point] if point is not None else []) + [len(tokens)]:
      anchor = self._read(tokens, done, stop, block)
      if stop == point: self.model.save_snapshot(tokens[:point], draft=self.draft)
      done = stop
    return anchor

  def _read(self, tokens: list[int], first: int, last: int, block: int) -> int | None:
    """Prompt rows first..last-1; returns the decision after the last one."""
    anchor, done = None, first
    if self.verifier == 'parallel' and block > 1:
      # Multirow prompt prefill in fixed-width chunks; the tail falls back
      # to one-row steps only when a chunk would overrun the context.
      verifier = self._parallel_verifier(block)
      while done < last and verifier.fits(self.observer.position):
        chunk = tokens[done:min(done + verifier.rows, last)]
        decisions, features = verifier.prefill(chunk)
        verifier.commit(len(chunk), decisions)
        self.stats.target_steps += 1
        self.draft.observe(features[:, :len(chunk)])
        anchor, done = decisions[len(chunk) - 1], done + len(chunk)
    for token in tokens[done:last]: anchor = self._step(token)
    return anchor

  def _flush(self, block: int):
    if self._held is None: return
    verifier = self._parallel_verifier(block)
    verifier.commit(self._held, self._held_decisions)
    self.draft.observe(_raw_features(verifier)[:, :self._held])
    self._held = None

  def _round(self, block: int, anchor: int, count: int, policy: _DraftPolicy):
    """One round after `anchor`: fused, parallel or sequential drafting, or a plain step when `count` is 0."""
    parallel = self.verifier == 'parallel'
    if parallel and count and self._held is not None:
      verifier = self._parallel_verifier(block)
      if self.fused and verifier.fused_eligible(self.draft) and verifier.fits(self.observer.position + self._held):
        return (yield from self._fused_round(verifier, anchor, count, policy))
    self._flush(block)
    proposals = self.draft.propose(anchor, self.observer.position, count) if count else []
    if len(proposals) != count or any(type(t) is not int or not 0 <= t < self.model.config.vocab_size for t in proposals):
      raise ValueError('invalid DFlash proposals')
    self.stats.rounds += 1  # plain steps count as rounds too
    self.stats.proposed += len(proposals)
    if parallel and proposals and self._parallel_verifier(block).fits(self.observer.position):
      return (yield from self._parallel_round(self._parallel_verifier(block), anchor, proposals, policy))
    if parallel and not proposals: return self._plain_step(anchor)
    return (yield from self._sequential_round(anchor, proposals))

  def _fused_round(self, verifier, anchor: int, count: int, policy: _DraftPolicy):
    """Commit the held block, draft and verify the next one on device (one readback)."""
    decisions, proposals = verifier.fused_round(self.draft, self._held, anchor)
    proposals = proposals[:count]
    accepted = _accepted(proposals, decisions)
    self.stats.rounds += 1
    self.stats.proposed += len(proposals)
    self.stats.target_steps += 1
    self.stats.accepted += accepted
    self._held, self._held_decisions = accepted + 1, decisions
    self._samples['fused'] += 1
    (policy.record if self._samples['fused'] > 2 else policy.record_untimed)(accepted)
    yield from proposals[:accepted]
    return decisions[accepted]

  def _parallel_round(self, verifier, anchor: int, proposals: list[int], policy: _DraftPolicy):
    """Verify host-drafted proposals in one multirow target forward; the block is held uncommitted."""
    decisions, _ = verifier.verify([anchor, *proposals])
    accepted = _accepted(proposals, decisions)
    self.stats.target_steps += 1
    self.stats.accepted += accepted
    self._held, self._held_decisions = accepted + 1, decisions
    # The unfused round that re-enters fused rounds is a probe; its time is not representative.
    (policy.record_untimed if self.fused and verifier.fused_eligible(self.draft) else policy.record)(accepted)
    yield from proposals[:accepted]
    return decisions[accepted]

  def _plain_step(self, anchor: int) -> int:
    # Time the step alone: the flush before it is a one-off per fallback and
    # compiles per held-row count. Skip capture samples and outliers.
    stepped = time.perf_counter()
    anchor = self._step(anchor)
    cost = time.perf_counter() - stepped
    self._samples['plain'] += 1
    if self._samples['plain'] > 2 and (self.plain_cost is None or cost < 4 * self.plain_cost):
      self.plain_cost = cost if self.plain_cost is None else 0.8 * self.plain_cost + 0.2 * cost
    return anchor

  def _sequential_round(self, anchor: int, proposals: list[int]):
    """Step the target one row per proposal, stopping at the first mismatch."""
    current = anchor
    for proposal in proposals:
      selected = self._step(current)
      if selected != proposal: return selected
      self.stats.accepted += 1
      yield selected
      current = selected
    # The bonus anchor after a fully accepted block (or the ordinary next anchor
    # when nothing was proposed). Never reached past an EOS or the limit.
    return self._step(current)

  def warmup(self, block_size: int | None = None) -> None:
    """Capture the graphs a request uses before the first one: drafting rounds
    (fused when enabled) and the plain-step fallback. Leaves no request state."""
    self._capture(block_size)

  def _capture(self, block_size: int | None):
    saved = self.min_tau, self.plain_cost
    prompt = [0] * min(16, self.model.config.context_length // 4)
    try:
      # Always draft, then fall back after every round (the tokens-per-round
      # rule decides while no plain step is timed) so plain steps are captured too.
      for min_tau, plain_cost in ((0.0, float('inf')), (float('inf'), None)):
        self.min_tau, self.plain_cost = min_tau, plain_cost
        for _ in range(2): list(self.stream(list(prompt), max_new_tokens=48, block_size=block_size))
    finally:
      self.min_tau, self.plain_cost = saved
      self.stats = DFlashStats()
      self.model.reset()  # no warmup prompt becomes a reusable prefix or snapshot

  def _parallel_verifier(self, block):
    from infurnace.dflash.parallel import ParallelVerifier
    if block not in self._parallel: self._parallel[block] = ParallelVerifier(self.observer, block, sampled=self.temperature > 0)
    return self._parallel[block]
