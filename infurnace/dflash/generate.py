"""Greedy DFlash generation with sequential or parallel (multirow) verification.

Sequential verification steps the unchanged upstream target one row at a time;
parallel verification runs the whole draft block through the target at once
(see parallel.py).
"""
import collections
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


class DFlashGenerator:
  def __init__(self, model, draft, *, verifier: str = 'sequential'):
    if model.device.split(':')[0] not in ('CPU', 'NV'):
      raise ValueError('DFlash reference currently requires CPU/NV one-row upstream prefill')
    if draft.target is not model or draft.device != model.device: raise ValueError('DFlash draft belongs to a different target/device')
    if verifier not in ('sequential', 'parallel'): raise ValueError('invalid DFlash verifier')
    self.model, self.draft, self.verifier = model, draft, verifier
    self.observer = TargetObserver(model, draft.config.feature_layers)
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
             eos_token_ids: tuple[int, ...] = (), temperature: float = 0.0):
    if temperature != 0.0 or isinstance(temperature, bool): raise ValueError('DFlash reference supports greedy temperature=0 only')
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
      emitted = 0
      # Every committed target row's features go to the draft as they are
      # produced (draft.observe copies them into fixed-shape staging).
      def step(token):
        decision, _ = observer.step(token, snapshot=False, consume=self.draft.observe)
        self.stats.target_steps += 1
        return decision
      try:
        if self.verifier == 'parallel' and block > 1:
          # Multirow prompt prefill in fixed-width chunks; the tail falls back
          # to one-row steps only when a chunk would overrun the context.
          verifier, done = self._parallel_verifier(block), 0
          while done < len(tokens) and verifier.fits(observer.position):
            chunk = tokens[done:done + verifier.rows]
            decisions, features = verifier.prefill(chunk)
            verifier.commit(len(chunk), decisions)
            self.stats.target_steps += 1
            self.draft.observe(features[:, :len(chunk)])
            anchor, done = decisions[len(chunk) - 1], done + len(chunk)
          for token in tokens[done:]: anchor = step(token)
        else:
          for token in tokens: anchor = step(token)
        # Adaptive drafting: draft only while parallel rounds cost less wall time
        # per emitted token than an ordinary one-row step. Otherwise decode
        # ordinary rows (still collecting features), backing off exponentially
        # while probes keep losing. Until a one-row step has been timed, a
        # tokens-per-round threshold decides the first fallback.
        # The cost is a ratio of running sums (round time / emitted tokens), so a
        # single round with nothing accepted cannot outweigh good rounds.
        tau, cooldown, backoff, probing = float(block), 0, 1, False
        spec_time = spec_tokens = None
        timed_rounds = 0  # timed rounds since the last probe
        # A parallel verify is held uncommitted: the next fused round commits it
        # on device; any other path flushes it (commit + keep its feature rows).
        held = held_decisions = None
        def flush():
          nonlocal held
          if held is None: return
          verifier = self._parallel_verifier(block)
          verifier.commit(held, held_decisions)
          self.draft.observe(_raw_features(verifier)[:, :held])
          held = None
        def account(accepted):
          nonlocal spec_time, spec_tokens, timed_rounds, tau, probing, cooldown, backoff
          elapsed = time.perf_counter() - started
          if spec_time is None or probing: spec_time, spec_tokens, timed_rounds = elapsed, accepted + 1, 0
          else:
            m = self.spec_memory
            spec_time, spec_tokens = m * spec_time + (1 - m) * elapsed, m * spec_tokens + (1 - m) * (accepted + 1)
          tau, probing, timed_rounds = 0.5 * tau + 0.5 * (accepted + 1), False, timed_rounds + 1
          if self.plain_cost is not None: losing = timed_rounds >= self.min_timed_rounds and spec_time / spec_tokens > self.plain_cost
          else: losing = tau < self.min_tau
          if losing: tau, cooldown, backoff = float(block), self.cooldown_steps * backoff, min(backoff * 2, self.max_backoff)
          elif timed_rounds >= self.min_timed_rounds: backoff = 1
        def account_untimed(accepted):
          # A round whose wall time is not representative (a capture, or the
          # unfused probe that re-enters fused rounds) only updates the
          # tokens-per-round estimate; `probing` stays set for the next timed round.
          nonlocal tau, cooldown, backoff
          tau = 0.5 * tau + 0.5 * (accepted + 1)
          if tau < self.min_tau: tau, cooldown, backoff = float(block), self.cooldown_steps * backoff, min(backoff * 2, self.max_backoff)
        while emitted < limit:
          emitted += 1
          yield anchor
          if anchor in eos_token_ids or emitted == limit: return
          count = min(block - 1, limit - emitted, self.model.config.context_length - observer.position - 1)
          if cooldown:
            cooldown, count = cooldown - 1, 0
            probing = not cooldown
          started = time.perf_counter()
          if self.verifier == 'parallel' and count and held is not None:
            verifier = self._parallel_verifier(block)
            if self.fused and verifier.fused_eligible(self.draft) and verifier.fits(observer.position + held):
              decisions, proposals = verifier.fused_round(self.draft, held, anchor)
              proposals = proposals[:count]
              self.stats.rounds += 1
              self.stats.proposed += len(proposals)
              self.stats.target_steps += 1
              accepted = 0
              while accepted < len(proposals) and proposals[accepted] == decisions[accepted]: accepted += 1
              self.stats.accepted += accepted
              held, held_decisions = accepted + 1, decisions
              self._samples['fused'] += 1
              (account if self._samples['fused'] > 2 else account_untimed)(accepted)
              for token in proposals[:accepted]:
                emitted += 1
                yield token
                if token in eos_token_ids or emitted == limit: return
              anchor = decisions[accepted]
              continue
          flush()
          if count:
            proposals = self.draft.propose(anchor, observer.position, count)
            if len(proposals) != count or any(type(t) is not int or not 0 <= t < self.model.config.vocab_size for t in proposals):
              raise ValueError('invalid DFlash proposals')
          else: proposals = []
          self.stats.rounds += 1
          self.stats.proposed += len(proposals)
          if self.verifier == 'parallel' and proposals:
            verifier = self._parallel_verifier(block)
            if verifier.fits(observer.position):
              decisions, _ = verifier.verify([anchor, *proposals])
              accepted = 0
              while accepted < len(proposals) and proposals[accepted] == decisions[accepted]: accepted += 1
              self.stats.target_steps += 1
              self.stats.accepted += accepted
              held, held_decisions = accepted + 1, decisions
              fused_next = self.fused and verifier.fused_eligible(self.draft)
              (account_untimed if fused_next else account)(accepted)
              for token in proposals[:accepted]:
                emitted += 1
                yield token
                if token in eos_token_ids or emitted == limit: return
              anchor = decisions[accepted]
              continue
          current = anchor
          if not proposals and self.verifier == 'parallel':
            # Time the step alone: flush() above is a one-off per fallback and
            # compiles per held-row count. Skip capture samples and outliers.
            stepped = time.perf_counter()
            anchor = step(current)
            cost = time.perf_counter() - stepped
            self._samples['plain'] += 1
            if self._samples['plain'] > 2 and (self.plain_cost is None or cost < 4 * self.plain_cost):
              self.plain_cost = cost if self.plain_cost is None else 0.8 * self.plain_cost + 0.2 * cost
            continue
          for proposal in proposals:
            selected = step(current)
            if selected != proposal:
              anchor = selected
              break
            self.stats.accepted += 1
            emitted += 1
            yield selected
            if selected in eos_token_ids or emitted == limit: return
            current = selected
          else:
            # Produce the bonus anchor after a fully accepted block (or the
            # ordinary next anchor for block=1). Never process the final output.
            anchor = step(current)
      finally:
        self.draft.reset()

  def _graphs(self, block: int) -> dict:
    """Every captured graph a parallel request can use, created up front with the
    buffers they touch, by a stable name."""
    self.observer._realize_state()
    verifier = self._parallel_verifier(block)
    verifier._allocate()
    graphs = {'observer': self.observer._jit, 'verify': verifier._forward_jit, 'commit': verifier._commit_jit,
              'absorb': self.draft._absorb_jit}
    if self.fused and verifier.fused_eligible(self.draft):
      pick, ids, pack = verifier._round_graphs(self.draft)
      graphs |= {'pick': pick, 'ids': ids, 'pack': pack, 'fused_absorb': verifier._fused_absorb, 'fused_draft': verifier._fused_draft}
    for count in range(1, block): graphs[f'denoise{count}'] = self.draft._denoise_graph(count)
    return graphs

  def _graph_cache_path(self, block: int):
    # Opt-in: a cached entry has been seen to hang its first replay in later processes (cause not yet found).
    if os.environ.get('INFURNACE_GRAPH_CACHE', '0') != '1' or self.verifier != 'parallel': return None
    if self.model.device.split(':')[0] != 'NV' or getattr(self.model, '_path', None) is None: return None
    from infurnace.dflash import graph_cache
    return graph_cache.cache_dir() / f'{graph_cache.key(self, block)}.pkl'

  def warmup(self, block_size: int | None = None) -> str:
    """Capture the graphs a request uses before the first one: drafting rounds
    (fused when enabled) and the plain-step fallback. Leaves no request state.
    Graphs come from the on-disk cache when it matches (returns 'cached'); otherwise
    they are captured (returns 'captured') and saved for the next process."""
    from infurnace.dflash import graph_cache
    block = self.draft.config.block_size if block_size is None else block_size
    path = self._graph_cache_path(block)
    if path is not None and path.is_file():
      graphs = self._graphs(block)
      loaded = graph_cache.load(path, graph_cache.named_buffers(self))
      if loaded is not None and set(loaded) <= set(graphs):
        for name, captured in loaded.items():
          graphs[name].captured, graphs[name].cnt = captured, 2
        # Replays the cached graphs once each and schedules the eager glue around them.
        self._capture(block_size, requests=1)
        return 'cached'
    self._capture(block_size)
    if path is not None: graph_cache.save(path, self._graphs(block), graph_cache.named_buffers(self))
    return 'captured'

  def _capture(self, block_size: int | None, requests: int = 2):
    saved = self.min_tau, self.plain_cost
    prompt = [0] * min(16, self.model.config.context_length // 4)
    try:
      # Always draft, then fall back after every round (the tokens-per-round
      # rule decides while no plain step is timed) so plain steps are captured too.
      for min_tau, plain_cost in ((0.0, float('inf')), (float('inf'), None)):
        self.min_tau, self.plain_cost = min_tau, plain_cost
        for _ in range(requests): list(self.stream(list(prompt), max_new_tokens=48, block_size=block_size))
    finally:
      self.min_tau, self.plain_cost = saved
      self.stats = DFlashStats()

  def _parallel_verifier(self, block):
    from infurnace.dflash.parallel import ParallelVerifier
    if block not in self._parallel: self._parallel[block] = ParallelVerifier(self.observer, block)
    return self._parallel[block]
