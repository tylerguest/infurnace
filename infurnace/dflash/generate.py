"""Greedy DFlash generation with sequential or transactional block verification.

Both schedules use unchanged upstream target blocks. Block verification is an
explicit layer-major captured schedule, not a native multirow GEMM replacement.
"""
from dataclasses import dataclass
import os
import time
from tinygrad import Context
from infurnace.dflash.target import TargetObserver


@dataclass
class DFlashStats:
  rounds: int = 0
  proposed: int = 0
  accepted: int = 0
  target_steps: int = 0
  speculative_rows: int = 0


def _raw_features(verifier):
  from infurnace.dflash.parallel import _raw
  return _raw(verifier._features)


class DFlashGenerator:
  def __init__(self, model, draft, *, verifier: str = 'sequential'):
    if model.device.split(':')[0] not in ('CPU', 'NV'):
      raise ValueError('DFlash reference currently requires CPU/NV one-row upstream prefill')
    if draft.target is not model or draft.device != model.device: raise ValueError('DFlash draft belongs to a different target/device')
    if verifier not in ('sequential', 'block', 'parallel'): raise ValueError('invalid DFlash verifier')
    self.model, self.draft, self.verifier = model, draft, verifier
    self.observer = TargetObserver(model, draft.config.feature_layers)
    from infurnace.dflash.verify import BlockVerifier
    self.block_verifier = BlockVerifier(self.observer) if verifier == 'block' else None
    self._parallel = {}
    # Break-even tokens/round for parallel verification vs one-row decode on
    # the 5060 (one round ~= 3.2 ordinary steps), and the fallback length.
    self.min_tau, self.cooldown_steps = 3.5, 8
    self.plain_cost = None  # measured seconds per one-row target step (kept across requests)
    # Device-side fused rounds are opt-in (INFURNACE_DFLASH_FUSED=1): faster,
    # but they can stall NV graph replay under sustained load (see parallel.py).
    self.fused = os.environ.get('INFURNACE_DFLASH_FUSED', '0') == '1'
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
      pending, emitted = [], 0
      def step(token):
        decision, features = observer.step(token)
        self.stats.target_steps += 1
        pending.append(features)
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
            with Context(DEV=self.model.device): pending.append(features[:, :len(chunk)].contiguous().realize())
            anchor, done = decisions[len(chunk) - 1], done + len(chunk)
          for token in tokens[done:]: anchor = step(token)
        else:
          for token in tokens: anchor = step(token)
        # Adaptive drafting: draft only while a parallel round costs less wall
        # time per emitted token than an ordinary one-row step. Otherwise decode
        # ordinary rows (still collecting features), backing off exponentially
        # while probes keep losing. Until a one-row step has been timed, a
        # tokens-per-round threshold decides the first fallback.
        tau, cooldown, backoff, spec_cost, probing = float(block), 0, 1, None, False
        # A parallel verify is held uncommitted: the next fused round commits it
        # on device; any other path flushes it (commit + keep its feature rows).
        held = held_decisions = None
        def flush():
          nonlocal held
          if held is None: return
          verifier = self._parallel_verifier(block)
          verifier.commit(held, held_decisions)
          with Context(DEV=self.model.device): pending.append(_raw_features(verifier)[:, :held].contiguous().realize())
          held = None
        def account(accepted):
          nonlocal spec_cost, tau, probing, cooldown, backoff
          cost = (time.perf_counter() - started) / (accepted + 1)
          spec_cost = cost if spec_cost is None or probing else 0.5 * spec_cost + 0.5 * cost
          tau, probing = 0.5 * tau + 0.5 * (accepted + 1), False
          if spec_cost > self.plain_cost if self.plain_cost is not None else tau < self.min_tau:
            tau, cooldown, backoff = float(block), self.cooldown_steps * backoff, min(backoff * 2, 16)
          else: backoff = 1
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
              account(accepted)
              for token in proposals[:accepted]:
                emitted += 1
                yield token
                if token in eos_token_ids or emitted == limit: return
              anchor = decisions[accepted]
              continue
          flush()
          if count:
            with Context(DEV=self.model.device):
              features = pending[0].cat(*pending[1:], dim=1).contiguous().realize()
            proposals = self.draft.propose(features, anchor, observer.position, count)
            if len(proposals) != count or any(type(t) is not int or not 0 <= t < self.model.config.vocab_size for t in proposals):
              raise ValueError('invalid DFlash proposals')
            pending.clear()
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
              account(accepted)
              for token in proposals[:accepted]:
                emitted += 1
                yield token
                if token in eos_token_ids or emitted == limit: return
              anchor = decisions[accepted]
              continue
          if self.block_verifier is not None and proposals:
            self.block_verifier.prepare(block)
            with self.block_verifier.verify([anchor, *proposals]) as packet:
              self.stats.speculative_rows += len(packet.inputs)
              def commit(row):
                packet.commit(row + 1)
                self.stats.target_steps += 1
                self.stats.speculative_rows -= 1
                pending.append(packet.feature(row))
                return packet.decisions[row]
              for row, proposal in enumerate(proposals):
                selected = commit(row)
                if selected != proposal:
                  anchor = selected
                  break
                self.stats.accepted += 1
                emitted += 1
                yield selected
                if selected in eos_token_ids or emitted == limit: return
              else: anchor = commit(len(proposals))
            continue
          current = anchor
          if not proposals and self.verifier == 'parallel':
            anchor = step(current)
            cost = time.perf_counter() - started
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

  def _parallel_verifier(self, block):
    from infurnace.dflash.parallel import ParallelVerifier
    if block not in self._parallel: self._parallel[block] = ParallelVerifier(self.observer, block)
    return self._parallel[block]
