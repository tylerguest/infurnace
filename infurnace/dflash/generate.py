"""Greedy DFlash generation with sequential or transactional block verification.

Both schedules use unchanged upstream target blocks. Block verification is an
explicit layer-major captured schedule, not a native multirow GEMM replacement.
"""
from dataclasses import dataclass
from tinygrad import Context
from infurnace.dflash.target import TargetObserver


@dataclass
class DFlashStats:
  rounds: int = 0
  proposed: int = 0
  accepted: int = 0
  target_steps: int = 0
  speculative_rows: int = 0


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
        # Adaptive drafting: a parallel round only pays off when enough tokens
        # are accepted per round. Below that, decode ordinary rows for a while
        # (still collecting features) and then probe the draft again.
        tau, cooldown = float(block), 0
        while emitted < limit:
          emitted += 1
          yield anchor
          if anchor in eos_token_ids or emitted == limit: return
          count = min(block - 1, limit - emitted, self.model.config.context_length - observer.position - 1)
          if cooldown:
            cooldown, count = cooldown - 1, 0
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
              decisions, features = verifier.verify([anchor, *proposals])
              accepted = 0
              while accepted < len(proposals) and proposals[accepted] == decisions[accepted]: accepted += 1
              verifier.commit(accepted + 1, decisions)
              self.stats.target_steps += 1
              self.stats.accepted += accepted
              tau = 0.5 * tau + 0.5 * (accepted + 1)
              if tau < self.min_tau: tau, cooldown = self.min_tau, self.cooldown_steps
              with Context(DEV=self.model.device):
                pending.append(features[:, :accepted + 1].contiguous().realize())
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
