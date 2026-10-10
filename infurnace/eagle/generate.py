"""Explicit opt-in, greedy EAGLE-3 chain generation.

The target owns every emitted decision. This API does not alter normal generate,
chat formatting, thinking tokens, or prefix reuse. Close the iterator on cancel.
"""
from dataclasses import dataclass
import math
from typing import Iterator

from tinygrad import Tensor

from infurnace.eagle import EagleDraft
from infurnace.eagle.reference import SequentialVerifier


@dataclass(frozen=True)
class EagleRound:
  tokens: tuple[int, ...]
  proposed: int
  accepted: int
  stop_reason: str


class EagleGenerator:
  """Explicit reusable generation session; no batching or concurrent ownership.

  Kernel captures survive requests, but tokens, KV visibility and draft state do
  not. Captures are invalidated if target packed weights or KV storage changes.
  """
  def __init__(self, model, draft: EagleDraft, *, fast: bool = False):
    if type(fast) is not bool: raise ValueError('fast must be boolean')
    self.model, self.draft = model, draft
    self._kernel_cache: dict = {}
    self._defaults = {}
    if fast:
      if model.device.split(':')[0] != 'NV' or draft.device != model.device:
        raise ValueError('fast EAGLE requires matching NV target and draft devices')
      names = ('attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate', 'ffn_up', 'ffn_down')
      layers = [model.transformer.output]+[getattr(block, name) for block in model.transformer.blk for name in names]
      # Only enable fused reductions for chunk counts with exact-order validation.
      fused = all((layer.in_features+1023)//1024 in (1, 2, 3, 4, 10) for layer in layers)
      draft.enable_fast_io()
      self._defaults = dict(draft_tokens=1, batched=True, loop_projections=True, fused_draft=True,
                            zero_copy_features=True, fused_projections=fused, fast_io=True, parallel_sampling=True)

  def generate(self, tokens: list[int], **kwargs) -> Iterator[EagleRound]:
    options = self._defaults | kwargs
    return generate_eagle(self.model, self.draft, tokens, kernel_cache=self._kernel_cache, **options)

  def stream(self, tokens: list[int], **kwargs) -> Iterator[int]:
    """Token-level streaming; closing mid-round releases verifier ownership."""
    rounds = self.generate(tokens, **kwargs)
    try:
      for result in rounds:
        yield from result.tokens
    finally:
      rounds.close()


def generate_eagle(model, draft: EagleDraft, tokens: list[int], *, draft_tokens: int = 2,
                   max_new_tokens: int | None = None, eos_token_ids: tuple[int, ...] = (),
                   batched: bool = False, kernel_cache: dict | None = None,
                   local_ffn_down: bool = False, loop_projections: bool = False,
                   fused_draft: bool = False, compact_verify: bool = False,
                   zero_copy_features: bool = False, draft_confidence: float | None = None,
                   fused_projections: bool = False, fast_io: bool = False,
                   parallel_sampling: bool = False) -> Iterator[EagleRound]:
  """Yield verified rounds, up to EOS, context capacity or an explicit budget.

  Sequential verification is the reference. ``batched=True`` selects the NV Q8
  chain adapter; neither path supports stochastic sampling. Draft observations
  always pair feature i with target token i+1, never with rejected proposals.
  """
  if type(local_ffn_down) is not bool or (local_ffn_down and not batched):
    raise ValueError('local_ffn_down requires batched verification and a boolean flag')
  if type(loop_projections) is not bool or (loop_projections and not batched):
    raise ValueError('loop_projections requires batched verification and a boolean flag')
  if type(fused_draft) is not bool or (fused_draft and not callable(getattr(draft, 'observe_propose', None))):
    raise ValueError('fused_draft requires a boolean flag and an observe_propose drafter')
  if (any(type(flag) is not bool for flag in (compact_verify, zero_copy_features, fused_projections, fast_io, parallel_sampling)) or
      ((compact_verify or zero_copy_features or fused_projections or fast_io or parallel_sampling) and not batched)):
    raise ValueError('verification optimizations require boolean flags and batched verification')
  if draft_confidence is not None and (type(draft_confidence) not in (int, float) or not math.isfinite(draft_confidence)
      or draft_confidence < 0 or not fused_draft or not callable(getattr(draft, 'observe_propose_scored', None))):
    raise ValueError('draft_confidence requires a nonnegative finite margin and fused scored drafting')
  if type(draft_tokens) is not int or not 0 <= draft_tokens <= 7:
    raise ValueError('draft_tokens must be between zero and seven')
  if max_new_tokens is not None and (type(max_new_tokens) is not int or max_new_tokens <= 0):
    raise ValueError('max_new_tokens must be positive')
  if draft.device != model.device or draft.capacity != model.config.context_length:
    raise ValueError('draft device and context capacity must match target')
  if batched:
    from infurnace.eagle.verify import BatchedVerifier
    verifier_type = BatchedVerifier
  else:
    verifier_type = SequentialVerifier
  # Ownership is acquired before touching draft state, so a rejected concurrent
  # request cannot reset a running session's drafter.
  with verifier_type(model, eos_token_ids=eos_token_ids,
                     **({'kernel_cache': kernel_cache, 'local_ffn_down': local_ffn_down,
                         'loop_projections': loop_projections, 'compact_verify': compact_verify,
                         'zero_copy_features': zero_copy_features, 'fused_projections': fused_projections,
                         'fast_io': fast_io, 'parallel_sampling': parallel_sampling} if batched else {})) as verifier:
    draft.reset()
    try:
      steps = verifier.prefill(tokens)
      pending_observation = None
      if (max_new_tokens != 1 and len(tokens)+1 < model.config.context_length and
          verifier.pending_token not in verifier.eos_token_ids):
        features = steps[0].features.cat(*(step.features for step in steps[1:]), dim=1)
        paired_ids = list(tokens[1:])+[steps[-1].next_token]
        if fused_draft and draft_tokens: pending_observation = (features, paired_ids)
        else: draft.observe(features, paired_ids)
      emitted = 0
      while True:
        remaining = model.config.context_length-verifier.committed_length
        if max_new_tokens is not None: remaining = min(remaining, max_new_tokens-emitted)
        if remaining == 1 or verifier.pending_token in verifier.eos_token_ids:
          # This root was already sampled by the target. No future token needs
          # its features/KV: avoid a redundant target pass and a size-one capture.
          token = verifier.pending_token
          assert token is not None
          reason = ('eos' if token in verifier.eos_token_ids else
                    'context' if verifier.committed_length+1 == model.config.context_length else 'limit')
          yield EagleRound((token,), 0, 0, reason)
          return
        count = min(draft_tokens, remaining-1)
        if pending_observation is not None:
          if draft_confidence is None:
            proposals = draft.observe_propose(*pending_observation, count)
          else:
            proposals, margins = draft.observe_propose_scored(*pending_observation, count)
            keep = next((i for i, margin in enumerate(margins) if margin < draft_confidence), len(proposals))
            proposals = proposals[:keep]
          pending_observation = None
        else:
          proposals = draft.propose(count, fast_loop=True)
        result = verifier.verify_chain(proposals, max_tokens=remaining,
                                       **({'transient_features': True} if batched else {}))
        emitted += len(result.tokens)
        done = result.next_token is None or (max_new_tokens is not None and emitted == max_new_tokens)
        yield EagleRound(result.tokens, len(proposals), result.accepted_drafts, result.stop_reason)
        if done: return
        if (verifier.committed_length+1 < model.config.context_length and result.next_token not in verifier.eos_token_ids and
            (max_new_tokens is None or emitted+1 < max_new_tokens)):
          features = (result.features if batched else
                      Tensor.cat(*(step.features for step in result.steps), dim=1))
          paired_ids = list(result.tokens[1:])+[result.next_token]
          if fused_draft and draft_tokens: pending_observation = (features, paired_ids)
          else: draft.observe(features, paired_ids)
    finally:
      draft.reset()
