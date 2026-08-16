from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
from infurnace.engine.request import SamplingParams

@dataclass(frozen=True)
class PrefillPlan:
    """Immutable prefill execution contract.
    Identifies the request, its full prompt, the assigned cache slot, and the
    chunk bounds that split the prompt into bounded prefill steps. Chunk bounds
    cover the entire prompt without overlap or gaps.
    """
    request_id: str
    input_ids: Tuple[int, ...]                 # full prompt token ids
    cache_slot: int
    chunk_bounds: Tuple[Tuple[int, int], ...]  # [(start, end), ...] covering prompt
    prefix_token_ids: Tuple[int, ...] = ()     # Phase 8 prefix cache prep
    prefix_cache_hit: bool = False

@dataclass(frozen=True)
class DecodePlan:
    """Immutable decode execution contract for one or more active requests.
    Carries per-request token, position, slot, and sampling metadata so Phase 4
    batched decode and sampling need no plan-structure change.
    """
    request_ids: Tuple[str, ...]
    input_ids: Tuple[int, ...]                 # last token per request (flat)
    positions: Tuple[int, ...]                 # decode position per request
    slots: Tuple[int, ...]                     # cache slot per request
    active_mask: Tuple[bool, ...]              # True for live rows (Phase 4 padding)
    block_tables: Tuple[Tuple[int, ...], ...] = ()  # Phase 5 paged KV prep
    sampling_params: Tuple[SamplingParams, ...] = ()  # Phase 4 batched sampling

ExecutionPlan = PrefillPlan | DecodePlan
