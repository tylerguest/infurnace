from __future__ import annotations
from collections import deque
from typing import Optional
from infurnace.engine.request import Request, RequestState
from infurnace.scheduler.plan import PrefillPlan, DecodePlan, ExecutionPlan
from infurnace.scheduler.policy import FIFOPolicy, chunk_prompt


class SchedulerError(ValueError):
    """Scheduler inputs or state do not satisfy the admission or scheduling contract."""


class Scheduler:
    """Admission and execution-plan generator.

    Single-request serial prefill in this phase: ``schedule()`` returns at most
    one plan per step (a prefill chunk, a decode batch, or nothing). The return
    type is a list so Phase 4 batched decode fits without a signature change.

    The scheduler owns logical cache-slot assignment and prefill progress. The
    engine transitions ``PREFILLING -> DECODING -> FINISHED`` and frees slots on
    terminal outcomes.
    """

    def __init__(self, num_slots: int, max_chunk_tokens: int = 512):
        if num_slots < 1:
            raise SchedulerError("num_slots must be >= 1")
        if max_chunk_tokens < 1:
            raise SchedulerError("max_chunk_tokens must be >= 1")
        self._policy = FIFOPolicy(max_chunk_tokens=max_chunk_tokens)
        self._num_slots = num_slots
        self._waiting: deque[str] = deque()
        self._requests: dict[str, Request] = {}
        self._active: dict[str, Request] = {}      # prefilling or decoding
        self._prefill_chunk: dict[str, int] = {}   # request_id -> next chunk idx
        self._free_slots: set[int] = set(range(num_slots))

    # --- Admission ---
    def add_request(self, req: Request) -> None:
        if req.request_id in self._requests:
            raise SchedulerError(f"duplicate request_id: {req.request_id}")
        # Admission = capacity check + slot reservation, before any execution.
        if not self._free_slots:
            raise SchedulerError("admission rejected: capacity exceeded")
        req.state = RequestState.WAITING
        req.cache_slot = self._free_slots.pop()
        self._requests[req.request_id] = req
        self._waiting.append(req.request_id)

    def cancel(self, request_id: str) -> None:
        req = self._requests.get(request_id)
        if req is None or req.is_terminal():
            return
        # Suppress in-flight output, free slot if assigned. Repeated or queued
        # references are dropped so the request cannot be scheduled again.
        if req.cache_slot is not None and req.cache_slot not in self._free_slots:
            self._free_slots.add(req.cache_slot)
        self._waiting = deque(r for r in self._waiting if r != request_id)
        self._active.pop(request_id, None)
        self._prefill_chunk.pop(request_id, None)
        req.transition(RequestState.CANCELLED)

    def get_request(self, request_id: str) -> Request:
        """Return the live request object (read access for the engine)."""
        return self._requests[request_id]

    @property
    def is_idle(self) -> bool:
        """True when nothing is waiting, active, or mid-prefill."""
        return not (self._waiting or self._active or self._prefill_chunk)

    # --- Scheduling ---
    def schedule(self) -> list[ExecutionPlan]:
        # 1) Continue an in-progress prefill (serial: one at a time)
        if self._prefill_chunk:
            rid = next(iter(self._prefill_chunk))
            return [self._make_prefill_plan(self._requests[rid])]
        # 2) Decode all active decoding requests
        decoding = [r for r in self._active.values() if r.state == RequestState.DECODING]
        if decoding:
            return [self._make_decode_plan(decoding)]
        # 3) Admit head of waiting queue to prefill (slot already reserved)
        if self._waiting:
            rid = self._waiting.popleft()
            req = self._requests[rid]
            req.transition(RequestState.PREFILLING)
            self._active[rid] = req
            self._prefill_chunk[rid] = 0
            return [self._make_prefill_plan(req)]
        return []  # idle

    def mark_prefill_chunk_done(self, request_id: str, chunk_idx: int) -> None:
        """Advance prefill progress after a chunk executes.

        Called by the engine. When the last chunk completes, the request moves
        to DECODING.
        """
        req = self._requests.get(request_id)
        if req is None or request_id not in self._prefill_chunk:
            return
        bounds = chunk_prompt(list(req.prompt_token_ids), self._policy.max_chunk_tokens)
        if chunk_idx + 1 >= len(bounds):
            self._prefill_chunk.pop(request_id, None)
            req.transition(RequestState.DECODING)
        else:
            self._prefill_chunk[request_id] = chunk_idx + 1

    def mark_finished(self, request_id: str) -> None:
        """Mark a request finished and free its cache slot. Called by engine."""
        req = self._requests.get(request_id)
        if req is None or req.is_terminal():
            return
        req.transition(RequestState.FINISHED)
        if req.cache_slot is not None:
            self._free_slots.add(req.cache_slot)
            req.cache_slot = None
        self._active.pop(request_id, None)
        self._prefill_chunk.pop(request_id, None)

    @property
    def num_free_slots(self) -> int:
        return len(self._free_slots)

    # --- Plan construction ---
    def _make_prefill_plan(self, req: Request) -> PrefillPlan:
        bounds = tuple(chunk_prompt(list(req.prompt_token_ids), self._policy.max_chunk_tokens))
        return PrefillPlan(
            request_id=req.request_id,
            input_ids=tuple(req.prompt_token_ids),
            cache_slot=req.cache_slot,
            chunk_bounds=bounds,
            prefix_token_ids=(),
            prefix_cache_hit=False,
        )

    def _make_decode_plan(self, reqs: list[Request]) -> DecodePlan:
        # input_ids: the most recent token of each sequence (the token whose KV
        # is about to be written). positions: absolute index where it is written,
        # equal to the current sequence length (len(all_token_ids) - 1).
        return DecodePlan(
            request_ids=tuple(r.request_id for r in reqs),
            input_ids=tuple(r.all_token_ids[-1] for r in reqs),
            positions=tuple(len(r.all_token_ids) - 1 for r in reqs),
            slots=tuple(r.cache_slot for r in reqs),
            active_mask=tuple(True for _ in reqs),
            block_tables=(),
            sampling_params=tuple(r.sampling_params for r in reqs),
        )
