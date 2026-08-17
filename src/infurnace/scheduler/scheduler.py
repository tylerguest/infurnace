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
    Single-request serial prefill: ``schedule()`` returns the decode batch and,
    when a waiting request holds a reserved slot, one new prefill (Phase 4C).
    Prefill stays one request at a time so concurrent requests build a real
    decode batch.

    The scheduler owns logical cache-slot assignment and prefill progress. The
    engine transitions ``PREFILLING -> DECODING -> FINISHED`` and frees slots on
    terminal outcomes.
    """

    def __init__(self, num_slots: int, max_context: int = 32768, max_chunk_tokens: int = 512):
        if num_slots < 1:
            raise SchedulerError("num_slots must be >= 1")
        if max_context < 1:
            raise SchedulerError("max_context must be >= 1")
        if max_chunk_tokens < 1:
            raise SchedulerError("max_chunk_tokens must be >= 1")
        self._policy = FIFOPolicy(max_chunk_tokens=max_chunk_tokens)
        self._num_slots = num_slots
        self._max_context = max_context
        self._waiting: deque[str] = deque()
        self._requests: dict[str, Request] = {}
        self._active: dict[str, Request] = {}      # prefilling or decoding
        self._prefill_chunk: dict[str, int] = {}   # request_id -> next chunk idx
        self._free_slots: set[int] = set(range(num_slots))

    # --- Admission ---
    def add_request(self, req: Request) -> None:
        if req.request_id in self._requests:
            raise SchedulerError(f"duplicate request_id: {req.request_id}")
        effective_limit = min(req.context_limit, self._max_context)
        prompt_len = len(req.prompt_token_ids)
        if prompt_len > effective_limit:
            self._reject(req, f"prompt length {prompt_len} exceeds context limit {effective_limit}")
            return
        total_tokens = prompt_len + req.sampling_params.max_tokens
        if total_tokens > effective_limit:
            self._reject(
                req,
                f"prompt length {prompt_len} + max_tokens {req.sampling_params.max_tokens} "
                f"exceeds context limit {effective_limit}",
            )
            return
        # Admission = capacity check + slot reservation, before any execution.
        if not self._free_slots:
            self._reject(req, "capacity exceeded")
            return
        req.state = RequestState.WAITING
        req.cache_slot = min(self._free_slots)  # lowest free slot keeps a prefix
        self._free_slots.discard(req.cache_slot)
        self._requests[req.request_id] = req
        self._waiting.append(req.request_id)

    def _reject(self, req: Request, reason: str) -> None:
        """Record a request that failed admission without consuming a cache slot.

        The request is registered with a terminal REJECTED state (no slot, not
        queued) so rejection stays observable through ``get_request``.
        """
        req.error = reason
        req.transition(RequestState.REJECTED)
        self._requests[req.request_id] = req

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
        plans: list[ExecutionPlan] = []
        # 2) Decode all active decoding requests
        decoding = [r for r in self._active.values() if r.state == RequestState.DECODING]
        if decoding:
            plans.append(self._make_decode_plan(decoding))
        # 3) Admit head of waiting queue to prefill even while decode runs, so
        #    concurrent requests build a real decode batch (Phase 4C). The
        #    waiting request already holds a reserved slot; prefill stays
        #    single-request and serial.
        if self._waiting:
            rid = self._waiting.popleft()
            req = self._requests[rid]
            req.transition(RequestState.PREFILLING)
            self._active[rid] = req
            self._prefill_chunk[rid] = 0
            plans.append(self._make_prefill_plan(req))
        return plans  # [decode, prefill], [decode], [prefill], or [] when idle

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

    def release(self, request_id: str) -> None:
        """Release a request's slot and drop it from queues on a terminal path.

        The engine sets the terminal state (e.g. ``FAILED``) before calling
        this; ``release`` frees the reserved slot so admission and idle state
        stay correct without overwriting that state.
        """
        req = self._requests.get(request_id)
        if req is None or req.cache_slot is None:
            return
        self._free_slots.add(req.cache_slot)
        req.cache_slot = None
        self._waiting = deque(r for r in self._waiting if r != request_id)
        self._active.pop(request_id, None)
        self._prefill_chunk.pop(request_id, None)

    def compact(self) -> list[tuple[str, int, int]]:
        """Return ``(request_id, from_slot, to_slot)`` moves that restore active
        requests to a contiguous prefix of slots ``0..B-1``.

        The engine applies the moves with ``runner.move_slot`` and updates each
        request's ``cache_slot``. Fixed-shape batched decode relies on this
        prefix (row ``i`` is bound to physical slot ``i``).
        """
        active = [r for r in self._active.values() if r.cache_slot is not None]
        if not active:
            return []
        occupied = sorted(r.cache_slot for r in active)
        moves: list[tuple[str, int, int]] = []
        for target in range(len(active)):
            if target in occupied:
                continue
            mover = max(occupied)  # move the highest-slot request into the hole
            req = next(r for r in active if r.cache_slot == mover)
            moves.append((req.request_id, mover, target))
            occupied.remove(mover)
            occupied.append(target)
        return moves

    @property
    def num_free_slots(self) -> int:
        return len(self._free_slots)

    def free_slot(self, slot: int) -> None:
        """Return a vacated cache slot to the free set (after a compaction move).

        Compaction relocates an active request into a lower slot; the slot it
        vacated is no longer owned by any request and must be reclaimable.
        """
        if slot < 0 or slot >= self._num_slots:
            raise SchedulerError(f"slot {slot} out of range [0, {self._num_slots})")
        self._free_slots.add(slot)

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
        # Row order is slot order (compaction keeps active requests in a prefix),
        # so batched decode can bind row ``i`` to physical slot ``i``.
        reqs = sorted(reqs, key=lambda r: r.cache_slot)
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
