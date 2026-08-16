from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from tinygrad import Tensor, dtypes

from infurnace.engine.request import Request, RequestState, RequestMetrics
from infurnace.scheduler.plan import ExecutionPlan, PrefillPlan, DecodePlan
from infurnace.scheduler.scheduler import Scheduler, SchedulerError
from infurnace.executor.runner import Runner
from infurnace.sampler import Sampler, GreedySampler


@dataclass(frozen=True)
class EngineStepResult:
    """Outcome of one engine step, for observability and the offline API."""

    completed_requests: tuple[str, ...]
    new_tokens: Mapping[str, tuple[int, ...]]
    metrics: Mapping[str, RequestMetrics]
    cache_slots_freed: tuple[int, ...]


@dataclass
class _PlanResult:
    """Raw result of executing one plan on the runner, before applying state."""

    plan: ExecutionPlan
    request_id: str
    token: int | None = None
    error: str | None = None
    is_prefill: bool = False


@dataclass
class RawStep:
    results: list[_PlanResult] = field(default_factory=list)


class Engine:
    """Offline inference engine driving the scheduler and a ``Runner``.

    The engine owns request lifecycle application, sampling, cache-slot
    clearance, and metrics. It is runner-agnostic: a ``FakeRunner`` (CPU-only)
    validates the control plane, and the real ``Qwen3Runner`` is injected in
    Phase 3E without engine changes.

    The per-iteration path is split into ``prepare_step`` (scheduling),
    ``execute_step`` (runner + sampler only), and ``commit_step`` (apply to
    requests/scheduler). ``step`` composes them; the split is the seam for
    future device-overlap (Phase 6).
    """

    def __init__(
        self,
        runner: Runner,
        scheduler: Scheduler | None = None,
        sampler: Sampler | None = None,
    ) -> None:
        self.runner = runner
        self.scheduler = scheduler or Scheduler(num_slots=runner.num_slots)
        self.sampler = sampler or GreedySampler()

    # --- Offline request API ---
    def add_request(self, req: Request) -> None:
        try:
            self.scheduler.add_request(req)
        except SchedulerError:
            # Admission failure surfaces as a terminal REJECTED outcome.
            req.transition(RequestState.REJECTED)

    def cancel(self, request_id: str) -> None:
        req = self._try_get(req_id=request_id)
        slot = req.cache_slot if req is not None else None
        self.scheduler.cancel(request_id)
        if slot is not None:
            self.runner.clear_slot(slot)

    def output_tokens(self, request_id: str) -> tuple[int, ...]:
        return self.scheduler.get_request(request_id).output_token_ids

    def metrics_of(self, request_id: str) -> RequestMetrics:
        return self.scheduler.get_request(request_id).metrics

    def is_done(self) -> bool:
        return self.scheduler.is_idle

    # --- Async seam ---
    def prepare_step(self) -> list[ExecutionPlan]:
        return self.scheduler.schedule()

    def execute_step(self, plans: list[ExecutionPlan]) -> RawStep:
        raw = RawStep()
        for plan in plans:
            if isinstance(plan, PrefillPlan):
                raw.results.extend(self._execute_prefill(plan))
            elif isinstance(plan, DecodePlan):
                raw.results.extend(self._execute_decode(plan))
        return raw

    def commit_step(self, raw: RawStep) -> EngineStepResult:
        completed: list[str] = []
        new_tokens: dict[str, list[int]] = {}
        metrics: dict[str, RequestMetrics] = {}
        freed: list[int] = []
        for r in raw.results:
            req = self.scheduler.get_request(r.request_id)
            if r.error is not None:
                slot = req.cache_slot
                req.error = r.error
                req.transition(RequestState.FAILED)
                if slot is not None:
                    self.runner.clear_slot(slot)
                    freed.append(slot)
                continue
            req.append_output_token(r.token)
            new_tokens.setdefault(r.request_id, []).append(r.token)
            if r.is_prefill:
                # Sampling the first token consumed the prefill logits; the
                # request now enters DECODING so the next step feeds it back.
                self.scheduler.mark_prefill_chunk_done(r.request_id, len(r.plan.chunk_bounds) - 1)
            reason = req.check_finished(r.token)
            if reason is not None:
                slot = req.cache_slot
                self.scheduler.mark_finished(r.request_id)
                metrics[r.request_id] = req.metrics
                completed.append(r.request_id)
                if slot is not None:
                    self.runner.clear_slot(slot)
                    freed.append(slot)
        return EngineStepResult(
            completed_requests=tuple(completed),
            new_tokens={k: tuple(v) for k, v in new_tokens.items()},
            metrics=metrics,
            cache_slots_freed=tuple(freed),
        )

    def step(self) -> EngineStepResult:
        return self.commit_step(self.execute_step(self.prepare_step()))

    # --- Execution helpers ---
    def _try_get(self, req_id: str) -> Request | None:
        try:
            return self.scheduler.get_request(req_id)
        except KeyError:
            return None

    def _execute_prefill(self, plan: PrefillPlan) -> list[_PlanResult]:
        logits = None
        try:
            for start, end in plan.chunk_bounds:
                chunk = list(plan.input_ids[start:end])
                logits = self.runner.prefill(
                    Tensor([chunk], dtype=dtypes.int32), slot=plan.cache_slot
                )
        except Exception as e:  # runner failures become a terminal FAILED request
            return [_PlanResult(plan, plan.request_id, error=str(e), is_prefill=True)]
        req = self.scheduler.get_request(plan.request_id)
        token = self.sampler.sample(logits, req.sampling_params)
        return [_PlanResult(plan, plan.request_id, token=token, is_prefill=True)]

    def _execute_decode(self, plan: DecodePlan) -> list[_PlanResult]:
        results: list[_PlanResult] = []
        for i, rid in enumerate(plan.request_ids):
            try:
                logits = self.runner.decode(
                    Tensor([[plan.input_ids[i]]], dtype=dtypes.int32),
                    position=plan.positions[i],
                    slot=plan.slots[i],
                )
                req = self.scheduler.get_request(rid)
                token = self.sampler.sample(logits, plan.sampling_params[i])
                results.append(_PlanResult(plan, rid, token=token, is_prefill=False))
            except Exception as e:
                results.append(_PlanResult(plan, rid, error=str(e), is_prefill=False))
        return results
