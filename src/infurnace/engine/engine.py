from __future__ import annotations
import traceback
from dataclasses import dataclass, field
from typing import Mapping
from tinygrad import Tensor, dtypes
from infurnace.engine.request import Request, RequestState, RequestMetrics
from infurnace.scheduler.plan import ExecutionPlan, PrefillPlan, DecodePlan
from infurnace.scheduler.scheduler import Scheduler
from infurnace.executor.runner import Runner
from infurnace.sampler import Sampler, GreedySampler
from infurnace.tokenizer import Tokenizer, StreamingDetokenizer, check_stop_strings

@dataclass(frozen=True)
class EngineStepResult:
    """Outcome of one engine step, for observability and the offline API."""
    completed_requests: tuple[str, ...]
    new_tokens: Mapping[str, tuple[int, ...]]
    new_text: Mapping[str, str] = field(default_factory=dict)
    metrics: Mapping[str, RequestMetrics] = field(default_factory=dict)
    cache_slots_freed: tuple[int, ...] = ()
    logprobs: Mapping[str, object] = field(default_factory=dict)  # Phase 7 stub

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
    Optionally tokenized: when a ``Tokenizer`` is supplied, prompts may be added
    as text (``add_text_request``), generated tokens are streamed as incremental
    text (``new_text``), and ``stop_strings`` are evaluated per request.
    """
    def __init__(
        self,
        runner: Runner,
        scheduler: Scheduler | None = None,
        sampler: Sampler | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.runner = runner
        self.scheduler = scheduler or Scheduler(
            num_slots=runner.num_slots, max_context=runner.max_context
        )
        self.sampler = sampler or GreedySampler()
        self.tokenizer = tokenizer
        self._detoks: dict[str, StreamingDetokenizer] = {}

    # --- Offline request API ---
    def add_request(self, req: Request) -> None:
        # Admission rejection is recorded by the scheduler (state REJECTED,
        # observable via get_request). Duplicate IDs still raise.
        self.scheduler.add_request(req)

    def add_text_request(
        self,
        prompt: str,
        sampling_params=None,
        stop_strings: list[str] | None = None,
        request_id: str | None = None,
        context_limit: int = 1024,
        **kw,
    ) -> Request:
        if self.tokenizer is None:
            raise RuntimeError("engine has no tokenizer; cannot add a text request")
        token_ids = self.tokenizer.encode(prompt)
        rid = request_id or f"req-{len(self._detoks) + 1}"
        req = Request(
            rid, token_ids, sampling_params or SamplingParams(),
            context_limit, prompt=prompt, stop_strings=stop_strings or [], **kw,
        )
        self._detoks[req.request_id] = StreamingDetokenizer(self.tokenizer, skip_prompt_len=len(token_ids))
        self.add_request(req)
        return req

    def cancel(self, request_id: str) -> None:
        req = self._try_get(request_id)
        slot = req.cache_slot if req is not None else None
        self.scheduler.cancel(request_id)
        if slot is not None:
            self.runner.clear_slot(slot)
        self._apply_compaction()

    def output_tokens(self, request_id: str) -> tuple[int, ...]:
        return self.scheduler.get_request(request_id).output_token_ids

    def metrics_of(self, request_id: str) -> RequestMetrics:
        return self.scheduler.get_request(request_id).metrics

    def final_text(self, request_id: str) -> str:
        detok = self._detoks.get(request_id)
        return detok.text if detok else ""

    def is_done(self) -> bool:
        return self.scheduler.is_idle

    def _apply_compaction(self) -> None:
        """Move active requests into a prefix of slots 0..B-1 (Phase 4A)."""
        for request_id, from_slot, to_slot in self.scheduler.compact():
            self.runner.move_slot(from_slot, to_slot)
            self.scheduler.get_request(request_id).cache_slot = to_slot
            # The source slot is vacated by the move and becomes free again.
            self.scheduler.free_slot(from_slot)

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
        new_text: dict[str, list[str]] = {}
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
                self.scheduler.release(r.request_id)
                continue
            req.append_output_token(r.token)
            new_tokens.setdefault(r.request_id, []).append(r.token)
            delta = ""
            if self.tokenizer is not None:
                detok = self._detoks.get(r.request_id)
                if detok is not None:
                    delta = detok.update(req.all_token_ids)
                    new_text.setdefault(r.request_id, []).append(delta)
            if r.is_prefill:
                # Sampling the first token consumed the prefill logits; the
                # request now enters DECODING so the next step feeds it back.
                self.scheduler.mark_prefill_chunk_done(r.request_id, len(r.plan.chunk_bounds) - 1)
            reason = req.check_finished(r.token)
            if reason is None and self.tokenizer is not None and self.tokenizer.is_end(r.token):
                reason = "eos"
            if reason is None and self.tokenizer is not None and req.stop_strings:
                detok = self._detoks.get(r.request_id)
                if detok is not None:
                    matched, _stop, trunc = check_stop_strings(
                        detok.text, len(delta), req.stop_strings,
                        req.sampling_params.include_stop_str_in_output)
                    if matched:
                        reason = "stop"
                        if not req.sampling_params.include_stop_str_in_output:
                            before = len(detok.text) - len(delta)
                            detok.truncate(trunc)
                            new_text[r.request_id][-1] = detok.text[before:]
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
            new_text={k: "".join(v) for k, v in new_text.items()},
            metrics=metrics,
            cache_slots_freed=tuple(freed),
        )

    def step(self) -> EngineStepResult:
        self._apply_compaction()
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
                    Tensor([chunk], dtype=dtypes.int32), slot=plan.cache_slot,
                    start_position=start,
                )
        except Exception as e:  # runner failures become a terminal FAILED request
            traceback.print_exc()
            return [_PlanResult(plan, plan.request_id, error=str(e), is_prefill=True)]
        req = self.scheduler.get_request(plan.request_id)
        token = self.sampler.sample(logits, req.sampling_params)
        return [_PlanResult(plan, plan.request_id, token=token, is_prefill=True)]

    def _execute_decode(self, plan: DecodePlan) -> list[_PlanResult]:
        if not plan.request_ids:
            return []
        try:
            logits = self.runner.decode_batch(
                Tensor([[tid] for tid in plan.input_ids], dtype=dtypes.int32),
                plan.positions,
                plan.slots,
            )
            tokens = self.sampler.sample_batch(logits, plan.sampling_params)
        except Exception as e:  # a batched decode failure is terminal for every row
            traceback.print_exc()
            return [_PlanResult(plan, rid, error=str(e), is_prefill=False) for rid in plan.request_ids]
        return [
            _PlanResult(plan, rid, token=token, is_prefill=False)
            for rid, token in zip(plan.request_ids, tokens)
        ]
