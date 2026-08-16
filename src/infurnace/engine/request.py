from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from typing import Optional


class RequestState(Enum):
    WAITING = "waiting"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"
    REJECTED = "rejected"


_VALID_TRANSITIONS: dict[RequestState, set[RequestState]] = {
    RequestState.WAITING: {
        RequestState.PREFILLING, RequestState.CANCELLED,
        RequestState.FAILED, RequestState.REJECTED,
    },
    RequestState.PREFILLING: {
        RequestState.DECODING, RequestState.FINISHED,
        RequestState.CANCELLED, RequestState.FAILED,
    },
    RequestState.DECODING: {
        RequestState.DECODING, RequestState.FINISHED,
        RequestState.CANCELLED, RequestState.FAILED,
    },
    RequestState.FINISHED: set(),
    RequestState.CANCELLED: set(),
    RequestState.FAILED: set(),
    RequestState.REJECTED: set(),
}


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 0.0
    top_k: int = 1
    top_p: float = 1.0
    max_tokens: int = 128
    # Stop conditions (3D adds string stop via tokenizer)
    stop_token_ids: list[int] = field(default_factory=list)
    min_tokens: int = 0
    include_stop_str_in_output: bool = False


@dataclass(slots=True)
class RequestMetrics:
    arrival_time: float = field(default_factory=monotonic)
    first_token_time: Optional[float] = None
    completion_time: Optional[float] = None

    @property
    def ttft(self) -> Optional[float]:
        return self.first_token_time - self.arrival_time if self.first_token_time else None

    @property
    def total_latency(self) -> Optional[float]:
        return self.completion_time - self.arrival_time if self.completion_time else None


@dataclass(slots=True)
class Request:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    context_limit: int
    eos_token_ids: list[int] = field(default_factory=list)
    session_id: Optional[str] = None

    # Mutable state
    state: RequestState = RequestState.WAITING
    computed_tokens: int = 0
    cache_slot: Optional[int] = None
    error: Optional[str] = None
    metrics: RequestMetrics = field(default_factory=RequestMetrics)

    # Token tracking (internal mutable, exposed read-only)
    _output_token_ids: list[int] = field(default_factory=list, init=False)
    _all_token_ids: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self._all_token_ids = list(self.prompt_token_ids)

    # --- Read-only views ---
    @property
    def output_token_ids(self) -> tuple[int, ...]:
        return tuple(self._output_token_ids)

    @property
    def all_token_ids(self) -> tuple[int, ...]:
        return tuple(self._all_token_ids)

    # --- State transitions ---
    def transition(self, new_state: RequestState) -> None:
        if new_state not in _VALID_TRANSITIONS[self.state]:
            raise ValueError(f"invalid transition: {self.state} -> {new_state}")
        if new_state == RequestState.FINISHED:
            self.metrics.completion_time = monotonic()
        self.state = new_state

    def is_terminal(self) -> bool:
        return self.state in (
            RequestState.FINISHED, RequestState.CANCELLED,
            RequestState.FAILED, RequestState.REJECTED,
        )

    def is_active(self) -> bool:
        return self.state in (RequestState.PREFILLING, RequestState.DECODING)

    # --- Token append + stop check ---
    def append_output_token(self, token_id: int) -> None:
        # TTFT is the time of the first *generated* token (standard definition).
        if self.metrics.first_token_time is None:
            self.metrics.first_token_time = monotonic()
        self._output_token_ids.append(token_id)
        self._all_token_ids.append(token_id)
        self.computed_tokens += 1

    def check_finished(self, new_token_id: int) -> Optional[str]:
        """Return stop reason or None. Integer-based only (no tokenizer)."""
        # min_tokens floor: force continuation until reached (blocks EOS/stop)
        if len(self._output_token_ids) < self.sampling_params.min_tokens:
            return None
        if new_token_id in self.eos_token_ids:
            return "eos"
        if new_token_id in self.sampling_params.stop_token_ids:
            return "stop_token_ids"
        if len(self._output_token_ids) >= self.sampling_params.max_tokens:
            return "max_tokens"
        return None
