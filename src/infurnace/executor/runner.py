from __future__ import annotations
from abc import ABC, abstractmethod
from tinygrad import Tensor

class Runner(ABC):
    """Execution contract shared by all model runners (tinygrad, fake, future).
    A runner owns model execution but not request lifecycle, scheduling, or
    cache ownership. The engine drives it through prefill and decode steps.
    """
    @property
    @abstractmethod
    def num_slots(self) -> int:
        """Number of independent cache slots the runner can execute concurrently."""
        ...

    @property
    @abstractmethod
    def max_context(self) -> int:
        """Maximum token context length the runner can store in its KV cache."""
        ...

    @abstractmethod
    def prefill(self, input_ids: Tensor, slot: int = 0, start_position: int = 0) -> Tensor:
        """Eager prefill. ``input_ids`` has shape ``[1, T]``.
        ``start_position`` is the absolute KV position where the chunk's first
        token is written (0 for the first chunk; advanced for chunked prefill).
        Returns logits of shape ``[1, vocab_size]`` for the last token.
        """
        ...

    @abstractmethod
    def decode(self, input_ids: Tensor, position: int, slot: int = 0) -> Tensor:
        """Single-token decode. ``input_ids`` has shape ``[1, 1]``.
        ``position`` is the absolute position of the decoded token. Returns
        logits of shape ``[1, vocab_size]``.
        """
        ...

    @abstractmethod
    def clear_slot(self, slot: int) -> None:
        """Reset the KV cache contents of ``slot`` so it can be reused.
        Called by the engine on every terminal transition (finished, cancelled,
        failed) after the request's cache slot is reclaimed.
        """
        ...

    @abstractmethod
    def move_slot(self, from_slot: int, to_slot: int) -> None:
        """Copy the KV contents of ``from_slot`` into ``to_slot`` and zero the
        source. The engine uses this to compact active requests into a prefix of
        slots so fixed-shape batched decode can bind row ``i`` to slot ``i``.
        """
        ...
