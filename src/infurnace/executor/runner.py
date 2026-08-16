from __future__ import annotations
from abc import ABC, abstractmethod
from tinygrad import Tensor


class Runner(ABC):
    """Execution contract shared by all model runners (tinygrad, fake, future).

    A runner owns model execution but not request lifecycle, scheduling, or
    cache ownership. The engine drives it through prefill and decode steps.
    """

    @abstractmethod
    def prefill(self, input_ids: Tensor, slot: int = 0) -> Tensor:
        """Eager prefill at position 0. ``input_ids`` has shape ``[1, T]``.

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
