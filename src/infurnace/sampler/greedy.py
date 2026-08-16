from __future__ import annotations
from abc import ABC, abstractmethod
from tinygrad import Tensor

class Sampler(ABC):
    """Turns model logits into a sampled token id, honoring SamplingParams.
    Sampling is an Infurnace policy implemented as tinygrad operations after
    model logits (see architecture.md). Greedy ``argmax`` is the first
    deterministic path; temperature, top-k, top-p, penalties, and per-request
    random state are added in Phase 7.
    """

    @abstractmethod
    def sample(self, logits: Tensor, params) -> int:
        """Return a single next-token id for ``logits`` of shape ``[1, V]``."""
        ...

class GreedySampler(Sampler):
    """Deterministic argmax sampler. Used to validate the server end-to-end."""

    def sample(self, logits: Tensor, params) -> int:
        return int(logits.argmax().item())
