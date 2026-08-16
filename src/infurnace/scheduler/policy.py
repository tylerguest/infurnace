from __future__ import annotations
from collections import deque
from dataclasses import dataclass


@dataclass
class FIFOPolicy:
    """Deterministic FIFO ordering; extensible to fairness/priority later."""

    max_chunk_tokens: int = 512

    def order_waiting(self, waiting: deque[str]) -> list[str]:
        return list(waiting)  # FIFO = insertion order


def chunk_prompt(prompt: list[int], max_chunk: int) -> list[tuple[int, int]]:
    """Split a prompt into [(start, end), ...] chunks of at most ``max_chunk`` tokens.

    An empty prompt yields ``[(0, 0)]``; a prompt shorter than ``max_chunk`` yields
    a single ``(0, len)`` bound (unchunked).
    """
    if not prompt:
        return [(0, 0)]
    return [
        (i, min(i + max_chunk, len(prompt)))
        for i in range(0, len(prompt), max_chunk)
    ]
