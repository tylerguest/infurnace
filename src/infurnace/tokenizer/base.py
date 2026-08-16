from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Optional


class Tokenizer(ABC):
    """Text<->token interface for the engine.

    Implemented by ``GGUFTokenizer`` (tinygrad ``SimpleTokenizer``) and by
    test fakes. The engine depends only on this contract.
    """

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        ...

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        ...

    @abstractmethod
    def is_end(self, token_id: int) -> bool:
        ...

    def stream_decoder(self) -> Optional[Callable[[Optional[int]], str]]:
        """Optional incremental UTF-8 decoder for streaming safety."""
        return None
