from __future__ import annotations
from pathlib import Path
from typing import Mapping, Union
from tinygrad import Tensor, dtypes
from tinygrad.llm.cli import SimpleTokenizer
from tinygrad.llm.gguf import gguf_load
from .base import Tokenizer

class GGUFTokenizer(Tokenizer):
    """Tokenizer backed by a GGUF-embedded vocabulary via tinygrad's parser.
    Wraps a ``SimpleTokenizer`` (composition, not subclassing: ``from_gguf_kv``
    is a staticmethod that returns the base type). Decoding joins token bytes
    and decodes UTF-8 with ``errors='replace'`` (byte-fallback safe), and
    ``stream_decoder`` exposes tinygrad's incremental UTF-8 decoder.
    """

    def __init__(self, inner: SimpleTokenizer) -> None:
        self._inner = inner

    def encode(self, text: str) -> list[int]:
        return self._inner.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._inner.decode(ids)

    def is_end(self, token_id: int) -> bool:
        return self._inner.is_end(token_id)

    def stream_decoder(self):
        return self._inner.stream_decoder()

    @classmethod
    def from_gguf_metadata(cls, metadata: Mapping) -> "GGUFTokenizer":
        return cls(SimpleTokenizer.from_gguf_kv(metadata))

    @classmethod
    def from_artifact(cls, path: Union[str, Path]) -> "GGUFTokenizer":
        path = Path(path)
        gguf_tensor = Tensor.empty(path.stat().st_size, dtype=dtypes.uint8, device=f"disk:{path}")
        metadata, _tensors = gguf_load(gguf_tensor)
        return cls(SimpleTokenizer.from_gguf_kv(metadata))

def load_tokenizer_from_gguf(source: Union[str, Path, Mapping]) -> GGUFTokenizer:
    if isinstance(source, (str, Path)):
        return GGUFTokenizer.from_artifact(source)
    return GGUFTokenizer.from_gguf_metadata(source)
