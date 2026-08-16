from __future__ import annotations
from typing import Optional
from .base import Tokenizer

def detokenize_incrementally(tokenizer: Tokenizer, all_token_ids: list[int],
                             prefix_offset: int, read_offset: int):
    """Incremental detokenize with a vLLM-style prefix/read offset interface.

    Re-decodes the full sequence each step (cheap for offline use) and returns
    only the characters past ``read_offset``. This guarantees the streaming
    invariant ``"".join(deltas) == tokenizer.decode(all_token_ids)`` even for
    per-character and byte-fallback tokenizers, where window-only decoding would
    drop characters at the prefix boundary. ``prefix_offset`` is accepted for
    signature compatibility and advanced callers; it is reset to the sequence
    length and ``read_offset`` advances by the emitted delta length.
    """
    full = tokenizer.decode(all_token_ids)
    if read_offset > len(full):
        read_offset = 0
    delta = full[read_offset:]
    return delta, len(all_token_ids), len(full)

class StreamingDetokenizer:
    """Per-request incremental detokenizer state.
    Feed it the full ``all_token_ids`` (including any prompt) after each new
    token; it returns only the newly decodable text. ``skip_prompt_len`` starts
    the window past the prompt so only generated text is streamed.
    """

    def __init__(self, tokenizer: Tokenizer, skip_prompt_len: int = 0) -> None:
        self._tokenizer = tokenizer
        self._skip = skip_prompt_len
        self._read = 0
        self._text = ""

    @property
    def text(self) -> str:
        return self._text

    def truncate(self, length: int) -> None:
        """Keep only the first ``length`` characters of the accumulated text."""
        self._text = self._text[:length]

    def update(self, all_token_ids: list[int]) -> str:
        full = self._tokenizer.decode(all_token_ids[self._skip:])
        delta = full[self._read:]
        self._read += len(delta)
        self._text += delta
        return delta

def check_stop_strings(text: str, new_char_count: int, stop_strings: list[str],
                       include_in_output: bool):
    """Return ``(matched, matched_string, truncate_len)`` for stop-string stops.

    ``new_char_count`` is accepted for API parity (bounds the search window in
    callers that need it); the whole ``text`` is searched here for correctness.
    ``truncate_len`` is the text length to keep when finishing: the full text if
    ``include_in_output`` else the index where the stop string begins.
    """
    for s in stop_strings:
        idx = text.find(s)
        if idx != -1:
            return (True, s, len(text) if include_in_output else idx)
    return (False, "", 0)
