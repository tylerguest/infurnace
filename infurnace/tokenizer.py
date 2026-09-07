"""Own BPE tokenizer, no ``tinygrad.llm.cli.SimpleTokenizer`` dependency.

Built from the GGUF ``tokenizer.ggml.*`` metadata (tokens, token_type, pre,
bos/eos/eot ids). Byte-level BPE: split text into words with the GPT-2-style
regex, then greedily merge byte pieces into known tokens. Decoding uses the
standard byte decoder. The ``pre`` preset drives the chat template rendered by
``render_chat`` (qwen2 -> ``<|im_start|>...<|im_end|>``, llama3 fallback).
"""

from __future__ import annotations

from typing import Callable, Mapping


class Tokenizer:
  """Byte-level BPE tokenizer read from GGUF vocab metadata."""

  def __init__(self, normal_tokens: dict[str, int], special_tokens: dict[str, int],
               preset: str = "qwen2", bos_id: int | None = None, eos_id: int = 0, eot_id: int | None = None):
    """``normal_tokens``: mergeable token text -> id; ``special_tokens``: control tokens."""
    raise NotImplementedError  # TODO: byte decoder map, split-to-word regex, token maps, stream state

  @classmethod
  def from_gguf(cls, metadata: Mapping) -> "Tokenizer":
    """Build from ``tokenizer.ggml.tokens`` / ``token_type`` / ``pre`` / bos-eos-eot ids."""
    raise NotImplementedError  # TODO: partition normal vs special (type == 1)

  def encode(self, text: str) -> list[int]:
    """Encode ``text`` to a list of token ids (special tokens passed through)."""
    raise NotImplementedError  # TODO: split into words, greedy byte-pair merge

  def decode(self, ids: list[int]) -> str:
    """Decode a list of token ids to text."""
    raise NotImplementedError  # TODO: byte decoder join + utf-8 decode

  def stream_decoder(self) -> Callable[[int | None], str]:
    """Incremental decoder for token-at-a-time output.

    Returns a callable: pass a token id to get its decoded text, or ``None``
    to flush any buffered partial utf-8 sequence.
    """
    raise NotImplementedError  # TODO: incremental utf-8 decoder over token bytes

  def is_end(self, token_id: int) -> bool:
    """True if ``token_id`` is an end-of-sequence / end-of-turn token."""
    return token_id in (self.eos_id, self.eot_id)

  def render_chat(self, messages: list[dict]) -> str:
    """Render a message list (``{"role", "content"}`` dicts) plus the assistant
    generation prompt, using the preset's chat template."""
    raise NotImplementedError  # TODO: qwen2 -> <|im_start|>role\n...<|im_end|>\n<|im_start|>assistant\n