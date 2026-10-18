"""Qwen3.5 GGUF BPE encoding and byte-stream decoding. Chat prompts come from the GGUF's template (chat.py)."""
from __future__ import annotations
from typing import Callable, Mapping
from tinygrad.llm.cli import SimpleTokenizer

class Qwen35Tokenizer:
  """Qwen3.5 text codec (HF tokenizers BPE)."""
  def __init__(self, normal_tokens: dict[str, int], special_tokens: dict[str, int], eos_id: int, eot_id: int | None):
    # SimpleTokenizer's byte decoder is shared with Qwen3.5. Its qwen2 encoder
    # is never used: Qwen3.5 needs its own pretokenization and NFC normalization.
    self._backend = SimpleTokenizer(normal_tokens, special_tokens, 'qwen2', None, eos_id, eot_id)
    self.bos_id, self.eos_id, self.eot_id = None, eos_id, eot_id

  @classmethod
  def from_gguf(cls, metadata: Mapping) -> "Qwen35Tokenizer":
    if metadata.get("tokenizer.ggml.model") != "gpt2": raise ValueError("only GPT-2 byte-level tokenizers are supported")
    if metadata.get("tokenizer.ggml.pre") != "qwen35": raise ValueError("expected Qwen3.5 tokenizer preset")
    tokens = metadata.get("tokenizer.ggml.tokens")
    types = metadata.get("tokenizer.ggml.token_type")
    if not isinstance(tokens, list) or not tokens or any(not isinstance(t, str) or not t for t in tokens):
      raise ValueError("invalid tokenizer vocabulary")
    if len(set(tokens)) != len(tokens): raise ValueError("duplicate tokenizer vocabulary entries")
    if not isinstance(types, list) or len(types) != len(tokens) or any(type(t) is not int or not 1 <= t <= 6 for t in types):
      raise ValueError("invalid tokenizer token types")
    if type(metadata.get("tokenizer.ggml.add_bos_token", False)) is not bool:
      raise ValueError("tokenizer.ggml.add_bos_token must be boolean")

    def token_id(key: str, default=None):
      value = metadata.get(key, default)
      if value is not None and (type(value) is not int or not 0 <= value < len(tokens)):
        raise ValueError(f"invalid {key}")
      return value

    token_id("tokenizer.ggml.bos_token_id")  # Validate if present; Qwen3.5 never prepends BOS.
    eos = token_id("tokenizer.ggml.eos_token_id")
    if eos is None: raise ValueError("missing tokenizer.ggml.eos_token_id")
    normal = {t: i for i, t in enumerate(tokens) if types[i] == 1}
    special = {t: i for i, t in enumerate(tokens) if types[i] != 1}
    eot = token_id("tokenizer.ggml.eot_token_id", special.get("<|im_end|>"))
    merges = metadata.get("tokenizer.ggml.merges")
    if not isinstance(merges, list) or any(not isinstance(m, str) or len(m.split(" ")) != 2 for m in merges):
      raise ValueError("invalid Qwen3.5 BPE merges")
    try:
      from tokenizers import Tokenizer as HFTokenizer, Regex, models, normalizers, pre_tokenizers, decoders
    except ImportError as exc:
      raise ValueError("Qwen3.5 requires the tokenizers package: reinstall infurnace (pip install -e .)") from exc
    try:
      result = cls(normal, special, eos, eot)
    except KeyError as exc:
      raise ValueError("invalid byte-level vocabulary encoding") from exc
    vocab = {word: i for i, word in enumerate(tokens)}
    result._hf = HFTokenizer(models.BPE(vocab, [tuple(m.split(" ")) for m in merges]))
    result._hf.normalizer = normalizers.NFC()
    pattern = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|"
               r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
    result._hf.pre_tokenizer = pre_tokenizers.Sequence([
      pre_tokenizers.Split(Regex(pattern), behavior="isolated"), pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    result._hf.decoder = decoders.ByteLevel()
    result._hf.add_special_tokens(list(special))
    result.eos_id = vocab.get("<|endoftext|>", result.eos_id)
    result.eot_id = vocab.get("<|im_end|>", result.eot_id)
    return result

  def encode(self, text: str) -> list[int]:
    """Raw text IDs, including recognized special tokens; no automatic BOS."""
    return self._hf.encode(text, add_special_tokens=False).ids

  def decode(self, ids: list[int]) -> str:
    return self._backend.decode(ids)

  def stream_decoder(self) -> Callable[[int | None], str]:
    """New UTF-8 decoder per stream; pass None to flush incomplete bytes."""
    return self._backend.stream_decoder()

  def is_end(self, token_id: int) -> bool:
    return token_id in (self.eos_id, self.eot_id)
