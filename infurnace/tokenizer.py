"""GGUF byte-level BPE encoding (HF tokenizers) and byte-stream decoding. Chat prompts come from the GGUF's
template (chat.py)."""
from __future__ import annotations
from typing import Callable, Mapping
from tinygrad.llm.cli import SimpleTokenizer

_CONTRACTIONS = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
# tokenizer.ggml.pre -> (pretokenizer regex, NFC normalization, BOS when add_bos_token asks for it), as llama.cpp's
# llama-vocab.cpp and the models' HF tokenizer.json define them. Qwen models never take a BOS, whatever converted
# metadata says.
PRESETS = {
  # Qwen3.5: letters keep their combining marks
  "qwen35": (_CONTRACTIONS + r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+", True, False),
  # Qwen2 and Qwen3: single digits
  "qwen2": (_CONTRACTIONS + r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+", True, False),
  # Llama 3: up to three digits, no normalization
  "llama-bpe": (_CONTRACTIONS + r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+", False, True),
}

class BPETokenizer:
  """Byte-level BPE text codec for GGUF vocabularies (GPT-2 style), with the pretokenizer of tokenizer.ggml.pre."""
  def __init__(self, normal_tokens: dict[str, int], special_tokens: dict[str, int], eos_id: int, eot_id: int | None,
               bos_id: int | None = None):
    # SimpleTokenizer supplies the byte decoder; its own encoder is never used (see from_gguf).
    self._backend = SimpleTokenizer(normal_tokens, special_tokens, 'qwen2', None, eos_id, eot_id)
    self.bos_id, self.eos_id, self.eot_id = bos_id, eos_id, eot_id

  @classmethod
  def from_gguf(cls, metadata: Mapping) -> "BPETokenizer":
    if metadata.get("tokenizer.ggml.model") != "gpt2": raise ValueError("only GPT-2 byte-level tokenizers are supported")
    if (preset := metadata.get("tokenizer.ggml.pre")) not in PRESETS:
      raise ValueError(f"unsupported tokenizer preset {preset!r}; supported: {', '.join(PRESETS)}")
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

    bos = token_id("tokenizer.ggml.bos_token_id")
    eos = token_id("tokenizer.ggml.eos_token_id")
    if eos is None: raise ValueError("missing tokenizer.ggml.eos_token_id")
    normal = {t: i for i, t in enumerate(tokens) if types[i] == 1}
    special = {t: i for i, t in enumerate(tokens) if types[i] != 1}
    eot = token_id("tokenizer.ggml.eot_token_id", special.get("<|im_end|>"))
    merges = metadata.get("tokenizer.ggml.merges")
    if not isinstance(merges, list) or any(not isinstance(m, str) or len(m.split(" ")) != 2 for m in merges):
      raise ValueError("invalid BPE merges")
    try:
      from tokenizers import Tokenizer as HFTokenizer, Regex, models, normalizers, pre_tokenizers, decoders
    except ImportError as exc:
      raise ValueError("the tokenizer requires the tokenizers package: reinstall infurnace (pip install -e .)") from exc
    try:
      result = cls(normal, special, eos, eot, bos if PRESETS[preset][2] and metadata.get("tokenizer.ggml.add_bos_token", False) else None)
    except KeyError as exc:
      raise ValueError("invalid byte-level vocabulary encoding") from exc
    vocab = {word: i for i, word in enumerate(tokens)}
    result._hf = HFTokenizer(models.BPE(vocab, [tuple(m.split(" ")) for m in merges]))
    pattern, nfc, _ = PRESETS[preset]
    if nfc: result._hf.normalizer = normalizers.NFC()
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
