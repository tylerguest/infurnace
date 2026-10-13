"""Qwen3.5 GGUF BPE encoding, byte-stream decoding and text-only chat formatting."""
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

  def render_chat(self, messages: list[dict], *, enable_thinking: bool = True, add_generation_prompt: bool = True) -> str:
    """Render Qwen3.5 text chat without executing arbitrary Jinja.

    Historical reasoning before the latest user turn is omitted. Thinking is
    enabled by default; tool use belongs to the GGUF template in HTTP serving.
    """
    if type(enable_thinking) is not bool or type(add_generation_prompt) is not bool:
      raise ValueError("chat rendering flags must be boolean")
    if not isinstance(messages, list) or not messages: raise ValueError("chat requires a nonempty message list")
    for message in messages:
      if not isinstance(message, dict): raise ValueError("chat messages must be dictionaries")
      role, content = message.get("role"), message.get("content")
      if role not in ("system", "user", "assistant"): raise ValueError(f"unsupported chat role: {role!r}")
      if not isinstance(content, str): raise ValueError("chat content must be text")
      if message.get("tool_calls") or message.get("tool_call_id"):
        raise ValueError("tool calls are not supported")
      if "<tool_response>" in content and "</tool_response>" in content:
        raise ValueError("tool responses are not supported")
      reasoning = message.get("reasoning_content")
      if reasoning is not None and (role != "assistant" or not isinstance(reasoning, str)):
        raise ValueError("reasoning_content must be assistant text")
    for marker in ("<|im_start|>", "<|im_end|>"):
      if marker not in self._backend._special_tokens: raise ValueError(f"chat vocabulary is missing {marker}")

    last_query = next((i for i in range(len(messages)-1, -1, -1) if messages[i]["role"] == "user"), len(messages)-1)
    parts = []
    for i, message in enumerate(messages):
      role, content = message["role"], message["content"]
      if role == "assistant":
        reasoning = message.get("reasoning_content")
        if reasoning is None:
          reasoning = ""
          if "</think>" in content:
            reasoning = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
            content = content.split("</think>")[-1].lstrip("\n")
        if i > last_query and (i == len(messages)-1 or reasoning):
          content = "<think>\n" + reasoning.strip("\n") + "\n</think>\n\n" + content.lstrip("\n")
      parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if add_generation_prompt:
      parts.append("<|im_start|>assistant\n")
      parts.append("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")
    return "".join(parts)
