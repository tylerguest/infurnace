"""Qwen3.5 GGUF text codec; separate from the unmodified tinygrad model."""
from infurnace.tokenizer import Tokenizer


class Qwen35Tokenizer(Tokenizer):
  @classmethod
  def from_gguf(cls, metadata) -> "Qwen35Tokenizer":
    if metadata.get("tokenizer.ggml.pre") != "qwen35": raise ValueError("expected Qwen3.5 tokenizer preset")
    try:
      from tokenizers import Tokenizer as HFTokenizer, Regex, models, normalizers, pre_tokenizers, decoders
    except ImportError as exc:
      raise ValueError("Qwen3.5 requires tokenizers: install infurnace[qwen35]") from exc
    adapted = dict(metadata, **{"tokenizer.ggml.pre": "qwen2", "tokenizer.ggml.add_bos_token": False})
    result = super().from_gguf(adapted)
    merges = metadata.get("tokenizer.ggml.merges")
    if not isinstance(merges, list) or any(not isinstance(m, str) or len(m.split(" ")) != 2 for m in merges):
      raise ValueError("invalid Qwen3.5 BPE merges")
    vocab = {word: i for i, word in enumerate(metadata["tokenizer.ggml.tokens"])}
    result._hf = HFTokenizer(models.BPE(vocab, [tuple(m.split(" ")) for m in merges]))
    result._hf.normalizer = normalizers.NFC()
    pattern = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|"
               r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
    result._hf.pre_tokenizer = pre_tokenizers.Sequence([
      pre_tokenizers.Split(Regex(pattern), behavior="isolated"), pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    result._hf.decoder = decoders.ByteLevel()
    result._hf.add_special_tokens(list(result._backend._special_tokens))
    result.eos_id = vocab.get("<|endoftext|>", result.eos_id)
    result.eot_id = vocab.get("<|im_end|>", result.eot_id)
    return result

  def encode(self, text: str) -> list[int]:
    return self._hf.encode(text, add_special_tokens=False).ids

  def render_chat(self, messages: list[dict], *, enable_thinking: bool = True, add_generation_prompt: bool = True) -> str:
    rendered = super().render_chat(messages, enable_thinking=enable_thinking, add_generation_prompt=add_generation_prompt)
    if enable_thinking and add_generation_prompt: rendered += "<think>\n"
    return rendered
