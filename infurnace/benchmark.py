"""Shared deterministic prompts for local and upstream-comparison benchmarks."""
DEFAULT_PROMPT_TOKENS = 512
_PROSE = (
  "A local language model processes a prompt before generating an answer. "
  "During prefill, it stores attention keys and values for the input tokens. "
  "During decoding, each new token uses the model weights and the growing cache. "
  "Performance depends on memory bandwidth, kernel efficiency, and active context length. "
  "A useful benchmark separates startup, first-token latency, and warm decoding, "
  "and records its model, hardware, input length, and output length.\n\n"
)


def benchmark_prompt_ids(tokenizer, prompt_tokens: int = DEFAULT_PROMPT_TOKENS, prompt_text: str | None = None) -> list[int]:
  """Exact-length repeated prose, or an unmodified raw-text prompt (no chat template).

  Synthetic length includes optional BOS. File/text prompts are never truncated.
  """
  if type(prompt_tokens) is not int or prompt_tokens <= 0: raise ValueError("prompt token count must be positive")
  prefix = [tokenizer.bos_id] if tokenizer.bos_id is not None else []
  if prompt_text is not None:
    encoded = tokenizer.encode(prompt_text)
    if not encoded: raise ValueError("benchmark prompt text must contain tokens")
    return prefix + encoded
  encoded = tokenizer.encode(_PROSE)
  if not encoded: raise ValueError("tokenizer produced an empty benchmark passage")
  length = prompt_tokens-len(prefix)
  return prefix + (encoded*((length+len(encoded)-1)//len(encoded)))[:length]


def validate_benchmark_context(prompt_tokens: int, output_tokens: int, context: int) -> None:
  if prompt_tokens <= 0 or output_tokens <= 0: raise ValueError("benchmark prompt and output must be nonempty")
  if prompt_tokens + output_tokens > context:
    raise ValueError(f"benchmark needs {prompt_tokens + output_tokens} context tokens "
                     f"({prompt_tokens} prompt + {output_tokens} output), but capacity is {context}")
