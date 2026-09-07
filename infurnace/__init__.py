"""infurnace: run a single LLM fast on your GPU with tinygrad.

No server, no batching, no scheduling. Load a GGUF, generate tokens, stream
them out. The module layout mirrors tinygrad's own ``tinygrad/llm`` package:

  gguf.py      — own GGUF parser (packed quantized views, no fp16 round-trip)
  tokenizer.py — own BPE tokenizer + streaming decode + chat template
  model.py     — Qwen3 forward/decode/generate with an internal KV cache
  kernels/     — custom decode kernels (Q4_K/Q6_K GEMV)
  cli.py       — single-shot, chat, and benchmark driver
"""

from infurnace.gguf import load_gguf
from infurnace.model import Qwen3Config, Qwen3Model, config_from_gguf
from infurnace.tokenizer import Tokenizer

__all__ = ["load_gguf", "Qwen3Config", "Qwen3Model", "config_from_gguf", "Tokenizer"]