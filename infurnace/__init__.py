"""infurnace: run a single LLM fast on your GPU with tinygrad.

One model, one request at a time: load a GGUF, generate tokens, stream them
out (chat, a raw prompt, or a single-user OpenAI-compatible server). The module layout mirrors tinygrad's own ``tinygrad/llm`` package:

  gguf.py      — tinygrad's GGUF parser, kept packed for the NV kernels
  tokenizer.py — tinygrad tokenizer adapter + streaming decode + Qwen3 chat template (+ Qwen3.5 codec)
  models/      — Qwen3 (validated Transformer adapter, bounded generation) and Qwen3.5 adapters
  kernels/     — custom NV kernels (quantized GEMV/GEMM, attention, gated delta net, row gather)
  dflash/      — DFlash block-diffusion speculative decoding
  cli.py       — single-shot, chat, and benchmark driver
"""

from infurnace.models import Qwen3Config, Qwen3Model, config_from_gguf
from infurnace.tokenizer import Tokenizer

__all__ = ["Qwen3Config", "Qwen3Model", "config_from_gguf", "Tokenizer"]