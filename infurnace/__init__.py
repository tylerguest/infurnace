"""infurnace: run a single LLM fast on your GPU with tinygrad.

No server, no batching, no scheduling. Load a GGUF, generate tokens, stream
them out. The module layout mirrors tinygrad's own ``tinygrad/llm`` package:

  gguf.py      — own GGUF parser (packed quantized views, no fp16 round-trip)
  tokenizer.py — tinygrad tokenizer adapter + streaming decode + Qwen3 chat template (+ Qwen3.5 codec)
  models/      — Qwen3 (validated Transformer adapter, bounded generation) and Qwen3.5 adapters
  kernels/     — custom NV kernels (quantized GEMV/GEMM, attention, gated delta net, row gather)
  dflash/      — DFlash block-diffusion speculative decoding
  eagle/       — EAGLE-3 speculative decoding
  cli.py       — single-shot, chat, and benchmark driver
"""

from infurnace.gguf import load_gguf
from infurnace.models import Qwen3Config, Qwen3Model, config_from_gguf
from infurnace.tokenizer import Tokenizer

__all__ = ["load_gguf", "Qwen3Config", "Qwen3Model", "config_from_gguf", "Tokenizer"]