"""Qwen3: attention-only, so no layers beyond what the engine already accelerates.

Prompts are read in chunks (the fused attention kernel handles them) and resume from the KV cache alone: it is
indexed by position, so any cached prefix stays valid and no snapshots are needed."""
from infurnace.models import Family
from infurnace.tokenizer import BPETokenizer

FAMILY = Family('qwen3', ('qwen3',), BPETokenizer.from_gguf, accelerate=lambda model: True)
