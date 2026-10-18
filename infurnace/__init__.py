"""Fast small language models on tinygrad, with optional DFlash speculation."""
import infurnace.device  # noqa: F401  first: it points tinygrad at NVIDIA's pip compiler before anything loads tinygrad's

from infurnace.engine import Model, ModelConfig
from infurnace.tokenizer import BPETokenizer

__all__ = ["Model", "ModelConfig", "BPETokenizer"]
