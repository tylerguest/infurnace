"""Single-user Qwen3.5-4B runner on tinygrad, with optional DFlash speculation."""
import infurnace.device  # noqa: F401  first: it points tinygrad at NVIDIA's pip compiler before anything loads tinygrad's

from infurnace.models import Qwen35Config, Qwen35Model
from infurnace.tokenizer import Qwen35Tokenizer

__all__ = ["Qwen35Config", "Qwen35Model", "Qwen35Tokenizer"]
