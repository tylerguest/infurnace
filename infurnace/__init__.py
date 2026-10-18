"""Single-user Qwen3.5-4B runner on tinygrad, with optional DFlash speculation."""
from infurnace._cuda import use_pip_compiler
use_pip_compiler()  # before anything loads tinygrad's CUDA compiler

from infurnace.models import Qwen35Config, Qwen35Model
from infurnace.tokenizer import Qwen35Tokenizer

__all__ = ["Qwen35Config", "Qwen35Model", "Qwen35Tokenizer"]
