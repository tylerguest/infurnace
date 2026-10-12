"""Model adapters: Qwen3 (infurnace's own packed decode path) and Qwen3.5 (tinygrad's model, packed NV projections)."""
from infurnace.models.qwen3 import Qwen3Config, Qwen3Model, config_from_gguf
from infurnace.models.qwen35 import Qwen35Model

__all__ = ["Qwen3Config", "Qwen3Model", "Qwen35Model", "config_from_gguf"]
