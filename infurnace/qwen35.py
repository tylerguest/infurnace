"""Thin local runner for tinygrad's existing Qwen3.5 GGUF implementation.

Tinygrad owns loading, transformer math, recurrence, projections, caches, RNG
and JIT. This adapter only exposes Infurnace's bounded generation/lifecycle API.
No block subclasses or target projection replacements are installed.
"""
from pathlib import Path
from tinygrad import Context
from tinygrad.llm.model import Transformer
from infurnace.gguf import read_gguf_metadata
from infurnace.model import Qwen3Config, Qwen3Model


class Qwen35Model(Qwen3Model):
  def __init__(self, transformer: Transformer, metadata: dict):
    if metadata.get('general.architecture') != 'qwen35': raise ValueError('expected dense Qwen3.5 GGUF')
    if not transformer.blk: raise ValueError('Qwen3.5 must contain transformer blocks')
    c = transformer.blk[0].config
    self.config = Qwen3Config(c.num_blocks, c.dim, c.hidden_dim, c.n_heads, c.n_kv_heads,
      c.head_dim, c.v_head_dim, c.rope_dim, c.rope_theta, c.norm_eps, c.vocab_size, transformer.max_context)
    self.transformer, self.metadata = transformer, metadata
    self.device = transformer.token_embd.weight.device
    self._generating = False

  @classmethod
  def from_gguf(cls, path: str | Path, max_context: int | None = None, device: str = 'CPU') -> 'Qwen35Model':
    if max_context is not None and (type(max_context) is not int or max_context <= 0): raise ValueError('invalid max_context')
    # Architecture dispatch is a cheap metadata-only check. All actual tensor
    # loading/configuration/dequantization remains upstream.
    if read_gguf_metadata(path).get('general.architecture') != 'qwen35': raise ValueError('expected dense Qwen3.5 GGUF')
    with Context(DEV=device):
      transformer, metadata = Transformer.from_gguf(Path(path), max_context=max_context, realize=False)
    return cls(transformer, metadata)
