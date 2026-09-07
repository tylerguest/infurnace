"""Own GGUF parser. No ``tinygrad.llm.gguf`` dependency.

Reads the GGUF binary container (header, typed KV metadata, tensor info, data)
and returns ``(metadata, state_dict)`` where quantized weights stay as *lazy
packed views* over the raw bytes. The dequant expression is fused into the
consuming kernel (matmul / embedding lookup) instead of materializing fp16, and
``kernels.quant_gemv.packed_weight`` digs the same packed view out for the
decode fast path — no dequant -> fp16 -> re-extract round trip.

Only the quant types this model needs are supported: F32 (native) and Q4_K /
Q6_K (block dequant in tinygrad ops). See ``docs/architecture.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tinygrad import Tensor, dtypes

# ggml quant block geometry: number of weights and packed bytes per block.
Q4_K, Q6_K = 12, 14
GGML_BLOCK_SIZE = 256
QUANT_BLOCK_BYTES = {Q4_K: 144, Q6_K: 210}

_NATIVE_DTYPES = {0: dtypes.float32}


def _round_up(value: int, alignment: int) -> int:
  """Round ``value`` up to a multiple of ``alignment`` (GGUF data alignment)."""
  return (value + alignment - 1) // alignment * alignment


def _read_str(handle) -> str:
  """Read a GGUF string: uint64 length prefix + utf-8 bytes."""
  raise NotImplementedError  # TODO: struct.unpack length, read bytes, decode


def _read_value(handle, value_type: int) -> Any:
  """Read one typed GGUF value (uint8..float64, string, or array)."""
  raise NotImplementedError  # TODO: value type table + recursive arrays


def _dequant(data: Tensor, n: int, ggml_type: int) -> Tensor:
  """Return a lazy tensor of ``n`` values for a raw GGUF byte slice ``data``.

  Native types bitcast directly; quantized types build the block dequant as a
  lazy expression over the packed bytes (never realized on its own).
  """
  raise NotImplementedError  # TODO: F32 bitcast, Q4_K + Q6_K dequant in tinygrad ops


def load_gguf(path: str | Path) -> tuple[dict[str, Any], dict[str, Tensor]]:
  """Parse a GGUF file and return ``(metadata, state_dict)``.

  ``metadata`` carries the ``qwen3.*`` config keys and ``tokenizer.ggml.*``
  vocab keys. ``state_dict`` maps tensor names to lazy fp16 views:
  quantized weights are packed views (fused dequant), native weights bitcast.
  """
  raise NotImplementedError  # TODO: parse header/kv/infos, disk tensor slices, reshape+cast