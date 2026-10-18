"""Model families: what differs between architectures. The engine (infurnace/engine.py) does the rest.

Each family module defines FAMILY; FAMILIES maps a GGUF's general.architecture to that module, imported only
when a model of that family loads."""
from __future__ import annotations
import importlib
from dataclasses import dataclass
from typing import Any, Callable

@dataclass(frozen=True)
class Family:
  name: str
  architectures: tuple[str, ...]
  # GGUF metadata -> tokenizer (encode, decode, stream_decoder, is_end, eos_id, eot_id, bos_id)
  tokenizer: Callable[[dict], Any]
  # Passes after the engine's on a packed NV model; True when prompts can then be read in chunks
  accelerate: Callable[[Any], bool] = lambda model: False
  # Recurrent state can't be rewound: prompts keep snapshots (see Model.save_snapshot)
  recurrent: bool = False
  # tokenizer -> (turn header token, user turn header tokens, tool result marker) for placing snapshots
  turn_markers: Callable[[Any], tuple[int | None, tuple[int, ...], int | None]] = lambda tokenizer: (None, (), None)

FAMILIES = {'qwen35': 'infurnace.models.qwen35', 'qwen3': 'infurnace.models.attention'}

def family_for(architecture: str | None) -> Family:
  if architecture not in FAMILIES:
    raise ValueError(f"unsupported model architecture {architecture!r}; supported: {', '.join(FAMILIES)}")
  return importlib.import_module(FAMILIES[architecture]).FAMILY
