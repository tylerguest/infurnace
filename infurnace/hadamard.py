"""Hadamard-folded weights (Bonsai's `prism.hadamard` GGUF metadata), as llama.cpp applies them.

Folded weights are stored with a blockwise rotation applied along their input dimension; the model is the
original one only if each such projection's input gets the matching transform first:

  x -> [GDN output projections: tiled (rep, nk, hd) heads to grouped (nk, rep, hd)] -> x * signs -> H x per block

H is the normalized Sylvester-Walsh-Hadamard matrix of the block size (entries +-1/sqrt(n), the sign from the
parity of row & col); it is its own inverse. Tables read by row lookup (the token embedding) store latent rows,
restored by the inverse: z -> H z per block -> * signs. With a tied output head (version 2) the same table is
also the output projection, with the forward transform.
"""
from __future__ import annotations
from dataclasses import dataclass
from tinygrad import Tensor, dtypes
from infurnace.device import nvidia
from infurnace.kernels.hadamard import fwht

def hadamard_matrix(n: int) -> list[float]:
  scale = n ** -0.5
  return [-scale if bin(row & col).count("1") % 2 else scale for row in range(n) for col in range(n)]

@dataclass(frozen=True)
class Rotation:
  """On NV the transform is kernels/hadamard.py's (in tinygrad's ops, the matmul fuses into whatever produced its
  input and recomputes it for every output: 21-125 us a rotation in a decode step instead of a few)."""
  matrix: Tensor                    # (block, block), float32
  signs: Tensor | None              # (width,), +-1
  perm: tuple[int, int, int] | None  # (rep, nk, hd): GDN value heads from tiled to grouped order

  @property
  def block(self) -> int: return self.matrix.shape[0]

  def _blocks(self, x: Tensor) -> Tensor:
    width = x.shape[-1]
    return (x.reshape(*x.shape[:-1], width // self.block, self.block) @ self.matrix).reshape(x.shape)

  def _kernel(self, x: Tensor) -> bool:
    return nvidia(x.device) and isinstance(x.numel(), int)

  def __call__(self, x: Tensor) -> Tensor:
    """A folded weight's input, in the weight's basis."""
    if self._kernel(x):
      signs = self.signs if self.signs is not None else Tensor.ones(x.shape[-1], device=x.device)
      return fwht(x, signs, block=self.block, perm=self.perm).cast(x.dtype)
    dtype, x = x.dtype, x.float()
    if self.perm is not None:
      rep, nk, hd = self.perm
      x = x.reshape(*x.shape[:-1], rep, nk, hd).transpose(-2, -3).reshape(x.shape)
    if self.signs is not None: x = x * self.signs
    return self._blocks(x).cast(dtype)

  def inverse(self, z: Tensor) -> Tensor:
    """A latent table row in the model's basis."""
    if self._kernel(z):
      signs = self.signs if self.signs is not None else Tensor.ones(z.shape[-1], device=z.device)
      return fwht(z, signs, block=self.block, signs_after=True).cast(z.dtype)
    dtype, z = z.dtype, self._blocks(z.float())
    return (z * self.signs if self.signs is not None else z).cast(dtype)

def rotations(metadata: dict, widths: dict[str, int], device: str, *, value_heads: int = 0,
              key_heads: int = 0) -> tuple[dict[str, Rotation], Rotation | None]:
  """(weight name -> the transform for its input, the token embedding's inverse transform or None), given each
  weight's input width. The tied output head is listed as output.weight. Empty without prism.hadamard metadata."""
  if (version := metadata.get("prism.hadamard.version")) is None: return {}, None
  if version not in (1, 2): raise ValueError(f"unsupported prism.hadamard.version {version}")
  block = metadata["prism.hadamard.block_size"]
  if metadata.get("prism.hadamard.transform") != "normalized-sylvester-walsh-hadamard" or \
     metadata.get("prism.hadamard.axis") != "input-last-dimension" or block <= 0 or block & (block - 1):
    raise ValueError("unsupported prism.hadamard transform")
  matrix = Tensor(hadamard_matrix(block), dtype=dtypes.float32, device=device).reshape(block, block).contiguous()
  signs: dict[int, Tensor] = {}
  if metadata.get("prism.hadamard.sign_mode") == "explicit":
    values, offset = metadata["prism.hadamard.sign_values"], 0
    for width in metadata["prism.hadamard.sign_widths"]:
      signs[width] = Tensor([float(v) for v in values[offset:offset + width]], dtype=dtypes.float32, device=device).contiguous()
      offset += width
    if offset != len(values): raise ValueError("prism.hadamard.sign_values length mismatch")
  elif metadata.get("prism.hadamard.sign_mode") != "identity": raise ValueError("unsupported prism.hadamard.sign_mode")
  grouped = metadata.get("prism.hadamard.gdn_v_grouped", False)

  shared: dict[tuple, Rotation] = {}  # one object per distinct transform: projections of one input rotate it once
  def rotation(name: str, width: int) -> Rotation:
    if width % block: raise ValueError(f"prism.hadamard block {block} does not divide {name}'s input width {width}")
    if signs and width not in signs: raise ValueError(f"prism.hadamard has no sign vector for width {width} ({name})")
    perm = None
    if grouped and ".ssm_out." in name:
      if not key_heads or value_heads % key_heads or width % value_heads: raise ValueError(f"bad GDN head geometry for {name}")
      perm = (value_heads // key_heads, key_heads, width // value_heads)
    return shared.setdefault((width, perm), Rotation(matrix, signs.get(width), perm))

  forward = {name: rotation(name, widths[name]) for name in metadata["prism.hadamard.weight_names"]}
  inverse = None
  if "token_embd.weight" in metadata.get("prism.hadamard.inverse_weight_names", []):
    inverse = rotation("token_embd.weight", widths["token_embd.weight"])
    if metadata.get("prism.hadamard.tied_output"): forward["output.weight"] = inverse
  return forward, inverse
