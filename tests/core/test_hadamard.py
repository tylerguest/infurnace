"""Bonsai's building blocks on the CPU: Q2_0 decoding, its prepared layout, and the Hadamard transforms."""
import random
import struct

import pytest
from tinygrad import Tensor, dtypes

from infurnace import gguf
from infurnace.hadamard import hadamard_matrix, rotations
from infurnace.kernels.quant_gemv import dequant_prepared_q2, prepare_q2

def q2_blocks(count: int, seed: int = 0) -> tuple[bytes, list[float]]:
  """Random Q2_0 blocks (codes 0-2) and their values as llama.cpp's dequantize_row_q2_0 computes them."""
  rng, raw, values = random.Random(seed), b'', []
  for _ in range(count):
    scale = struct.unpack('<e', struct.pack('<e', rng.uniform(0.01, 0.2)))[0]
    codes = [rng.randrange(3) for _ in range(64)]
    raw += struct.pack('<e', scale) + bytes(sum(codes[4 * i + j] << (2 * j) for j in range(4)) for i in range(16))
    values += [(c - 1) * scale for c in codes]
  return raw, values

def test_q2_0_decode_and_prepared_layout():
  raw, values = q2_blocks(8)
  data = Tensor(raw, dtype=dtypes.uint8)
  assert gguf.decode(data, 512, gguf.Q2_0).tolist() == pytest.approx(values, abs=1e-6)
  assert dequant_prepared_q2(prepare_q2(data), in_features=128).tolist() == pytest.approx(values, abs=1e-6)  # 4 rows of 128
  assert gguf.block_size(gguf.Q2_0) == (64, 18)

def metadata(**extra):
  return {'prism.hadamard.version': 2, 'prism.hadamard.block_size': 4, 'prism.hadamard.transform': 'normalized-sylvester-walsh-hadamard',
          'prism.hadamard.axis': 'input-last-dimension', 'prism.hadamard.sign_mode': 'explicit', 'prism.hadamard.sign_widths': [8],
          'prism.hadamard.sign_values': [1, -1, -1, 1, 1, 1, -1, 1], 'prism.hadamard.weight_names': ['blk.0.ffn_down.weight', 'blk.0.ssm_out.weight'],
          'prism.hadamard.inverse_weight_names': ['token_embd.weight'], 'prism.hadamard.tied_output': True,
          'prism.hadamard.gdn_v_grouped': True, **extra}

def test_hadamard_matrix_is_its_own_inverse():
  h = Tensor(hadamard_matrix(8)).reshape(8, 8)
  identity = [[1.0 if i == j else 0.0 for j in range(8)] for i in range(8)]
  assert all(row == pytest.approx(want, abs=1e-6) for row, want in zip((h @ h).tolist(), identity))

def test_rotations_follow_llama_cpp():
  widths = {'blk.0.ffn_down.weight': 8, 'blk.0.ssm_out.weight': 8, 'token_embd.weight': 8, 'output.weight': 8}
  forward, inverse = rotations(metadata(), widths, 'CPU', value_heads=4, key_heads=2)
  assert forward['output.weight'] is inverse and forward['blk.0.ffn_down.weight'] is not forward['blk.0.ssm_out.weight']
  assert forward['blk.0.ssm_out.weight'].perm == (2, 2, 2)
  x = [0.5, -1.0, 2.0, 0.25, 1.5, -0.75, 3.0, 1.0]
  signs, h = [1, -1, -1, 1, 1, 1, -1, 1], hadamard_matrix(4)
  def transform(v): return [sum(h[i * 4 + k] * v[b * 4 + k] for k in range(4)) for b in range(2) for i in range(4)]
  # forward: (reorder) -> * signs -> H per block; the GDN reorder takes tiled (rep, nk, hd) to grouped (nk, rep, hd)
  assert forward['blk.0.ffn_down.weight'](Tensor([x])).tolist()[0] == pytest.approx(transform([a * s for a, s in zip(x, signs)]), abs=1e-6)
  grouped = [x[i] for i in (0, 1, 4, 5, 2, 3, 6, 7)]
  assert forward['blk.0.ssm_out.weight'](Tensor([x])).tolist()[0] == pytest.approx(transform([a * s for a, s in zip(grouped, signs)]), abs=1e-6)
  # inverse (embedding rows): H per block -> * signs, undoing the forward transform
  assert inverse.inverse(Tensor([x])).tolist()[0] == pytest.approx([a * s for a, s in zip(transform(x), signs)], abs=1e-6)
  assert inverse.inverse(inverse(Tensor([x]))).tolist()[0] == pytest.approx(x, abs=1e-6)

def test_rotations_reject_what_they_cannot_apply():
  widths = {'blk.0.ffn_down.weight': 8, 'blk.0.ssm_out.weight': 8, 'token_embd.weight': 8, 'output.weight': 8}
  assert rotations({}, widths, 'CPU') == ({}, None)
  for bad in ({'prism.hadamard.version': 3}, {'prism.hadamard.block_size': 6}, {'prism.hadamard.transform': 'dct'},
              {'prism.hadamard.sign_values': [1] * 7}):
    with pytest.raises(ValueError): rotations(metadata(**bad), widths, 'CPU', value_heads=4, key_heads=2)
  with pytest.raises(ValueError, match='GDN head geometry'): rotations(metadata(), widths, 'CPU', value_heads=3, key_heads=2)
