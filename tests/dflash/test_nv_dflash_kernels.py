import random
import struct
import pytest
from tinygrad import Context, Tensor, dtypes
from infurnace.dflash.parallel import _decays, _unit_lower_inverse
from infurnace.gguf import _dequant
from infurnace.kernels.embedding import packed_gather
from infurnace.kernels.gdn import gdn_chunk
from infurnace.kernels.quant_gemv import prepare_q4, prepare_q6

pytestmark = pytest.mark.nv


def test_gdn_chunk_matches_tensor_closed_form():
  H, T, D, eps = 32, 16, 128, 1e-6
  with Context(DEV='NV'):
    Tensor.manual_seed(1)
    k, q = Tensor.randn(H, T, D).normalize(dim=-1).realize(), (Tensor.randn(H, T, D).normalize(dim=-1) * D**-0.5).realize()
    v, beta = Tensor.randn(H, T, D).realize(), Tensor.rand(H, T).realize()
    g = (-Tensor.rand(H, T) * 0.3).cumsum(-1).realize()
    S0, gate = (Tensor.randn(H, D, D) * 0.1).realize(), Tensor.randn(T, H, D).half().realize()
    weight = (Tensor.rand(D) + 0.5).realize()
    for keep in (1.0, 0.0):  # 0 = sequence start: the initial state is ignored
      kb, qb, vb, bb, gb = (t.unsqueeze(0) for t in (k, q, v, beta, g))
      Sb = (S0 * keep).unsqueeze(0)
      strict, inclusive = _decays(gb, T)
      A = bb.unsqueeze(-1) * strict * (kb @ kb.transpose(-1, -2))
      u_ref = _unit_lower_inverse(A, T) @ (bb.unsqueeze(-1) * (vb - gb.exp().unsqueeze(-1) * (kb @ Sb.transpose(-1, -2))))
      out = gb.exp().unsqueeze(-1) * (qb @ Sb.transpose(-1, -2)) + (inclusive * (qb @ kb.transpose(-1, -2))) @ u_ref
      core = out.transpose(1, 2)[0]
      z_ref = core * (core.square().mean(-1, keepdim=True) + eps).rsqrt() * weight * gate.float().silu()
      z, u = gdn_chunk(q, k, v, beta, g, S0, gate, weight, Tensor([keep]), eps)
      assert ((u - u_ref[0]).abs().max() / u_ref.abs().max()).item() < 1e-5
      # z is stored as FP16 for the output projection.
      assert ((z.float() - z_ref).abs().max() / z_ref.abs().max()).item() < 2e-3


def _q6_rows(rows, width):
  rng = random.Random(7)
  raw = bytearray(rng.getrandbits(8) for _ in range(rows*width//256*210))
  for offset in range(0, len(raw), 210): raw[offset+208:offset+210] = struct.pack('<e', 0.001)
  return bytes(raw)


def _q4_rows(rows, width):
  rng = random.Random(9)
  raw = bytearray(rng.getrandbits(8) for _ in range(rows*width//256*144))
  for offset in range(0, len(raw), 144): raw[offset:offset+4] = struct.pack('<ee', 0.002, 0.0004)
  return bytes(raw)


@pytest.mark.parametrize('typ,swizzled', [(14, False), (14, True), (12, True), (12, False)])
def test_packed_gather_decodes_only_requested_rows(typ, swizzled):
  rows, width = 40, 1280
  raw_bytes = _q6_rows(rows, width) if typ == 14 else _q4_rows(rows, width)
  ids = [0, 39, 7, 7, 21, 1, 38, 12]
  with Context(DEV='CPU'):
    table = _dequant(Tensor(raw_bytes, dtype=dtypes.uint8, device='CPU'), rows*width, typ).reshape(rows, width).half().float().tolist()
  with Context(DEV='NV'):
    raw = Tensor(raw_bytes, dtype=dtypes.uint8, device='NV').realize()
    if typ == 14:
      packed = prepare_q6(raw, in_features=width, swizzled=swizzled)
      options = dict(q6_aligned=True, q6_swizzled=swizzled)
    else:
      packed = prepare_q4(raw, in_features=width) if swizzled else raw.bitcast(dtypes.uint32).flatten().realize()
      options = dict(q4_swizzled=swizzled)
    got = packed_gather(Tensor(ids, dtype=dtypes.int32), packed, width=width, ggml_type=typ, **options).tolist()
  for i, row in zip(ids, got): assert row == table[i]
