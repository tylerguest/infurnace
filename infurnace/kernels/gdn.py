"""Fused gated delta-net chunk for multirow verification (NV, one block per head).

Computes, for T rows of one GatedDeltaNet layer and initial state S0 (V x K):
  u_t   = β_t (v_t - γ_t S0 k_t - Σ_{s<t} (γ_t/γ_s)(k_t·k_s) u_s)     (forward substitution)
  o_t   = γ_t S0 q_t + Σ_{s<=t} (γ_t/γ_s)(q_t·k_s) u_s
  z_t   = RMSNorm_V(o_t) * silu(gate_t)
with γ_t = exp(g_t) from cumulative log-decays g. Each of the 128 threads owns one
value row v: it streams S0[v, :] once and keeps its column of U in registers, so
the triangular solve needs no synchronization. Only K/Q tiles, the T x T Gram
matrices and the norm partials live in shared memory. U is written for `commit`.
"""
import functools

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo

from infurnace.kernels.quant_gemv import warp_reduce


def _silu(x: UOp) -> UOp:
  return x / (UOp.const(1.0, dtypes.float32) + (-x).exp())


@functools.cache
def _gdn_chunk_kernel(z: UOp, u: UOp, q: UOp, k: UOp, v: UOp, beta: UOp, g: UOp, state: UOp, gate: UOp,
                      weight: UOp, keep: UOp, *, heads: int, rows: int, eps: float) -> UOp:
  T, D = rows, 128
  head = UOp.range(heads, 0, axis_type=AxisType.GLOBAL)
  warp = UOp.range(4, 1, axis_type=AxisType.LOCAL)
  lane = UOp.range(32, -1, axis_type=AxisType.WARP)
  col = warp*32+lane

  # Stage this head's K and Q rows (coalesced over the 128 columns).
  ksh = UOp.placeholder((T, D), dtypes.float32, slot=0, addrspace=AddrSpace.LOCAL)
  qsh = UOp.placeholder((T, D), dtypes.float32, slot=1, addrspace=AddrSpace.LOCAL)
  staged = UOp.group(*(ksh[t, col].store(k[head, t, col].load()) for t in range(T)),
                     *(qsh[t, col].store(q[head, t, col].load()) for t in range(T))).barrier()
  ksh, qsh = ksh.after(staged), qsh.after(staged)

  # Gram matrices K Kᵀ and Q Kᵀ (T x T): each thread computes two entries of each.
  pairs = [(col//T, col%T), ((col+128)//T, (col+128)%T)] if T == 16 else None
  assert pairs is not None, "fused GDN chunk is specialized to 16 rows"
  gram = UOp.placeholder((4,), dtypes.float32, slot=2, addrspace=AddrSpace.REG)
  gram = gram.after(UOp.group(*(gram[i].store(UOp.const(0.0, dtypes.float32)) for i in range(4))))
  d = UOp.range(D, 2, axis_type=AxisType.LOOP)
  acc = gram.after(d)
  terms = [ksh[pairs[0][0], d].load()*ksh[pairs[0][1], d].load(), ksh[pairs[1][0], d].load()*ksh[pairs[1][1], d].load(),
           qsh[pairs[0][0], d].load()*ksh[pairs[0][1], d].load(), qsh[pairs[1][0], d].load()*ksh[pairs[1][1], d].load()]
  gram = gram.after(UOp.group(*(acc[i].store(acc[i].load()+term) for i, term in enumerate(terms))).end(d))
  kk = UOp.placeholder((T, T), dtypes.float32, slot=3, addrspace=AddrSpace.LOCAL)
  qk = UOp.placeholder((T, T), dtypes.float32, slot=4, addrspace=AddrSpace.LOCAL)
  grams = UOp.group(kk[pairs[0]].store(gram[0].load()), kk[pairs[1]].store(gram[1].load()),
                    qk[pairs[0]].store(gram[2].load()), qk[pairs[1]].store(gram[3].load())).barrier()
  kk, qk = kk.after(grams), qk.after(grams)

  # This thread's value row of S0 against every K and Q row.
  proj = UOp.placeholder((2*T,), dtypes.float32, slot=5, addrspace=AddrSpace.REG)
  proj = proj.after(UOp.group(*(proj[i].store(UOp.const(0.0, dtypes.float32)) for i in range(2*T))))
  j = UOp.range(D, 3, axis_type=AxisType.LOOP)
  s0 = state[head, col, j].load()*keep[0].load()
  acc = proj.after(j)
  proj = proj.after(UOp.group(*(acc[t].store(acc[t].load()+ksh[t, j].load()*s0) for t in range(T)),
                              *(acc[T+t].store(acc[T+t].load()+qsh[t, j].load()*s0) for t in range(T))).end(j))
  ks, qs = [proj[t].load() for t in range(T)], [proj[T+t].load() for t in range(T)]

  # Forward substitution and outputs, entirely in this thread's registers.
  gs = [g[head, t].load() for t in range(T)]
  bs = [beta[head, t].load() for t in range(T)]
  us, outs = [], []
  for t in range(T):
    value = bs[t]*(v[head, t, col].load()-gs[t].exp()*ks[t])
    for s in range(t): value = value-bs[t]*(gs[t]-gs[s]).exp()*kk[t, s].load()*us[s]
    us.append(value)
  for t in range(T):
    value = gs[t].exp()*qs[t]
    for s in range(t+1): value = value+(gs[t]-gs[s]).exp()*qk[t, s].load()*us[s]
    outs.append(value)

  # RMSNorm over the 128 value columns: warp shuffle, then combine four warps.
  partial = UOp.placeholder((4, T), dtypes.float32, slot=6, addrspace=AddrSpace.LOCAL)
  reduced = UOp.group(*(partial[warp, t].store(warp_reduce(outs[t]*outs[t], full_wave=True)) for t in range(T))).barrier()
  partial = partial.after(reduced)
  stores = []
  for t in range(T):
    mean = sum((partial[w, t].load() for w in range(1, 4)), partial[0, t].load())/D
    scale = (mean+eps).sqrt().reciprocal()*weight[col].load()
    stores.append(z[t, head, col].store((outs[t]*scale*_silu(gate[t, head, col].load().float())).cast(dtypes.float16)))
    stores.append(u[head, t, col].store(us[t]))
  return UOp.group(*stores).end(head, warp, lane).sink(arg=KernelInfo(name="gdn_chunk", opts_to_apply=()))


def gdn_chunk(q: Tensor, k: Tensor, v: Tensor, beta: Tensor, g: Tensor, state: Tensor, gate: Tensor,
              weight: Tensor, keep: Tensor, eps: float) -> tuple[Tensor, Tensor]:
  """q, k, v: (H, T, 128) FP32 (q pre-scaled); beta, g: (H, T); state: (H, 128, 128)
  as (V, K); gate: (T, H, 128); weight: (128,); keep: (1,) 0/1 multiplier on S0.
  Returns z (T, H, 128) FP16 and u (H, T, 128) FP32."""
  heads, rows, dim = q.shape
  if dim != 128 or rows != 16: raise ValueError("fused GDN chunk requires 16 rows of 128-dimensional heads")
  z = Tensor.empty(rows, heads, dim, dtype=dtypes.float16, device=q.device)
  u = Tensor.empty(heads, rows, dim, dtype=dtypes.float32, device=q.device)
  inputs = [t.float().contiguous() for t in (q, k, v, beta, g, state)] + [gate.contiguous(), weight.float().contiguous(),
                                                                         keep.float().contiguous()]
  z, u = Tensor.custom_kernel(z, u, *inputs, fxn=functools.partial(_gdn_chunk_kernel, heads=heads, rows=rows, eps=eps))[:2]
  return z, u
