"""Fused gated delta-net chunk for multirow verification and prompt prefill (NV, one block per head).

Computes, for T rows of one GatedDeltaNet layer and initial state S0 (V x K):
  u_t   = β_t (v_t - γ_t S0 k_t - Σ_{s<t} (γ_t/γ_s)(k_t·k_s) u_s)     (forward substitution)
  o_t   = γ_t S0 q_t + Σ_{s<=t} (γ_t/γ_s)(q_t·k_s) u_s
  z_t   = RMSNorm_V(o_t) * silu(gate_t)
with γ_t = exp(g_t) from cumulative log-decays g. Each of the 128 threads owns one
value row v: it streams S0[v, :] once and keeps its column of U in registers, so
the triangular solve needs no synchronization. Only K/Q tiles, the T x T Gram
matrices and the norm partials live in shared memory. U is written for `commit`.

Prefill keeps every row, so `gdn_prefill` also advances the state in place:
  S_T   = γ_T S0 + Σ_t (γ_T/γ_t) u_t k_tᵀ
each thread writing its own value row from the U column it already holds.
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
                      weight: UOp, keep: UOp, *, heads: int, rows: int, eps: float, advance: bool = False) -> UOp:
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
  if advance:
    # S_T[v, :] = γ_T S0[v, :] + Σ_t (γ_T/γ_t) u_t[v] k_t; this thread read only row v of S0 above.
    weights = [(gs[T-1]-gs[t]).exp()*us[t] for t in range(T)]
    i = UOp.range(D, 4, axis_type=AxisType.LOOP)
    total = sum((weights[t]*ksh[t, i].load() for t in range(T)), gs[T-1].exp()*state[head, col, i].load()*keep[0].load())
    stores.append(state[head, col, i].store(total).end(i))
  name = "gdn_prefill" if advance else "gdn_chunk"
  return UOp.group(*stores).end(head, warp, lane).sink(arg=KernelInfo(name=name, opts_to_apply=()))

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

def gdn_prefill(q: Tensor, k: Tensor, v: Tensor, beta: Tensor, g: Tensor, state: Tensor, gate: Tensor,
                weight: Tensor, keep: Tensor, eps: float) -> Tensor:
  """`gdn_chunk` for prompt prefill: every row is kept, so `state` (H, 128, 128) FP32 is advanced
  past all 16 rows in place. Padding rows (β = 0, log-decay 0) leave it unchanged. Returns z (T, H, 128) FP16."""
  heads, rows, dim = q.shape
  if dim != 128 or rows != 16: raise ValueError("fused GDN prefill requires 16 rows of 128-dimensional heads")
  z = Tensor.empty(rows, heads, dim, dtype=dtypes.float16, device=q.device)
  u = Tensor.empty(heads, rows, dim, dtype=dtypes.float32, device=q.device)
  inputs = [t.float().contiguous() for t in (q, k, v, beta, g)] + [state, gate.contiguous(), weight.float().contiguous(),
                                                                  keep.float().contiguous()]
  return Tensor.custom_kernel(z, u, *inputs, fxn=functools.partial(_gdn_chunk_kernel, heads=heads, rows=rows, eps=eps, advance=True))[0]

@functools.cache
def _gdn_decode_kernel(z: UOp, state: UOp, qkv: UOp, conv: UOp, conv_w: UOp, x: UOp, w_beta: UOp, w_alpha: UOp, dt_bias: UOp,
                       a: UOp, gate: UOp, weight: UOp, keep: UOp, *, heads: int, kheads: int, width: int, eps: float) -> UOp:
  D = 128
  head = UOp.range(heads, 0, axis_type=AxisType.GLOBAL)
  warp = UOp.range(4, 1, axis_type=AxisType.LOCAL)
  lane = UOp.range(32, -1, axis_type=AxisType.WARP)
  col = warp*32+lane
  kept = keep[0].load()

  # Causal conv over [conv state | this row] and silu, for this thread's q, k and v channels.
  # Value heads tile the key heads (upstream repeats them), so head h reads key head h % kheads.
  def conv_silu(channel: UOp) -> UOp:
    terms = [conv[i, channel].load()*kept*conv_w[channel, i].load() for i in range(3)] + [qkv[channel].load()*conv_w[channel, 3].load()]
    total = ((terms[0]+terms[1])+terms[2])+terms[3]
    return total*(UOp.const(1.0, dtypes.float32) + (-total).exp()).reciprocal()
  kh = head % kheads
  qv, kv, vv = conv_silu(kh*D+col), conv_silu(kheads*D+kh*D+col), conv_silu(2*kheads*D+head*D+col)

  # This head's beta and decay projections (rows of 2560-wide weights; the threads split the width).
  def project(w: UOp) -> UOp:
    return sum((x[d*D+col].load().float()*w[head, d*D+col].load().float() for d in range(1, width//D)),
               x[col].load().float()*w[head, col].load().float())

  # Block sums: q², k² (L2 norms) and the two projections.
  sums = UOp.placeholder((4, 4), dtypes.float32, slot=0, addrspace=AddrSpace.LOCAL)
  sums = sums.after(UOp.group(*(sums[i, warp].store(warp_reduce(value, full_wave=True))
                                for i, value in enumerate((qv*qv, kv*kv, project(w_beta), project(w_alpha))))).barrier())
  def total(i: int) -> UOp: return sum((sums[i, w].load() for w in range(1, 4)), sums[i, 0].load())
  def norm(i: int) -> UOp: return total(i).sqrt().maximum(UOp.const(1e-6, dtypes.float32))
  one, zero = UOp.const(1.0, dtypes.float32), UOp.const(0.0, dtypes.float32)
  def half(value: UOp) -> UOp: return value.cast(dtypes.float16).cast(dtypes.float32)  # upstream rounds these to FP16
  b = half((one + (-half(total(2))).exp()).reciprocal())                      # sigmoid, in FP16 like upstream
  pre = half(total(3)) + dt_bias[head].load().float()
  softplus = pre.maximum(zero) + (one + (-(pre.maximum(-pre))).exp()).log()  # log(1 + e^pre), stable
  log_alpha = softplus*a[head].load().float()
  ksh = UOp.placeholder((D,), dtypes.float32, slot=1, addrspace=AddrSpace.LOCAL)
  qsh = UOp.placeholder((D,), dtypes.float32, slot=2, addrspace=AddrSpace.LOCAL)
  vsh = UOp.placeholder((D,), dtypes.float32, slot=3, addrspace=AddrSpace.LOCAL)
  staged = UOp.group(qsh[col].store(qv/norm(0)*D**-0.5), ksh[col].store(kv/norm(1)), vsh[col].store(vv)).barrier()
  ksh, qsh, vsh = ksh.after(staged), qsh.after(staged), vsh.after(staged)

  # Recurrence: each warp streams 32 value rows, one coalesced read and write per row.
  # s = decay*S[row]; delta = b(v_row - s·k); S[row] = s + delta*k; out_row = s·q + delta(k·q)
  cols = [lane*4+c for c in range(4)]
  kc, qc = [ksh[c].load() for c in cols], [qsh[c].load() for c in cols]
  kq = warp_reduce(sum((kc[i]*qc[i] for i in range(1, 4)), kc[0]*qc[0]), full_wave=True)
  decay = log_alpha.exp()*kept  # keep=0 at sequence start: the old state is ignored
  outsh = UOp.placeholder((D,), dtypes.float32, slot=4, addrspace=AddrSpace.LOCAL)
  i = UOp.range(32, 2, axis_type=AxisType.LOOP)
  row = warp*32+i
  s = [state[head, row, c].load()*decay for c in cols]
  sk = warp_reduce(sum((s[n]*kc[n] for n in range(1, 4)), s[0]*kc[0]), full_wave=True)
  sq = warp_reduce(sum((s[n]*qc[n] for n in range(1, 4)), s[0]*qc[0]), full_wave=True)
  delta = b*(vsh[row].load()-sk)
  rows_done = UOp.group(*(state[head, row, c].store(s[n]+delta*kc[n]) for n, c in enumerate(cols)),
                        outsh[row].store(sq+delta*kq)).end(i).barrier()
  outsh = outsh.after(rows_done)

  # RMSNorm over the head's 128 outputs, times silu(gate), as FP16.
  out = outsh[col].load()
  partial = UOp.placeholder((4,), dtypes.float32, slot=5, addrspace=AddrSpace.LOCAL)
  partial = partial.after(partial[warp].store(warp_reduce(out*out, full_wave=True)).barrier())
  mean = sum((partial[w].load() for w in range(1, 4)), partial[0].load())/D
  scale = (mean+eps).sqrt().reciprocal()*weight[col].load()
  store = z[head, col].store((out*scale*_silu(gate[head, col].load().float())).cast(dtypes.float16))
  return store.end(head, warp, lane).sink(arg=KernelInfo(name="gdn_decode", opts_to_apply=()))

@functools.cache
def _conv_shift_kernel(out: UOp, conv: UOp, qkv: UOp, keep: UOp, z: UOp, *, channels: int, size: int) -> UOp:
  # Runs after gdn_decode (it consumes its output): pass z through and shift the conv window in place.
  # One thread per output value; each also shifts channels//size conv channels.
  block = UOp.range(size//128, 0, axis_type=AxisType.GLOBAL)
  lane = UOp.range(128, 1, axis_type=AxisType.LOCAL)
  t = block*128+lane
  kept = keep[0].load()
  stores = [out[block, lane].store(z[block, lane].load())]  # out, z: (heads, 128)
  for j in range(channels//size):
    c = t+j*size
    r1, r2 = conv[1, c].load(), conv[2, c].load()
    after = conv.after(r1, r2)
    stores += [after[0, c].store(r1*kept), after[1, c].store(r2*kept), after[2, c].store(qkv[c].load())]
  return UOp.group(*stores).end(block, lane).sink(arg=KernelInfo(name="gdn_conv_shift", opts_to_apply=()))

def gdn_decode(qkv: Tensor, conv_state: Tensor, conv_w: Tensor, x: Tensor, w_beta: Tensor, w_alpha: Tensor, dt_bias: Tensor,
               a: Tensor, state: Tensor, gate: Tensor, weight: Tensor, keep: Tensor, eps: float, kheads: int) -> Tensor:
  """One gated delta-net decode row from the raw qkv projection, in two kernels.

  qkv: (C,) projection output; conv_state: (3, C) FP32 realized; conv_w: (C, 4); x: (width,) layer
  input; w_beta, w_alpha: (H, width) projection weights (beta = sigmoid(x·w_beta), log decay =
  softplus(x·w_alpha + dt_bias)*a); state: (H, 128, 128) FP32 as (V, K), realized; gate: (H, 128);
  weight: (128,); keep: (1,) 0/1 multiplier on the old conv and recurrent state.
  The recurrent and conv states are updated in place; returns z (H, 128) FP16."""
  heads, channels, width = w_beta.shape[0], qkv.shape[0], x.shape[0]
  if (state.shape != (heads, 128, 128) or conv_state.shape != (3, channels) or channels != 2*kheads*128 + heads*128
      or channels % (heads*128) or width % 128 or w_beta.shape != (heads, width) or w_alpha.shape != (heads, width)):
    raise ValueError("fused GDN decode requires 128-dimensional heads and a 4-tap conv")
  qkv, keep = qkv.float().contiguous(), keep.float().contiguous()
  z = Tensor.empty(heads, 128, dtype=dtypes.float16, device=qkv.device)
  z = Tensor.custom_kernel(z, state, qkv, conv_state, conv_w.float().contiguous(), x.contiguous(), w_beta.contiguous(),
                           w_alpha.contiguous(), dt_bias.contiguous(), a.contiguous(), gate.contiguous(),
                           weight.float().contiguous(), keep,
                           fxn=functools.partial(_gdn_decode_kernel, heads=heads, kheads=kheads, width=width, eps=eps))[0]
  out = Tensor.empty(heads, 128, dtype=dtypes.float16, device=qkv.device)
  return Tensor.custom_kernel(out, conv_state, qkv, keep, z, fxn=functools.partial(
    _conv_shift_kernel, channels=channels, size=heads*128))[0]
