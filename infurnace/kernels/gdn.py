"""Fused gated delta-net chunk for multirow verification and prompt prefill (NV).

Computes, for T rows of one GatedDeltaNet layer and initial state S0 (V x K), 16 rows at a time:
  u_t   = β_t (v_t - γ_t S0 k_t - Σ_{s<t} (γ_t/γ_s)(k_t·k_s) u_s)     (forward substitution)
  o_t   = γ_t S0 q_t + Σ_{s<=t} (γ_t/γ_s)(q_t·k_s) u_s
  z_t   = RMSNorm_V(o_t) * silu(gate_t)
with γ_t = exp(g_t) from cumulative log-decays g. The substitution is (I + A) u = rhs with A strictly lower and
the same for every value row, so a block inverts it once (T = (I + A)^-1) and every row's u = T rhs and o are
independent sums. A head is split over 4 blocks of 32 value rows; the projections of S0 on K and Q and the
state update are register-tiled. U is written for `commit`.

Prefill keeps every row, so `gdn_prefill` also advances the state in place, all rows in one launch:
  S_T   = γ_T S0 + Σ_t (γ_T/γ_t) u_t k_tᵀ
the block's state rows staying in shared memory between the 16-row steps.
"""
import functools

from tinygrad import Tensor, UOp, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo

from infurnace.kernels.quant_gemv import _load_lanes, warp_reduce

def _silu(x: UOp) -> UOp:
  return x / (UOp.const(1.0, dtypes.float32) + (-x).exp())

@functools.cache
def _gdn_chunk_kernel(o: UOp, u: UOp, q: UOp, k: UOp, v: UOp, beta: UOp, g: UOp, state: UOp, keep: UOp, *,
                      heads: int, rows: int, advance: bool = False, cumulative: bool = True) -> UOp:
  """The closed form over `rows` rows (a multiple of 16), 16 at a time, one launch. A head is split over 4 blocks
  of 32 value rows (S0's rows), 128 threads each. The block's 32 state rows stay in shared memory from one 16-row
  step to the next, so the state is read and (with `advance`) written once a call.

  The products are register-tiled from 16-byte shared loads: the projections S0 Kᵀ and S0 Qᵀ as 2 value rows x 4
  key rows a thread, the state update S += Wᵀ K as 4 x 8 a thread; the decayed Gram matrices and the
  substitution's inverse, which don't depend on the value row, once a block. A lane a value row then takes its u
  and outputs as independent sums (each thread solving the substitution itself, a 16-row step took ~41 us; RTX
  5060: now ~26, a 64-row prompt chunk 105 us against four launches of 43). g is the cumulative log-decay within
  each 16 rows, or with `cumulative=False` the per-row log-decay, summed here."""
  T, D, R = 16, 128, 32  # rows a step, head dim, value rows a block
  assert rows % T == 0, "fused GDN chunk takes a multiple of 16 rows"
  F32, zero = dtypes.float32, UOp.const(0.0, dtypes.float32)
  item = UOp.range(heads*(D//R), 0, axis_type=AxisType.GLOBAL)
  warp = UOp.range(4, 1, axis_type=AxisType.LOCAL)
  lane = UOp.range(32, -1, axis_type=AxisType.WARP)
  tid = warp*32+lane
  head, row0 = item // (D//R), (item % (D//R))*R
  row = row0+lane  # the substitution's value row
  def load4(ptr: UOp) -> list[UOp]: return list(_load_lanes(ptr, 4))
  # The block's 32 rows of S0, carried across the steps (rows padded by 4: 16-byte loads of eight rows at one
  # column hit different banks)
  ssh = UOp.placeholder((R, D+4), F32, slot=8, addrspace=AddrSpace.LOCAL)
  init = []
  for n in range(R*D//4//128):
    r, c = n*4+warp, lane*4
    init += [ssh[r, c+i].store(x*keep[0].load()) for i, x in enumerate(load4(state[head, row0+r, c]))]
  ssh = ssh.after(UOp.group(*init).barrier())
  step = UOp.range(rows//T, 5, axis_type=AxisType.LOOP)
  base = step*T
  current = ssh.after(step)

  # This step's K rows (0-15) and Q rows (16-31) in shared memory
  kq = UOp.placeholder((2*T, D+4), F32, slot=0, addrspace=AddrSpace.LOCAL)
  staged = []
  for n in range(2*T*D//4//128):
    t, c = n*4+warp, lane*4
    source = k[head, base+t, c] if t.vmax < T else q[head, base+t-T, c]
    staged += [kq[t, c+i].store(x) for i, x in enumerate(load4(source))]
  kq = kq.after(UOp.group(*staged).barrier())

  # This step's cumulative log-decays and betas (every thread holds all 16)
  gs = [g[head, base+t].load() for t in range(T)]
  if not cumulative:
    for t in range(1, T): gs[t] = gs[t-1]+gs[t]
  bs = [beta[head, base+t].load() for t in range(T)]
  def pick(values: list[UOp], t: UOp) -> UOp: return sum((t.eq(i).where(x, zero) for i, x in enumerate(values)), zero)

  # Decayed Gram matrices, a block's once: kk[t, s] = β_t (γ_t/γ_s) k_t·k_s for s < t, qk[t, s] = (γ_t/γ_s) q_t·k_s for s <= t
  pairs = [((tid+128*m) // T, (tid+128*m) % T) for m in range(2)]
  dots = UOp.placeholder((4,), F32, slot=2, addrspace=AddrSpace.REG)
  dots = dots.after(kq).after(UOp.group(*(dots.after(kq)[i].store(zero) for i in range(4))))
  c = UOp.range(D//4, 3, axis_type=AxisType.LOOP)
  acc = dots.after(c)
  terms = []
  for m in range(4):
    (t, s), x = pairs[m % 2], m // 2
    a, b = load4(kq[t+T*x, c*4]), load4(kq[s, c*4])
    terms.append(acc[m].store(acc[m].load()+sum((a[i]*b[i] for i in range(1, 4)), a[0]*b[0])))
  dots = dots.after(UOp.group(*terms).end(c))
  grams = UOp.placeholder((2, T, T), F32, slot=3, addrspace=AddrSpace.LOCAL)
  entries = []
  for m in range(4):
    (t, s), x = pairs[m % 2], m // 2
    factor = (pick(gs, t)-pick(gs, s)).exp()
    if x == 0: entries.append(grams[0, t, s].store((s < t).where(pick(bs, t)*factor*dots[m].load(), zero)))
    else: entries.append(grams[1, t, s].store((s <= t).where(factor*dots[m].load(), zero)))
  grams = grams.after(UOp.group(*entries).barrier())

  # Projections of S0's rows on this step's K and Q rows: thread (rp, tq) takes value rows 2rp, 2rp+1 x kq rows
  # 4tq..4tq+3, 32 FMAs per six 16-byte loads
  rp, tq = tid % 16, tid // 16
  proj = UOp.placeholder((8,), F32, slot=6, addrspace=AddrSpace.REG)
  proj = proj.after(grams).after(UOp.group(*(proj.after(grams)[i].store(zero) for i in range(8))))
  c4 = UOp.range(D//4, 2, axis_type=AxisType.LOOP)
  srows = [load4(current.after(grams)[2*rp+r, c4*4]) for r in range(2)]
  krows = [load4(kq[4*tq+i, c4*4]) for i in range(4)]
  acc = proj.after(c4)
  proj = proj.after(UOp.group(*(acc[r*4+i].store(acc[r*4+i].load()+sum((srows[r][j]*krows[i][j] for j in range(1, 4)), srows[r][0]*krows[i][0]))
                                for r in range(2) for i in range(4))).end(c4))
  pk = UOp.placeholder((R, 2*T+1), F32, slot=5, addrspace=AddrSpace.LOCAL)  # value row x (K's 16, Q's 16), +1: no conflicts
  pk = pk.after(UOp.group(*(pk[2*rp+r, 4*tq+i].store(proj[r*4+i].load()) for r in range(2) for i in range(4))).barrier())
  ks, qs = [pk[lane, t].load() for t in range(T)], [pk[lane, T+t].load() for t in range(T)]

  # The substitution (I + A) u = rhs (A = kk, strictly lower) for every value row at once: T = (I + A)^-1, a column a
  # lane of warp 0, then M = qk T, so a row's u = T rhs and outputs γ qs + M rhs are independent sums, not a chain.
  inverse = UOp.placeholder((2, T, T), F32, slot=9, addrspace=AddrSpace.LOCAL)  # T, then M
  column = []  # T[t, j] for j = lane: 1 on the diagonal, 0 above it, -Σ_s A[t, s] T[s, j] below
  for t in range(T):
    total = lane.eq(t).where(UOp.const(1.0, F32), zero)
    for s in range(t): total = total-grams[0, t, s].load()*column[s]
    column.append(total)
  inv_stores = [inverse[0, t, lane.valid(warp.eq(0) & (lane < T))].store(column[t]) for t in range(T)]
  inverse = inverse.after(UOp.group(*inv_stores).barrier())
  mixed = []
  for m in range(2):
    n = tid+128*m
    t, j = n // T, n % T
    total = sum((grams[1, t, s].load()*inverse[0, s, j].load() for s in range(1, T)), grams[1, t, 0].load()*inverse[0, 0, j].load())
    mixed.append(inverse[1, t, j].store(total))
  inverse = inverse.after(UOp.group(*mixed).barrier())
  rhs = [bs[t]*(v[head, base+t, row].load()-gs[t].exp()*ks[t]) for t in range(T)]
  us = [sum((inverse[0, t, s].load()*rhs[s] for s in range(t)), rhs[t]) for t in range(T)]
  outs = [sum((inverse[1, t, s].load()*rhs[s] for s in range(t+1)), gs[t].exp()*qs[t]) for t in range(T)]
  first = warp.eq(0)
  stores = [o[base+t, head, row.valid(first)].store(outs[t]) for t in range(T)]
  if not advance: stores += [u[head, base+t, row.valid(first)].store(us[t]) for t in range(T)]
  else:
    # S_T = γ_T S0 + Σ_t (γ_T/γ_t) u_t k_tᵀ: the rows' weights through shared memory, then thread (rq, cq) updates
    # value rows 4rq..4rq+3 x key columns 8cq..8cq+7, a rank-1 update per K row from three 16-byte loads
    wsh = UOp.placeholder((T, R), F32, slot=7, addrspace=AddrSpace.LOCAL)
    wsh = wsh.after(UOp.group(*(wsh[t, lane.valid(first)].store((gs[T-1]-gs[t]).exp()*us[t]) for t in range(T))).barrier())
    rq, cq = tid // 16, tid % 16
    now = current.after(wsh)
    decay = gs[T-1].exp()
    tile = [[x*decay for half in range(2) for x in load4(now[4*rq+r, 8*cq+4*half])] for r in range(4)]
    for t in range(T):
      w, kr = load4(wsh[t, 4*rq]), load4(kq[t, 8*cq])+load4(kq[t, 8*cq+4])
      tile = [[tile[r][c]+w[r]*kr[c] for c in range(8)] for r in range(4)]
    stores += [now[4*rq+r, 8*cq+c].store(tile[r][c]) for r in range(4) for c in range(8)]
  # Every thread must be done with this step's shared memory before the next one is staged over it.
  done = UOp.group(*stores).barrier().end(step)
  final = [done]
  if advance:
    for n in range(R*D//4//128):
      r, c = n*4+warp, lane*4
      final += [state[head, row0+r, c+i].store(ssh.after(done)[r, c+i].load()) for i in range(4)]
  name = "gdn_prefill" if advance else "gdn_chunk"
  return UOp.group(*final).end(item, warp, lane).sink(arg=KernelInfo(name=name, opts_to_apply=()))

def _gated_norm(o: Tensor, gate: Tensor, weight: Tensor, eps: float) -> Tensor:
  """RMSNorm over each head's 128 values, times the SiLU gate: z (T, H, 128) FP16. A separate step since a head's
  values are spread over four blocks of the chunk kernel."""
  # contiguous: fused across a prompt chunk's four scans and their concatenation, tinygrad builds an invalid kernel
  return (o * (o.square().mean(-1, keepdim=True) + eps).rsqrt() * weight.float() * gate.float().silu()).cast(dtypes.float16).contiguous()

def gdn_chunk(q: Tensor, k: Tensor, v: Tensor, beta: Tensor, g: Tensor, state: Tensor, gate: Tensor,
              weight: Tensor, keep: Tensor, eps: float) -> tuple[Tensor, Tensor]:
  """q, k, v: (H, T, 128) FP32 (q pre-scaled); beta, g: (H, T); state: (H, 128, 128)
  as (V, K); gate: (T, H, 128); weight: (128,); keep: (1,) 0/1 multiplier on S0.
  Returns z (T, H, 128) FP16 and u (H, T, 128) FP32."""
  heads, rows, dim = q.shape
  if dim != 128 or rows != 16: raise ValueError("fused GDN chunk requires 16 rows of 128-dimensional heads")
  o = Tensor.empty(rows, heads, dim, dtype=dtypes.float32, device=q.device)
  u = Tensor.empty(heads, rows, dim, dtype=dtypes.float32, device=q.device)
  inputs = [t.float().contiguous() for t in (q, k, v, beta, g, state)] + [keep.float().contiguous()]
  o, u = Tensor.custom_kernel(o, u, *inputs, fxn=functools.partial(_gdn_chunk_kernel, heads=heads, rows=rows))[:2]
  return _gated_norm(o, gate, weight, eps), u

@functools.cache
def _gates_kernel(out: UOp, x: UOp, w: UOp, *, rows: int, outs: int, width: int) -> UOp:
  # One warp per (row, output): lanes take 8 consecutive inputs a step (16-byte loads), FP32 sums, one shuffle reduction.
  warps = 4
  item = UOp.range(rows*outs//warps, 0, axis_type=AxisType.GLOBAL)
  warp = UOp.range(warps, 1, axis_type=AxisType.LOCAL)
  lane = UOp.range(32, -1, axis_type=AxisType.WARP)
  index = item*warps + warp
  row, o = index // outs, index % outs
  total = None
  for step in range(width // 256):
    xs, ws = _load_lanes(x[row, step*256 + lane*8], 8), _load_lanes(w[o, step*256 + lane*8], 8)
    for i in range(8):
      term = xs[i].float() * ws[i].float()
      total = term if total is None else total + term
  total = warp_reduce(total, full_wave=True)
  return out[row, o.valid(lane.eq(0))].store(total.cast(dtypes.float16)).end(item, warp, lane).sink(
    arg=KernelInfo(name=f"gdn_gates_{outs}", opts_to_apply=()))

def gdn_gates(x: Tensor, weight: Tensor) -> Tensor:
  """x (rows, width) @ weight (outs, width)ᵀ for the GDN β and α projections (width → 2 × heads, both weights
  stacked), FP16 out like upstream's FP16 Linear. tinygrad's generic matmul spends ~90 us a call on these
  (2,560 → 32, 64 rows; RTX 5060); one warp per output takes a few."""
  rows, width = x.shape
  outs = weight.shape[0]
  if width % 256 or outs % 4: raise ValueError("gdn_gates needs a width multiple of 256 and outputs a multiple of 4")
  out = Tensor.empty(rows, outs, dtype=dtypes.float16, device=x.device)
  return Tensor.custom_kernel(out, x.half().contiguous(), weight.half().contiguous(),
                              fxn=functools.partial(_gates_kernel, rows=rows, outs=outs, width=width))[0]

def gdn_prefill(q: Tensor, k: Tensor, v: Tensor, beta: Tensor, log_decay: Tensor, state: Tensor, gate: Tensor,
                weight: Tensor, keep: Tensor, eps: float) -> Tensor:
  """The closed form for prompt prefill, any multiple of 16 rows in one launch: every row is kept, so `state`
  (H, 128, 128) FP32 is advanced past them all in place. log_decay (H, T) is per row (summed in the kernel within
  each 16 rows). Padding rows (β = 0, log-decay 0) leave the state unchanged. Returns z (T, H, 128) FP16."""
  heads, rows, dim = q.shape
  if dim != 128 or rows % 16: raise ValueError("fused GDN prefill requires a multiple of 16 rows of 128-dimensional heads")
  o = Tensor.empty(rows, heads, dim, dtype=dtypes.float32, device=q.device)
  u = Tensor.empty(1, dtype=dtypes.float32, device=q.device)  # not written: prefill keeps every row
  inputs = [t.float().contiguous() for t in (q, k, v, beta, log_decay)] + [state, keep.float().contiguous()]
  o = Tensor.custom_kernel(o, u, *inputs, fxn=functools.partial(_gdn_chunk_kernel, heads=heads, rows=rows, advance=True,
                                                               cumulative=False))[0]
  return _gated_norm(o, gate, weight, eps)

@functools.cache
def _gdn_decode_kernel(z: UOp, state: UOp, qkv: UOp, conv: UOp, conv_w: UOp, x: UOp, w_beta: UOp, w_alpha: UOp, dt_bias: UOp,
                       a: UOp, gate: UOp, weight: UOp, keep: UOp, *, heads: int, kheads: int, width: int, eps: float,
                       gate_offset: int = 0) -> UOp:
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
  store = z[head, col].store((out*scale*_silu(gate[gate_offset + head*128 + col].load().float())).cast(dtypes.float16))
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
               a: Tensor, state: Tensor, gate: Tensor | None, weight: Tensor, keep: Tensor, eps: float, kheads: int) -> Tensor:
  """One gated delta-net decode row from the raw qkv projection, in two kernels.

  qkv: (C,) projection output; conv_state: (3, C) FP32 realized; conv_w: (C, 4); x: (width,) layer
  input; w_beta, w_alpha: (H, width) projection weights (beta = sigmoid(x·w_beta), log decay =
  softplus(x·w_alpha + dt_bias)*a); state: (H, 128, 128) FP32 as (V, K), realized; gate: (H, 128);
  weight: (128,); keep: (1,) 0/1 multiplier on the old conv and recurrent state. With gate None, qkv is the merged
  projection's output (C + H x 128,): the qkv channels, then the gate, read from the one buffer.
  The recurrent and conv states are updated in place; returns z (H, 128) FP16."""
  heads, width = w_beta.shape[0], x.shape[0]
  channels = qkv.shape[0] - (heads*128 if gate is None else 0)
  gate_offset, gate = (channels, qkv) if gate is None else (0, gate.reshape(-1))
  if (state.shape != (heads, 128, 128) or conv_state.shape != (3, channels) or channels != 2*kheads*128 + heads*128
      or channels % (heads*128) or width % 128 or w_beta.shape != (heads, width) or w_alpha.shape != (heads, width)):
    raise ValueError("fused GDN decode requires 128-dimensional heads and a 4-tap conv")
  qkv, keep = qkv.float().contiguous(), keep.float().contiguous()
  z = Tensor.empty(heads, 128, dtype=dtypes.float16, device=qkv.device)
  z = Tensor.custom_kernel(z, state, qkv, conv_state, conv_w.float().contiguous(), x.contiguous(), w_beta.contiguous(),
                           w_alpha.contiguous(), dt_bias.contiguous(), a.contiguous(), gate.contiguous(),
                           weight.float().contiguous(), keep,
                           fxn=functools.partial(_gdn_decode_kernel, heads=heads, kheads=kheads, width=width, eps=eps,
                                                 gate_offset=gate_offset))[0]
  out = Tensor.empty(heads, 128, dtype=dtypes.float16, device=qkv.device)
  return Tensor.custom_kernel(out, conv_state, qkv, keep, z, fxn=functools.partial(
    _conv_shift_kernel, channels=channels, size=heads*128))[0]

@functools.cache
def _commit_kernel(state: UOp, conv: UOp, g: UOp, k: UOp, u: UOp, window: UOp, params: UOp, *, heads: int, kd: int, vd: int,
                   rows: int, conv_rows: int, channels: int, tile: int) -> UOp:
  """A verified block's first `count` rows (params[0]) into one GDN layer's state, in place:
  S = exp(g_last) S + sum_{j<count} exp(g_last - g_j) u_j k_j^T (S zeroed first when params[1], the block at position 0),
  and the conv state = the window's rows count .. count + conv_rows - 1. A CTA per head and `tile` value rows, a
  thread per key column: each thread weighs its 16 key values once and updates its column of the tile's rows."""
  tiles = vd // tile
  block = UOp.range(heads * tiles, 0, axis_type=AxisType.GLOBAL)
  lane = UOp.range(kd, 1, axis_type=AxisType.LOCAL)
  head, first_row = block // tiles, (block % tiles) * tile
  count, initial = params[0].load(), params[1].load().ne(0)
  last = g[0, head, (count - 1).maximum(0)].load()
  weighted = [(UOp.const(j, dtypes.int32) < count).where((last - g[0, head, j].load()).exp(), UOp.const(0.0, dtypes.float32))
              * k[0, head, j, lane].load() for j in range(rows)]
  decay = last.exp()
  stores = []
  for r in range(tile):
    row = first_row + r
    previous = initial.where(UOp.const(0.0, dtypes.float32), state[0, head, row, lane].load() * decay)
    total = functools.reduce(lambda a, b: a + b, (u[0, head, j, row].load() * weighted[j] for j in range(rows)), previous)
    stores.append(state[0, head, row, lane].store(total))
  # the conv state: one value per thread of the first CTAs
  index = block * kd + lane
  size = conv_rows * channels
  clamped = index.minimum(size - 1)
  shifted = window[0, count + clamped // channels, clamped % channels].load()
  stores.append(conv[0, clamped // channels, (clamped % channels).valid(index < size)].store(shifted))
  return UOp.group(*stores).end(block, lane).sink(arg=KernelInfo(name="gdn_commit", opts_to_apply=()))

def gdn_commit(state: Tensor, conv: Tensor, g: Tensor, k: Tensor, u: Tensor, window: Tensor, params: Tensor, *, tile: int = 4):
  """One GDN layer's commit of a verified block (see _commit_kernel), in place: returns (state, conv) after it. RTX 5060,
  Bonsai-4B's 32 x 128 x 128 FP32 state: 13.1 us with 4 value rows a CTA (16: 16.2, 1: 22.8); reading and writing
  the state takes ~9.4 us at the card's bandwidth."""
  _, heads, vd, kd = state.shape
  rows, conv_rows, channels = g.shape[-1], conv.shape[1], conv.shape[2]
  if vd % tile or heads * (vd // tile) * kd < conv_rows * channels:
    raise ValueError("gdn_commit: tile must divide the value rows and its threads cover the conv state")
  fxn = functools.partial(_commit_kernel, heads=heads, kd=kd, vd=vd, rows=rows, conv_rows=conv_rows, channels=channels, tile=tile)
  return tuple(Tensor.custom_kernel(state, conv, g, k, u, window, params, fxn=fxn)[:2])
