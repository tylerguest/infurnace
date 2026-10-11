"""Multirow DFlash verification: one target forward over the whole draft block.

Every candidate row goes through each target layer together, so projection
weights are read once per block instead of once per row (`_NVLinear` uses its
packed multirow GEMM). Full-attention layers write K/V for all rows; rows past
the accepted prefix stay outside the authorized causal prefix and are
overwritten later. Gated delta-net layers do not advance state during the
forward. The block uses the chunked (WY/UT) closed form of the gated delta rule:
with u_t = β_t(v_t - α_t S_{t-1} k_t), S_t = α_t S_{t-1} + u_t k_tᵀ, so
(I + A) U = β(V - γ·K S0ᵀ) with strictly lower A_tj = β_t (γ_t/γ_j) k_t·k_j,
solved exactly by a nilpotent product. Outputs need no per-row loop, and
`commit` forms S_m = γ_m S0 + Σ_{j<m} (γ_m/γ_j) u_j k_jᵀ for the accepted
prefix from stored U, K and log-decays; no per-row state copies are kept.

Decisions are greedy argmax over multirow logits. They match one-row decoding
up to floating-point reduction order; no RNG is consumed.
"""
import functools
from tinygrad import Context, Tensor, TinyJit, UOp, dtypes
from tinygrad.llm.model import GatedDeltaNetBlock, apply_rope
from infurnace.dflash.target import TargetObserver
from infurnace.kernels.gdn import gdn_chunk
from infurnace.kernels.embedding import packed_gather


def _raw(tensor):
  return Tensor(tensor.uop.buf_uop).reshape(tensor.shape)


def _stored(destination, source):
  # Lazy in-place store into an owned buffer; reading the result orders the
  # write inside the same graph (no extra schedule break).
  return Tensor(destination.uop.after(destination.uop.store(source.cast(destination.dtype).uop)))


def _decays(g: Tensor, rows: int) -> tuple[Tensor, Tensor]:
  """exp(g_t - g_j) for j<t (strict) and j<=t (inclusive); zero elsewhere."""
  index = Tensor.arange(rows)
  diff = g.unsqueeze(-1) - g.unsqueeze(-2)
  strict = (index.reshape(rows, 1) > index.reshape(1, rows)).where(diff, float('-inf')).exp()
  inclusive = (index.reshape(rows, 1) >= index.reshape(1, rows)).where(diff, float('-inf')).exp()
  return strict, inclusive


def _unit_lower_inverse(A: Tensor, rows: int) -> Tensor:
  """(I + A)^-1 for strictly lower-triangular A: A is nilpotent, so
  (I - A)(I + A^2)(I + A^4)... is exact once the power reaches `rows`."""
  eye = Tensor.eye(rows)
  result, power, n = eye - A, A, 2
  while n < rows:
    power = power @ power
    result = result @ (eye + power)
    n *= 2
  return result


class _Operands:
  """Per-GDN-layer scratch for one verification block."""
  def __init__(self, block: GatedDeltaNetBlock, rows: int, device: str):
    heads, kd, vd = block.num_v_heads, block.head_k_dim, block.head_v_dim
    self.window = Tensor.empty(1, block.ssm_conv_kernel - 1 + rows, block.conv_channels, device=device).realize()
    self.k = Tensor.empty(1, heads, rows, kd, device=device).realize()
    self.u = Tensor.empty(1, heads, rows, vd, device=device).realize()
    self.g = Tensor.empty(1, heads, rows, device=device).realize()


class ParallelVerifier:
  def __init__(self, observer: TargetObserver, rows: int):
    if type(rows) is not int or not 2 <= rows <= 16: raise ValueError('parallel verifier supports 2..16 rows')
    model = observer.model
    if any(getattr(b, 'ssm_g_a', None) is not None for b in model.transformer.blk): raise ValueError('KDA blocks are unsupported')
    if model.config.context_length < rows + 1: raise ValueError('context too small for parallel verification')
    self.observer, self.rows, self.device = observer, rows, model.device
    self._operands: dict[int, _Operands] = {}
    self._position = UOp.variable('start_pos', 0, model.config.context_length - rows)
    # The fused round commits the previous block and verifies the next one in
    # one graph, so the committed block's start needs its own variable.
    self._commit_position = UOp.variable('dflash_commit_pos', 0, model.config.context_length - rows)
    self._count = UOp.variable('dflash_commit_rows', 1, rows)
    self._forward_jit, self._commit_jit = TinyJit(self._forward_store), TinyJit(self._commit)
    self._round_jit = None
    self._fused = rows == 16 and self.device.split(':')[0] == 'NV' and all(
      b.head_k_dim == 128 and b.head_v_dim == 128 for b in model.transformer.blk if isinstance(b, GatedDeltaNetBlock))

  def fits(self, position: int) -> bool:
    return 0 <= position <= self.observer.model.config.context_length - self.rows

  def _gdn(self, index: int, block: GatedDeltaNetBlock, x: Tensor, start_pos: UOp) -> Tensor:
    # Mirrors upstream GatedDeltaNetBlock._attention (non-KDA, generic scan) for
    # a concrete row count, but leaves conv/recurrent state untouched.
    ops = self._operands[index]
    B, T, _ = x.shape
    H, K, V = block.num_v_heads, block.head_k_dim, block.head_v_dim
    initial = Tensor(start_pos).eq(0)
    x = x.half()
    out_gate = block.attn_gate(x).reshape(B, T, H, V)
    beta = block.ssm_beta(x).sigmoid().reshape(B, T, H)
    log_alpha = (block.ssm_alpha(x).float() + block.ssm_dt['bias']).softplus().reshape(B, T, H, -1) * block.ssm_a.reshape(H, -1)
    conv_state = initial.where(0, _raw(block.conv_state))
    window = conv_state.float().cat(block.attn_qkv(x).float(), dim=1)
    window = _stored(ops.window, window)
    conv_out = functools.reduce(lambda a, b: a + b,
      (window[:, i:i+T] * block.ssm_conv1d['weight'][:, i] for i in range(block.ssm_conv_kernel))).silu()
    q, k, v = conv_out.split([block.q_dim, block.q_dim, block.conv_channels - 2*block.q_dim], dim=-1)
    q, k = (z.reshape(B, T, block.num_k_heads, K).normalize(dim=-1, eps=1e-6).repeat(1, 1, H // block.num_k_heads, 1) for z in (q, k))
    v = v.reshape(B, T, H, V)
    q, k, v, beta = (z.transpose(1, 2).float() for z in (q, k, v, beta))
    q = q * K**-0.5
    # Chunked closed form of the gated delta rule (WY/UT form; see module doc).
    g = _stored(ops.g, log_alpha.transpose(1, 2)[..., 0].cumsum(-1))                 # (B,H,T) log decay
    k = _stored(ops.k, k)
    if self._fused:
      # One NV kernel per layer: closed-form chunk, RMSNorm and gate (kernels/gdn.py).
      keep = (1 - initial.float()).reshape(1)
      z, u = gdn_chunk(q[0], k[0], v[0], beta[0], g[0], _raw(block.recurrent_state).reshape(H, V, K), out_gate[0],
                       block.ssm_norm.weight, keep, block.ssm_norm.eps)
      u = _stored(ops.u, u.unsqueeze(0))
      return block.ssm_out(Tensor(z.uop.after(u.uop)).reshape(B, T, -1))
    state = initial.where(0, _raw(block.recurrent_state).float())                       # S0 (B,H,V,K)
    strict, inclusive = _decays(g, T)
    A = beta.unsqueeze(-1) * strict * (k @ k.transpose(-1, -2))
    rhs = beta.unsqueeze(-1) * (v - g.exp().unsqueeze(-1) * (k @ state.transpose(-1, -2)))
    u = _stored(ops.u, _unit_lower_inverse(A, T) @ rhs)                                # (B,H,T,V)
    out = g.exp().unsqueeze(-1) * (q @ state.transpose(-1, -2)) + (inclusive * (q @ k.transpose(-1, -2))) @ u
    core = out.transpose(1, 2)                                                          # (B,T,H,V)
    z = (block.ssm_norm(core) * out_gate.silu()).cast(x.dtype).contiguous()
    return block.ssm_out(z.reshape(B, T, -1))

  def _attention(self, block, x: Tensor, start_pos: UOp) -> Tensor:
    # Upstream TransformerBlock._attention, but attending over the fixed-length
    # cache with a causal mask: symbolic `start_pos+T` key lengths compile to
    # slow kernels, and masked rows contribute exact zeros.
    c, (B, T, _) = block.config, x.shape
    q, k, v = block.attn_q(x), block.attn_k(x), block.attn_v(x)
    if c.qk_norm and c.qk_norm != c.head_dim: q, k = block.attn_q_norm(q), block.attn_k_norm(k)
    if c.attn_output_gate:
      qg = q.reshape(B, T, c.n_heads, 2, c.head_dim)
      q, gate = qg[:, :, :, 0, :], qg[:, :, :, 1, :].reshape(B, T, c.n_heads * c.head_dim)
    q = q.reshape(B, T, c.n_heads, c.head_dim).transpose(1, 2)
    k = k.reshape(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
    v = v.reshape(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
    if c.qk_norm == c.head_dim: q, k = block.attn_q_norm(q), block.attn_k_norm(k)
    freqs = block.freqs_cis[start_pos:start_pos + T]
    q = apply_rope(q[..., :c.rope_dim], freqs).cat(q[..., c.rope_dim:], dim=-1)
    k = apply_rope(k[..., :c.rope_dim], freqs).cat(k[..., c.rope_dim:], dim=-1)
    store = block.cache_kv[:, :, :, start_pos:start_pos + T, :].uop.store(Tensor.stack(k, v).cast(block.cache_kv.dtype).uop)
    cache = Tensor(block.cache_kv.uop.after(store))
    length = c.max_context
    visible = Tensor.arange(length).reshape(1, length) <= (Tensor.arange(T).reshape(T, 1) + Tensor(start_pos))
    mask = visible.where(0.0, float('-inf')).cast(x.dtype).reshape(1, 1, T, length)
    attn = q.scaled_dot_product_attention(cache[0], cache[1], attn_mask=mask, enable_gqa=True).transpose(1, 2).reshape(B, T, -1)
    return block.attn_output(attn * gate.sigmoid() if c.attn_output_gate else attn)

  def _forward(self, x: Tensor, start_pos: UOp):
    target, layers = self.observer.model.transformer, self.observer.layers
    features = []
    for index, block in enumerate(target.blk):
      if index in layers: features.append(x)
      block._init_state(x)
      norm = block.attn_norm(x)
      h = x + (self._gdn(index, block, norm, start_pos) if isinstance(block, GatedDeltaNetBlock) else self._attention(block, norm, start_pos))
      x = (h + block._feed_forward(block.ffn_norm(h))).contiguous()
    logits = target.output(target.output_norm(x))
    return logits.argmax(-1).cast(dtypes.int32).contiguous(), features[0].cat(*features[1:], dim=-1).contiguous()

  def _forward_store(self, x: Tensor, start_pos: UOp):
    decisions, features = self._forward(x, start_pos)
    decisions, features = _stored(self._decisions, decisions), _stored(self._features, features)
    Tensor.realize(decisions, features)
    return decisions, features

  def _gather(self, ids: Tensor) -> Tensor:
    head = self.observer.model.transformer.output
    return packed_gather(ids, head.packed, width=head.in_features, ggml_type=head.ggml_type, q6_aligned=head.q6_aligned,
                         q6_swizzled=head.q6_swizzled, q4_swizzled=head.q4_swizzled)

  def _pick(self, count: UOp, *, mask: int):
    # Anchor = the held block's last accepted decision, then noise embeddings.
    index = Tensor.arange(self.rows)
    anchor = (_raw(self._decisions)[0] * (index == Tensor(count) - 1)).sum().cast(dtypes.int32).reshape(1)
    anchor = _stored(self._anchor, anchor).realize()
    masks = Tensor.full((self.rows - 1,), mask, dtype=dtypes.int32)
    noise_ids = _stored(self._noise_ids, anchor.cat(masks)).realize()
    return _stored(self._noise, self._gather(noise_ids).reshape(1, self.rows, -1)).realize()

  def _ids(self, proposals: Tensor):
    proposals = _stored(self._proposals, proposals.reshape(self.rows - 1).cast(dtypes.int32)).realize()
    verify_ids = _stored(self._verify_ids, _raw(self._anchor).cat(proposals)).realize()
    return _stored(self._embeddings, self._gather(verify_ids).reshape(1, self.rows, -1)).realize()

  def _pack(self):
    readback = _raw(self._decisions).reshape(self.rows).cat(_raw(self._proposals), _raw(self._anchor))
    return _stored(self._readback, readback).realize()

  def fused_eligible(self, draft) -> bool:
    head = self.observer.model.transformer.output
    return (self.device.split(':')[0] == 'NV' and getattr(head, 'packed', None) is not None and self.rows == draft.config.block_size)

  def fused_round(self, draft, count: int, anchor: int) -> tuple[list[int], list[int]]:
    """Commit `count` rows of the pending block (whose last decision is `anchor`),
    draft from them, verify the next block. Returns (decisions, proposals) with
    one device readback; the new block stays pending."""
    observer = self.observer
    previous, position = observer.position, observer.position + count
    if type(count) is not int or not 1 <= count <= self.rows or not self.fits(position): raise ValueError('invalid fused round')
    try:
      with Context(DEV=self.device):
        # The round is a chain of small captured graphs launched back to back
        # with one readback at the end. KNOWN ISSUE: back-to-back fused rounds
        # can stall the GPU inside a captured NV graph (timeline stuck one batch
        # short, no MMU fault; masked by profiling timestamps or JIT=2; also seen
        # with newer tinygrad). Fused rounds are therefore opt-in.
        if self._round_jit is None:
          self._round_jit = (TinyJit(functools.partial(self._pick, mask=draft.config.mask_token)), TinyJit(self._ids),
                             TinyJit(self._pack))
          self._draft_position = UOp.variable('dflash_position', 1, draft.capacity - self.rows)
        pick, ids, pack = self._round_jit
        self._commit_jit(self._count.bind(count), self._position.bind(previous))
        draft._absorb_jit(_raw(self._features), draft._start.bind(previous))
        noise = pick(self._count.bind(count))
        if self.rows - 1 not in draft._jits: draft._jits[self.rows - 1] = TinyJit(draft._forward)
        proposals = draft._jits[self.rows - 1](noise, self._draft_position.bind(position))
        embeddings = ids(proposals)
        self._forward_jit(embeddings, self._position.bind(position))
        pack()
        values = [int(v) for v in _raw(self._readback).tolist()]
    except BaseException:
      observer.close()
      raise
    observer.position, observer.next_decision = position, anchor
    draft.length = position  # transient noise K/V never authorizes prefix reuse
    self.last_device_anchor = values[-1]
    return values[:self.rows], values[self.rows:2*self.rows-1]

  def _commit(self, count: UOp, start_pos: UOp):
    # S_m = γ_m S0 + Σ_{j<m} (γ_m/γ_j) u_j k_jᵀ for the m accepted rows.
    rows, initial, outputs = Tensor(count), Tensor(start_pos).eq(0), []
    index = Tensor.arange(self.rows)
    for i, ops in self._operands.items():
      block = self.observer.model.transformer.blk[i]
      g, k, u = _raw(ops.g), _raw(ops.k), _raw(ops.u)
      last = (g * (index == rows - 1)).sum(-1, keepdim=True)                            # g_{m-1} (B,H,1)
      weight = (index < rows).where(last - g, float('-inf')).exp()                      # (B,H,T)
      state = initial.where(0, _raw(block.recurrent_state).float())
      state = last.exp().unsqueeze(-1) * state + (u * weight.unsqueeze(-1)).transpose(-1, -2) @ k
      recurrent, conv = _raw(block.recurrent_state), _raw(block.conv_state)
      window = _raw(ops.window)[:, count:count + block.ssm_conv_kernel - 1]
      outputs.append(Tensor(recurrent.uop.after(recurrent.uop.store(state.cast(recurrent.dtype).uop))))
      outputs.append(Tensor(conv.uop.after(conv.uop.store(window.cast(conv.dtype).uop))))
    Tensor.realize(*outputs)
    return tuple(outputs)

  def verify(self, inputs: list[int]) -> tuple[list[int], Tensor]:
    """Run the target over `inputs` (anchor + proposals, padded to `rows`).

    Returns per-row greedy decisions and per-row feature taps. Target state is
    not advanced until `commit`.
    """
    if inputs and inputs[0] != self.observer.next_decision: raise ValueError('verification must start with the known target anchor')
    return self.run(inputs)

  def prefill(self, tokens: list[int]) -> tuple[list[int], Tensor]:
    """Process up to `rows` known prompt tokens; caller commits all of them."""
    return self.run(tokens)

  def run(self, inputs: list[int]) -> tuple[list[int], Tensor]:
    observer = self.observer
    if not observer.active: raise RuntimeError('use verifier inside the target observer')
    if not isinstance(inputs, list) or not 1 <= len(inputs) <= self.rows: raise ValueError('invalid verification block')
    if any(type(t) is not int or not 0 <= t < observer.model.config.vocab_size for t in inputs): raise ValueError('invalid target token')
    if not self.fits(observer.position): raise ValueError('block does not fit the remaining context')
    padded = inputs + [inputs[-1]] * (self.rows - len(inputs))
    try:
      with Context(DEV=self.device):
        if not self._operands:
          x = Tensor.empty(1, 1, observer.model.config.embedding_length, device=self.device)
          for index, block in enumerate(observer.model.transformer.blk):
            if isinstance(block, GatedDeltaNetBlock):
              block._init_state(x)
              self._operands[index] = _Operands(block, self.rows, self.device)
          width = observer.model.config.embedding_length * len(observer.layers)
          self._decisions = Tensor.empty(1, self.rows, dtype=dtypes.int32, device=self.device).realize()
          self._features = Tensor.empty(1, self.rows, width, device=self.device).realize()
          self._anchor = Tensor.empty(1, dtype=dtypes.int32, device=self.device).realize()
          self._proposals = Tensor.empty(self.rows - 1, dtype=dtypes.int32, device=self.device).realize()
          self._readback = Tensor.empty(2 * self.rows, dtype=dtypes.int32, device=self.device).realize()
          dim = observer.model.config.embedding_length
          self._noise_ids, self._verify_ids = (Tensor.empty(self.rows, dtype=dtypes.int32, device=self.device).realize() for _ in range(2))
          self._noise, self._embeddings = (Tensor.empty(1, self.rows, dim, device=self.device).realize() for _ in range(2))
        embeddings = observer.model.embed(padded).contiguous().realize()
        self._forward_jit(embeddings, self._position.bind(observer.position))
        decisions = [int(v) for v in _raw(self._decisions).tolist()[0]]
        features = _raw(self._features)
      return decisions, features
    except BaseException:
      observer.close()
      raise

  def commit(self, count: int, decisions: list[int]):
    observer = self.observer
    if type(count) is not int or not 1 <= count <= self.rows: raise ValueError('invalid commit row count')
    try:
      with Context(DEV=self.device): self._commit_jit(self._count.bind(count), self._position.bind(observer.position))
    except BaseException:
      observer.close()
      raise
    observer.position += count
    observer.next_decision = decisions[count - 1]
