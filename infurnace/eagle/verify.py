"""Opt-in short-chain NV Q8 verifier, sharing target weights and KV storage.

Integer dots and FP32 reductions preserve single-token arithmetic. Projection
sums are materialized before batched Q/K norms to prevent nested-reduction
reassociation. Packed GEMV CTAs visit adjacent tokens to improve cache reuse.
This is an opt-in prototype, not the ordinary CLI generation path.
"""
import copy
from dataclasses import dataclass

from tinygrad import Context, Tensor, TinyJit, UOp, function, dtypes

from infurnace.eagle.io import mapped_tensor, mapped_output, upload_tokens

from infurnace.kernels.attention import fused_chain_attention
from infurnace.kernels.quant_gemv import q8_quantize, quant_gemv
from infurnace.model import _NVAttentionBlock, _NVLinear, apply_rope
from infurnace.eagle.reference import SequentialVerifier


def norm_rows(norm, x):
  # Row-independent reduction over the hidden axis; validated against the
  # one-token path on the real target, not just a small synthetic block.
  return norm(x)


def norm_head_rows(norm, x, heads):
  # Cut the projection sum before its nested RMSNorm reduction is scheduled.
  return norm(x.contiguous().reshape(x.shape[0], x.shape[1], heads, 128).transpose(1, 2))


class _ChainBlock(_NVAttentionBlock):
  reuse_weights = False
  local_ffn_down = False
  loop_projections = False
  fused_projections = False
  batch_residuals = True
  batch_qk_norm = True
  def __call__(self, x, start_pos):
    self.cache_kv, self.freqs_cis = self.source.cache_kv, self.source.freqs_cis
    @function(precompile=True, allow_implicit=True)
    def run(x, start_pos):
      if self.fused_projections:
        # The pinned target schedules residual + partial[0] + partial[1] ...,
        # not residual + sum(partials). Carry that exact order into the CTA.
        h = self._attention(norm_rows(self.attn_norm, x), start_pos, residual=x)
        return self._feed_forward(self.ffn_norm(h), residual=h).contiguous()
      parts = self._attention(norm_rows(self.attn_norm, x), start_pos)
      if self.batch_residuals:
        h = x+parts.sum(-1).reshape(x.shape)
        feed = self._feed_forward(self.ffn_norm(h))
        return (h+feed.sum(-1).reshape(x.shape)).contiguous()
      hs = [x[:, i:i+1].contiguous()+parts[i:i+1].contiguous().sum(-1).reshape(1, 1, self.config.dim) for i in range(x.shape[1])]
      normalized = self.ffn_norm(Tensor.cat(*(h.contiguous() for h in hs), dim=1).contiguous())
      feed = self._feed_forward(normalized)
      return Tensor.cat(*((h+feed[i:i+1].contiguous().sum(-1).reshape(1, 1, self.config.dim)).contiguous()
                          for i, h in enumerate(hs)), dim=1).contiguous()
    return run(x, start_pos)

  def linear(self, x, name, q8=None, return_parts=False, residual=None):
    layer = getattr(self, name)
    result = quant_gemv(x, layer.packed, out_features=layer.out_features, in_features=layer.in_features,
      ggml_type=layer.ggml_type, q6_aligned=layer.q6_aligned, q6_swizzled=layer.q6_swizzled,
      q4_swizzled=layer.q4_swizzled, q8=q8, interleave_tokens=True, exact_rows=False, return_parts=return_parts,
      reuse_tokens=self.reuse_weights and x.shape[1] > 1,
      local_tokens=(self.local_ffn_down or self.loop_projections) and not self.reuse_weights
                   and name == 'ffn_down' and layer.ggml_type == 14 and x.shape[1] > 1,
      loop_chunks=self.loop_projections, fuse_chunks=self.fused_projections, residual=residual)
    return result if layer.bias is None else result+layer.bias

  def _attention(self, x, start_pos, residual=None):
    size = x.shape[1]
    q8 = q8_quantize(x, size, self.config.dim)
    q, k, v = (self.linear(x, name, q8) for name in ('attn_q', 'attn_k', 'attn_v'))
    if self.batch_qk_norm:
      # Materialize projection sums before RMSNorm. Without this boundary the
      # nested reduction can be rescheduled and change Q8 target arithmetic.
      q = norm_head_rows(self.attn_q_norm, q, self.config.n_heads)
      k = norm_head_rows(self.attn_k_norm, k, self.config.n_kv_heads)
      frequencies = self.freqs_cis[start_pos:start_pos+size]
      q, k = apply_rope(q, frequencies).contiguous(), apply_rope(k, frequencies).contiguous()
    else:
      queries, keys = [], []
      for i in range(size):
        qi = self.attn_q_norm(q[:, i:i+1].contiguous().reshape(1, 1, self.config.n_heads, 128).transpose(1, 2))
        ki = self.attn_k_norm(k[:, i:i+1].contiguous().reshape(1, 1, self.config.n_kv_heads, 128).transpose(1, 2))
        frequencies = self.freqs_cis[start_pos+i:start_pos+i+1]
        queries.append(apply_rope(qi, frequencies).contiguous())
        keys.append(apply_rope(ki, frequencies).contiguous())
      q, k = Tensor.cat(*queries, dim=2).contiguous(), Tensor.cat(*keys, dim=2).contiguous()
    v = v.reshape(1, size, self.config.n_kv_heads, 128).transpose(1, 2)
    store = self.cache_kv[:, :, :, start_pos:start_pos+size, :].uop.store(Tensor.stack(k, v).cast(self.cache_kv.dtype).uop)
    assigned = Tensor(self.cache_kv.uop.after(store))
    attended = fused_chain_attention(q, assigned, start_pos).transpose(1, 2).reshape(1, size, -1)
    return self.linear(attended, 'attn_output', return_parts=residual is None, residual=residual)

  def _feed_forward(self, x, residual=None):
    q8 = q8_quantize(x, x.shape[1], self.config.dim)
    gated = self.linear(x, 'ffn_gate', q8).silu().contiguous()*self.linear(x, 'ffn_up', q8)
    return self.linear(gated, 'ffn_down', return_parts=residual is None, residual=residual)


@dataclass(frozen=True)
class BatchChain:
  tokens: tuple[int, ...]
  accepted_drafts: int
  features: Tensor
  next_token: int | None
  stop_reason: str


class BatchedVerifier(SequentialVerifier):
  reuse_weights = False
  def __init__(self, model, *, kernel_cache: dict | None = None, local_ffn_down: bool = False,
               loop_projections: bool = False, fuse_root_sampling: bool = False, batch_residuals: bool = True,
               zero_copy_features: bool = False, batch_qk_norm: bool = True, compact_verify: bool = False,
               fused_projections: bool = False, fast_io: bool = False, parallel_sampling: bool = False, **kwargs):
    super().__init__(model, **kwargs)
    if type(local_ffn_down) is not bool: raise ValueError('local_ffn_down must be boolean')
    self.local_ffn_down = local_ffn_down
    if type(fast_io) is not bool or type(parallel_sampling) is not bool:
      raise ValueError('fast_io and parallel_sampling must be boolean')
    if (fast_io or parallel_sampling) and model.device.split(':')[0] != 'NV':
      raise ValueError('fast EAGLE sampling/I/O requires NV')
    self.fast_io, self.parallel_sampling = fast_io, parallel_sampling
    self._decision_outputs = {}
    if type(loop_projections) is not bool: raise ValueError('loop_projections must be boolean')
    if type(fused_projections) is not bool: raise ValueError('fused_projections must be boolean')
    self.fused_projections = fused_projections
    self.loop_projections = loop_projections or fused_projections
    if type(fuse_root_sampling) is not bool or type(batch_residuals) is not bool:
      raise ValueError('fuse_root_sampling and batch_residuals must be boolean')
    # Fused sampling and zero-copy handoff passed isolated replay checks but a
    # combined long-request trial timed out. Keep both off pending attribution.
    self.fuse_root_sampling, self.batch_residuals = fuse_root_sampling, batch_residuals
    if type(zero_copy_features) is not bool: raise ValueError('zero_copy_features must be boolean')
    self.zero_copy_features = zero_copy_features
    if type(batch_qk_norm) is not bool: raise ValueError('batch_qk_norm must be boolean')
    self.batch_qk_norm = batch_qk_norm
    if type(compact_verify) is not bool: raise ValueError('compact_verify must be boolean')
    self.compact_verify = compact_verify
    if kernel_cache is not None and not isinstance(kernel_cache, dict):
      raise ValueError('kernel_cache must be a dictionary')
    self._kernel_cache = kernel_cache
    if not (isinstance(model.transformer.output, _NVLinear) and all(
      isinstance(block, _NVAttentionBlock) and block.fused_attention and all(
        isinstance(getattr(block, name), _NVLinear) for name in
        ('attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate', 'ffn_up', 'ffn_down'))
      for block in model.transformer.blk)):
      raise ValueError('batched verification requires eligible NV packed linears and fused attention')
    self.blocks = []
    for original in model.transformer.blk:
      block = copy.copy(original)
      block.__class__ = _ChainBlock
      block.source = original
      block.reuse_weights = self.reuse_weights
      block.local_ffn_down = self.local_ffn_down
      block.loop_projections = self.loop_projections
      block.fused_projections = self.fused_projections
      block.batch_residuals = self.batch_residuals
      block.batch_qk_norm = self.batch_qk_norm
      self.blocks.append(block)
    self._batch_jits = {}
    self._feature_jits = {}
    self._token_inputs = {}
    self._sampling_jits = {}
    with Context(DEV=model.device): self._temperature = Tensor([0.0], device=model.device).realize()

  def _cache_signature(self):
    target = self.model.transformer
    return (id(self.model), self.feature_layers, self.chunk_size, self.model.config.context_length,
            self.reuse_weights, self.local_ffn_down, self.loop_projections, self.fused_projections, self.fuse_root_sampling,
            self.batch_residuals, self.zero_copy_features, self.batch_qk_norm, self.compact_verify, self.fast_io, self.parallel_sampling,
            tuple(sorted(self.eos_token_ids)) if self.compact_verify else (),
            tuple((getattr(block, 'cache_kv', None).uop.buf_uop if getattr(block, 'cache_kv', None) is not None else None,
                   tuple(getattr(block, name).packed.uop for name in
                         ('attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate', 'ffn_up', 'ffn_down')),
                   tuple(getattr(block, name).weight.uop for name in
                         ('attn_norm', 'attn_q_norm', 'attn_k_norm', 'ffn_norm')),
                   getattr(block, 'freqs_cis', None).uop if getattr(block, 'freqs_cis', None) is not None else None)
                  for block in target.blk), target.output.packed.uop,
            target.output_norm.weight.uop, target.token_embd.weight.uop)

  def __enter__(self):
    super().__enter__()
    try:
      if self._kernel_cache is not None:
        if self._kernel_cache.get('signature') != self._cache_signature(): self._kernel_cache.clear()
        for name in ('_prefill_jit', '_batch_jits', '_sampling_jits', '_feature_jits'):
          if name in self._kernel_cache: setattr(self, name, self._kernel_cache[name])
      return self
    except BaseException:
      if self._kernel_cache is not None: self._kernel_cache.clear()
      self.close()
      raise

  def prefill(self, tokens):
    steps = super().prefill(tokens)
    if self._kernel_cache is not None:
      self._kernel_cache['signature'] = self._cache_signature()
      for name in ('_prefill_jit', '_batch_jits', '_sampling_jits', '_feature_jits'):
        self._kernel_cache[name] = getattr(self, name)
    return steps

  def _forward_many(self, tokens, position):
    target = self.model.transformer
    x, features = target.token_embd(tokens).float(), []
    for index, block in enumerate(self.blocks):
      if index in self.feature_layers: features.append(x)
      x = block(x, position)
    normalized = norm_rows(target.output_norm, x)
    layer = target.output
    logits = quant_gemv(normalized, layer.packed, out_features=layer.out_features, in_features=layer.in_features,
      ggml_type=layer.ggml_type, q6_aligned=layer.q6_aligned, q6_swizzled=layer.q6_swizzled,
      q4_swizzled=layer.q4_swizzled, interleave_tokens=True, exact_rows=False,
      reuse_tokens=self.reuse_weights and tokens.shape[1] > 1,
      local_tokens=self.loop_projections and not self.reuse_weights and layer.ggml_type == 14 and tokens.shape[1] > 1,
      loop_chunks=self.loop_projections, fuse_chunks=self.fused_projections)
    return logits.contiguous(), features[0].cat(*features[1:], dim=-1).contiguous()

  def _verify_forward(self, tokens, position, temperature):
    logits, features = self._forward_many(tokens, position)
    # The known root is always committed. Fuse exactly its RNG draw/sampling
    # into verification; other rows still draw only after prefix acceptance.
    decision = self._sample(logits, temperature, UOp.const(0))
    return logits, features, decision

  def _commit_forward(self, tokens, position, temperature, key, counter):
    from infurnace.eagle.sampling import sample_chain
    logits, features = self._forward_many(tokens, position)
    Tensor.realize(logits, features)
    packet = sample_chain(logits, tokens, temperature, key, counter, eos_token_ids=tuple(sorted(self.eos_token_ids)))
    # Keep the materialized projection output live through capture/replay. Not
    # returning it let capture reuse its storage while sampling views still read
    # it (three-row toy continuation changed decisions from 10 to 0).
    return packet, features, logits

  @staticmethod
  def _sample(logits, temperature, index):
    # Select inside the capture: a caller-side contiguous slice can retain a
    # capture-time buffer view and silently sample the wrong row on replay.
    row = logits[:, index:index+1, :].reshape(1, logits.shape[-1]).contiguous()
    # Draw only for committed nodes. Rejected speculative rows consume no RNG.
    return (row/temperature.maximum(1e-12)
            - (Tensor.rand_like(row).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

  def _sample_optimized(self, logits, temperature, index):
    if self.parallel_sampling:
      from infurnace.kernels.parallel_argmax import parallel_score_argmax
      row = logits[:, index:index+1, :].reshape(1, logits.shape[-1]).contiguous()
      decision = parallel_score_argmax(row, temperature, Tensor.rand_like(row))
    else:
      decision = self._sample(logits, temperature, index)
    return mapped_output(self._decision_outputs, logits.shape[1], decision) if self.fast_io else decision

  @staticmethod
  def _copy_features(features, *, count):
    return features[:, :count, :].contiguous().clone()

  def verify_chain(self, proposals: list[int], *, max_tokens: int | None = None,
                   transient_features: bool = False) -> BatchChain:
    """Verify a chain; transient features are valid only until the next round.

    The generation adapter consumes these immediately in draft.observe before
    running verification again. Default snapshots remain independent.
    """
    self._require_active()
    if not self._prefilled: raise RuntimeError('prefill a prompt first')
    if self._next_token is None: raise RuntimeError('verification session has finished')
    values = self._validate_tokens(proposals)
    if len(values) > 7: raise ValueError('batched verification supports at most seven draft descendants')
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens <= 0): raise ValueError('max_tokens must be positive')
    size = min(len(values)+1, self.model.config.context_length-len(self._tokens),
               max_tokens if max_tokens is not None else len(values)+1)
    candidates = [self._next_token]+list(values[:size-1])
    try:
      with Context(DEV=self.model.device):
        position = UOp.variable('verify_position', 0, self.model.config.context_length-size).bind(len(self._tokens))
        if size not in self._token_inputs:
          self._token_inputs[size] = (mapped_tensor((1, size), dtypes.int32, self.model.device) if self.fast_io else
                                      Tensor.empty(1, size, dtype=dtypes.int32, device=self.model.device).realize())
        inputs = self._token_inputs[size]
        # Each prior round samples its committed root via item(), completing
        # target reads before this dedicated input buffer can be overwritten.
        upload_tokens(inputs, candidates, mapped=self.fast_io)
        if size not in self._batch_jits:
          self._batch_jits[size] = TinyJit(self._commit_forward if self.compact_verify else
                                         self._verify_forward if self.fuse_root_sampling else self._forward_many)
        committed_count, decisions = None, None
        if self.compact_verify:
          device = self.model.device
          if device not in Tensor._device_seeds: Tensor._next_counter(device, 0)
          key, counter = Tensor._device_seeds[device], Tensor._device_rng_counters[device]
          Tensor.realize(key, counter)
          counter = Tensor(counter.uop.buf_uop).reshape(counter.shape)
          packet, features, _ = self._batch_jits[size](inputs, position, self._temperature, key, counter)
          # The assigned counter depends on GPU commitment and therefore the
          # entire target graph. Keep its storage, not that executed graph, or
          # the next RNG preparation walks all 36 layers again every round.
          Tensor._device_rng_counters[device] = Tensor(counter.uop.buf_uop).reshape(counter.shape)
          decisions = Tensor(packet.uop.buf_uop).reshape(packet.shape).tolist()[0]
          committed_count = decisions[-1]
          expected_count = 1
          while (expected_count < size and candidates[expected_count] == decisions[expected_count-1]
                 and candidates[expected_count-1] not in self.eos_token_ids): expected_count += 1
          if committed_count != expected_count: raise RuntimeError('invalid device commitment packet')
          logits, root_decision = None, None
        elif self.fuse_root_sampling:
          logits, features, root_decision = self._batch_jits[size](inputs, position, self._temperature)
        else:
          logits, features = self._batch_jits[size](inputs, position)
          root_decision = None
        # TinyJit has already executed and realized both outputs.
        # Outputs are concrete contiguous buffers. Drop the already-executed
        # target graph before host-driven sampling/snapshot scheduling; walking
        # all 36 blocks again for each tiny slice dominated verifier overhead.
        if logits is not None: logits = Tensor(logits.uop.buf_uop).reshape(logits.shape)
        features = Tensor(features.uop.buf_uop).reshape(features.shape)
        if not self.compact_verify and size not in self._sampling_jits:
          self._sampling_jits[size] = TinyJit(self._sample_optimized if self.fast_io or self.parallel_sampling else self._sample)
        row_index = UOp.variable('verify_row', 0, size-1)
        committed, reason = [], 'proposals'
        for index, token in enumerate(candidates):
          if (committed_count is not None and index >= committed_count) or (index and token != self._next_token):
            reason = 'mismatch'
            break
          if self.compact_verify:
            self._next_token = decisions[index]
          else:
            decision = (root_decision if index == 0 and root_decision is not None else
                        self._sampling_jits[size](logits, self._temperature, row_index.bind(index)))
            self._next_token = int(Tensor(decision.uop.buf_uop).reshape(decision.shape).item())
          self._tokens.append(token)
          committed.append(token)
          if token in self.eos_token_ids:
            self._next_token, reason = None, 'eos'
            break
          if len(self._tokens) == self.model.config.context_length:
            self._next_token, reason = None, 'context'
            break
        else:
          if max_tokens is not None and size == max_tokens: reason = 'limit'
        # Rejected tail KV remains physically present, but active prefix length
        # excludes it. A later round overwrites those slots before reading them.
        if transient_features and self.zero_copy_features:
          # A committed prefix is contiguous in the captured feature output.
          # Expose its storage directly, without another copy graph or kernel.
          # The draft consumes it before this target capture can run again.
          shape = (1, len(committed), features.shape[-1])
          storage = features.uop.buffer.view(len(committed)*features.shape[-1], features.dtype, 0)
          snapshot = Tensor(UOp.from_buffer(storage, self.model.device)).reshape(shape)
        elif transient_features:
          from functools import partial
          key = (size, len(committed))
          if key not in self._feature_jits:
            self._feature_jits[key] = TinyJit(partial(self._copy_features, count=len(committed)))
          snapshot = self._feature_jits[key](features)
          snapshot = Tensor(snapshot.uop.buf_uop).reshape(snapshot.shape)
        else:
          snapshot = features[:, :len(committed), :].contiguous().clone().realize()
        return BatchChain(tuple(committed), len(committed)-1, snapshot, self._next_token, reason)
    except BaseException:
      if self._kernel_cache is not None: self._kernel_cache.clear()
      self.close()
      raise
