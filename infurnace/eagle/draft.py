"""Checkpoint-matched EAGLE-3 chain drafter, entirely on tinygrad.

The Qwen3 AngelSlim head uses explicit head_dim, separate embedding/feature
normalization, no Q/K norm, and offset-valued reduced-vocabulary mappings.
"""
import json
import math
from pathlib import Path

from tinygrad import Context, Tensor, TinyJit, UOp, dtypes
from tinygrad.nn.state import safe_load
from infurnace.eagle.io import mapped_tensor, mapped_output, upload_tokens
from infurnace.kernels.dense_gemv import bf16_gemv


class EagleDraft:
  def __init__(self, config: dict, weights: dict[str, Tensor], embedding, *, capacity: int, device: str):
    integer_fields = ('hidden_size', 'intermediate_size', 'num_attention_heads', 'num_key_value_heads',
                      'head_dim', 'vocab_size', 'draft_vocab_size', 'max_position_embeddings')
    if not isinstance(config, dict) or any(type(config.get(k)) is not int or config[k] <= 0 for k in integer_fields):
      raise ValueError('invalid EAGLE dimensions')
    if any(type(config.get(k)) not in (int, float) or not math.isfinite(config[k]) or config[k] <= 0
           for k in ('rms_norm_eps', 'rope_theta')):
      raise ValueError('invalid EAGLE normalization or RoPE configuration')
    self.dim = config['hidden_size']
    self.hidden = config['intermediate_size']
    self.heads, self.kv_heads = config['num_attention_heads'], config['num_key_value_heads']
    self.head_dim = config['head_dim']
    self.vocab, self.draft_vocab = config['vocab_size'], config['draft_vocab_size']
    self.eps, self.theta = config['rms_norm_eps'], config['rope_theta']
    if (type(config.get('num_hidden_layers')) is not int or config['num_hidden_layers'] != 1
        or config.get('rope_scaling') is not None or config.get('attention_bias', False)
        or config.get('mlp_bias', False) or config.get('use_sliding_window', False)
        or config.get('sliding_window') is not None or config.get('pretraining_tp', 1) != 1
        or self.draft_vocab > self.vocab
        or config.get('hidden_act') != 'silu' or self.heads%self.kv_heads or self.head_dim%2
        or type(capacity) is not int or not 1 <= capacity <= config['max_position_embeddings']):
      raise ValueError('unsupported EAGLE-3 architecture or capacity')
    self.device, self.capacity, self.embedding = device, capacity, embedding
    self.fast_gemv = device.split(':')[0] == 'NV'
    shapes = {'fc.weight': (self.dim, 3*self.dim), 'lm_head.weight': (self.draft_vocab, self.dim),
      'midlayer.self_attn.q_proj.weight': (self.heads*self.head_dim, 2*self.dim),
      'midlayer.self_attn.k_proj.weight': (self.kv_heads*self.head_dim, 2*self.dim),
      'midlayer.self_attn.v_proj.weight': (self.kv_heads*self.head_dim, 2*self.dim),
      'midlayer.self_attn.o_proj.weight': (self.dim, self.heads*self.head_dim),
      'midlayer.mlp.gate_proj.weight': (self.hidden, self.dim),
      'midlayer.mlp.up_proj.weight': (self.hidden, self.dim),
      'midlayer.mlp.down_proj.weight': (self.dim, self.hidden),
      'd2t': (self.draft_vocab,), 't2d': (self.vocab,)}
    for name in ('midlayer.hidden_norm', 'midlayer.input_layernorm', 'midlayer.post_attention_layernorm', 'norm'):
      shapes[name+'.weight'] = (self.dim,)
    if set(weights) != set(shapes): raise ValueError(f'EAGLE weight keys differ: {set(weights)^set(shapes)}')
    for name, shape in shapes.items():
      dtype = dtypes.int64 if name == 'd2t' else dtypes.bool if name == 't2d' else dtypes.bfloat16
      if weights[name].shape != shape or weights[name].dtype != dtype:
        raise ValueError(f'invalid EAGLE tensor {name}: expected {shape}, {dtype}')
    offsets, membership = weights['d2t'].to('CPU').tolist(), weights['t2d'].to('CPU').tolist()
    mapped = [i+offset for i, offset in enumerate(offsets)]
    if (any(not 0 <= i < self.vocab for i in mapped) or len(set(mapped)) != self.draft_vocab
        or set(mapped) != {i for i, yes in enumerate(membership) if yes}):
      raise ValueError('invalid EAGLE vocabulary offsets/membership')
    with Context(DEV=device):
      self.weights = {name: tensor.to(device).contiguous() for name, tensor in weights.items() if name not in ('d2t', 't2d')}
      Tensor.realize(*self.weights.values())
      self.mapping = Tensor(mapped, dtype=dtypes.int32, device=device).realize()
      self.cache = Tensor.zeros(2, 1, self.kv_heads, capacity, self.head_dim, dtype=dtypes.bfloat16, device=device).realize()
      positions = Tensor.arange(capacity).to(device).float()
      inverse = self.theta**(-Tensor.arange(0, self.head_dim, 2).to(device).float()/self.head_dim)
      phase = positions[:, None]*inverse[None, :]
      self.cos, self.sin = phase.cos().bfloat16().realize(), phase.sin().bfloat16().realize()
    self._position = UOp.variable('draft_position', 0, capacity-1)
    self._decode_jit = TinyJit(self._forward)
    self._head_jit = TinyJit(self._head)
    self._chain_jits = {}
    self._observe_jits = {}
    self._observe_propose_jits = {}
    self._token_inputs = {}
    self.fast_io = False
    self._proposal_outputs = {}
    self.reset()

  @classmethod
  def from_checkpoint(cls, path: str | Path, model):
    path = Path(path)
    config = json.loads((path/'config.json').read_text())
    target = model.config
    expected = {'hidden_size': target.embedding_length, 'intermediate_size': target.feed_forward_length,
                'num_attention_heads': target.attention_head_count, 'num_key_value_heads': target.attention_head_count_kv,
                'head_dim': target.key_length, 'vocab_size': target.vocab_size}
    if any(config.get(k) != v for k, v in expected.items()): raise ValueError('EAGLE checkpoint does not match target dimensions')
    return cls(config, safe_load(path/'model.safetensors'), model.transformer.token_embd,
               capacity=target.context_length, device=model.device)

  def enable_fast_io(self):
    """Configure before first use; never replace scratch underlying live captures."""
    if self.fast_io: return
    if self.device.split(':')[0] != 'NV': raise ValueError('fast draft I/O requires NV')
    if self._token_inputs or self._observe_propose_jits or self.last_hidden is not None:
      raise ValueError('configure fast draft I/O before first use')
    self.fast_io = True

  def reset(self):
    self.length = 0
    self.last_hidden = None

  def _linear(self, x, name):
    weight = self.weights[name+'.weight']
    if self.fast_gemv and isinstance(x.numel(), int) and 1 <= x.numel()//weight.shape[1] <= 8:
      return bf16_gemv(x, weight)
    return (x @ weight.T).bfloat16()

  def _norm(self, x, name):
    normalized = (x.float()*(x.float().square().mean(-1, keepdim=True)+self.eps).rsqrt()).bfloat16()
    return (normalized*self.weights[name+'.weight']).bfloat16()

  def _rope(self, x, position, size):
    cos = self.cos[position:position+size].reshape(1, 1, size, self.head_dim//2)
    sin = self.sin[position:position+size].reshape(1, 1, size, self.head_dim//2)
    first, second = x.chunk(2, dim=-1)
    return ((first*cos).bfloat16()-(second*sin).bfloat16()).bfloat16().cat(
      ((second*cos).bfloat16()+(first*sin).bfloat16()).bfloat16(), dim=-1)

  def _forward(self, features, token_ids, position, *, last_only=False):
    size = token_ids.shape[1]
    embedded = self.embedding(token_ids).bfloat16()
    h = features.bfloat16()
    if h.shape[-1] == 3*self.dim: h = self._linear(h, 'fc')
    joined = self._norm(embedded, 'midlayer.input_layernorm').cat(self._norm(h, 'midlayer.hidden_norm'), dim=-1)
    query_size = 1 if last_only else size
    query_position = position+size-1 if last_only else position
    query_input = joined[:, -1:].contiguous() if last_only else joined
    q = self._linear(query_input, 'midlayer.self_attn.q_proj').reshape(1, query_size, self.heads, self.head_dim).transpose(1, 2)
    k = self._linear(joined, 'midlayer.self_attn.k_proj').reshape(1, size, self.kv_heads, self.head_dim).transpose(1, 2)
    v = self._linear(joined, 'midlayer.self_attn.v_proj').reshape(1, size, self.kv_heads, self.head_dim).transpose(1, 2)
    q, k = self._rope(q, query_position, query_size), self._rope(k, position, size)
    if last_only: h = h[:, -1:].contiguous()
    store = self.cache[:, :, :, position:position+size, :].uop.store(Tensor.stack(k, v).uop)
    assigned = Tensor(self.cache.uop.after(store))
    keys = assigned[0].repeat_interleave(self.heads//self.kv_heads, dim=1)
    values = assigned[1].repeat_interleave(self.heads//self.kv_heads, dim=1)
    scores = (q.matmul(keys.transpose(-2, -1), dtype=dtypes.float32).bfloat16().float()/math.sqrt(self.head_dim)).bfloat16()
    allowed = Tensor.arange(self.capacity).to(self.device)[None, :] <= Tensor.arange(query_size).to(self.device)[:, None]+query_position
    probabilities = allowed.reshape(1, 1, query_size, self.capacity).where(scores.float(), float('-inf')).softmax(-1).bfloat16()
    attended = probabilities.matmul(values, dtype=dtypes.float32).bfloat16().transpose(1, 2).reshape(1, query_size, -1)
    h = (h+self._linear(attended, 'midlayer.self_attn.o_proj')).bfloat16()
    n = self._norm(h, 'midlayer.post_attention_layernorm')
    gated = (self._linear(n, 'midlayer.mlp.gate_proj').float().silu().bfloat16()*self._linear(n, 'midlayer.mlp.up_proj')).bfloat16()
    return (h+self._linear(gated, 'midlayer.mlp.down_proj')).bfloat16().contiguous()

  def _head(self, hidden):
    reduced = self._linear(self._norm(hidden, 'norm'), 'lm_head').argmax(-1)
    return self.mapping[reduced].reshape(1, 1).contiguous()

  def _head_scored(self, hidden):
    logits = self._linear(self._norm(hidden, 'norm'), 'lm_head').float()
    reduced = logits.argmax(-1)
    best = logits.max(-1, keepdim=True)
    second = (Tensor.arange(self.draft_vocab).to(self.device) != reduced.unsqueeze(-1)).where(
      logits, float('-inf')).max(-1, keepdim=True)
    return self.mapping[reduced].reshape(1, 1).contiguous(), (best-second).reshape(1, 1).contiguous()

  def _observe(self, features, ids, position):
    # All accepted rows need K/V, but only the last row feeds proposal logits.
    # Earlier queries, attention outputs and FFNs are dead observation work.
    return self._forward(features, ids, position, last_only=True).contiguous()

  def _validate_observation(self, features, paired_ids):
    size = len(paired_ids)
    if (size <= 0 or features.shape != (1, size, 3*self.dim) or features.device != self.device
        or self.length+size > self.capacity or any(type(i) is not int or not 0 <= i < self.vocab for i in paired_ids)):
      raise ValueError('invalid EAGLE feature/token alignment or capacity')
    return size

  def _observation_inputs(self, features, paired_ids):
    size = len(paired_ids)
    if size not in self._token_inputs:
      self._token_inputs[size] = (mapped_tensor((1, size), dtypes.int32, self.device) if self.fast_io else
                                  Tensor.empty(1, size, dtype=dtypes.int32, device=self.device).realize())
    ids = self._token_inputs[size]
    upload_tokens(ids, paired_ids, mapped=self.fast_io)
    # A concrete storage view gives stable JIT metadata even for prefix aliases.
    features = features.contiguous().realize()
    return Tensor(UOp.from_buffer(features.uop.buffer, self.device)).reshape(features.shape), ids

  def observe(self, features: Tensor, paired_ids: list[int]):
    """Add actual target features paired with their following tokens, including pending root."""
    size = self._validate_observation(features, paired_ids)
    with Context(DEV=self.device):
      features, ids = self._observation_inputs(features, paired_ids)
      if size not in self._observe_jits: self._observe_jits[size] = TinyJit(self._observe)
      position = UOp.variable('draft_observe_position', 0, self.capacity-size).bind(self.length)
      hidden = self._observe_jits[size](features, ids, position)
      self.last_hidden = Tensor(hidden.uop.buf_uop).reshape(hidden.shape)
    self.length += size

  def _observe_and_chain(self, features, ids, position, *, count, scored=False):
    hidden = self._observe(features, ids, position).realize()
    tokens = (self._chain_scored if scored else self._chain)(hidden, position+ids.shape[1], count=count)
    if self.fast_io: tokens = mapped_output(self._proposal_outputs, (ids.shape[1], count, scored), tokens)
    return hidden, tokens

  def observe_propose(self, features: Tensor, paired_ids: list[int], count: int) -> list[int]:
    """Observe and draft in one capture, retaining only the true observed hidden."""
    return self._observe_propose(features, paired_ids, count, scored=False)

  def observe_propose_scored(self, features: Tensor, paired_ids: list[int], count: int) -> tuple[list[int], list[float]]:
    """Return identical proposals plus top-one/top-two logit margins."""
    return self._observe_propose(features, paired_ids, count, scored=True)

  def _observe_propose(self, features, paired_ids, count, *, scored):
    size = self._validate_observation(features, paired_ids)
    if type(count) is not int or not 1 <= count <= 7 or self.length+size+count-1 > self.capacity:
      raise ValueError('invalid fused draft length')
    with Context(DEV=self.device):
      from functools import partial
      features, ids = self._observation_inputs(features, paired_ids)
      key = (size, count, scored)
      if key not in self._observe_propose_jits:
        self._observe_propose_jits[key] = TinyJit(partial(self._observe_and_chain, count=count, scored=scored))
      position = UOp.variable('draft_fused_position', 0, self.capacity-size-count+1).bind(self.length)
      hidden, tokens = self._observe_propose_jits[key](features, ids, position)
      self.last_hidden = Tensor(hidden.uop.buf_uop).reshape(hidden.shape)
      proposals = Tensor(tokens.uop.buf_uop).reshape(tokens.shape).tolist()[0]
    self.length += size
    return ([int(i) for i in proposals[:count]], proposals[count:]) if scored else proposals

  def _chain_scored(self, hidden, position, *, count):
    tokens, margins = [], []
    for index in range(count):
      token, margin = self._head_scored(hidden)
      Tensor.realize(token, margin)
      tokens.append(token)
      margins.append(margin)
      if index+1 < count: hidden = self._forward(hidden, token, position+index).realize()
    return tokens[0].cat(*tokens[1:], dim=1).float().cat(margins[0].cat(*margins[1:], dim=1), dim=1).contiguous()

  def _chain(self, hidden, position, *, count):
    # One capture owns all recurrent intermediates. Do not feed mutable outputs
    # between separately captured head/decode graphs or store into a JIT input.
    tokens = []
    for index in range(count):
      token = self._head(hidden).realize()
      tokens.append(token)
      if index+1 < count:
        hidden = self._forward(hidden, token, position+index).realize()
    return tokens[0].cat(*tokens[1:], dim=1).contiguous()

  def propose(self, count: int, *, fast_loop: bool = False) -> list[int]:
    if self.last_hidden is None: raise RuntimeError('observe target features before drafting')
    if type(count) is not int or count < 0 or self.length+max(0, count-1) > self.capacity:
      raise ValueError('invalid draft length')
    if not count: return []
    with Context(DEV=self.device):
      if fast_loop:
        if count not in self._chain_jits:
          from functools import partial
          self._chain_jits[count] = TinyJit(partial(self._chain, count=count))
        position = UOp.variable('draft_chain_position', 0, self.capacity-max(1, count-1))
        result = self._chain_jits[count](self.last_hidden, position.bind(min(self.length, self.capacity-1)))
        return Tensor(result.uop.buf_uop).reshape(result.shape).tolist()[0]
      hidden, tokens = self.last_hidden, []
      for index in range(count):
        token = self._head_jit(hidden).clone().realize()
        tokens.append(token)
        if index+1 < count:
          hidden = self._decode_jit(hidden, token, self._position.bind(self.length+index)).clone().realize()
      # One transfer for the entire chain; no host .item() at each draft step.
      return tokens[0].cat(*tokens[1:], dim=1).tolist()[0]
