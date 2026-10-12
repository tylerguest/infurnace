"""Independent scalar DFlash forward oracle derived from z-lab/dflash model.py.

No torch/MLX/NumPy runtime. Actual rounded weights and half-cache storage are
shared inputs, not implementation helpers. Covers nonuniform Q/K, RoPE, GQA,
context fusion, both residuals, final norm, masks and incremental cache overwrite.
"""
import math
import struct
from tinygrad import Context, Tensor, UOp
from tinygrad.nn.state import get_state_dict
from test_dflash import tiny_target, tiny_draft

def linear(x, weight): return [sum(a*b for a, b in zip(x, row)) for row in weight]
def norm(x, weight, epsilon):
  scale = 1 / math.sqrt(sum(v*v for v in x)/len(x) + epsilon)
  return [a*scale*b for a, b in zip(x, weight)]
def half(x): return struct.unpack('<e', struct.pack('<e', x))[0]
def silu(x): return x / (1 + math.exp(-x))

class ScalarDraft:
  def __init__(self, config, weights):
    self.c, self.w = config, weights
    self.cache = [{} for _ in range(config.layers)]

  def rope(self, x, position):
    mid = len(x)//2
    out = [0.]*len(x)
    for i in range(mid):
      angle = position / self.c.theta**(2*i/len(x))
      co, si = math.cos(angle), math.sin(angle)
      out[i], out[mid+i] = x[i]*co-x[mid+i]*si, x[mid+i]*co+x[i]*si
    return out

  def forward(self, features, noise, position):
    c, w = self.c, self.w
    context = [norm(linear(row, w['fc.weight']), w['enc.output_norm.weight'], c.epsilon) for row in features]
    x = noise
    for layer in range(c.layers):
      prefix = f'blk.{layer}.'
      def weight(name): return w[prefix+name+'.weight']
      normalized = [norm(row, weight('attn_norm'), c.epsilon) for row in x]
      # Context K/V never passes through the noise input-layer norm.
      for row, index in zip(context+normalized, range(position-len(context), position+len(x))):
        key, value = linear(row, weight('attn_k')), linear(row, weight('attn_v'))
        keys, values = [], []
        for head in range(c.kv_heads):
          ks = key[head*c.head_dim:(head+1)*c.head_dim]
          keys.append([half(v) for v in self.rope(norm(ks, weight('attn_k_norm'), c.epsilon), index)])
          values.append([half(v) for v in value[head*c.head_dim:(head+1)*c.head_dim]])
        self.cache[layer][index] = keys, values
      attention = []
      for row, query_position in zip(normalized, range(position, position+len(x))):
        query = linear(row, weight('attn_q'))
        # Spell out upstream's layer default, independent of layer_causal().
        causal = c.sliding[layer] if c.causal is None else c.causal
        visible = [k for k in range(position+len(x)) if (not causal or k <= query_position)
                   and (not c.sliding[layer] or query_position-k < c.window)
                   and (not c.sliding[layer] or causal or k-query_position < c.window)]
        all_heads = []
        for head in range(c.heads):
          group = head // (c.heads//c.kv_heads)
          qs = query[head*c.head_dim:(head+1)*c.head_dim]
          qs = self.rope(norm(qs, weight('attn_q_norm'), c.epsilon), query_position)
          scores = [sum(a*b for a, b in zip(qs, self.cache[layer][k][0][group]))/math.sqrt(c.head_dim) for k in visible]
          probabilities = [math.exp(s-max(scores)) for s in scores]
          total = sum(probabilities)
          all_heads.extend(sum(p*self.cache[layer][k][1][group][i] for p, k in zip(probabilities, visible))/total
                           for i in range(c.head_dim))
        attention.append(linear(all_heads, weight('attn_output')))
      x = [[a+b for a, b in zip(row, update)] for row, update in zip(x, attention)]
      next_x = []
      for row in x:
        normalized = norm(row, weight('ffn_norm'), c.epsilon)
        gate, up = linear(normalized, weight('ffn_gate')), linear(normalized, weight('ffn_up'))
        update = linear([silu(a)*b for a, b in zip(gate, up)], weight('ffn_down'))
        next_x.append([a+b for a, b in zip(row, update)])
      x = next_x
    return [norm(row, w['output_norm.weight'], c.epsilon) for row in x[1:]]

def test_nonuniform_forward_and_cache_against_upstream_scalar_contract(tmp_path):
  import pytest
  with Context(DEV='CPU'):
    model = tiny_target(tmp_path)
    draft = tiny_draft(model)
    weights = {k: v.tolist() for k, v in get_state_dict(draft.weights).items()}
    oracle = ScalarDraft(draft.config, weights)
    inputs = [(Tensor.randn(1, 2, 8).realize(), Tensor.randn(1, 4, 8).realize(), pos) for pos in (2, 4)]
    for features, noise, pos in inputs:
      expected = oracle.forward(features.tolist()[0], noise.tolist()[0], pos)
      position = UOp.variable('reference_position', 2, 12).bind(pos)
      actual = draft._observe(features, noise, position).tolist()[0]
      for got, wanted in zip(actual, expected): assert got == pytest.approx(wanted, abs=4e-5, rel=4e-5)
      head = model.transformer.output.weight.tolist()
      expected_tokens = [max(range(len(head)), key=linear(row, head).__getitem__) for row in expected]
      assert draft._observe(features, noise, position, tokens=True).tolist()[0] == expected_tokens
