"""Independent scalar BF16 reference for the pinned AngelSlim architecture.

No EagleDraft math helpers and no ML runtime besides the implementation under
 test. This checks two causal positions, shifted embeddings, split-half RoPE,
 feature projection/residual, separate norms and the reduced vocabulary head.
"""
import math
import struct

from tinygrad import Context, Tensor, dtypes

from infurnace.eagle import EagleDraft

def bf(value):
  bits = struct.unpack('<I', struct.pack('<f', value))[0]
  return struct.unpack('<f', struct.pack('<I', ((bits+0x7fff+((bits >> 16) & 1)) >> 16) << 16))[0]

def test_scalar_checkpoint_architecture():
  dim, width = 4, 8
  config = dict(hidden_size=dim, intermediate_size=width, num_attention_heads=1, num_key_value_heads=1,
                head_dim=dim, vocab_size=8, draft_vocab_size=4, rms_norm_eps=1e-6, rope_theta=1e6,
                num_hidden_layers=1, hidden_act='silu', max_position_embeddings=8)
  shapes = {'fc': (dim, 3*dim), 'lm_head': (4, dim),
            'midlayer.self_attn.q_proj': (dim, 2*dim), 'midlayer.self_attn.k_proj': (dim, 2*dim),
            'midlayer.self_attn.v_proj': (dim, 2*dim), 'midlayer.self_attn.o_proj': (dim, dim),
            'midlayer.mlp.gate_proj': (width, dim), 'midlayer.mlp.up_proj': (width, dim),
            'midlayer.mlp.down_proj': (dim, width)}
  matrices = {name: [[((i*7+j*3+n)%13-6)/32 for j in range(cols)] for i in range(rows)]
              for n, (name, (rows, cols)) in enumerate(shapes.items())}
  norms = {name: [1.0, 0.75, 1.25, 0.875] for name in
           ('midlayer.hidden_norm', 'midlayer.input_layernorm', 'midlayer.post_attention_layernorm', 'norm')}
  embedding = [[((i*5+j*3)%11-5)/16 for j in range(dim)] for i in range(8)]
  features = [[((i*5+j*2)%17-8)/16 for j in range(3*dim)] for i in range(2)]
  ids = [2, 6]

  def linear(x, name): return [bf(sum(a*b for a, b in zip(row, x))) for row in matrices[name]]
  def norm(x, name):
    scale = 1/math.sqrt(sum(a*a for a in x)/len(x)+config['rms_norm_eps'])
    return [bf(bf(a*scale)*w) for a, w in zip(x, norms[name])]
  def rope(x, pos):
    c = [bf(math.cos(pos/config['rope_theta']**(2*i/dim))) for i in range(dim//2)]
    s = [bf(math.sin(pos/config['rope_theta']**(2*i/dim))) for i in range(dim//2)]
    return ([bf(bf(x[i]*c[i])-bf(x[i+2]*s[i])) for i in range(2)] +
            [bf(bf(x[i+2]*c[i])+bf(x[i]*s[i])) for i in range(2)])
  expected, keys, values = [], [], []
  for pos, (feature, token) in enumerate(zip(features, ids)):
    h = linear(feature, 'fc')
    joined = norm(embedding[token], 'midlayer.input_layernorm')+norm(h, 'midlayer.hidden_norm')
    q = rope(linear(joined, 'midlayer.self_attn.q_proj'), pos)
    keys.append(rope(linear(joined, 'midlayer.self_attn.k_proj'), pos))
    values.append(linear(joined, 'midlayer.self_attn.v_proj'))
    scores = [bf(bf(sum(a*b for a, b in zip(q, k)))/math.sqrt(dim)) for k in keys]
    exps = [math.exp(s-max(scores)) for s in scores]
    probs = [bf(e/sum(exps)) for e in exps]
    attended = [bf(sum(p*v[j] for p, v in zip(probs, values))) for j in range(dim)]
    h = [bf(a+b) for a, b in zip(h, linear(attended, 'midlayer.self_attn.o_proj'))]
    n = norm(h, 'midlayer.post_attention_layernorm')
    gate, up = linear(n, 'midlayer.mlp.gate_proj'), linear(n, 'midlayer.mlp.up_proj')
    gated = [bf(bf(a/(1+math.exp(-a)))*b) for a, b in zip(gate, up)]
    expected.append([bf(a+b) for a, b in zip(h, linear(gated, 'midlayer.mlp.down_proj'))])

  with Context(DEV='CPU'):
    weights = {name+'.weight': Tensor(value, device='CPU').bfloat16() for name, value in (matrices | norms).items()}
    weights.update(d2t=Tensor([0, 1, 2, 3], dtype=dtypes.int64),
                   t2d=Tensor([True, False]*4, dtype=dtypes.bool))
    emb = Tensor(embedding, device='CPU').bfloat16().realize()
    draft = EagleDraft(config, weights, lambda ids: emb[ids], capacity=8, device='CPU')
    actual = draft._forward(Tensor([features], device='CPU'), Tensor([ids], device='CPU'), 0).float().tolist()[0]
    # One BF16 rounding unit allows FP32 reduction order differences; gross
    # architecture/layout mistakes are not hidden behind a loose relative bound.
    assert max(abs(a-b) for row, ref in zip(actual, expected) for a, b in zip(row, ref)) <= 0.001953125
    logits = linear(norm(expected[-1], 'norm'), 'lm_head')
    assert draft._head(Tensor([[expected[-1]]], device='CPU').bfloat16()).item() == 2*max(range(4), key=logits.__getitem__)
