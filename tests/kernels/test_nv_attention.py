import os

import pytest
from tinygrad import Context, Tensor, TinyJit, UOp, dtypes, function

from infurnace.kernels.attention import decode_attention, fused_decode_attention, AttentionError

NV = pytest.mark.skipif(not os.environ.get("INFURNACE_TEST_NV"), reason="set INFURNACE_TEST_NV=1 on an NV GPU")

@pytest.mark.parametrize("active", [0, 17, True])
def test_invalid_active_length(active):
  with pytest.raises(AttentionError, match="active length"):
    decode_attention(Tensor.empty(1, 4, 1, 128, device="CPU"),
                     Tensor.empty(2, 1, 1, 16, 128, dtype=dtypes.float16, device="CPU"), active)

def test_symbolic_length_bounds():
  with pytest.raises(AttentionError, match="symbolic active length"):
    decode_attention(Tensor.empty(1, 4, 1, 128, device="CPU"),
                     Tensor.empty(2, 1, 1, 16, 128, dtype=dtypes.float16, device="CPU"), UOp.variable("length", 0, 16))

def test_attention_requires_nv_quant_before_loading():
  from infurnace.models.qwen3 import Qwen3Model
  with pytest.raises(ValueError, match="requires NV quantized"):
    Qwen3Model.from_gguf("does-not-exist.gguf", nv_attention=True)

def test_fused_graph_has_shared_kv_and_bounded_partials():
  from tinygrad.dtype import AddrSpace
  from tinygrad.uop.ops import Ops
  from infurnace.kernels.attention import _fused_partial, _fused_merge, _kernel_scalar
  out = UOp.placeholder((32, 32, 130), dtypes.float32, slot=0)
  q = UOp.placeholder((1, 32, 1, 128), dtypes.float32, slot=1)
  kv = UOp.placeholder((2, 1, 8, 4096, 128), dtypes.float16, slot=2)
  active = _kernel_scalar(UOp.variable("graph_active", 1, 4096))
  graph = _fused_partial(out, q, kv, active=active, heads=32, capacity=4096, partition=128)
  ops = graph.toposort()
  shared = [u for u in ops if u.op is Ops.BUFFER and u.addrspace == AddrSpace.LOCAL]
  assert len(shared) == 1 and shared[0].numel()*shared[0].dtype.itemsize == 8192
  assert sum(u.op is Ops.BARRIER for u in ops) == 2
  assert len([u for u in ops if u.op is Ops.PARAM and u.addrspace == AddrSpace.GLOBAL]) == 3
  result = UOp.placeholder((1, 32, 1, 128), dtypes.float32, slot=0)
  partial = UOp.placeholder((32, 32, 130), dtypes.float32, slot=1)
  merged = _fused_merge(result, partial, heads=32, live=(active+127)//128)
  assert not any(u.op is Ops.BUFFER and u.addrspace == AddrSpace.LOCAL for u in merged.toposort())

@pytest.mark.parametrize("partition", [0, 128.0, True])
def test_fused_rejects_invalid_partitions(monkeypatch, partition):
  import infurnace.kernels.attention as attention
  monkeypatch.setattr(attention, "nv_attention_supported", lambda device: True)
  with pytest.raises(AttentionError, match="partition must"):
    fused_decode_attention(Tensor.empty(1, 4, 1, 128, device="CPU"),
                           Tensor.empty(2, 1, 1, 128, 128, dtype=dtypes.float16, device="CPU"), 1, partition=partition)

def test_fused_rejects_noncontiguous_cache(monkeypatch):
  import infurnace.kernels.attention as attention
  monkeypatch.setattr(attention, "nv_attention_supported", lambda device: True)
  kv = Tensor.empty(2, 1, 1, 128, 128, dtype=dtypes.float16, device="CPU").transpose(-1, -2)
  with pytest.raises(AttentionError, match="contiguous native KV"):
    fused_decode_attention(Tensor.empty(1, 4, 1, 128, device="CPU"), kv, 1)

def reference(q, cache, length):
  # Snapshot mutable/JIT-reused buffers before building CPU reference operations.
  # Lazy to()/slice graphs can reuse stale host copies after a cache write.
  q = Tensor(q.tolist(), device="CPU", dtype=dtypes.float64)
  snapshot = Tensor(cache.tolist(), device="CPU", dtype=dtypes.float64)
  k = snapshot[0, :, :, :length, :]
  v = snapshot[1, :, :, :length, :]
  return q.scaled_dot_product_attention(k, v, enable_gqa=True)

@NV
@pytest.mark.parametrize("capacity,length", [(33, 1), (33, 31), (33, 32), (33, 33), (256, 127),
                                            (256, 128), (256, 129), (512, 512), (2048, 2048)])
@pytest.mark.parametrize("fused", [False, True])
def test_attention_reference_and_poisoned_future(capacity, length, fused):
  q = Tensor.randn(1, 8, 1, 128, dtype=dtypes.float32, device="NV").realize()
  cache = Tensor.randn(2, 1, 2, capacity, 128, device="NV").half()
  future = (Tensor.arange(capacity).to("NV") >= length).reshape(1, 1, 1, capacity, 1)
  cache = future.where(float("nan"), cache).half().realize()
  expected = reference(q, cache, length)
  result = (fused_decode_attention if fused else decode_attention)(q, cache, length).to("CPU").double()
  assert result.shape == q.shape
  assert (result-expected).abs().max().item() < 3e-5

@NV
@pytest.mark.parametrize("scale", [0.0, 1.0, 20.0])
@pytest.mark.parametrize("fused", [False, True])
def test_dynamic_jit_cache_writes_and_reset(scale, fused):
  capacity = 256
  query = (Tensor.randn(1, 8, 1, 128, device="NV")*scale).realize()
  cache = Tensor.zeros(2, 1, 2, capacity, 128, dtype=dtypes.float16, device="NV").realize()
  nt = UOp.variable("attention_length", 1, capacity)
  @function(allow_implicit=True)
  def run(q, kv, update, length):
    # Same ordered write/read dependency as upstream; the latest V is distinctive.
    store = kv[:, :, :, length-1:length, :].uop.store(update.uop)
    return (fused_decode_attention if fused else decode_attention)(q, Tensor(kv.uop.after(store)), length)
  jit = TinyJit(run)
  for iteration, length in enumerate((1, 31, 32, 33, 127, 128, 129, 256, 1, 33)):
    # Returning to length 1 exercises reset semantics without reading stale suffixes.
    update = (Tensor.randn(2, 1, 2, 1, 128, device="NV")*(iteration+1)).half().realize()
    actual = jit(query, cache, update, nt.bind(length)).realize()
    expected = reference(query, cache, length)
    snapshot = Tensor(actual.tolist(), device="CPU", dtype=dtypes.float64)
    assert (snapshot-expected).abs().max().item() < 2e-4, (iteration, length, scale, fused)

@NV
@pytest.mark.parametrize("partition", [64, 128, 256])
def test_fused_standalone_binding_gqa_partitions_and_reset(partition):
  q = Tensor.zeros(1, 8, 1, 128, device="NV").realize()
  keys = Tensor.zeros(1, 2, 513, 128, device="NV")
  labels = Tensor([1.0, -3.0], device="NV").reshape(1, 2, 1, 1).expand(1, 2, 513, 128)
  cache = Tensor.stack(keys, labels).half()
  future = (Tensor.arange(513).to("NV") >= 257).reshape(1, 1, 1, 513, 1)
  cache = future.where(float("nan"), cache).half().realize()
  sp = UOp.variable("standalone_position", 0, 512)
  def run(q, kv, position): return fused_decode_attention(q, kv, position+1, partition=partition)
  jit = TinyJit(run)
  for count in (1, 32, 129, 256, 257, 1):
    result = jit(q, cache, sp.bind(count-1)).tolist()[0]
    for head, vector in enumerate(result):
      assert vector[0] == pytest.approx([1.0 if head < 4 else -3.0]*128, abs=1e-6)

@NV
@pytest.mark.parametrize("fused", [False, True])
def test_decode_block_matches_upstream_and_retains_state(fused):
  from tinygrad.llm.model import TransformerBlock, TransformerConfig
  from infurnace.models.qwen3 import enable_decode_attention, _NVAttentionBlock
  config = TransformerConfig(num_blocks=1, dim=256, hidden_dim=256, n_heads=4, n_kv_heads=1,
                             vocab_size=32, norm_eps=1e-6, rope_theta=1e6,
                             head_dim=128, v_head_dim=128, rope_dim=128, qk_norm=128, max_context=64)
  with Context(DEV="NV"):
    block = TransformerBlock(config)
  x = Tensor.randn(1, 1, 256, device="NV").realize()
  block._init_state(x)
  block.cache_kv.assign(Tensor.zeros_like(block.cache_kv)).realize()
  weights = block.attn_q.weight
  cache = block.cache_kv
  # Compare the same projections/norms/RoPE/cache at each position, before JIT.
  for position in (0, 1, 31, 32):
    expected = TransformerBlock._attention(block, x, position).realize()
    if position == 0:
      assert enable_decode_attention([block], fused=fused) == 1
      assert type(block) is _NVAttentionBlock
      assert block.attn_q.weight is weights and block.cache_kv is cache
    actual = block._attention(x, position).realize()
    assert (actual-expected).abs().max().item() < 2e-4
  # Multi-token prefill stays entirely upstream, including causal masking.
  prefill = Tensor.randn(1, 3, 256, device="NV").realize()
  expected = TransformerBlock._attention(block, prefill, 0).realize()
  actual = block._attention(prefill, 0).realize()
  assert (actual-expected).abs().max().item() < 1e-6

@pytest.mark.skipif(not (os.environ.get("INFURNACE_TEST_MODEL") and
                        (os.environ.get("INFURNACE_TEST_NV_ATTENTION") or os.environ.get("INFURNACE_TEST_NV_FUSED_ATTENTION"))),
                    reason="set INFURNACE_TEST_MODEL and INFURNACE_TEST_NV_ATTENTION=1 for real-model logits")
def test_local_model_teacher_forced_attention_logits():
  from infurnace.models.qwen3 import Qwen3Model
  from infurnace.tokenizer import Tokenizer
  from infurnace.models.qwen3 import enable_decode_attention
  model = Qwen3Model.from_gguf(os.environ["INFURNACE_TEST_MODEL"], max_context=2048,
                              device="NV", nv_quant=True, nv_prefill=True)
  transformer = model.transformer
  ids = Tokenizer.from_gguf(model.metadata).encode(" ".join(map(str, range(40))))[:40]
  # Reuse upstream embedding, blocks, norms and head; omit only the sampler to
  # expose teacher-forced logits. Both paths use the same weights and token IDs.
  def logits(tokens, position):
    x = transformer.token_embd(tokens).float()
    for block in transformer.blk: x = block(x, position)
    return transformer.output(transformer.output_norm(x[:, -1:]))[:, -1, :]
  position = UOp.variable("logit_position", 0, 2047)
  def sequence():
    logits(Tensor([ids[:32]], device="NV"), 0).realize()
    jit = TinyJit(logits)
    # Copy host values, not retained Tensor views of a reused JIT output buffer.
    return [jit(Tensor([[ids[p]]], device="NV"), position.bind(p)).realize().tolist() for p in range(32, 36)]
  reference_logits = sequence()
  fused = bool(os.environ.get("INFURNACE_TEST_NV_FUSED_ATTENTION"))
  assert enable_decode_attention(transformer.blk, fused=fused) == model.config.block_count
  actual_logits = sequence()  # Rewrites the prefix and tail in order; no stale suffix participates.
  metrics = []
  for token, expected_values, actual_values in zip(ids[32:36], reference_logits, actual_logits):
    expected, actual = Tensor(expected_values, device="CPU"), Tensor(actual_values, device="CPU")
    error = (actual-expected).abs()
    mean, maximum = error.mean().item(), error.max().item()
    logp, logq = expected.double().log_softmax(-1), actual.double().log_softmax(-1)
    kl = (logp.exp()*(logp-logq)).sum().item()
    nll_delta = abs((logp[0, token]-logq[0, token]).item())
    same_top1 = actual.argmax(-1).item() == expected.argmax(-1).item()
    print(f"attention logits: mean_abs={mean:.6g}, max_abs={maximum:.6g}, KL={kl:.6g}, NLL_delta={nll_delta:.6g}, top1={same_top1}")
    metrics.append((mean, maximum, kl, nll_delta, same_top1))
  # Parallel FP32 sums can alter downstream Q8 rounding. Absolute logits alone
  # are not the quality oracle: constrain distribution/NLL drift and top token.
  # These smoke bounds (0.005 nats KL, 0.1 nats NLL) are NOT held-out perplexity
  # or universal quality parity. Keep this path opt-in pending a larger corpus.
  assert all(mean < 0.1 and maximum < 0.5 and kl < 0.005 and nll < 0.1 and top1
             for mean, maximum, kl, nll, top1 in metrics), metrics
