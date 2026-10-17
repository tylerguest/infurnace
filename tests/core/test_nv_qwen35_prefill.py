"""Real-model checks for multirow prompt prefill (Qwen3.5-4B Q4_K_M, NV).

Marked nv (INFURNACE_TEST_NV=1); needs the model file below; loads ~3 GB once.
"""
from pathlib import Path
import pytest

TARGET = Path('models/qwen35-validation/Qwen3.5-4B-Q4_K_M.gguf')
pytestmark = [pytest.mark.nv, pytest.mark.skipif(not TARGET.is_file(), reason='needs the Qwen3.5-4B target under models/')]

@pytest.fixture(scope='module')
def loaded():
  from infurnace.loader import load_model
  model, tokenizer = load_model(str(TARGET), 1024, 'NV')
  assert model._multirow_prefill
  # Outside warmup, short prompts never capture the 16-row graph; warm up so these tests exercise it.
  model.warmup()
  assert model.transformer.prefill_jit.captured is not None
  return model, tokenizer

def ids(tokenizer, text):
  return tokenizer.encode(tokenizer.render_chat([{'role': 'user', 'content': text}], enable_thinking=False))

def run(model, tokens, count=16, *, multirow=True, reuse_cache=False):
  model._multirow_prefill = multirow
  try: return list(model.generate(list(tokens), count, reuse_cache=reuse_cache))
  finally: model._multirow_prefill = True

def last_logits(model, tokens, chunks):
  from tinygrad import Context, Tensor
  t, position = model.transformer, 0
  with Context(DEV='NV'):
    for rows in chunks:
      x = t.token_embd(Tensor([tokens[position:position + rows]], dtype='int32')).float()
      for block in t.blk: x = block(x, position)
      x, position = x.realize(), position + rows
    return t.output(t.output_norm(x[:, -1:]))[0, -1].float().realize()

def test_multirow_logits_match_one_row(loaded):
  model, tokenizer = loaded
  tokens = ids(tokenizer, 'List five facts about the planet Mars, one per line, each with a short explanation of why it matters.')
  one = last_logits(model, tokens, [1] * len(tokens))
  multi = last_logits(model, tokens, [16, 16] + [1] * (len(tokens) - 32))  # two chunks through the 16-row kernel
  # One-row projections quantize activations to 8 bits; the 16-row GEMM multiplies in FP16. The results
  # agree to ~1%, so near-ties can flip: this prompt's first token is "1" vs "M", 0.04 logits apart.
  assert ((one - multi).abs().max() / one.abs().max()).item() < 0.03
  assert one.argmax().item() in multi.topk(2)[1].tolist() and multi.argmax().item() in one.topk(2)[1].tolist()

def test_multirow_prompt_matches_one_row_prompt(loaded):
  model, tokenizer = loaded
  # Several full 16-row chunks plus a partial one, through generate.
  tokens = ids(tokenizer, 'Explain how a refrigerator keeps food cold, in simple terms suitable for a ten year old child, using an analogy.')
  assert len(tokens) > 2 * 16 and (len(tokens) - 1) % 16
  multi, one = run(model, tokens), run(model, tokens, multirow=False)
  if multi != one:
    # The two paths' logits differ by ~1% (see above), so they may flip a near-tie, but only that: the two
    # choices must be the top two candidates, closer than that difference (here " heat" vs " vacuum").
    i = next(k for k in range(len(multi)) if multi[k] != one[k])
    n = len(tokens) + i
    values, top = (x.tolist() for x in last_logits(model, tokens + multi[:i], [16] * (n // 16) + [1] * (n % 16)).topk(2))
    assert {multi[i], one[i]} == set(top) and values[0] - values[1] < 0.01 * abs(values[0]), (i, top, values)

def test_extending_the_cached_prefix_prefills_only_new_tokens(loaded):
  model, tokenizer = loaded
  first = ids(tokenizer, 'Name three rivers in Europe.')
  reply = run(model, first, 8)
  # Continue the same conversation: the cached prefix (prompt + reply) is reused, and the
  # new tokens go through multirow prefill starting mid-sequence.
  extended = first + reply + tokenizer.encode('\nNow name three in Asia, with their countries.')
  assert run(model, first, 8) == reply  # leaves prompt + reply as the cached prefix
  assert model.transformer.get_start_pos(extended) > 0
  assert run(model, extended, reuse_cache=True) == run(model, extended, multirow=False)
