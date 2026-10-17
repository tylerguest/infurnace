"""Real-model checks for multirow prompt prefill (Qwen3.5-4B Q4_K_M, NV).

Marked nv (INFURNACE_TEST_NV=1); needs the model file below; loads ~3 GB once.
"""
from pathlib import Path
import pytest
from infurnace.models.qwen35 import PROMPT_ROWS

TARGET = Path('models/qwen35-validation/Qwen3.5-4B-Q4_K_M.gguf')
pytestmark = [pytest.mark.nv, pytest.mark.skipif(not TARGET.is_file(), reason='needs the Qwen3.5-4B target under models/')]

@pytest.fixture(scope='module')
def loaded():
  from infurnace.loader import load_model
  model, tokenizer = load_model(str(TARGET), 1024, 'NV')
  assert model._multirow_prefill
  # Outside warmup, short prompts never capture the 16-row graph; warm up so these tests exercise it.
  model.warmup()
  assert model._prompt_jit.captured is not None
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

def one_row_logits(model, tokens):
  """Logits after `tokens`, read one row at a time through a captured step (eager steps take seconds each)."""
  from tinygrad import Context, Tensor, TinyJit, UOp, dtypes
  t = model.transformer
  if not hasattr(model, '_test_step'):
    def step(token, position):
      x = t.token_embd(token).float()
      for block in t.blk: x = block(x, position)
      return t.output(t.output_norm(x)).reshape(-1).float().contiguous()
    model._test_step = TinyJit(step)
  position = UOp.variable('start_pos', 0, t.max_context - 1)
  with Context(DEV='NV'):
    model.reset()
    for i, token in enumerate(tokens): logits = model._test_step(Tensor([[token]], dtype=dtypes.int32), position.bind(i))
    return Tensor(logits.tolist())

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
  # Several full prompt chunks plus a partial one, through generate.
  question = 'Explain how a refrigerator keeps food cold, in simple terms suitable for a ten year old child, using an analogy.'
  notes = ' '.join(f'Note {i}: the compressor, the coils and the refrigerant each play a part.' for i in range(9))
  tokens = ids(tokenizer, notes + '\n\n' + question)
  assert len(tokens) > 2 * PROMPT_ROWS and (len(tokens) - 1) % PROMPT_ROWS  # full prompt chunks and a partial one
  multi, one = run(model, tokens), run(model, tokens, multirow=False)
  if multi != one:
    # The two paths' logits differ by up to 3% (see above), so they may flip a near-tie, but only that: the two
    # choices must be the top two candidates, closer than that difference (here 0.19 on 13.9).
    i = next(k for k in range(len(multi)) if multi[k] != one[k])
    values, top = (x.tolist() for x in one_row_logits(model, tokens + multi[:i]).topk(2))
    assert {multi[i], one[i]} == set(top) and values[0] - values[1] < 0.03 * abs(values[0]), (i, top, values)

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
