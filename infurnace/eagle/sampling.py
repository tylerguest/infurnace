"""GPU prefix commitment with the target's exact zero-temperature RNG semantics."""
from tinygrad import Tensor, dtypes

def advance_counter(counter, amount):
  low = counter[:1]+amount.cast(dtypes.uint32)
  return low.cat(counter[1:]+(low < counter[:1]).cast(dtypes.uint32))

def score_argmax(row, temperature, random, *, materialize_scores=False):
  """Preserve the target score; optionally prevent expensive reduction recomputation."""
  if type(materialize_scores) is not bool: raise ValueError('materialize_scores must be boolean')
  scores = row/temperature.maximum(1e-12) - (random.maximum(1e-12).log().neg()).log()
  return (scores.contiguous() if materialize_scores else scores).argmax(-1, keepdim=True)

def sample_chain(logits, candidates, temperature, key, counter, *, eos_token_ids=(), materialize_scores=False):
  """Return [row decisions..., committed count]; advance RNG for committed rows only.

  Rejected rows may compute stateless draws, but never advance shared RNG state.
  Passing seed/counter as inputs also avoids capturing a stale manual_seed buffer.
  """
  if type(materialize_scores) is not bool: raise ValueError('materialize_scores must be boolean')
  size, vocabulary = candidates.shape[1], logits.shape[-1]
  if logits.dtype != dtypes.float32: raise ValueError('chain sampling requires FP32 target logits')
  start = counter.clone().realize()
  decisions = []
  for index in range(size):
    row = logits[:, index:index+1, :].reshape(1, vocabulary).contiguous()
    offset = Tensor([index*vocabulary], device=logits.device, dtype=dtypes.uint32)
    random = Tensor._rand(key, advance_counter(start, offset), row.shape, row.dtype)
    decisions.append(score_argmax(row, temperature, random, materialize_scores=materialize_scores))
  selected = decisions[0].cat(*decisions[1:], dim=1).contiguous().realize()
  alive = Tensor.ones(1, dtype=dtypes.bool, device=logits.device)
  count = Tensor.ones(1, dtype=dtypes.int32, device=logits.device)
  for index in range(1, size):
    alive = alive & candidates[:, index].eq(selected[:, index-1])
    for token in eos_token_ids: alive = alive & candidates[:, index-1].ne(token)
    count = count+alive.cast(dtypes.int32)
  # Realize the prefix result before the mutation, so replay cannot reload start
  # from an overwritten counter. The cloned start remains live through all draws.
  result = selected.cast(dtypes.int32).cat(count.reshape(1, 1), dim=1).contiguous().realize()
  counter.assign(advance_counter(start, result[:, -1].cast(dtypes.uint32)*vocabulary)).realize()
  return result
