"""Experimental quantization-aware draft-head calibration; never changes the target.

Run collect and fit in separate processes. NumPy is optional experiment tooling,
not an inference dependency. Teacher-forced accuracy is NOT application throughput.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from tinygrad import Context, Tensor, TinyJit, dtypes
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import safe_load, safe_save

from infurnace.cli import _build
from infurnace.eagle import EagleDraft
from infurnace.kernels.dense_gemv import bf16_gemv
from infurnace.eagle.reference import SequentialVerifier

TRAIN = [
  'Calculate 13 times 29, explaining the arithmetic.',
  'Calculate 47 times 16, explaining the arithmetic.',
  'Explain how to check whether an integer is prime.',
  'Write a Python function that returns the factorial of an integer.',
  'Explain the difference between a list and a tuple in Python.',
  'Explain why the sky appears blue.',
  'Give a step-by-step plan to make a small vegetable garden.',
  'Describe three ways to save water at home.',
]
VALIDATION = [
  'Calculate 19 times 31, explaining the arithmetic.',
  'Explain the difference between a Python list and a set.',
]


def file_hash(path):
  with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def collect(args):
  model, tokenizer = _build(args.model, args.context, 'NV')
  draft = EagleDraft.from_checkpoint(args.checkpoint, model)
  # Serialization only; keep optional NumPy initialization outside engine loading.
  import numpy as np
  mapping = draft.mapping.tolist()
  inverse = {token: index for index, token in enumerate(mapping)}

  def head_forward(hidden):
    normalized = draft._norm(hidden, 'norm')
    return normalized, draft._linear(normalized, 'lm_head')

  head_jit = TinyJit(head_forward)
  states, logits_rows, labels, splits = [], [], [], []
  captures = {}
  for prompt_index, prompt in enumerate(TRAIN + VALIDATION):
    Tensor.manual_seed(args.seed + prompt_index)
    ids = tokenizer.encode(tokenizer.render_chat([{'role': 'user', 'content': prompt}]))
    draft.reset()
    correct = count = 0
    with SequentialVerifier(model) as verifier:
      if captures:
        verifier._prefill_jit, verifier._decode_jit = captures['prefill'], captures['decode']
      step = verifier.prefill(ids)[-1]
      draft.observe(step.features, ids[1:] + [step.next_token])
      for _ in range(min(args.tokens, args.context-len(ids))):
        normalized, logits = head_jit(draft.last_hidden)
        Tensor.realize(normalized, logits)
        x = Tensor(normalized.uop.buf_uop).reshape(normalized.shape).float().numpy().reshape(-1)
        base = Tensor(logits.uop.buf_uop).reshape(logits.shape).float().numpy().reshape(-1)
        result = verifier.verify_chain([])
        if result.next_token is None: break
        gold = result.next_token
        count += 1
        correct += mapping[int(base.argmax())] == gold
        states.append(x)
        logits_rows.append(base)
        labels.append(inverse.get(gold, -1))
        splits.append(prompt_index >= len(TRAIN))
        draft.observe(result.steps[-1].features, [gold])
      captures = {'prefill': verifier._prefill_jit, 'decode': verifier._decode_jit}
    print(f'collected {prompt_index}: {count} samples, accuracy {correct/max(count, 1):.4f}', flush=True)
  if not labels: raise ValueError('no teacher samples collected')
  metadata = {
    'train_prompts': TRAIN, 'validation_prompts': VALIDATION, 'seed': args.seed,
    'target': str(args.model), 'context': args.context, 'tokens_per_prompt': args.tokens,
    'target_sha256': file_hash(args.model),
    'checkpoint_sha256': file_hash(args.checkpoint/'model.safetensors'),
  }
  args.dataset.parent.mkdir(parents=True, exist_ok=True)
  np.savez(args.dataset, hidden=np.array(states, dtype=np.float32), base=np.array(logits_rows, dtype=np.float32),
           labels=np.array(labels, dtype=np.int32), validation=np.array(splits), metadata=np.array(json.dumps(metadata)))
  print(f'saved {len(labels)} samples to {args.dataset}', flush=True)


def fit(args):
  import numpy as np
  dataset = np.load(args.dataset, allow_pickle=False)
  hidden, base, labels, validation = (dataset[key] for key in ('hidden', 'base', 'labels', 'validation'))
  metadata = json.loads(str(dataset['metadata']))
  if file_hash(args.checkpoint/'model.safetensors') != metadata['checkpoint_sha256']:
    raise ValueError('dataset/checkpoint identity mismatch')
  train_indices = np.where((~validation) & (labels >= 0))[0]
  validation_indices = np.where(validation)[0]
  if not len(train_indices) or not len(validation_indices): raise ValueError('empty training/validation split')
  Tensor.manual_seed(args.seed)
  rng = random.Random(args.seed)
  states = Tensor(hidden, device='NV').realize()
  base_logits = Tensor(base, device='NV').realize()
  targets = Tensor(labels, device='NV', dtype=dtypes.int32).realize()
  a = (Tensor.randn(args.rank, hidden.shape[-1], device='NV')*.01).realize()
  b = Tensor.zeros(base.shape[-1], args.rank, device='NV').realize()
  a.requires_grad = b.requires_grad = True
  optimizer = Adam([a, b], lr=.001)

  @TinyJit
  def update(x, original_logits, y):
    with Context(TRAINING=1):
      optimizer.zero_grad()
      logits = original_logits + (x @ a.T) @ b.T
      loss = logits.sparse_categorical_crossentropy(y) + 1e-4*(a.square().sum() + b.square().sum())
      loss.backward()
      optimizer.step()
      return loss.realize()

  def accuracy(indices):
    index = Tensor(indices, device='NV', dtype=dtypes.int32)
    predicted = (base_logits[index] + (states[index] @ a.T) @ b.T).argmax(-1)
    return (predicted == targets[index]).float().mean().item()

  initial = float(np.mean(np.argmax(base[validation_indices], axis=-1) == labels[validation_indices]))
  best, best_parameters = initial, None
  print(f'initial validation {initial:.6f}; train {len(train_indices)}, validation {len(validation_indices)}', flush=True)
  for step in range(args.steps+1):
    index = Tensor([int(rng.choice(train_indices)) for _ in range(64)], device='NV', dtype=dtypes.int32).realize()
    loss = update(states[index].contiguous().realize(), base_logits[index].contiguous().realize(),
                  targets[index].contiguous().realize())
    if step % 25 == 0 or step == args.steps:
      score = accuracy(validation_indices)
      print(f'step {step}: loss {loss.item():.6f}, validation {score:.6f}', flush=True)
      if score > best:
        best, best_parameters = score, (a.detach().clone().realize(), b.detach().clone().realize())
  if best_parameters is None:
    print('no held-out improvement; no head saved', flush=True)
    return
  a_best, b_best = best_parameters
  original_head = safe_load(args.checkpoint/'model.safetensors')['lm_head.weight'].to('NV').realize()
  merged = (original_head.float() + b_best @ a_best).bfloat16().contiguous().realize()
  predictions = []
  for start in range(0, len(validation_indices), 8):
    index = validation_indices[start:start+8]
    x = Tensor(hidden[index], device='NV').bfloat16().reshape(1, len(index), hidden.shape[-1]).realize()
    predictions.extend(bf16_gemv(x, merged).argmax(-1).tolist()[0])
  merged_accuracy = float(np.mean(np.array(predictions) == labels[validation_indices]))
  print(f'actual merged BF16 validation {merged_accuracy:.6f}', flush=True)
  if merged_accuracy <= initial:
    print('merged head failed improvement check; no head saved', flush=True)
    return
  args.head.parent.mkdir(parents=True, exist_ok=True)
  safe_save({'lm_head.weight': merged.to('CPU').realize()}, args.head)
  metadata.update(initial_validation=initial, merged_validation=merged_accuracy, rank=args.rank, steps=args.steps)
  args.head.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
  print(f'saved experimental head to {args.head}; no application speedup is inferred', flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('mode', choices=('collect', 'fit'))
  parser.add_argument('--model', type=Path, default=Path('models/Qwen3-4B-Q4_K_M.gguf'))
  parser.add_argument('--checkpoint', type=Path, default=Path('models/eagle3-qwen3-4b'))
  parser.add_argument('--dataset', type=Path, required=True)
  parser.add_argument('--head', type=Path)
  parser.add_argument('--context', type=int, default=512)
  parser.add_argument('--tokens', type=int, default=128)
  parser.add_argument('--rank', type=int, default=16)
  parser.add_argument('--steps', type=int, default=300)
  parser.add_argument('--seed', type=int)
  args = parser.parse_args()
  if min(args.context, args.tokens, args.rank, args.steps) <= 0: parser.error('counts must be positive')
  if args.dataset.suffix != '.npz': parser.error('dataset must have .npz suffix')
  if args.seed is None: args.seed = 101 if args.mode == 'collect' else 42
  if args.mode == 'collect':
    if args.dataset.exists(): parser.error('refusing to overwrite dataset')
    collect(args)
  else:
    if args.head is None or args.head.suffix != '.safetensors': parser.error('fit requires a .safetensors head path')
    if args.head.exists() or args.head.with_suffix('.json').exists(): parser.error('refusing to overwrite head/metadata')
    if args.head.resolve() == (args.checkpoint/'model.safetensors').resolve(): parser.error('cannot replace original checkpoint')
    fit(args)


if __name__ == '__main__': main()
