"""Synchronized baseline/DFlash/baseline exact-output protocol, all in tinygrad.

Times whole requests, including prefill (plain decoding prefills one row at a time).
Requires the Qwen3.5 target and its draft; no external model runtime.
RNG storage identities remain stable across captured replays.
"""
import argparse
import json
import os
from pathlib import Path
import time
from tinygrad import Device, Tensor, dtypes
from infurnace.loader import load_model as _build
from infurnace.dflash import DFlashDraft, DFlashGenerator


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('model', type=Path)
  parser.add_argument('draft', type=Path)
  parser.add_argument('--device', default='NV')
  parser.add_argument('--context', type=int, default=512)
  parser.add_argument('--output', type=int, default=64)
  parser.add_argument('--block-size', type=int, default=4)
  parser.add_argument('--verifier', choices=('sequential', 'parallel'), default='parallel')
  parser.add_argument('--legacy-noncausal', action='store_true')
  parser.add_argument('--warmup-pairs', type=int, default=2)
  parser.add_argument('--repeats', type=int, default=2)
  parser.add_argument('--seed', type=int, default=101)
  parser.add_argument('--prompt', default='The capital of France is')
  parser.add_argument('--chat', action='store_true')
  parser.add_argument('--json', type=Path, required=True)
  args = parser.parse_args()
  if min(args.context, args.output, args.block_size, args.warmup_pairs, args.repeats) <= 0: parser.error('counts must be positive')
  report = {'status': 'running', 'stage': 'loading', 'model': str(args.model), 'draft': str(args.draft), 'device': args.device,
    'context': args.context, 'output_limit': args.output, 'block_size': args.block_size, 'seed': args.seed, 'temperature': 0.0,
    'target_implementation': 'unmodified tinygrad Transformer.from_gguf',
    'verifier': args.verifier, 'legacy_noncausal': args.legacy_noncausal, 'timing': 'synchronized whole request including prefill and drafting',
    'hcq_cache_thresh': os.environ.get('HCQ_CACHE_THRESH', '64'), 'warmups': [], 'records': [], 'comparisons': []}
  def save():
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + '\n')
  save()
  try:
    model, tokenizer = _build(str(args.model), args.context, args.device)
    draft = DFlashDraft.from_gguf(args.draft, model, legacy_noncausal=args.legacy_noncausal)
    engine = DFlashGenerator(model, draft, verifier=args.verifier)
    if args.block_size > draft.config.block_size: raise ValueError('block exceeds trained draft size')
    text = tokenizer.render_chat([{'role': 'user', 'content': args.prompt}], enable_thinking=False) if args.chat else args.prompt
    prompt = tokenizer.encode(text)
    if len(prompt) + args.output > model.config.context_length: raise ValueError('prompt plus output exceeds context')
    report.update(stage='requests', prompt=text, prompt_tokens=prompt, feature_layers=draft.config.feature_layers,
                  mask_token=draft.config.mask_token, draft_packed_linears=draft.packed_linears,
                  draft_layer_causal=[draft.config.layer_causal(i) for i in range(draft.config.layers)])
    # Initialize the pinned tinygrad key/counter once. Reset counter contents,
    # not the dictionary or buffer identities captured by either target JIT.
    Tensor.manual_seed(args.seed)
    Tensor.rand(1, device=model.device).realize()
    counter = Tensor._device_rng_counters[model.device]
    reference = reference_rng = None
    def run(mode):
      nonlocal reference, reference_rng
      counter.assign(Tensor([0, 0], dtype=dtypes.uint32, device=model.device)).realize()
      Device[model.device].synchronize()
      start = time.perf_counter()
      stream = model.generate(prompt, args.output, reuse_cache=False) if mode == 'baseline' else engine.stream(
        prompt, max_new_tokens=args.output, block_size=args.block_size)
      try: tokens = list(stream)
      finally: stream.close()
      Device[model.device].synchronize()
      elapsed = time.perf_counter() - start
      rng = counter.tolist()
      if reference is None: reference, reference_rng = tokens, rng
      if tokens != reference: raise RuntimeError('DFlash/baseline token mismatch; protocol aborted')
      if rng != reference_rng: raise RuntimeError('DFlash/baseline RNG advancement mismatch; protocol aborted')
      stats = vars(engine.stats).copy() if mode == 'dflash' else {}
      result = {'mode': mode, 'seconds': elapsed, 'tokens': len(tokens), 'tok_s': len(tokens)/elapsed,
                'exact_output': True, 'rng_counter': rng, **stats}
      print(f'{mode:8s} {elapsed:.4f}s {result["tok_s"]:.2f} tok/s {stats}', flush=True)
      return result
    for _ in range(args.warmup_pairs):
      for mode in ('baseline', 'dflash'):
        report['warmups'].append(run(mode))
        save()
    report['records'].append(run('baseline'))
    save()
    for _ in range(args.repeats):
      for mode in ('dflash', 'baseline'):
        report['records'].append(run(mode))
        save()
    for i, record in enumerate(report['records']):
      if record['mode'] != 'dflash': continue
      before, after = report['records'][i-1], report['records'][i+1]
      baseline = (before['tok_s'] + after['tok_s']) / 2
      report['comparisons'].append({'baseline_tok_s': baseline, 'dflash_tok_s': record['tok_s'], 'ratio': record['tok_s']/baseline})
    report.update(status='complete', output_tokens=reference, output_text=tokenizer.decode(reference))
    save()
  except BaseException as exc:
    report.update(status='aborted', error=f'{type(exc).__name__}: {exc}', comparisons=[])
    save()
    raise


if __name__ == '__main__': main()
