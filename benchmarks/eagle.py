"""Bounded, thinking-enabled, exact-output EAGLE comparison in one process.

Run from the repository root with the installed environment. Warmup costs are
reported separately; measured baseline/EAGLE runs alternate and synchronize.
This does not update the published 1024-output chart or imply a speedup.
"""
import argparse
import json
import os
import time
from pathlib import Path

import tinygrad
from tinygrad import Device, Tensor
from tinygrad.device import Compiled

from infurnace.loader import load_model as _build
from infurnace.eagle import EagleDraft
from infurnace.eagle.generate import EagleGenerator


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('model', type=Path)
  parser.add_argument('checkpoint', type=Path)
  parser.add_argument('--context', type=int, default=1024)
  parser.add_argument('--output', type=int, default=256)
  parser.add_argument('--draft-tokens', type=int, default=None)
  parser.add_argument('--fast-eagle', action='store_true', help='use the integrated CLI EAGLE fast profile')
  parser.add_argument('--local-ffn-down', action='store_true', help='experimental CTA-local Q6 verification projection')
  parser.add_argument('--loop-projections', action='store_true', help='experimental whole-row verification CTAs')
  parser.add_argument('--fused-draft', action='store_true', help='observe and propose inside one draft capture')
  parser.add_argument('--compact-verify', action='store_true', help='commit the verified prefix on device with one decision transfer')
  parser.add_argument('--zero-copy-features', action='store_true', help='consume captured target features without a copy capture')
  parser.add_argument('--draft-confidence', type=float, help='minimum draft logit margin; requires --fused-draft')
  parser.add_argument('--fused-projections', action='store_true', help='sum exact projection chunks inside their CTA')
  parser.add_argument('--warmup-pairs', type=int, default=2)
  parser.add_argument('--repeats', type=int, default=2)
  parser.add_argument('--seed', type=int, default=101)
  parser.add_argument('--prompt', default='Calculate 17 times 23, explaining the arithmetic.')
  parser.add_argument('--json', type=Path)
  args = parser.parse_args()
  if args.draft_tokens is None: args.draft_tokens = 1 if args.fast_eagle else 2
  if args.fast_eagle and (args.local_ffn_down or args.loop_projections or args.fused_draft or args.compact_verify
                         or args.zero_copy_features or args.draft_confidence is not None or args.fused_projections):
    parser.error('--fast-eagle selects its own optimization profile; omit individual optimization flags')
  if min(args.output, args.warmup_pairs, args.repeats) <= 0: parser.error('counts must be positive')
  if not 0 <= args.draft_tokens <= 7: parser.error('draft-tokens must be between zero and seven')
  runtime = dict(tinygrad_path=str(Path(tinygrad.__file__).resolve().parent),
                 hcq_cache_thresh=os.environ.get('HCQ_CACHE_THRESH', '64'), rtalloc_pool_bytes=Compiled.rtalloc_size)
  loading = dict(model=str(args.model), checkpoint=str(args.checkpoint), status='running', stage='loading',
                 runtime=runtime, error=None, warmups=[], records=[], comparisons=[])
  if args.json: args.json.write_text(json.dumps(loading, indent=2)+'\n')
  try:
    model, tokenizer = _build(args.model, args.context, 'NV')
    draft = EagleDraft.from_checkpoint(args.checkpoint, model)
    engine = EagleGenerator(model, draft, fast=True) if args.fast_eagle else EagleGenerator(model, draft)
    prompt = tokenizer.encode(tokenizer.render_chat([{'role': 'user', 'content': args.prompt}]))
    if len(prompt)+args.output > args.context: parser.error('prompt plus output exceeds context')
  except BaseException as exc:
    loading.update(status='aborted', error=f'{type(exc).__name__}: {exc}')
    if args.json: args.json.write_text(json.dumps(loading, indent=2)+'\n')
    raise
  reference = None
  warmups, measured = [], []

  def run(mode):
    nonlocal reference
    Tensor.manual_seed(args.seed)
    Device[model.device].synchronize()
    start = time.perf_counter()
    times, counts, tokens, accepted, proposed = [], [], [], 0, 0
    first, first_count = None, 0
    if mode == 'baseline':
      for token in model.generate(prompt, args.output, temperature=0.0, reuse_cache=False):
        tokens.append(token)
        if first is None: first, first_count = time.perf_counter(), 1
    else:
      last = start
      options = {} if args.fast_eagle else dict(
        batched=True, local_ffn_down=args.local_ffn_down, loop_projections=args.loop_projections,
        fused_draft=args.fused_draft, compact_verify=args.compact_verify,
        zero_copy_features=args.zero_copy_features, draft_confidence=args.draft_confidence, fused_projections=args.fused_projections)
      for result in engine.generate(prompt, draft_tokens=args.draft_tokens, max_new_tokens=args.output, **options):
        now = time.perf_counter()
        if first is None: first, first_count = now, len(result.tokens)
        times.append(now-last)
        counts.append(len(result.tokens))
        last = now
        tokens.extend(result.tokens)
        accepted += result.accepted
        proposed += result.proposed
    Device[model.device].synchronize()
    elapsed = time.perf_counter()-start
    if reference is None: reference = tokens
    if tokens != reference: raise RuntimeError('verified output differs from baseline; comparison aborted')
    record = dict(mode=mode, seconds=elapsed, tokens=len(tokens), tok_s=len(tokens)/elapsed,
                  accepted=accepted, proposed=proposed, acceptance=accepted/proposed if proposed else None,
                  rounds=len(times), committed_per_round=len(tokens)/len(times) if times else None,
                  ttft_seconds=first-start if first is not None else None,
                  decode_tok_s=(len(tokens)-first_count)/(start+elapsed-first) if first is not None and len(tokens)>first_count else None,
                  round_seconds=times, round_tokens=counts)
    # Printing is outside the measurement window.
    print(f'{mode:8s} {elapsed:.4f}s {record["tok_s"]:.2f} tok/s accepted={accepted} rounds={len(times)}', flush=True)
    return record

  def save(status, error=None):
    comparisons = []
    for index, record in enumerate(measured):
      if record['mode'] != 'eagle' or index == 0 or index+1 == len(measured): continue
      before, after = measured[index-1], measured[index+1]
      baseline_rate = (before['tok_s']+after['tok_s'])/2
      comparisons.append(dict(eagle_tok_s=record['tok_s'], baseline_tok_s=baseline_rate,
                              speedup=record['tok_s']/baseline_rate))
    if status != 'complete': comparisons = []  # Never publish a speedup from an incomplete protocol.
    payload = dict(model=str(args.model), checkpoint=str(args.checkpoint), prompt=args.prompt,
                   prompt_tokens=len(prompt), output_tokens=args.output, context=args.context,
                   draft_tokens=args.draft_tokens, local_ffn_down=args.local_ffn_down, loop_projections=args.loop_projections,
                   fused_draft=args.fused_draft, compact_verify=args.compact_verify,
                   zero_copy_features=args.zero_copy_features, draft_confidence=args.draft_confidence,
                   fused_projections=args.fused_projections, thinking=True, temperature=0.0, seed=args.seed,
                   timing='synchronized whole request, including prompt and draft observation',
                   status=status, stage='requests', runtime=runtime, error=error,
                   warmups=warmups, records=measured, comparisons=comparisons)
    payload['fast_eagle'] = args.fast_eagle
    if args.fast_eagle: payload.update({k: v for k, v in engine._defaults.items() if k != 'draft_tokens'})
    if args.json: args.json.write_text(json.dumps(payload, indent=2)+'\n')

  status, error = 'complete', None
  save('running')
  try:
    for _ in range(args.warmup_pairs):
      for mode in ('baseline', 'eagle'):
        warmups.append(run(mode))
        save('running')
    measured.append(run('baseline'))
    save('running')
    for _ in range(args.repeats):
      measured.append(run('eagle'))
      save('running')
      measured.append(run('baseline'))
      save('running')
  except BaseException as exc:
    status, error = 'aborted', f'{type(exc).__name__}: {exc}'
    raise
  finally:
    save(status, error)


if __name__ == '__main__': main()
