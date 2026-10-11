# DFlash correctness and experimental results

All current model inference is tinygrad. The target is now the unmodified
upstream Qwen3.5 `Transformer.from_gguf`; our drafter/verifier lives in
`infurnace/dflash/`. Sequential verification is the default reference. `--verifier block` selects experimental
layer-major captured verification with incremental prefix-state/RNG commitment;
neither is a validated acceleration.

Run a synchronized exact-output/RNG baseline/DFlash/baseline protocol:

```sh
PYTHONPATH=. .venv/bin/python benchmarks/dflash.py \
  models/qwen35-validation/Qwen3.5-4B-Q4_K_M.gguf \
  models/dflash-qwen35/qwen35-4b-dflash-Q4_K_M.gguf \
  --context 64 --output 8 --block-size 4 --warmup-pairs 2 --repeats 2 \
  --json /tmp/dflash.json
```

## Upstream audit / attention policy

See the [pinned source-contract audit](upstream-audit.md). Current official
attention defaults are causal for sliding layers and noncausal for full layers;
this corrected a mismatch with our former all-noncausal default. Explicit GGUF
metadata still overrides the default. `--legacy-noncausal` reproduces the older
policy. New JSON reports record effective per-layer flags.

**All reports listed below predate this correction and use legacy all-noncausal
draft attention.** Exact target-output/RNG checks remain valid, but their draft
proposals, acceptance and timings cannot be attributed to current layer defaults.
No corrected-policy full-model performance report has completed.

## Reports

- **Aborted experimental block schedule:**
  [layer-major context64/output8](results/qwen35-upstream-layer-major-block4-context64-output8.json).
  First warmup matched output/RNG; subsequent synchronization failed with the
  existing NV signal timeout/`NV_ERR_INVALID_ARGUMENT`. No measured records or
  ratios. This attempt predates RNG graph cleanup; small NV controls passed
  afterward, but no completed full-model block protocol is claimed.
- **Current target / sequential:** [upstream context64/output8](results/qwen35-upstream-block4-context64-output8.json).
  Complete, exact output and RNG. Baseline about2.95 tok/s; DFlash about2.38.
  Narrow correctness evidence; no performance success.
- **Superseded local target:** [context128/output16](results/qwen35-q4-block4-context128-output16.json).
  Complete, exact output/RNG against the former local target, not upstream.
- **Superseded local target / first capture screen:**
  [one warmup pair](results/qwen35-q4-block4-initial-warmup1.json).
  Retained unchanged for history; captures were not fully warmed.

Neither historical local-target reports nor their speed numbers transfer to the
upstream implementation. These results do not update the published decode chart.

[Model boundary](../../../docs/qwen35.md) · [DFlash design/limits](../../../docs/dflash.md)
