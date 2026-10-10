# Sampler PTX/SASS and VIZ follow-up

[Experiments](../README.md) · [Reports](../results/README.md)

Report filenames below refer to `../results/`; commands run from the repository root.

The installed tinygrad pin is unchanged. Ordinary inference defaults and original checkpoint files remain unchanged. All timings below are diagnostics or separate experimental reports, not updates to the published decode chart.

## Reproduce kernel inspection

```sh
mkdir -p /tmp/eagle-sampling-traces
TMPDIR=/tmp/eagle-sampling-traces PYTHONPATH=. DEV=NV HCQ_CACHE_THRESH=0 VIZ=2 \
  .venv/bin/python benchmarks/experiments/eagle3/inspect_sampling.py \
  --output-dir /tmp/eagle-sampling-traces/kernels --tokens 1 2 3 --debug 6
```

`VIZ=2` saves rewrite/profile pickles, rather than simply printing CUDA. The script also prints CUDA and separately compiled PTX and saves actual runtime cubin disassembly. It explicitly sets `Context(PMA=0)` before NV device initialization: this pin enables privileged PMA hardware counters with VIZ=2, and environment `PMA=0` does not disable the bool-valued context variable (`bool('0')` is true). Two initial VIZ attempts failed with `NV_ERR_INSUFFICIENT_PERMISSIONS`; no driver permissions were changed. The corrected run completed six cases and dumped 31 kernels, plus both trace files, under `/tmp/eagle-sampling-viz-complete/`.

Inspect the captured timeline from the terminal:

```sh
TMPDIR=/tmp/eagle-sampling-viz-complete .venv/bin/python -m tinygrad.viz.cli --list
TMPDIR=/tmp/eagle-sampling-viz-complete .venv/bin/python -m tinygrad.viz.cli -s NV -t 12
```

Both commands were exercised; the aggregate reduction group dominates this **sampling-only** trace. CLI aggregation merges shader names across variants, so use the separately named `.sass`/`.cu` files to distinguish implementations. For the interactive rewrite/graph UI, `python -m tinygrad.viz.serve` accepts explicit `--profile-path` and `--rewrites-path` arguments; it was not started during this investigation. This pin's server binds all interfaces by default, so do not expose it on an untrusted network.

## Actual scheduling problem

The original batched sampler's max and index passes both launch **one 32-thread block** and scan 151,936 vocabulary scores. Each lane executes 1,187 loop iterations with four values per iteration. Both passes recompute the score's two logarithms. Source includes:

```cuda
for (int Ridx0 = 0; Ridx0 < 1187; Ridx0++) {
  // ... four score expressions and running reductions ...
  float score = logit * inverse_temperature
              + log2(log2(clamped_random) * -0.6931471824645996f)
                * -0.6931471824645996f;
}
```

This is real serialized GPU work, not register spilling. Materializing the **same FP32 Gumbel score** before the existing argmax removes recomputation without dropping RNG draws, replacing sampling with plain argmax, or changing the first-index tie rule. The helper `score_argmax(..., materialize_scores=False)` leaves this opt-in; the isolated wrapper enables it with `--materialize-scores`.

Isolated score/argmax inspection excludes random-number generation. With VIZ instrumentation, realized TinyJit medians were:

| Input rows | Original | Materialized |
|---|---:|---:|
| 1 | 800.89 us | 324.14 us |
| 2 | 809.72 us | 342.98 us |
| 3 | 897.86 us | 338.43 us |

Original GPU max/index loops each took approximately 288 us. Materialization took approximately 6 us, while the remaining max/index loops took approximately 38 / 64 us. They are still single-block reductions. These figures are not whole-request speedups.

An explicit block-reshape argmax prototype was rejected: a positional NaN check changed the existing CPU decision from 32 to 6. Keeping the existing reduction semantics matters even when a finite-logit benchmark happens to pass.

## Whole-request follow-ups

The protocols below preserve exact target output, original target/draft files, and ordinary baseline decode. They use context 512, 256 outputs, one proposal, two warmup pairs, two measured repeats, chunk looping, fused drafting, and `HCQ_CACHE_THRESH=0`.

| Report / experiment | EAGLE tok/s | Bracketed baseline tok/s | Ratios |
|---|---:|---:|---:|
| `qwen3-4b-eagle3-mapped-compact.json`: mapped I/O + compact commitment + zero-copy features | 106.24 / 105.11 | 105.82 / 103.99 | 1.004 / 1.011 |
| `qwen3-4b-eagle3-mapped-materialized.json`: mapped I/O + materialized scores + zero-copy features | 120.27 / 121.52 | 106.90 / 107.40 | 1.125 / 1.131 |
| `qwen3-4b-eagle3-mapped-calibrated-materialized.json`: same optimizations plus the separately identified calibrated draft head | 129.18 / 129.29 | 109.18 / 109.32 | 1.183 / 1.183 |

The first two completed and retained **94 accepted descendants / 162 rounds**. The calibrated-head protocol also completed, with **101 / 155**, exactly as in the earlier calibrated-head comparison. Thus the acceptance increase is not responsible for this follow-up gain. The first materialized-score attempt without zero-copy features aborted during EAGLE warmup with the existing NV synchronization failure; the subsequent zero-copy protocol completed. That does not prove zero-copy fixes the failure.

Compact commitment alone is not a meaningful gain. The combined materialized-score result is a useful 12–13% whole-request improvement on this request, but not a general performance or reliability claim. The two reports do not isolate every changed component. The calibrated-head combination reaches **18.3%** on this request, still with exact original target output. Head/target/checkpoint identities were verified by the archived calibration wrapper. Its first attempt failed during draft loading before any request measurements; the isolated retry completed. A separate new-prompt bicycle-gears evaluation also completed on an isolated retry: **115.87 / 111.96 versus 105.89 / 106.35 tok/s**, ratios **1.094 / 1.053**, with **86 accepted descendants / 170 rounds**. Report: `qwen3-4b-eagle3-mapped-calibrated-gears.json`. This prompt is absent from the calibration training/validation sets and uses the same 256-output, two-warmup-pair/two-repeat exact-output protocol. Its first attempt failed during draft loading before measurements. The smaller 5–9% gain makes clear that the arithmetic request's 18.3% result is not general.

To reproduce the score/I/O combination, run `mapped_io_benchmark.py` with `--materialize-scores --zero-copy-features` in addition to the usual loop/fused-draft benchmark flags. To explicitly use the existing calibration artifact, add `--head models/eagle3-head-calibration-q8.safetensors`. The direct-copy bypass is restricted to owned mapped token buffers, not other NV/runtime or baseline buffers.

## Validation and remaining work

Sampling tests check decisions, first-index rounding ties, rejection/EOS commitment, counter rollover, continuation RNG, and four capture replays. Both original and materialized chain modes passed the focused CPU/NV sampling suite (61 tests before additional nonfinite cases). CPU positional NaN/infinity comparisons also matched the materialized implementation; six additional CPU/NV nonfinite cases passed. Wrapper patches restore original descriptors/functions on cancellation and do not bypass unowned buffer copies. Final full CPU regression: **411 passed, 259 skipped**. Focused sampling validation: **61 CPU/NV tests plus six nonfinite CPU/NV cases passed**. Six experimental-tooling guards and touched-file Ruff / `git diff --check` also passed. Further ordinary-generation enablement requires broader prompt/context evaluation and resolution of recurring NV timeouts.
