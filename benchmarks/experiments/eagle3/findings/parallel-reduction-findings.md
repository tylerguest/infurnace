# Parallel argmax follow-up

[Experiments](../README.md) · [Reports](../results/README.md)

Report filenames below refer to `../results/`; commands run from the repository root.

Experimental only; ordinary inference and target arithmetic are unchanged.

`parallel_argmax.py` implements a two-stage reduction over the already materialized **unchanged FP32 Gumbel scores**. One warp handles each 512-score tile, followed by one merge warp. It carries score/index pairs, preferring the first index on ties. Masked tail entries and NaNs do not become valid winners; an all-NaN row retains the target's vocabulary-size sentinel.

## A real codegen dependency bug caught by output checks

The first custom prototype returned index **489 instead of 163** on a 511-element random row. Dumped CUDA showed the mutable accumulator loaded **before** the loop; each iteration compared against the initial state rather than the previous iteration. Packing score/index into a single uint64 register prevented independently updated state but did not alone fix the hoist. The fix is an explicit range dependency: `accumulator.after(step)[0].load()`, with an opaque selection expression. Subsequent random/replay and special-value tests passed. This was a bug in the new prototype, not evidence that the existing production target has this bug.

## Diagnostics

`/tmp/eagle-parallel-inspect/` contains a completed nine-case `VIZ=2` inspection, including CUDA, separately compiled PTX, actual runtime cubin/SASS, resources, and decision/reference outputs. The log `/tmp/eagle-parallel-inspect.log` prints all three source/assembly forms with `--debug 7`.

| Logit rows | Original capture | Materialized capture | Parallel capture |
|---|---:|---:|---:|
| 1 | 825.9 us | 353.5 us | 272.4 us |
| 2 | 835.4 us | 364.3 us | 276.5 us |
| 3 | 852.9 us | 363.9 us | 283.7 us |

The two custom kernels each timed around **5.7–6.0 us** in the isolated call harness. Runtime resource reports show **30 registers for tiles / 50 for merge**, with **zero local or stack memory**. The tile kernel launches 297 independent warps for vocabulary 151936, rather than serializing the vocabulary scan in one warp.

These synchronized, instrumented score/argmax captures **exclude RNG generation**. They are not end-to-end performance evidence. Each inspected decision and replay matched its BEAM=0 reference.

Reproduce inspection:

```sh
mkdir -p /tmp/parallel-inspection
TMPDIR=/tmp/parallel-inspection PYTHONPATH=. DEV=NV HCQ_CACHE_THRESH=0 VIZ=2 \
  .venv/bin/python benchmarks/experiments/eagle3/inspect_sampling.py \
  --output-dir /tmp/parallel-inspection/kernels --tokens 1 2 3 \
  --parallel-argmax --debug 7 --trials 5
```

The mapped-I/O wrapper enables this path only with both `--materialize-scores --parallel-argmax`; it restores the original sampler on exit. It can also be combined with `--compact-verify` to measure GPU prefix commitment. Completed full-request reports, not the table above, determine whether either is useful.

## Completed application protocols

Context 512 / 256 outputs, calibrated head, one proposal, loop projections,
fused drafting, zero-copy features, two warmup pairs and two measured repeats.
All reports below completed exact-output baseline/EAGLE/baseline comparisons.

| Configuration | EAGLE tok/s | Bracketed baseline | Ratios |
|---|---:|---:|---:|
| Parallel reduction | 119.41 / 120.16 | 101.96 / 102.07 | 1.171 / 1.177 |
| Parallel + compact GPU commitment | 114.36 / 120.58 | 103.59 / 102.58 | 1.104 / 1.176 |
| Parallel + mapped draft proposals | 129.61 / 129.68 | 107.14 / 107.09 | 1.210 / 1.211 |
| Same, confidence cutoff 1 | 122.04 / 121.81 | 107.23 / 107.21 | 1.138 / 1.136 |
| Mapped draft proposals + fused root sampling | 129.74 / 129.35 | 106.80 / 107.41 | 1.215 / 1.204 |
| Mapped draft proposals + exact fused projections, no root fusion | **131.65 / 131.82** | **107.25 / 107.15** | **1.228 / 1.230** |

All configurations except confidence filtering retained **101 accepted descendants / 155 rounds**. Confidence filtering lost useful proposals: **84 / 172**, and was slower. Reports are
`qwen3-4b-eagle3-parallel-reduction.json`,
`qwen3-4b-eagle3-parallel-compact.json`, and
`qwen3-4b-eagle3-parallel-mapped-draft.json`, and
`qwen3-4b-eagle3-parallel-confidence.json`, and
`qwen3-4b-eagle3-parallel-fused-root.json`. The final and fastest configuration is
`qwen3-4b-eagle3-parallel-fused-projections.json`.

The additional `--mapped-draft-output` option copies the fused draft's small
proposal output into owned host-mapped storage inside its existing capture;
previous experiments mapped draft input IDs but not its output proposals.
Hidden state and observed cache length are unchanged. Buffers are scoped by
draft instance, observation size, proposal count and scored/unscored mode.

Baseline speed varied across sessions, so cross-run absolute deltas are not
isolated component effects. The mapped-draft protocol shows **21% over its
bracketed baseline**. Enabling the already-validated exact fused projection
reductions improves the latest completed combination to **131.7–131.8 tok/s**,
**23% over its bracketed baseline**. This is still only one arithmetic request;
200 tok/s has not been reached, and successful protocols are not an NV reliability fix.

Reproduce the fastest completed configuration:

```sh
PYTHONPATH=. DEV=NV HCQ_CACHE_THRESH=0 .venv/bin/python \
  benchmarks/experiments/eagle3/mapped_io_benchmark.py \
  --head models/eagle3-head-calibration-q8.safetensors \
  models/Qwen3-4B-Q4_K_M.gguf models/eagle3-qwen3-4b \
  --context 512 --output 256 --draft-tokens 1 \
  --loop-projections --fused-projections --fused-draft \
  --materialize-scores --parallel-argmax --mapped-draft-output --zero-copy-features \
  --warmup-pairs 2 --repeats 2 --json /tmp/parallel-fused-projections.json
```
 Compact commitment still fails to provide an
established gain. Root sampling fusion (`--fused-root`) also showed no clear
incremental gain and remains off. The wrapper rejects combining root fusion
with compact commitment, restores constructor patches on exit, and leaves
ordinary decode unchanged.

## Validation

- `tests/test_nv_parallel_argmax.py`: **36 NV tests passed**, covering random scores with five changing-input JIT calls, sizes 1/32/511/512/513/151936, ties, NaNs, all-NaNs, positive/negative infinity, and signed zero. Twelve additional chain cases compare original/parallel decisions and committed counters over four replays, including acceptance/rejection/EOS and counter rollover; subsequent RNG draws match.
- Existing scoped-wrapper tests cover restoration of `score_argmax`, draft capture methods, and verifier initialization, including exceptional exit with the parallel/mapped-draft/fused-root options. **14 tooling guards passed**.
- Final focused CPU suite: **48 passed, 69 skipped** (including the NV-only skips). Touched-file Ruff and `git diff --check` passed. No full-suite rerun this iteration.
