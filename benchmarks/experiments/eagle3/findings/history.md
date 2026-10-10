# EAGLE3 follow-up history

[Current status and tools](../README.md) · [Report index](../results/README.md)

Report filenames below refer to `../results/`. Tool filenames refer to the
parent experiment directory. Run all shell commands from the repository root.
This historical overview preserves the sequence of results; later findings
supersede earlier performance conclusions.

These reports are separate from the published 1024-output decode chart.
The three projection/prefill protocols below completed synchronized baseline/EAGLE/baseline runs,
verified identical target output, used two warmup pairs and two measured repeats,
and retained the original target and draft checkpoints. They used context 512,
256 requested outputs, one proposal, loop projections, fused drafting, and
`HCQ_CACHE_THRESH=0`. **None demonstrates an application speedup.**

| Report | EAGLE tok/s | Bracketed baseline tok/s | Ratio |
| --- | --- | --- | --- |
| `qwen3-4b-eagle3-fused-projections.json` | 93.33 / 93.19 | 107.83 / 106.47 | 0.865 / 0.875 |
| `qwen3-4b-eagle3-detached-prefill.json` | 94.19 / 94.51 | 109.63 / 109.81 | 0.859 / 0.861 |
| `qwen3-4b-eagle3-meta12-experiment.json` | 92.33 / 91.08 | 104.96 / 106.06 | 0.880 / 0.859 |

Every EAGLE run accepted 94 descendants over 162 rounds (1.58 committed tokens
per round). Differences across protocols are small and not statistical evidence
of improvement. GPU clocks, process order and desktop activity were not controlled.

## Candidates

- **Fused projections:** opt-in repository implementation. A completed warp sum
  must be rounded before accumulation (`__fadd_rn` prevents flattening the final
  shuffle addition). Residual projections accumulate residual first, then chunks,
  matching the pinned target scheduler. Summing first failed real-model exactness
  despite passing isolated projection tests. The corrected version passed exact
  real-model logits/features and replay for 2/3/4 rows, plus 30 focused NV tests.
- **Detached prefill snapshots:** copy realized buffer views rather than retaining
  target graph dependencies. This does not change target arithmetic or output.
- **12-bit Q4 metadata:** isolated-process prototype, not integrated. Repacking
  the scale/minimum pairs without increasing weight size saved roughly 10–15% on
  some standalone projections, with exact partial outputs. Preparation/restoration
  and the full request erased the benefit. In-place metadata roundtrips and all
  bracketing target outputs matched; model files were never changed.

## Assembly inspection and small-buffer I/O

See [assembly-findings.md](assembly-findings.md) for reproducible CUDA/PTX/SASS
inspection and the subsequent real-verifier profile. No register spills were
found. Static weight reuse loses activation-load vectorization, while tiny
host transfers cost approximately 1.4–1.5 ms each with caching disabled.

A host-mapped scratch I/O retry completed exact-output comparisons:
`qwen3-4b-eagle3-mapped-io.json` records **112.37 / 111.64 EAGLE versus
105.99 / 106.43 bracketed baseline tok/s**, **1.060 / 1.049** ratios, and
unchanged **94/162** acceptance/rounds. This modest gain on one 256-output
request is not the substantial or general speedup sought. An earlier attempt
aborted on an NV synchronization failure; a completed 64-output wrapper smoke
remained slower. `mapped_io_benchmark.py` is an isolated experiment, not an
inference default or reliability fix. The published chart is unchanged.

See [sampling-findings.md](sampling-findings.md) for the subsequent **VIZ=2**
rewrite/profile traces and 31 additional sampler PTX/SASS dumps. The sampler
was recomputing both Gumbel logarithms in two single-block vocabulary scans.
Opt-in score materialization preserves the expression and RNG behavior.

Completed exact-output 256-output follow-ups:

- Mapped compact commitment: **106.24 / 105.11 vs 105.82 / 103.99 tok/s**,
  essentially break-even; unchanged 94/162 acceptance/rounds.
- Mapped I/O + materialized scores + zero-copy features:
  **120.27 / 121.52 vs 106.90 / 107.40**, **1.125 / 1.131** ratios; unchanged
  94/162. Report: `qwen3-4b-eagle3-mapped-materialized.json`.
- The same optimizations with the explicitly calibrated draft head:
  **129.18 / 129.29 vs 109.18 / 109.32**, **1.183 / 1.183** ratios; 101/155.
  Report: `qwen3-4b-eagle3-mapped-calibrated-materialized.json`.

These are useful gains on the excluded arithmetic application request, not
broad performance/reliability claims. A completed new bicycle-gears prompt
comparison reached **115.87 / 111.96 vs 105.89 / 106.35 tok/s**, **1.094 / 1.053**
ratios, with 86/170 acceptance/rounds; see
`qwen3-4b-eagle3-mapped-calibrated-gears.json`. Loading/warmup timeouts still occur;
all ordinary defaults and published chart data remain unchanged.

### Further implementation screening

[implementation-ideas.md](implementation-ideas.md) records tinygrad/TensorRT-LLM
source guidance, selected-capture PTX/SASS inspection, and two further completed
protocols. Two proposals reached **110.8–111.0 tok/s**; sampler-only BEAM=2 reached
**118.2–126.7 tok/s**. Neither establishes an improvement over the prior 129.2 tok/s
result. Defaults remain unchanged. The final inspection printed CUDA/PTX/SASS
and exact decision/reference outputs for six cases, with **31 selected JIT shader
dumps** under `/tmp/eagle-sampler-selected-keys/`.

### Parallel reduction and draft-output I/O

[parallel-reduction-findings.md](parallel-reduction-findings.md) documents a new
opt-in two-pass NV argmax, its accumulator dependency fix, nine completed
CUDA/PTX/SASS inspections, and exact-output request comparisons. With mapped
draft proposal outputs it reached **129.61 / 129.68 vs 107.14 / 107.09 tok/s**
(**1.210 / 1.211**). Compact commitment and confidence filtering did not help;
root fusion showed no clear additional gain. Adding exact fused projections
then reached **131.65 / 131.82 vs 107.25 / 107.15 tok/s** (**1.228 / 1.230**).
See `qwen3-4b-eagle3-parallel-fused-projections.json`. No 200 tok/s result is claimed.

Add `--materialize-scores --parallel-argmax --mapped-draft-output` to the mapped
I/O wrapper, alongside the calibrated head, fused draft, loop projections and
zero-copy-feature options, to reproduce this experimental combination. Add
`--fused-projections` for the fastest completed configuration.
The parallel reducer passed **36 NV correctness/replay/RNG tests**.

## Quantization-aware head calibration

An isolated rank-16 LoRA experiment trained only the draft vocabulary projection
on teacher-forced states from the original Q8-arithmetic target. Eight training
prompts supplied 1,024 positions (1,000 with supported labels); two separate
validation prompts supplied 256 positions. The original arithmetic benchmark
(17 × 23) was excluded from both sets. The best merged BF16 head improved
validation one-step accuracy from **56.64% to 72.27%**. This is a small,
prompt-limited calibration result, not a general quality or throughput claim.
The original checkpoint and target files are unchanged; the new head is a separate
local artifact under `models/` with a JSON identity/training manifest.

Completed exact-output baseline/EAGLE/baseline protocols using the calibrated head:

| Report | Proposals | EAGLE tok/s | Bracketed baseline tok/s | Ratio |
| --- | ---: | --- | --- | --- |
| `qwen3-4b-eagle3-calibrated-one.json` | 1 | 98.12 / 97.65 | 109.52 / 109.39 | 0.896 / 0.893 |
| `qwen3-4b-eagle3-calibrated-two.json` | 2 | 91.93 / 92.19 | 109.54 / 109.78 | 0.839 / 0.840 |

One proposal accepted **101 descendants / 155 rounds** (1.65 committed tokens per
round), versus **94 / 162** (1.58) in the original-head control above. This is a
real but modest useful-acceptance improvement, insufficient to beat baseline.
Two proposals accepted **129 / 127** (2.02 tokens per round), but verification and
additional drafting cost more than the saved rounds. There is no matched
uncalibrated-two-proposal control in this follow-up set. Both protocols used two
warmup pairs and two measured repeats, context 512/output 256, loop projections,
fused drafting, and disabled HCQ caching. Source target/checkpoint hashes matched
before loading the override, and every committed output matched baseline.

The calibration tools are deliberately isolated here, not part of ordinary
inference or CLI defaults. NumPy is optional tooling and can be installed into a
temporary directory rather than changing the inference environment:

```sh
.venv/bin/python -m pip install --target /tmp/eagle-calibration-deps numpy
PYTHONPATH=.:/tmp/eagle-calibration-deps DEV=NV HCQ_CACHE_THRESH=0 \
  .venv/bin/python benchmarks/experiments/eagle3/calibrate_head.py collect \
  --dataset /tmp/eagle-training.npz
PYTHONPATH=.:/tmp/eagle-calibration-deps DEV=NV HCQ_CACHE_THRESH=0 \
  .venv/bin/python benchmarks/experiments/eagle3/calibrate_head.py fit \
  --dataset /tmp/eagle-training.npz --head /tmp/eagle-head.safetensors
PYTHONPATH=. DEV=NV HCQ_CACHE_THRESH=0 \
  .venv/bin/python benchmarks/experiments/eagle3/calibrated_benchmark.py \
  --head /tmp/eagle-head.safetensors \
  models/Qwen3-4B-Q4_K_M.gguf models/eagle3-qwen3-4b \
  --context 512 --output 256 --draft-tokens 1 --loop-projections --fused-draft \
  --warmup-pairs 2 --repeats 2 --json /tmp/calibrated-eagle.json
```

Collection and fitting must run in separate processes to release the resident
teacher before training. Unsupported target labels count as incorrect validation
predictions and are excluded only from training. Saving requires improvement with
the actual merged BF16 GEMV, not only the float training expression. Tools refuse
to overwrite data/head files or replace the original checkpoint. Six CPU tests
cover head shape/dtype/payload checks, replacement isolation, hashes and the
original-checkpoint overwrite guard. Replaying the archived trainer reproduced
identical saved head bytes; the archived wrapper also completed a 64-output
exact-output repeated-request smoke protocol.

Two initial collection attempts failed during draft loading with NV signal
timeouts, before collecting any samples. A subsequent isolated retry completed;
this does not establish the cause or a reliability fix. No GPU reset or unrelated
process termination was performed.

Ordinary decoding remains the default. Keep experimental flags opt-in, and do
not infer end-to-end gains from kernel timings, acceptance percentage or aborted
protocols.
