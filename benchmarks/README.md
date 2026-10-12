# Benchmarks

From the checkout with Infurnace installed in the active environment:

```sh
python benchmarks/compare.py --model models/Qwen3-4B-Q4_K_M.gguf \
  --device NV --context 2048 --tokens 256 --runs 3 \
  --output /tmp/infurnace-vs-tinygrad.json
```

Stop other model processes first. Runners load independently in sequential
subprocesses, avoiding simultaneous model residency. Both must support the
chosen local GGUF; Infurnace currently validates dense Qwen3 F32/Q4_K/Q6_K.
NV quantization is enabled for Infurnace by default; `--no-nv-quant` compares
generic loading/generation instead. For CPU use `--device CPU --no-nv-quant`.

## Experimental DFlash reference

`benchmarks/dflash.py` compares ordinary upstream-tinygrad Qwen3.5 generation
against our tinygrad DFlash verifier, bracketed by baseline runs. Sequential is
the default; `--verifier block` selects experimental layer-major capture with
prefix-state/RNG rollback. The full-model block protocol currently has only an
aborted report (NV timeout), not a measured comparison.
It checks exact tokens and RNG advancement, with synchronized whole-request
measurement and explicit running/aborted/complete reports. It requires no other
model runtime. The reference verifier is currently slower, not a speedup claim.
See [commands and reports](experiments/dflash/README.md) and [design/limits](../docs/dflash.md).
Historical local-Qwen3.5-target results are superseded, not current evidence.

## Experimental EAGLE-3 comparison

```sh
DEV=NV PYTHONPATH=. .venv/bin/python benchmarks/eagle.py \
  models/Qwen3-4B-Q4_K_M.gguf models/eagle3-qwen3-4b \
  --fast-eagle --context 512 --output 256 \
  --json /tmp/eagle-comparison.json
```

This keeps one target/draft resident, warms both paths, then brackets EAGLE with
baseline runs. Thinking stays enabled and every output must match the baseline
exactly. Timing is synchronized whole-request time, including prompt and draft
observation; it is not decode-only throughput. JSON separates warmups from
measurements and checkpoints completed runs with `status=running`; failures
are marked `aborted`. Only `complete` runs constitute a completed comparison.
Reports include TTFT, post-first-burst throughput, proposed/accepted counts,
committed tokens per round and bracketing speedup ratios. Ratios are suppressed
unless the entire protocol completes; raw records remain available on failure.
Loading failures also checkpoint `status=aborted`, `stage=loading`. Reports record
HCQ cache threshold, runtime pool size and imported tinygrad module path.
Do not run alongside another full-model inference process.

`--fast-eagle` selects the integrated profile now used by CLI `--eagle3`:
one proposal by default, fused drafting, owned mapped I/O, parallel sampling,
zero-copy features, and validated exact fused projections. Omit individual
optimization switches when using this profile. `--draft-tokens` may still be
overridden. The original checkpoint is used unchanged; a calibrated head is
not automatically selected. Without `--fast-eagle`, this benchmark retains
its historical explicit configuration and two-proposal default.

Optimized experimental comparisons reached about **132 vs 107 tok/s** with a
separately calibrated head. See [latest results](experiments/eagle3/README.md).
These are workload-specific gains, not a reliability guarantee.

### Historical controls and individual optimization switches

The original arithmetic and JSON trials were slower than their matched baseline. Newer
attention/sampling candidates require completed repeated-request validation;
an NV signal timeout is an aborted experiment, not performance evidence.
The published 1024-output chart is unchanged. Test a smaller draft budget with
`--draft-tokens 1`; a projection-level improvement is not evidence of an
application win. `--local-ffn-down` explicitly enables the new CTA-local Q6
projection candidate; it is disabled by default because repeated application
measurements aborted with NV synchronization errors. `--loop-projections` tests
whole-row projection CTAs, including local-token Q6 FFN-down/output-head loops.
These projection variants preserve partial sums but do not establish a tok/s win.
`--fused-draft` combines draft observation and proposals in one capture; it is
separate from experimental target/root sampling and remains disabled by default.
A completed exact-output original-target trial (256 outputs, one proposal, two
warmup pairs and two repeats, loop projections and fused draft, HCQ cache disabled)
measured **94.75 / 94.29 EAGLE versus 109.39–109.70 baseline tok/s**, still slower.
`--fused-projections` additionally folds chunk sums and residual updates into the
projection CTA, preserving the target's residual-first accumulation and completed
warp-sum rounding boundaries. Real-model 2/3/4-row logits/features and replay match
exactly. A completed 256-output trial measured **93.33 / 93.19 EAGLE versus
106–109 baseline tok/s**, so this option also remains disabled. Follow-up reports
are under [`experiments/eagle3`](experiments/eagle3/README.md), not in the published
chart directory.

`--compact-verify` tests device-side prefix commitment; `--zero-copy-features`
tests direct feature views. `--draft-confidence MARGIN` requires `--fused-draft`
and truncates proposals at the first low-confidence token. Higher acceptance
percentage alone is insufficient: filtering can discard useful accepted tokens
and create additional rounds. These options do not establish an application gain.

Normal-cache repeated requests remain unreliable; disabling caching is neither
an application speedup nor a universal fix.
The updated tinygrad pin fixes submission-signal reinitialization; a later combined
fused-sampling/zero-copy trial still timed out, so those modes remain disabled.
Full-row residual reductions and materialized batched Q/K normalization passed
real-model exactness checks and are enabled within the opt-in verifier.
`--eagle3 CHECKPOINT` exposes the experimental path in CLI chat and raw
single-shot generation, but ordinary decode remains the default.

## Method

Defaults to **512 tokens of deterministic repeated prose**, including optional
BOS. `--prompt-tokens 2048 --context 4096` tests a larger active window;
`--prompt-file prompt.txt` uses raw UTF-8 text without truncation or chat formatting.
Both runners use identical IDs, verified before accepting the comparison.
Synthetic repeated prose controls context length; it is not representative task
quality evaluation. Use a real prompt file for representative content.

Uses tinygrad's CLI timing boundaries: reset `GlobalCounters` and time each
`next(generate)` using tinygrad `Timing`, but with a configurable longer prompt
instead of upstream's one-token seed.
Generation synchronizes through upstream `.item()`. Temperature is zero and
EOS is ignored. Each runner executes a cold pass, a capture pass, then three
measured warm passes, resetting prefix reuse each time. First-token timing is
reported separately; warm decode excludes that token. Reports include the actual
prompt length and decode input-position range. Prompt length plus requested
output must fit the actual model context; invalid runs fail rather than silently
shortening output. TTFT includes prefill and first-token work, not pure prefill.
Longer prompts can make all five passes significantly slower; `--runs 1 --tokens 32`
is useful for a quicker comparison. This is not an HTTP-server benchmark.

The table reports load time, cold TTFT, aggregate warm tok/s, median/p95 latency,
whole-process host peak RSS and current tinygrad allocated bytes. Full JSON
includes each run's TTFT, per-step token IDs, timing, estimated bytes/ops/kernel
counts, model SHA256, Python and installed tinygrad source/version. Aggregate
throughput is tokens divided by total measured time, NOT averaged instantaneous
rates. Warm pass active positions match across runners.

The parent disables profiling/debugging in children and selects lazy FP16
upstream loading (`realize=False`), matching the default upstream CLI. Compiler
caches may already be populated: "cold" means process/JIT cold, not necessarily
compiler-cache cold. GPU clocks, thermal state and other workloads aren't
controlled; repeat comparisons before judging small improvements. Infurnace's
packed math/Q8 activations differ from upstream generic math; output quality
requires separate validation. Counters estimate work and may incompletely
account for custom kernels; estimated GB/s is NOT measured DRAM bandwidth.
Current allocation is NOT GPU peak. Device probing is excluded from load timing;
load includes tokenizer construction for both runners.

## Other measurements

`infurnace --benchmark [PROMPT,]COUNT` is tinygrad's per-step benchmark: after warmup it
times prefill of a PROMPT-token numbers prompt (default 512), then each of COUNT synchronized
decode steps (default 20), printing ms, tok/s, tinygrad's estimated traffic GB/s, traffic/allocated
MB and the text so far. Runs longer than 40 steps show only the first and last 20. Traffic is a
compiler estimate, not measured DRAM bandwidth; allocation is not GPU peak.

`benchmarks/decode.py --tokens 32` is the machine-readable report: one cold run, two capture
passes and three warm runs as JSON (`--prompt-tokens N` for repeated prose, or `--prompt-file`
for an untruncated raw-text prompt). `compare.py` and `wrapper.py` share its prompt and
measurement helpers.

Commands further down this file record the flags of the time; the `--nv-*`,
`--warmup` and `--benchmark-prompt-*` CLI flags have since been removed (NV
acceleration is automatic; `infurnace.loader.load_model` still accepts the `nv_*`
options for comparisons). Previous measurements used an 8-token
prompt (CLI) or a one-token seed (comparison); do not compare their rates directly
to these longer-context results.
`python benchmarks/wrapper.py --device NV --tokens 20` measures wrapper overhead
on the SAME loaded generic transformer; it is not an independent engine test.

## Experimental packed prefill

The CLI and comparison benchmark enable tensor-core prefill automatically on
eligible NV devices; `compare.py --no-nv-prefill` / `--no-nv-quant` select generic
prefill or the fully generic baseline. Programmatic `from_gguf` defaults remain
generic; opt in with `nv_quant=True, nv_prefill=True`.

```sh
infurnace --device NV --benchmark 32
python benchmarks/compare.py --model models/Qwen3-4B-Q4_K_M.gguf \
  --tokens 32 --runs 1 --nv-prefill --output /tmp/comparison-prefill.json
```

The kernel decodes Q4_K/Q6_K directly into NVIDIA m16n8k16 FP16 fragments with
FP32 accumulation. No persistent dense weight copy or Q8 activation quantization
is used for prefill. The initial implementation uses four warps per block where
rows allow, pads token tiles to 16, and has no shared-memory staging/autotuning.
The adapter pads upstream symbolic token counts to their maximum and slices
valid outputs; one-token decode keeps its existing GEMV path.

RTX 5060, bundled Qwen3-4B, 512-token synthetic prompt, 32 generated tokens,
2048 context capacity, three warm CLI runs:

| Metric | Generic prefill | Packed prefill v1 |
| --- | ---: | ---: |
| Mean warm TTFT | 16.66 s | 0.979 s |
| Warm decode | 80.5 tok/s | 81.2 tok/s |
| Current tinygrad allocation | 3.43 GiB | 3.14 GiB |

These are separate local runs, not an alternating statistical comparison. TTFT
includes the first output token; allocation is not measured process GPU peak.
Results: `/tmp/infurnace-512-prompt-benchmark.json` and
`/tmp/infurnace-prefill-gemm-v1.json`. FP16 activation/weight rounding can alter
logits. Kernel parity and a generation smoke test are not full-model quality
parity. This experimental path is now the eligible-NV application default despite
those pending checks; `--no-nv-prefill` is the escape hatch.

## Experimental fused decode attention

```sh
infurnace --nv-fused-attention --benchmark 32 --benchmark-prompt-tokens 2048 --max-context 4096
python benchmarks/compare.py --nv-fused-attention --tokens 32 --runs 1 --context 4096 --prompt-tokens 2048
```

The new opt-in path supersedes `--nv-attention` when both flags are supplied.
It fuses QK scoring, stable online softmax and weighted-V accumulation per context
partition, then launches a small merge kernel. Four query warps share FP16 K/V
staging (16 tokens × 128 dimensions × two tensors = 8 KiB) per KV-head block.
Queries, scores and accumulators remain FP32; no query FP16 conversion occurs.

Default partition size is 64 tokens (initial version: 128). Only live partitions launch and merge;
tail loads are masked, and shared memory has producer/consumer barriers before
reuse. Scratch is `(query_heads, ceil(capacity/64), 130)` FP32 values, bounded by
capacity rather than session length. The last two values hold max/normalization
statistics. No full score or probability matrix is retained. Native cache layout
and the ordered latest-token write remain unchanged. Prefill stays upstream.

Initial supported shape: batch 1, one query token, D=128, four query heads per KV
head. Unsupported blocks remain upstream. Programmatic loading selects this path
with `nv_quant=True, nv_fused_attention=True`; CLI automatically selects fused
attention on eligible NV runs (`--no-nv-fused-attention` opts out).

One same-process comparison, Qwen3-4B/RTX 5060, 4096 capacity, identical synthetic
prompt IDs, 32 output tokens, packed prefill, no profiler:

| Prompt tokens | Parallel-softmax decode | Fused decode |
| --- | ---: | ---: |
| 512 | 76.6 tok/s | 109.3 tok/s |
| 2048 | 48.3 tok/s | 96.3 tok/s |

Both modes were warmed separately; fresh rollout JIT was used after switching.
There was one measured pass per mode/context, sequential parallel→fused, not an
alternating statistical comparison. Clocks/thermal state were not controlled.
TTFT stayed about 1 s at 512 and 8.1 s at 2048; this does not accelerate prefill.
Current tinygrad allocation stayed about 3.44 GiB, not measured GPU peak.
Results: `/tmp/infurnace-fused-matched-benchmark.json`.

Isolated synchronized attention at 2048 active tokens measured 0.350 ms generic,
0.185 ms parallel-softmax and 0.176 ms fused. CPU submission and cache residency
limit interpretation of these wall-clock microbenchmarks; the whole model has
different scheduling and weight/KV cache pressure. These are not DRAM counters.

The four-position teacher-forced smoke check had identical top-1 IDs; fused mean
absolute logit error was 0.037–0.061, maximum 0.202–0.236, KL 0.000033–0.000529
nats, and NLL differences 0.003–0.055 nats. This is not held-out quality parity.
The new path remains experimental and opt-in.

Existing attention correctness/performance tests cover both paths. Select fused
real-model logits with:

```sh
DEV=NV INFURNACE_TEST_MODEL=models/Qwen3-4B-Q4_K_M.gguf INFURNACE_TEST_NV_FUSED_ATTENTION=1 \
  python -m pytest -q -s tests/test_nv_attention.py::test_local_model_teacher_forced_attention_logits
```

The generation smoke test also accepts `INFURNACE_TEST_NV_FUSED_ATTENTION=1`.

### Decode scheduling follow-up

KV staging now loads four adjacent halfwords per lane, one row per warp,
reducing the loader loop from 16 trips to four. Partition size 64 gives more
parallel blocks and shorter per-block work than the original 128. The FP32
attention arithmetic is unchanged within a partition; different merge grouping
can change rounding. Scratch per partial buffer doubles versus partition 128
but remains bounded by capacity; explicit partition 128/256 remains supported.

Alternating captured whole-model decode replays, same prefilled prefix per
context, `item()` synchronization per token, 32 decode steps per batch, four
pairs, no profiler (RTX 5060, context capacity 4096):

| Prompt tokens | Original fused path | Tuned fused path | Gain |
| --- | ---: | ---: | ---: |
| 512 | 92.4 tok/s | 95.2 tok/s | 3.1% |
| 2048 | 84.5 tok/s | 89.9 tok/s | 6.4% |

Rates are 32 divided by median batch duration. Contexts were checked in separate
processes; modes alternated within each process. The 512-token batches completed
before that process was interrupted, so its text log rather than final JSON is
retained. An earlier single-pass full-generation comparison was mixed, motivating
these paired checks. Absolute rates vary with clocks and other desktop workloads;
do not compare directly to earlier sessions' peak rates. This tunes the opt-in
`--nv-fused-attention` path only; prefill and generic attention are unchanged.

Artifacts: `/tmp/infurnace-decode-replay-512-output.txt`,
`/tmp/infurnace-decode-replay-ab.json`, and the earlier
`/tmp/infurnace-decode-tuning-ab.json`. Tile-softmax and subgroup-dot variants were
screened and rejected because their additional staging/synchronization lost time.
Broader regression and model-quality checks remain follow-up work.

## Prepared Q4/Q6 weight layouts (experimental)

NV quantized runs automatically prepare Q6 weights once during model loading, adding
zero alignment bytes to each 210-byte GGUF block (216 bytes/block). Values,
scales and FP16 block multipliers are unchanged. Prepared Q6 uses row-local
four-block tiles with word-pair/block interleaving (described below), not a new
quantization format. Q4 now also uses four-block tiles, but interleaves four-word
groups to preserve 128-bit payload loads without adding storage (described below).
The resident buffer is uint32, enabling 64-bit vectorized payload loads; packed
prefill also reads this layout. Generic decode/embedding interprets the padded
stride directly, rather than constructing a second tight GGUF allocation.
Tied embedding/output share the prepared backing and lazy decoded weight.

```sh
infurnace --nv-prepared-weights --benchmark 512,128 --warmup
```

Programmatic loading: `Qwen3Model.from_gguf(..., nv_quant=True,
nv_prepared_weights=True)`. Preparation now defaults on with `nv_quant=True`;
pass `nv_prepared_weights=False` or CLI `--no-nv-prepared-weights` to opt out.
`--no-nv-quant` also disables automatic preparation. It remains experimental:
the measured gain is too small to establish a reliable throughput improvement.
The model load still has temporary staging allocations; no GPU peak was measured.

Initial 212-byte candidate comparison (historical), fused attention, packed prefill, capacity 4096,
512 prompt tokens, 128 outputs, three warmed runs per mode, native process first
then prepared process (no profiler):

| Layout | Decode rates | Median | Current tinygrad allocation |
| --- | --- | ---: | ---: |
| Native Q6 | 111.71 / 111.72 / 111.64 tok/s | 111.71 tok/s | 3,691,969,884 bytes |
| Prepared Q6 | 112.32 / 111.88 / 112.32 tok/s | 112.32 tok/s | 3,698,896,244 bytes |

Approximately 0.5% higher median and 6.9 MB additional allocation; separate
sequential processes do not establish a causal gain at this scale. Q6 storage
padding is 0.95%, distinct from whole-model allocation overhead. Artifacts:
`/tmp/infurnace-q6-prepared-native.json` and
`/tmp/infurnace-q6-prepared-prepared.json`. Existing benchmark charts are not
updated with these incomparable measurements. Focused tests cover byte-preserving
preparation, generic decoding and NV eager/JIT decode and prefill parity, including
odd/even block positions and partial final chunks.

### UOp packing / vector-load iteration

Current preparation uses a single `_prepare_q6_kernel` UOp STORE/SINK call,
not a chain of tensor padding/contiguous/bitcast kernels. Tensor is only used as
the buffer allocation/realization facade. The pinned tinygrad coalescer
(`codegen/late/coalesce.py`) requires provable alignment to form vector loads.
A 216-byte block stride guarantees 8-byte alignment; explicit eight-word
`_load_lanes()` payload accesses lower to uint2 loads. A 224-byte stride enabled
uint4 loads but was rejected after whole-model throughput decreased.

Generated cubins, inspected with `cuobjdump`, for Q6 FFN-down and vocabulary
head: 212-byte layout has 22 global-load instructions total per kernel body,
216-byte layout 14, 224-byte layout 10 (including activation/metadata loads).
These counts are static instructions, not measured DRAM traffic. No register
spills in the inspected 212/224 kernels. The 224 FFN-down variant used 48
registers versus 47 for 212; both head variants used 40.

Same workload as above: 512-token prompt, 128 outputs, capacity 4096,
three warmed runs per layout, separate processes in order 212, 224, 216:

| Block stride | Decode rates | Median | Current allocation |
| --- | --- | ---: | ---: |
| 212 bytes | 113.24 / 113.39 / 113.19 tok/s | 113.24 tok/s | 3,698,896,244 bytes |
| 224 bytes (rejected) | 112.19 / 111.74 / 112.04 tok/s | 112.04 tok/s | 3,740,307,292 bytes |
| 216 bytes (retained, now default for NV quantized runs) | 114.46 / 114.06 / 116.36 tok/s | 114.46 tok/s | 3,712,669,532 bytes |

The retained candidate is approximately 1.1% faster than the saved 212-byte
baseline in this session, not an alternating statistical confirmation. No
baseline rerun or extrapolation to the longer-context published chart.
Padding is 2.86% of Q6 storage, about 20.7 MB above tight native Q6 storage.
Actual allocation shown above includes graph/scratch differences and is not
GPU peak. The next iteration below adds cross-block swizzling without changing
the 216-byte storage budget.
Artifacts: `/tmp/infurnace-q6-vector-{212,216,224}.json`, with generated
CUDA/cubins/SASS in `/tmp/infurnace-q6-{212,216,224}-{down,head}.*`.

### Four-block Q6 word-pair swizzle

Current prepared Q6 layout groups four adjacent 256-value blocks within each
weight row. Each tile stores `[word_pair, block_within_tile, pair_element]`;
tiles at a row's end use their actual block count (1–3), without extra tile
padding. Block storage remains 216 bytes. This brings the same word pairs from
the four blocks assigned to a decode warp into contiguous addresses, retaining
aligned 64-bit loads. Preparation, GEMV and prefill addressing are explicit
UOps; generic/tied embedding consumers use inverse views and lazy decode of
that same allocation. No inverse tight-GGUF buffer is explicitly realized.

The 216-byte aligned-only kernel remains available internally for comparison
(`q6_swizzled=False`); normal prepared adapters select the swizzle. CLI JSON
records the prepared layout (initially `"q6_tile4_pair_swizzle"`, now
`"q4_tile4_quad_q6_tile4_pair"`), or `"gguf"` when
preparation is disabled. This is a format/shape-specific Q6 layout, not a
second inference engine.

Bounded comparison: RTX 5060, capacity 4096, 512-token prompt, 128 outputs,
fused attention and packed prefill, three warmed runs in each of two sequential
processes (aligned first, swizzled second), no profiler:

| Layout | Decode rates | Median | Current allocation |
| --- | --- | ---: | ---: |
| Aligned 216-byte | 113.60 / 107.97 / 113.22 tok/s | 113.22 tok/s | 3,712,669,532 bytes |
| Four-block swizzle | 117.51 / 117.50 / 117.72 tok/s | 117.51 tok/s | 3,712,698,204 bytes |

Approximately 3.8% median improvement in this session; process order, GPU clocks
and desktop workload remain confounders. No statistical confidence claim or
long-context extrapolation. Current allocation differs by about 29 KB, not an
additional full weight copy; transient/GPU peak allocation is not measured.
TTFT snapshots were 0.983–1.010 s aligned versus 0.963–0.979 s swizzled; this is
not a separate statistically established prefill improvement.

Inspected cubins contain 14 global-load instructions for both layouts. Q6
FFN-down registers decreased 49->46; head remains 40, no spills. The intended
improvement is cross-lane locality, not fewer instructions or proven DRAM
bandwidth. Tests cover complete/partial tiles, row boundaries, byte-preservation,
generic decode, one/eight-warp eager/JIT GEMV, and concrete/symbolic prefill.
Artifacts: `/tmp/infurnace-q6-swizzle-{aligned,swizzled}.json` and generated
`/tmp/infurnace-q6-{aligned,swizzled}-{down,head}.{cu,cubin,sass}`.
Published 1024-output chart remains the earlier aligned-layout measurement.

### Four-block Q4 four-word swizzle

Prepared Q4 now stores `[word_group_of_four, block_within_tile, group_element]`
for four adjacent GGUF blocks in each row. Partial tiles use their actual block
count. All 144 bytes/block are preserved with no padding; preparation and kernel
addressing use UOps. Generic decode and prefill consume the same resident layout.
`--no-nv-prepared-weights` disables both Q4 and Q6 preparation.

A Q4 word-pair variant was rejected (more narrow payload loads). The retained
four-word variant preserves native 128-bit payload loads while improving intended
cross-block locality. Inspected gate/down cubins have 10 global-load instructions
for both native and four-word variants, but 38->36 registers, no spills.
These are static instruction/resource observations, not DRAM counter measurements.

RTX 5060, 512 prompt tokens, 128 outputs, capacity 4096, fused attention,
packed prefill, Q6 swizzle enabled throughout, three warmed runs per process:

| Q4 layout | Decode rates | Median | Current allocation |
| --- | --- | ---: | ---: |
| Native | 113.38 / 114.30 / 114.57 tok/s | 114.30 tok/s | 3,712,698,204 bytes |
| Word-pair swizzle (rejected) | 110.91 / 110.97 / 108.43 tok/s | 110.91 tok/s | 3,712,709,676 bytes |
| Four-word swizzle (retained) | 114.28 / 117.05 / 117.50 tok/s | 117.05 tok/s | 3,712,709,676 bytes |

Sequential processes, native first, then pair, then quad. About 2.4% median gain
against the saved native baseline, not an alternating statistical confirmation.
An intermediate runner invocation accidentally selected native rather than quad;
its report is retained as `/tmp/infurnace-q4-extra-native-control.json` and is
not used as the candidate. The corrected quad run is the one reported above.
TTFT snapshots are 0.973–0.990 s native versus 0.973–1.017 s quad; no prefill
speedup claim. Current allocation changes about 11 KB, not a second full weight
allocation; peak/transient allocation is not measured.

Parity tests cover every byte, generic decode, partial tile/row boundaries,
eager/JIT GEMV and concrete/symbolic prefill. Artifacts:
`/tmp/infurnace-q4-swizzle-{native,swizzled,quad}.json` and generated
`/tmp/infurnace-q4-{native,quad}-{gate,down}.{cu,cubin,sass}`.
The published 1024-output chart is unchanged and predates these layout changes.

## Shared Q8 activation packing

Supported `_NVAttentionBlock` single-token decode now packs its already-normalized
input once for Q/K/V and once for FFN gate/up. `_NVLinear.decode()` passes that
explicit Q8 tuple into the GEMV UOp call, with dtype/shape/device validation.
No Python activation cache, norm replacement or change to the upstream function
boundary. Prefill/unsupported shapes fall back unchanged. On Qwen3-4B this reduces
packing calls from 253 to 145 per token (108 removed across 36 layers).
CLI JSON/text records `nv_shared_q8_blocks`. Sharing applies to eligible NV
parallel/fused attention blocks, not the plain upstream-attention block path.

RTX 5060, 512 prompt tokens, 128 outputs, capacity 4096, current Q4/Q6 prepared
layouts and fused attention, three warmed runs per mode in separate processes
(baseline first, shared second), no profiler:

| Packing | Decode rates | Median |
| --- | --- | ---: |
| Separate per projection | 111.24 / 111.40 / 111.38 tok/s | 111.38 tok/s |
| Shared Q/K/V and gate/up | 113.12 / 113.08 / 113.27 tok/s | 113.12 tok/s |

Approximately 1.6% median gain in this session; sequential process order and
clocks remain confounders, so not statistical confirmation. Tests verify a single
packing CALL feeding three projections, exact individual projection parity,
whole toy-block parity through JIT/cache writes/reset, independent KV buffers,
and unchanged upstream prefill. Published long-output charts are unchanged.
Reports: `/tmp/infurnace-activation-{baseline,shared}.json`.

An RMSNorm/Q8 UOp fusion prototype passed eight isolated reference/JIT cases but
its whole-model benchmark hit an NV synchronization timeout after warmup. No
throughput result or proven cause was obtained; that integration and kernel were
removed rather than enabled. Prototypes are saved outside the repo as
`/tmp/infurnace-rmsnorm-q8-prototype.py` and
`/tmp/infurnace-model-rmsnorm-prototype.py`; failed run logs:
`/tmp/infurnace-activation-fused{,-stderr}.txt`. RMSNorm fusion remains follow-up
work requiring reliable whole-model validation.

## Parallel-softmax reference attention

```sh
infurnace --nv-attention --benchmark 32 --benchmark-prompt-tokens 2048 --max-context 4096
python benchmarks/compare.py --nv-attention --tokens 32 --runs 1 --context 4096 --prompt-tokens 2048
```

This opt-in first iteration replaces dynamic-length softmax with concrete-capacity,
masked parallel reductions. QK scoring and weighted V still operate on the active
prefix; scores/probabilities are not yet eliminated or fused with those operations.
Unused positions are masked before max/exp. Native FP16 cache storage and its
ordered latest-token write are preserved. Prefill stays upstream.

The pinned runtime lacks a dispatch hook. `infurnace/models/qwen3.py` installs a
per-model decode-only block subclass, retaining parameter/cache identities and
reusing upstream projections, norms and RoPE; unsupported configurations and
prefill call the original method. No installed dependency or global Tensor method
is patched. This small supported decode setup should migrate to an upstream hook
when available.

One unprofiled run, RTX 5060/Qwen3-4B, 4096 capacity, packed prefill, 32 outputs:

| Prompt tokens | Warm TTFT | Decode tok/s |
| --- | ---: | ---: |
| 512 | 1.020 s | 80.0 |
| 2048 | 8.075 s | 49.8 |

`/tmp/infurnace-parallel-softmax-benchmark.json` records results. The earlier
2048-token diagnostic measured 40.6 tok/s WITH VIZ=1; that is not a matched
unprofiled speedup comparison. TTFT should not improve: this affects decode only.
An alternating, synchronized isolated attention test at 2048 active tokens measured
0.359 ms generic vs 0.186 ms optimized (ratio 0.518). That cache-resident test is
not whole-model throughput or measured DRAM bandwidth.

Four teacher-forced real-model decode positions after a 32-token synthetic prefix
had identical top-1 IDs; mean absolute logit error was 0.034–0.051, maximum
0.180–0.217, reference-to-candidate KL 0.00025–0.00053 nats, and teacher-forced
NLL differences 0.003–0.086 nats. These are bounded smoke checks, not held-out
quality parity. Parallel FP32 reductions change ordering and downstream Q8 rounding
can amplify perturbations; broader quality/reliability checks remain necessary.

Tests:

```sh
DEV=NV INFURNACE_TEST_NV=1 python -m pytest -q tests/test_nv_attention.py
DEV=NV INFURNACE_TEST_NV_PERF=1 python -m pytest -q -s tests/test_nv_attention_performance.py
DEV=NV INFURNACE_TEST_MODEL=models/Qwen3-4B-Q4_K_M.gguf INFURNACE_TEST_NV_ATTENTION=1 \
  python -m pytest -q -s tests/test_nv_attention.py::test_local_model_teacher_forced_attention_logits
```

The relative performance gate requires at least 5% improvement and is calibrated
for an idle RTX 5060 with profiling disabled. The real-model logits test is opt-in;
standard tests never load the whole model. The existing generation smoke test also
accepts `INFURNACE_TEST_NV_ATTENTION=1` alongside its model/generation/NV flags.

## Regression checks

```sh
DEV=CPU python -m pytest -q
DEV=NV INFURNACE_TEST_NV=1 python -m pytest -q \
  tests/test_nv_q8.py tests/test_nv_decode_layout.py tests/test_tinygrad_compat.py \
  tests/test_nv_prefill.py
DEV=NV INFURNACE_TEST_NV_PERF=1 \
  INFURNACE_TEST_MODEL=models/Qwen3-4B-Q4_K_M.gguf \
  python -m pytest -q -s tests/test_nv_performance.py
```

CPU graph tests protect the eight integer dots, explicit warp/lane separation,
and native Q6 halfword storage. NV tests cover signed per-16 activation sums
(including cancelling halves), eager/JIT parity, multi-token indexing, partial
chunks, packed-byte preservation, and tied embedding/output storage.

The optional performance test loads only the real Q6 output head, not the whole
model. It compares current GEMV against frozen old bytewise Q6 arithmetic and a
one-warp layout, sharing current Q8 packing. It warms both JITs and alternates
six pairs of batches; optimized median latency must be at least 5% lower.
This gate is intended for an idle RTX 5060 and the tested head shape, with
profiling disabled. It is a relative microbenchmark, NOT a fixed tok/s guarantee,
a full-model quality test, or measured DRAM bandwidth. Investigate/repeat failures
with the GPU idle; clocks, contention and thermal state can affect results.
No full-model benchmark runs automatically in the normal test suite.

Prefill tests cover tensor-core fragment structure, exact packed storage,
FP16-rounded activation/reference math, tile tails, multiple GGML blocks and
symbolic upstream-function/JIT replay. The optional real-model generation test
also accepts `INFURNACE_TEST_NV_PREFILL=1`:

```sh
DEV=NV INFURNACE_TEST_MODEL=models/Qwen3-4B-Q4_K_M.gguf \
  INFURNACE_TEST_GENERATION=1 INFURNACE_GENERATION_DEVICE=NV \
  INFURNACE_TEST_NV_QUANT=1 INFURNACE_TEST_NV_PREFILL=1 \
  python -m pytest -q tests/test_runner.py::test_local_generation
```
