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

`infurnace --benchmark 32` now also defaults to 512 prompt tokens. Customize it:

```sh
infurnace --benchmark 32 --benchmark-prompt-tokens 512
infurnace --benchmark 32 --benchmark-prompt-tokens 2048 --max-context 4096
infurnace --benchmark 32 --benchmark-prompt-file prompt.txt
```

CLI prompt options require `--benchmark`. Previous measurements used an 8-token
prompt (CLI) or a one-token seed (comparison); do not compare their rates directly
to these longer-context results.
`python benchmarks/wrapper.py --device NV --tokens 20` measures wrapper overhead
on the SAME loaded generic transformer; it is not an independent engine test.

## Experimental packed prefill

The CLI and comparison benchmark enable tensor-core prefill automatically on
eligible NV devices. Use `--no-nv-prefill` to keep accelerated decode with generic
prefill, or `--no-nv-quant` for the fully generic baseline. `--nv-prefill` explicitly
requires support instead of falling back. Programmatic `from_gguf` defaults remain
generic; opt in with `nv_quant=True, nv_prefill=True`.

```sh
infurnace --device NV --nv-quant --nv-prefill --benchmark 32
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

Default partition size is 128 tokens. Only live partitions launch and merge;
tail loads are masked, and shared memory has producer/consumer barriers before
reuse. Scratch is `(query_heads, ceil(capacity/128), 130)` FP32 values, bounded by
capacity rather than session length. The last two values hold max/normalization
statistics. No full score or probability matrix is retained. Native cache layout
and the ordered latest-token write remain unchanged. Prefill stays upstream.

Initial supported shape: batch 1, one query token, D=128, four query heads per KV
head. Unsupported blocks remain upstream. Programmatic loading selects this path
with `nv_quant=True, nv_fused_attention=True`; CLI/benchmark selection is explicit.

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

The pinned runtime lacks a dispatch hook. `infurnace/model.py` installs a
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
