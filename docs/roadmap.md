# Roadmap

Build a small **local Qwen3 application on tinygrad**, not a second inference
engine. Target: bundled Qwen3-4B Q4_K_M on the 8 GB RTX 5060, one user/model/GPU.
Reuse upstream transformer/tokenizer/JIT; own narrow, measured improvements.
No scheduler, request batching, multi-GPU or model-family expansion.

## Working foundations

- [x] Validated GGUF descriptors, packed staging and lazy F32/Q4_K/Q6_K decoding;
  Qwen3 config/shape validation, tied weights, context cap and KV estimate.
- [x] Tinygrad Transformer integration and tokenizer adapter; sampled tokenizer
  parity with llama.cpp and bundled chat-template parity with Jinja.
- [x] Local CLI, automatic usable-device selection, default NV acceleration on
  supported devices, thinking, startup warmup and `--no-nv-quant` fallback.
- [x] Chat runs until EOS/remaining context unless explicitly capped; visible
  oldest-turn rollover and token/context stop notices.
- [x] CPU integration tests, isolated NV eager/JIT/Q8-reference tests, generic
  multi-token fallback and bundled-model NV generation smoke tests.
- [x] Short benchmark: ~93–94 accelerated vs ~10.75 generic decode tok/s.
  Wrapper vs the same upstream Transformer adds no meaningful overhead.

These are working foundations, **not full-model quality or long-session gates**.
See [baseline](baseline.md) for settings and limitations: 8 prompt tokens,
20 generated tokens, allocated context 2048. That is NOT decode at 2048 active
cached tokens. Current tinygrad allocation is not process GPU peak.

## What the code review establishes

Reviewed installed tinygrad pin `ce9c84605e94de9667fb578c1ff3fa8678725a87`
and sibling `tinygrad/` checkout `bec565d15786115279fe869e437f3b43b1e80010`.
They are different runtimes; this review does not upgrade the dependency.

| Code | Finding and implication |
|---|---|
| `infurnace/kernels/quant_gemv.py` | Separate activation quantization, packed GEMV and partial-sum reduction. Q4 recomputes activation sums for every output row; Q6 unpacks/repackages individual bytes. One 32-thread block per output/chunk. |
| `infurnace/models/qwen3.py::_NVLinear` | 253 eligible linears; explicit packed identity and generic/fast storage sharing. Only concrete single-token inputs accelerate; prefill remains generic except the last-token output head. |
| `tinygrad/llm/kernels/amd.py` | Reusable designs: quantize each activation once, precompute activation sums, wordwise quant math, grouped output rows, tiled prefill and online-softmax attention. Some improvements already exist in our pin. Newer Q6 uses native 210-byte blocks via halfword storage instead of padded copies. These are AMD implementations, not drop-in NV kernels. |
| `tinygrad/llm/model.py` | Separate prefill/decode JITs already exist. Every prefill chunk computes output logits and sampling even when its result is discarded. NV Qwen3 uses generic attention; the AMD flash dispatch is also restricted to hybrid models. |
| `tinygrad/engine/jit.py`, NV runtime | The pin already batches supported captured work into graphs. Newer upstream has substantial HCQ2/runtime changes. Profile replay/dispatch before proposing another graph layer or assuming an upgrade is faster. |
| `infurnace/cli.py`, `benchmarks/compare.py` | Existing timing is a smoke baseline; comparison script measures wrapper overhead, not independent engines. No per-kernel bottleneck attribution yet. |

These are source-level optimization candidates, not measured speedups.

## The bandwidth budget

CPU-only inspection of the bundled GGUF descriptors gives **2,490,539,520 bytes**
of packed linear weights per ordinary decode step, counting the tied output
matrix once. This excludes metadata/file padding, tiny norms, embedding lookup,
activations and scratch traffic. It is a useful minimum-work model, not a hardware
DRAM counter: repeated loads and cache behavior must be profiled.

- Q4_K: **1.765 GB / 70.9%**; Q6_K: **0.725 GB / 29.1%**.
- FFN gate/up/down: **65.4%** of linear weight bytes; output head: **12.8%**.
  Prioritize those shapes, not just small attention projections.
- FP16 KV storage is **147,456 bytes per cached token** across all 36 layers.
  An efficient decode attention pass ideally reads K/V once per step, sharing
  each KV head across its four query heads. Generic GQA may incur extra traffic.

Assuming the advertised **448 GB/s**, weights plus one active-KV read imply:

| Active cached tokens | Idealized bytes/step | Bandwidth-only tok/s |
|---:|---:|---:|
| Near zero | 2.491 GB | 180 |
| 512 | 2.566 GB | 175 |
| 2048 | 2.793 GB | 160 |
| 4096 | 3.095 GB | 145 |
| 8192 | 3.698 GB | 121 |

These optimistic estimates omit compute, launch/sync costs, extra memory passes
and bandwidth inefficiency. Allocated context capacity and active length are
separate variables. At ~93 tok/s, latency is ~10.75 ms and **weight-equivalent**
throughput is ~232 GB/s; that is not measured DRAM throughput or proof that the
remaining time is all memory stalls. The weight-only ideal is ~5.56 ms.

Measure sustainable large-buffer bandwidth on this GPU. For illustration,
312 GB/s would give only ~125 tok/s before other costs at short context; the
kernel comment citing llama.cpp at that bandwidth is not a validated Infurnace
measurement. Do not promise 180 tok/s, or 500 tok/s through kernel tuning.

## 1 — Establish correctness and an attributable performance baseline

Do these together, before changing arithmetic or launch geometry.

- [ ] Full-model **teacher-forced logits** on identical token IDs/cache positions:
  generic vs accelerated, eager vs JIT, chunk boundaries and one-token tails.
  Obtain logits with a test hook rather than maintain a second transformer.
  Run models sequentially/in separate processes to fit VRAM.
- [ ] Separate kernel correctness against Q8-rounded activations from quality
  changes against generic FP16 weights. The fast path also avoids generic
  decoded-weight FP16 rounding. Record logit error, top-k agreement and held-out
  loss/perplexity; document tolerances and test real prompts. Free-running text
  can diverge after a small logit change; identical prose is not the sole oracle.
- [ ] Extend benchmarks to 128–256 generated tokens, short/medium/long prompts,
  and decode near active positions 128/512/2048/4096 (8192 only if memory fits).
  Allocate room beyond the tested position for generation. Report actual ranges,
  token IDs, chunk size, sampling, model hash, runtime/driver and GPU state.
- [ ] Separate load/transfer/compile, pure prefill, TTFT and decode; compare cold
  processes and repeated warm runs, median/tail inter-token latency and variance.
  Measure process GPU peak, host peak and steady-state allocation separately;
  label sampled peaks honestly. Keep EOS policy and prefix reuse explicit.
- [ ] Use tinygrad profiling to break down Q4/Q6 GEMV, Q8 packing, reductions,
  norms/FFN, embedding, attention, sampling, transfers and host/device gaps.
  Inspect generated CUDA/PTX and replay coverage; confirm whether repeated packs,
  expanded weights or large embedding intermediates survive compilation.
  Timing runs must also run **without profiler overhead**.
- [ ] Build shape microbenchmarks from actual descriptors: input widths
  2560/4096/9728, output widths 1024/2560/4096/9728/151936, each applicable type.
  Include full-sized FFN and Q6 output-head weights. Rotate enough buffers to
  exceed L2; repeatedly timing a cache-resident tiny matrix is misleading.
- [ ] Independently compare upstream loading/generation and llama.cpp with the
  same GGUF, token IDs, active context and GPU placement. Record differences in
  KV dtype, kernels and sampling instead of implying mathematically identical
  engines. Measure sustained streaming bandwidth as a separate reference.

**Gate:** reproducible latency breakdown, real-shape kernel baselines and a
numerical/quality regression suite. Rank work by whole-model milliseconds, not
isolated percentage wins. Fix correctness failures before enabling new paths.

## 2 — Make the existing packed NV GEMV efficient

Highest-value initial experiments, reordered if the profile disagrees:

1. **Remove redundant activation work.** Quantize one value per lane and pack
   with shuffles rather than reloading/requantizing groups. Produce per-16 sums
   once; Q4 can use their total instead of eight extra `dp4a` sums per group per
   output row. Test zero inputs, extremes and rounding ties explicitly.
2. **Improve packed loads and Q6 math.** Vectorize/coalesce Q4 loads and simplify
   scale extraction. For Q6, operate on packed words and subtract `32*xsum`
   after dot products rather than rebuilding signed bytes individually. Respect
   210-byte block alignment: native halfword loads or a deliberately benchmarked
   padded layout, never an unsafe aligned-u32 assumption. Preserve shared packed
   storage and account for any repacking cost/extra bytes.
3. **Tune scheduling by shape.** Compare 2/4/8 warps per block handling neighboring
   rows, a separate row/chunk grid, and different reduction splits. Today widths
   2560/4096/9728 produce 3/4/10 partials per row. Test fewer partials or in-block
   completion against split-K parallelism; removing a reduction kernel is not
   a win if occupancy collapses. Inspect register pressure/spills and tail lanes.
4. **Test NV cache policy only with evidence.** Streaming weight loads may protect
   activations/KV, but AMD's `nontemporal` renderer rule does not implement an NV
   policy. Evaluate supported CUDA/PTX load policies; verify generated code and
   both short-/long-context effects. Do not equate wider loads with fewer DRAM bytes.

**Gate per experiment:** kernel and full-model tests pass, repeated end-to-end
improvement exceeds noise, memory stays within budget. Retain simple per-shape
choices only when justified; no large autotuning framework. Preserve generic
fallback and make one change at a time so gains remain attributable.

## 3 — Remove measured non-weight overhead

- [ ] Share activation packing across Q/K/V and across gate/up when inputs are
  identical. They currently call separate adapters; first verify compiler reuse.
  If all packs survive, sharing could remove **108 pack calls per step** across
  36 layers, without reducing weight reads. Use bounded per-forward/capture
  lifetimes, not an unbounded Python cache keyed by tensor identity.
- [ ] Fuse proven hot boundaries: RMSNorm→Q8 packing, partial reduction→consumer,
  or SiLU/gate multiply→down-projection packing. Consider grouped projections
  only if launch costs justify them. Keep residual/norm rounding testable.
- [ ] Audit sampling at temperature zero: upstream still constructs Gumbel noise.
  Propose a true greedy fast path upstream if material, with explicit tie/numeric
  semantics. Do not silently change the sampler while reporting a kernel speedup.
- [ ] Confirm captured decode replay has no accidental weight copies, repeated
  compilation or avoidable CPU synchronizations. `.item()` returns each token;
  measure that cost before changing streaming/EOS/cancellation behavior.
- [ ] Benchmark newer tinygrad in an isolated environment if profiling points
  to runtime overhead or an upstream fix. Port custom UOp APIs and rerun all
  gates before changing the pin; HCQ2 changes are not a free guaranteed speedup.

**Ownership:** prefer small upstream hooks/patches or narrow adapters. Do not
copy Transformer/attention/FFN wholesale just to share a pack or fuse an epilogue.

## 4 — Keep decode fast as context grows

Move this ahead of Phase 3 if attention dominates the target workload.

- [ ] Profile attention vs **active** length and allocated capacity; inspect
  generic GQA expansion, score/softmax intermediates and actual KV traffic.
- [ ] If warranted, add a narrow NV decode-attention path using tinygrad UOps:
  GQA KV reuse, tiled K/V reads, stable online softmax, and split-context combine
  when needed for occupancy. Start with this model's head dimension 128 and
  32 query / 8 KV heads. AMD flash kernels are an algorithm reference, not a
  renderer-only port; NVIDIA scheduling/reduction layouts need separate tuning.
- [ ] Preserve upstream KV layout, cache writes and prefix rules; validate
  masking, partial tiles, start-position boundaries, reset and replay. Keep a
  generic fallback. Avoid quantized KV until FP16 traffic is demonstrably limiting.

**Gate:** correctness plus flatter long-context latency and bounded scratch
memory. Report short-context regressions; dispatch by workload if justified.

## 5 — Improve prefill and first-token latency separately

Decode tok/s alone does not make chat responsive.

- [ ] First try a small upstream prefill-only hook to skip output projection and
  sampling for nonfinal chunks while still executing all KV writes. Each skipped
  head avoids scanning ~319 MB of Q6 weights. Verify final logits/cache parity.
- [ ] Sweep chunk sizes (e.g. 32/64/128), including partial/one-token tails, warmup
  and symbolic replay. Measure TTFT, pure prefill, compilation and peak memory.
- [ ] If generic linears dominate, prototype packed Q4_K/Q6_K **NV tiled GEMM**:
  decode weight tiles for reuse across prompt tokens, use supported NVIDIA tensor
  core layouts through tinygrad, and retain generic fallback. AMD WMMA layouts
  cannot simply be relabeled CUDA. No persistent whole-model FP16 copy.
- [ ] Optimize embedding lookup only if its scheduled work is material; audit
  lazy packed-row access before adding another custom kernel.

**Gate:** materially better realistic-prompt TTFT/prefill without decode, quality,
storage-sharing or memory regressions. Prompt-token tiling is not request batching.

## 6 — Reliability, startup and frontend work

Run reliability checks throughout the performance work, not just at the end.

- [ ] Hundreds of chat turns, rollover, abandoned/interrupted streams, repeated
  resets and varying warmup/prompt lengths; bounded host/GPU/JIT growth.
- [ ] Configurable answer-space reservation, reset/context-usage commands and
  clear OOM/device-failure recovery. State when a fresh process is necessary.
- [ ] Profile slower accelerated loading separately: parse, packed transfers,
  typed-view preparation, tokenizer and compilation. Preserve no full FP16 copy,
  no duplicate ties, and packed-byte lifetime tests. Startup savings are not
  steady-state decode savings.
- [ ] Complete GGUF metadata/alignment/truncation/unsupported-input fixtures;
  retain numerical reference, packed-byte and preallocation validation tests.
- [ ] After local reliability gates pass, evaluate tinygrad's server before a
  thin localhost API for Pi. Test streaming, cancellation and errors; defer tools.
  Optional history summarization is separate from simple visible rollover.

## Execution and stopping rules

1. Next work package: Phase 1 quality harness, benchmark matrix and profile.
2. Next: the best measured Phase 2 experiment, one independently tested change.
3. Re-profile; choose Phase 3 or 4 by remaining milliseconds. Treat Phase 5 as
   a separate TTFT objective. Keep Phase 6 reliability checks running throughout.

Use sustainable measured bandwidth and the best matched engine as reference,
not advertised bandwidth alone. At short context, 120/140/150 tok/s would require
~299/349/374 GB/s of weight-equivalent throughput **before** extra traffic.
These are useful checkpoints, not forecasts. Consider the path near its practical
ceiling when packed kernels approach measured streaming capability, remaining
non-weight costs are explained, and successive changes no longer beat noise.
Keep per-context results; there is no single context-independent ceiling.

Do not chase improvements by rewriting the tokenizer/loader for decode speed,
adding another scheduler/graph engine, or assuming tensor cores solve M=1 GEMV.
Smaller/differently quantized models, speculative decoding and quantized KV change
the assumptions and require separate quality/memory justification; they are not
shortcuts to this model's ordinary-decode target.

## Maintenance rules

- Keep the exact tinygrad pin until a tested upgrade wins or fixes a needed bug.
  Force reinstall the VCS dependency if pip retains a same-version older wheel.
- Keep strict loading while its validation/storage control earns its maintenance
  cost; revisit preflight validation plus upstream loading later.
- Direct offset Buffer views previously hung on NV. Do not remove the safe packed
  storage path without dedicated offset/JIT/lifetime regressions.
- Preserve `--no-nv-quant`; revisit default enablement on quality/reliability
  failures. Publish settings, timings and tradeoffs, not unsupported speed claims.
- [Architecture](architecture.md) describes ownership; this roadmap tracks gates.
