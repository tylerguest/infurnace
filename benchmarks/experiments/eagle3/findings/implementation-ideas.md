# Implementation follow-up: tinygrad and TensorRT-LLM

[Experiments](../README.md) · [Reports](../results/README.md)

Report filenames below refer to `../results/`; commands run from the repository root.

Sources read during this follow-up:

- Local tinygrad `docs/developer/speed.md` and `docs/developer/runtime.md`.
- https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/speculative-decoding.md
- https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog11_GPT_OSS_Eagle3.md
- First split-statistics/output stages of https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/fusedSampling/fusedSamplingMultiCta.cu

These upstream main-branch documents are implementation references, not evidence that our installed tinygrad pin or consumer GPU supports every described capability. No TensorRT dependency or driver change was installed.

## Transferable ideas

1. **Separate scheduler, codegen, and submission costs.** tinygrad's speed guide explicitly discusses recomputation versus materialization and BEAM's equivalent-kernel search. The previous materialized-score result is a concrete scheduler win. `inspect_sampling.py --beam 2` now screens codegen separately, prints decision/reference outputs, and records search width. PTX is separately compiled; SASS comes from the actual runtime binary.
2. **Split vocabulary across CTAs.** TensorRT-LLM's small-batch sampler launches multiple CTAs per row, reducing contiguous slices into scratch partials, then merging statistics. This directly addresses our single-block vocabulary scans. However, its online-softmax and greedy semantics are not drop-in equivalents to our target's Gumbel expression and RNG advancement. The earlier reshape prototype changed NaN behavior; do not promote it without explicit semantic validation.
3. **Reuse preallocated workspaces and capture submissions.** TensorRT's dynamic-tree buffers are preallocated; its deployment guide configures CUDA graphs. Our mapped input/output scratch and TinyJit captures pursue this without swapping runtimes. Pointer lifetime, cancellation, and previous-reader synchronization remain necessary.
4. **Evaluate draft length by total request time.** TensorRT exposes `max_draft_len` and optional tree budgets. A second descendant is being re-evaluated with materialized scores and mapped I/O, rather than extrapolating from older high-overhead runs. More accepted tokens is not sufficient if three-row verification costs too much.
5. **Trees and suffix matching are subsequent options, not free speedups.** TensorRT documents dynamic trees and a GPU suffix-automaton enhancer. Trees require correct tree attention, accepted-path KV commitment, and branching draft state; our current linear verifier cannot simply consume sibling tokens. Prior n-gram screening lost useful proposals, so a suffix override needs measured validation before integration.

The GB200/B200 guide uses different hardware and explicitly describes greedy-only serving. Its configurations do not justify dropping our RNG expression or importing its performance expectations to the RTX 5060.

## Completed iterations

All application comparisons below use exact-output baseline/EAGLE/baseline bracketing, two warmup pairs, two measured repeats, 256 outputs/context 512, calibrated head, mapped I/O, materialized scores, and zero-copy features. Defaults remain unchanged.

| Iteration | EAGLE tok/s | Bracketed baseline | Ratios | Acceptance / rounds |
|---|---:|---:|---:|---:|
| Two proposals | 111.00 / 110.83 | 105.12 / 104.36 | 1.056 / 1.062 | 129 / 127 |
| One proposal, sampler-only BEAM=2 | 126.69 / 118.16 | 104.91 / 103.13 | 1.208 / 1.146 | 101 / 155 |

Reports: `qwen3-4b-eagle3-mapped-calibrated-two.json` and `qwen3-4b-eagle3-sampling-beam.json`. The BEAM application run used `/tmp/eagle-sampling-beam-benchmark.py`; its scoped equivalent is archived as `sampling_beam_benchmark.py`, with CPU scope/exception-restoration coverage. It only searches mapped sampler captures, never baseline or target projections.

Neither iteration establishes an improvement over the earlier **129.18 / 129.29 tok/s** one-proposal result. Baseline speed also varied across runs; do not interpret cross-session differences as a precisely isolated regression. More proposals clearly failed to pay for themselves here. No claim of reaching 200 tok/s.

### Assembly observations

Matched VIZ sampler-only runs (`/tmp/eagle-sampler-beam-control/` and `/tmp/eagle-sampler-beam-replay/`) measured approximately **353 us materialized BEAM=0 versus 279 us BEAM=2** for synchronized capture replay. These exclude RNG generation and are not application comparisons.

An important diagnostic correction: dumping the standalone expression is insufficient to describe JIT codegen. BEAM produced additional capture-specific kernels. The inspector now saves selected JIT CUDA/PTX/cubin/SASS separately as `*_jit_*`, using HCQ capture metadata's shader keys to identify programs, and checks warmup/replay decisions against a BEAM=0 reference outside timing. A first attempt also picked up CPU HCQ submission code and failed NVRTC compilation; the tool now filters CPU sources rather than mislabeling them GPU shaders. That report is explicitly aborted.

The selected-program capture dump (`/tmp/eagle-sampler-selected-keys/`) shows a max pass split across **64 blocks of two threads**, versus the previous single 32-thread block. Its index pass instead uses **128 threads in one block**, still looping 1,187 times. Resource reports show **53 / 39 registers, zero local memory** for those two shaders. Parallelization is better in one pass but still poorly balanced; this is not evidence of spills. The standalone dumps retain the older schedules, illustrating why the actual capture kernels matter.

Use `--debug 7` to **print CUDA, PTX, and runtime SASS**, plus `OUTPUT ... decision=... reference=...`. `--debug 6` prints CUDA/PTX and saves SASS to disk. Earlier `/tmp/eagle-sampler-beam-jit-fixed/` and `/tmp/eagle-sampler-beam-final/` lists were cache misses, including rejected BEAM candidates—not proof of selected execution. A direct PROGRAM graph traversal also found zero shaders because HCQ had already lowered the submission. The final tool instead matches HCQ metadata keys to runtime programs, asserts complete matching, and saves unique selected shaders; it does not infer execution counts from compiler-cache size.
