# Roadmap

Scope: **Qwen3.5-4B Q4_K_M + its DFlash draft**, one user/model/GPU, on tinygrad.
No other model families, batching, scheduler or multi-GPU implementation.

## Working foundations

- [x] Upstream Qwen3.5 transformer loading and bounded generation lifecycle.
- [x] One named preset with first-use target/draft downloads and local-file overrides.
- [x] Qwen3.5 tokenizer, text chat, context rollover, warmup and raw prompts.
- [x] Tinygrad-based HTTP serving and GGUF-template tool formatting/parsing.
- [x] Packed NV projections, tensor-core multirow GEMM, packed embedding gather
  and supported fused one-row GDN execution.
- [x] DFlash sequential reference, parallel NV verification and adaptive fallback.
- [x] CPU fixture coverage and opt-in NV kernel/real-model tests.
- [x] Remove the alternate model-family implementation and its attention adapters;
  preserve shared packed kernels for Qwen3.5 and DFlash.

These are implementation milestones, not full-model quality or reliability gates.

## Correctness and reliability gates

- [ ] Teacher-forced logits on identical tokens/positions: generic versus packed
  projections and fused GDN, including longer contexts and near-tie decisions.
- [ ] DFlash output parity across prompt types, EOS positions, tail blocks,
  repeated requests and context exhaustion.
- [ ] Long-session reset/cache/ownership tests, interrupts and device timeouts.
- [ ] Real-model HTTP tool-use and truncated-response validation.
- [ ] Peak GPU memory during load, warmup, ordinary chat and DFlash at several windows.

## Performance priorities

1. Multirow prefill for plain decoding and serving. Upstream recurrent generation
   currently forces one-row execution on NV; DFlash already has multirow prefill.
2. Bound DFlash attention work by active length before enlarging default context.
3. Measure whether reusable prompt/draft state can avoid full-history prefill
   without exposing rejected recurrent state or stale features.
4. Reduce startup capture cost while preserving stable JIT inputs and buffer identities.
5. Measure adaptive fallback against matched ordinary-decoding workloads.

Each optimization needs output/quality checks and synchronized cold/warm timings,
not only a kernel microbenchmark. Separate allocated capacity from active length,
and estimated traffic from hardware measurements.

## Usability

- [ ] Make the required default-model tokenizer dependency part of the base install.
- [ ] Add plain-decoding temperature controls; speculative sampling needs its own design.
- [ ] Publish supported-device and minimum-memory measurements before claiming broader GPU support.

See [architecture](architecture.md), [baseline](baseline.md),
[benchmarks](../benchmarks/README.md) and the detailed [modest-GPU plan](gpu-poor-plan.md).
