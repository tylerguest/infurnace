# Phase 0E Baseline Report

Phase 0E establishes the current upstream tinygrad baseline for the pinned
Qwen3-0.6B Q8_0 checkpoint on the NVIDIA GeForce RTX 5060. It records separate
checkpoint/model loading, steady prefill, steady decode, and closed-loop sequential
generation measurements under lazy Q8_0 and realized-FP16 weight policies.

## Evidence

- [Functional baseline](phase0d-upstream-functional.md)
- [Steady prefill baseline](phase0e-upstream-prefill.md)
- [Steady decode baseline](phase0e-upstream-decode.md)
- [Closed-loop upstream generation baseline](phase0e-upstream-end-to-end.md)

All benchmark entry points synchronize device work, emit schema-versioned validated
JSON, retain raw timing and output samples, verify exact checkpoint identity, and
record device, driver, execution, and memory-source metadata. Raw JSON belongs in
the ignored `results/` directory; the linked documents are the committed summaries.

## Summary

| Metric | Lazy Q8_0 expressions | Realized FP16 |
| --- | ---: | ---: |
| Model loading | 1.21 s | 3.52 s |
| Steady 16-token prefill TTFT | 105.53 ms | 134.10 ms |
| Steady decode latency | 20.91 ms/token | 16.61 ms/token |
| Decode throughput | 47.82 tok/s | 60.21 tok/s |
| Closed-loop 16-token latency | 413.49 ms | 374.04 ms |
| Closed-loop generated throughput | 38.69 tok/s | 42.77 tok/s |
| Benchmark startup-to-ready | 18.00 s | 15.97 s |
| Sampled process GPU peak | 1,220 MiB | 2,365 MiB |

Recorded functional, decode, and closed-loop outputs agreed exactly across both
weight policies.

## Policy Decision

Realized FP16 is the provisional latency-oriented policy for the first constrained
single-request implementation. It improves steady decode and complete generation
latency, and its additional memory fits the measured 8 GiB device budget at the
current 1,024-token baseline context.

Lazy Q8_0 remains an explicit memory-oriented alternative. It loads faster, improves
short-prompt prefill, and preserves approximately 1.1 GiB for future KV capacity.
Phase 1B implements both policies and retains realized FP16 as the provisional
default. Phase 1C must revisit the choice against Infurnace's stateless model rather
than treating upstream behavior as a permanent serving policy.

## Phase Gate

The Phase 0 baseline gate is satisfied for the selected checkpoint, current tinygrad
contracts, NV backend, and single-device topology:

- output is recorded and repeatable;
- startup and model loading are measured;
- prefill, decode, and end-to-end latency are measured separately;
- generated-token throughput is recorded from a complete closed-loop window;
- host, tinygrad-requested, and sampled driver-visible memory are distinguished;
- lazy and realized-FP16 policies are measured in clean processes.

## Non-Claims

Phase 0 contains no Infurnace model runner or server runtime. The end-to-end result
directly invokes upstream `tinygrad.llm.Transformer.generate()` with fixed token IDs.
It does not validate external KV, tokenization, request lifecycle, scheduling,
queueing, HTTP, streaming, cancellation, backpressure, concurrency, or production
readiness. Those capabilities remain gated by later roadmap phases.
