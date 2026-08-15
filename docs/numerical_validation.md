# Numerical Validation

Every optimized or batched implementation starts with a simpler independent
reference. End-to-end token agreement is necessary but not sufficient: a cache bug
can remain hidden until a later position or request schedule.

## Test Context

Every numerical result records:

- checkpoint source and SHA-256;
- quantization format and weight realization policy;
- model, accumulation, and KV-cache dtypes;
- prompt tokens, context length, and execution shape;
- JIT mode, device backend, and execution topology;
- absolute and relative tolerances and the reason they are appropriate.

Infurnace follows current tinygrad and uses contract tests rather than a recorded
source checkout or supported-version matrix.

## Validation Chain

1. Validate GGUF metadata, required tensor names, shapes, layouts, and dequantized
   values against the checkpoint manifest.
2. Compare stateless eager Qwen3 intermediates, last-token logits, and greedy tokens
   with an independent PyTorch implementation using the same dequantized GGUF
   values.
3. Compare final behavior with current `tinygrad.llm` while accounting for its
   internal cache and sampled-token interface. This is a useful same-runtime
   reference, not a substitute for the independent PyTorch reference.
4. Compare external contiguous cached decode with full-sequence recomputation at
   every position, chunk boundary, and configured context boundary.
5. Compare eager, first warmup, capture, immediate post-capture execution, and replay
   using both the capture cache and a different compatible cache allocation.
6. Compare fixed batched decode with independent single-request decode for every
   active row, including row movement, inactive padding, slot reuse, and requests
   finishing at different lengths.
7. Compare greedy and stochastic sampler primitives with CPU references. Use
   statistical tests for distributions and exact tests for seeded request isolation.
8. Compare paged KV store and attention with the contiguous dense path across page
   boundaries, partial pages, all GQA head mappings, and every supported cache dtype.
9. Compare deterministic end-to-end server output with the offline engine under the
   same request arrival and cancellation schedule.
10. Compare locally compiled and loaded-artifact outputs if artifact support is ever
    enabled.

## Comparison Rules

- Report the maximum absolute and relative error, not only pass or fail.
- Inspect the first layer and position where an error exceeds tolerance.
- Compare logits before argmax because identical tokens can hide large logit error.
- Use fixed token IDs for model tests so tokenizer changes do not alter the workload.
- Test cache replacement by checking both returned logits and the caller-provided
  cache contents.
- Test maximum and minimum symbolic values used by a captured contract.
- Never loosen a tolerance solely to make an optimization pass. Explain the expected
  precision change and validate it across representative prompts first.

## State Invariants

Serving correctness includes non-numerical state guarantees:

- a request can read or write only cache locations assigned to it;
- page reference counts equal active, prefix-cache, and in-flight owners;
- allocation failure does not partially modify a request or page pool;
- completion, failure, and cancellation reclaim all unshared pages after in-flight
  work completes;
- inactive JIT rows do not write live, shared, or concurrently written dummy state;
- KV stores complete before attention reads the corresponding positions;
- one request starting, moving, finishing, or cancelling does not alter another
  request's output;
- request-slot reuse resets token history, sampling state, and private cache state;
- JIT warmup, capture, replay, and output-buffer reuse do not leak state;
- a captured input buffer appears only once in the TinyJit input tree;
- prefix-cache hash matches cannot select a different token block;
- queued, executing, and completed plans have unambiguous cache ownership.

## Server Invariants

- protocol and offline callers use the same engine execution path;
- malformed or over-capacity requests fail before model execution;
- disconnect cancellation works from waiting, prefill, and decode states;
- output backpressure cannot block scheduler progress indefinitely;
- incremental decoding does not emit malformed UTF-8;
- terminal output is emitted once and cache release occurs once;
- health endpoints distinguish startup, ready, draining, and failed workers;
- deterministic requests produce the same result independent of unrelated clients.

Correctness is a phase gate. Optimization and new runtime features do not proceed
while numerical differences or state invariants remain unexplained.
