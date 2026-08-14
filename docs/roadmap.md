# Roadmap

Infurnace is built as an inference server whose production compute path is entirely
tinygrad. The roadmap proves model and mutation correctness first, exposes a limited
real server early, and then improves concurrency and cache utilization without
changing the request or protocol contracts.

Every phase has a correctness gate. A feature is not part of a release until its
gate passes on the selected model, the current tinygrad codebase, backend, and GPU.

## Phase 0: Current NV Baseline

- Define the Python package, dependency installation, and test configuration.
- Develop against current tinygrad rather than pinning a commit or maintaining a
  supported-version matrix.
- Treat Infurnace's tinygrad contract tests as the compatibility boundary and adapt
  them when upstream APIs or semantics change.
- Use the tinygrad `NV` backend explicitly on the NVIDIA GeForce RTX 5060.
- Pin the official Qwen3-0.6B Q8_0 GGUF by URL, revision, byte size, and SHA-256.
- Record model configuration and tokenizer metadata from the pinned artifact.
- Measure tinygrad's existing Qwen load, prompt, and decode path as a baseline.
- Measure GGUF loading peak memory, lazy-weight behavior, and fully realized FP16
  weight behavior instead of assuming Q8 storage remains resident during execution.
- Establish separate model, decode, and end-to-end benchmark entry points.

**Gate:** The current environment produces a documented baseline with recorded
output, latency, generated-token throughput, startup time, and memory measurements.

## Phase 1: Stateless Qwen3 Model Runner

- Document dimensions, RoPE, Q/K normalization, biases, tied embeddings, dtypes,
  quantization, and every required GGUF tensor.
- Load GGUF metadata and weights through tinygrad.
- Decide and document whether serving uses lazy quantized expressions or realized
  FP16 weights based on measured memory and latency.
- Implement Qwen3 model operations with tinygrad Tensor APIs.
- Return last-token logits and keep token history and KV state outside the model.
- Keep PyTorch restricted to tests and development comparison tools.
- Compare configuration, mapped weights, layer intermediates, logits, and greedy
  tokens with a PyTorch implementation using the same dequantized GGUF values.
- Compare final behavior with the current `tinygrad.llm` baseline, accounting for its
  model-owned cache and sampling interface.

**Gate:** Stateless prompt logits and greedy tokens match the independent reference
within documented per-dtype tolerances. Missing and unexpected GGUF tensors fail
loading instead of being silently ignored.

## Phase 2: External Contiguous KV

- Select explicit model, accumulation, and KV-cache dtypes.
- Allocate initialized, contiguous, realized KV storage outside the model.
- Implement single-request bounded chunked prefill and one-token decode.
- Use a conservative context limit derived from the RTX 5060 memory budget.
- Represent cache writes and write-before-read dependencies explicitly.
- Validate eager execution before adding TinyJit.
- Capture independent prefill and decode contracts only after their input shapes,
  views, and symbolic variables are documented.
- Give each contract its own `TinyJit` instance.
- Use disposable cache state during both warmup and capture.
- Replay with a different cache allocation matching the complete input contract.
- Compare cached decode logits with full-sequence recomputation at every position,
  including chunk and cache boundaries.

**Gate:** Single-request cached generation remains correct through eager execution,
warmup, capture, replay, cache replacement, and repeated conversations. The model
owns no conversation history or KV state.

## Phase 3: Minimal Streaming Server

- Implement request IDs, waiting, prefilling, decoding, and terminal outcomes.
- Support rejection, failure, and cancellation from every nonterminal state.
- Add immutable prefill and decode execution plans.
- Add bounded input, output, context, and generated-token limits.
- Expose an offline `add_request`, `step`, cancellation, and token-output API.
- Put a small documented HTTP completion and streaming adapter over that same engine.
- Add tokenizer and detokenizer integration, including incremental UTF-8 decoding.
- Cancel disconnected clients and bound output queues to enforce backpressure.
- Add readiness, liveness, and deterministic capacity errors.
- Keep prefill single-request and execution serial in this phase. Concurrent clients
  may queue, but the server makes no throughput claim yet.
- Use greedy sampling so end-to-end behavior is deterministic.

**Gate:** Concurrent clients can queue, stream, finish, fail, and cancel independently
without leaked cache state or a protocol path that bypasses the engine. Repeated
runs produce the same schedule and greedy output.

**Release:** `v0.1`, constrained single-GPU tinygrad inference server.

## Phase 4: Fixed Batched Decode

- Add persistent request slots and per-request positions and active masks.
- Select an initial slot count and context limit that fit the measured memory budget.
- Add decode contracts for batch sizes justified by measured workloads, beginning
  with 1, 2, and 4 rather than an unbounded bucket sequence.
- Keep prefill one request per execution batch.
- Use contiguous per-request cache regions as the numerical baseline.
- Mask inactive cache writes or provide a unique dummy write slot per padded row.
- Keep changing metadata in persistent fixed-shape device inputs.
- Batch the existing tinygrad greedy sampler and return only selected token IDs to
  the engine.
- Test row movement, slot reuse, requests finishing at different lengths, and EOS on
  the first sampled token.
- Verify that steady decode creates no new persistent device buffers and does not
  recapture a contract.

**Gate:** Fixed ragged decode batches match independent single-request execution
without cache corruption, inactive-row mutation, recapture, or cross-request output
changes.

**Release:** `v0.2`, fixed batched tinygrad inference server.

## Phase 5: Paged Decode KV

- Choose and benchmark a page size using the selected model and backend.
- Add a logical page pool, request block tables, reference counts, and reserved dummy
  read state.
- Preallocate the physical KV pool as initialized tinygrad tensors.
- Derive pool capacity from measured weight, workspace, and backend reserve.
- Plan multi-page allocations atomically and roll back failed admission.
- Track active-request and in-flight execution ownership separately.
- Test randomized allocate, retain, release, cancel, exhaustion, and stale-ID traces.
- Implement a custom UOp indexed KV-store operation with masked inactive writes.
- Implement custom UOp paged one-token decode attention that follows block tables.
- Preserve explicit store-before-attention dependencies.
- Compare every paged result with contiguous dense attention across page boundaries,
  partial pages, GQA head mapping, and supported dtypes.

**Gate:** Paged and contiguous decode agree numerically, allocator invariants hold
under stress, and pages are reclaimed only after active and in-flight ownership ends.

## Phase 6: Continuous Batching

- Admit and retire requests between engine iterations.
- Prioritize decode sufficiently to bound inter-token latency.
- Schedule one single-request prefill chunk under a bounded token budget when decode
  policy permits it.
- Keep prefill and decode as separate execution plans.
- Add cache-aware admission, bounded waiting, and explicit fairness metrics.
- Replay steady, bursty, long-prompt, and mixed prefill/decode workloads.
- Introduce control-plane/device overlap only with an explicit completion mechanism
  that prevents early page reuse.
- Measure a unified token-budget policy only after the separate policy is stable.

**Gate:** Continuous batching improves measured throughput over fixed batching
without output changes, leaked pages, unbounded starvation, or latency outside the
documented policy.

**Release:** `v0.3`, paged continuous-batching inference server.

## Phase 7: Production Sampling

- Define sampler behavior for temperature zero, invalid values, ties, NaN and Inf
  logits, and unsupported parameter combinations.
- Add temperature, fixed-contract top-k, top-p, penalties, and requested logprobs.
- Store explicit random key and counter state per request.
- Keep random streams stable across batch reordering and request-slot reuse.
- Measure built-in tinygrad operations before writing custom sampling UOps.
- Decide by profiling whether production contracts combine model execution and
  sampling to avoid materializing full logits between stages.
- Preserve a logits-returning validation path.

**Gate:** Sampling distributions pass statistical tests, seeded requests are
reproducible independently of scheduling, and one request cannot consume another's
random stream or token history.

## Phase 8: Prefix Cache

- Cache only complete immutable token blocks.
- Namespace entries by checkpoint and all execution settings that affect KV values.
- Verify token identity on a hash match.
- Share pages by reference and keep partial active tails private.
- Protect active and in-flight pages from reclamation during cache eviction.
- Define capacity, eviction order, and invalidation on model reload.

**Gate:** Prefix hits preserve numerical output, collisions cannot select unrelated
tokens, and all references remain valid under cancellation, eviction, and concurrent
execution.

## Phase 9: Server Hardening and Isolation

- Separate API and tokenization from the GPU worker process.
- Define a versioned IPC request, cancellation, result, and health protocol.
- Add worker startup failure, crash, timeout, and graceful-shutdown behavior.
- Add an OpenAI-compatible chat-completions subset and exact chat-template contract.
- Add structured metrics for queueing, time to first token, inter-token latency,
  throughput, cache occupancy, rejection, cancellation, and compilation.
- Add authentication and deployment policy only at the protocol boundary.

**Gate:** Concurrent clients stream independently under backpressure, worker failure
does not hang API requests, and process isolation does not change engine output.

**Release:** `v0.4`, hardened single-GPU network server.

## Phase 10: Batched Prefill Evaluation

- Profile whether single-request chunked prefill is a measured bottleneck.
- Choose padded batch-by-sequence or packed-token execution based on target workload.
- For padded prefill, define valid-query and key masks and inactive-write behavior.
- For packed prefill, define query offsets, request mapping, absolute positions,
  per-query context, offset causal boundaries, and per-request logits indices.
- Compare the candidate with repeated single-request chunked prefill.
- Keep decode scheduling independent of the selected prefill representation.

**Gate:** Batched prefill improves measured time to first token or throughput and
matches single-request prefill without cross-request attention or cache writes.

## Phase 11: Compiled Function Artifacts

- Represent local execution as versioned model, prefill, decode, and sampler
  contracts.
- Record GGUF hash, tinygrad-defined compatibility metadata, device target, complete
  input contract, workspace, and benchmark metadata.
- Load artifacts only if tinygrad defines a validated format for the selected
  backend.
- Verify artifact output before benchmarking or serving it.
- Compare artifact load time and performance with local JIT compilation.

**Gate:** A compatible artifact replaces local compilation without changing model,
scheduler, cache-manager, sampling, or request semantics.

## Deferred Scope

- Multi-GPU and multi-node execution
- MoE and multimodal models
- Speculative decoding
- LoRA
- Broad compatibility with unrelated decoder architectures
- Remote or tiered KV storage
- A tensor framework or independent device compiler
