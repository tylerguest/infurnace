# Roadmap

Infurnace is built as an inference server whose production compute path is entirely
tinygrad. The roadmap proves model and mutation correctness first, exposes a limited
real server early, and then improves concurrency and cache utilization without
changing the request or protocol contracts.

Every phase has a correctness gate. A feature is not part of a release until its
gate passes on the selected model, current tinygrad codebase, backend, and execution
topology.

Each phase is divided into lettered subphases. A subphase is a reviewable,
independently tested implementation increment; completing one does not imply that
the parent phase gate has passed. CPU-only contract and state tests should remain
separate from model-dependent and accelerator tests so normal development does not
require the checkpoint or execution hardware.

## Phase 0: Current NV Baseline

### Phase 0A: Repository Foundation

- Define the Python package, dependency installation, and test configuration.
- Develop against current tinygrad rather than pinning a commit or maintaining a
  supported-version matrix.
- Treat Infurnace's tinygrad contract tests as the compatibility boundary and adapt
  them when upstream APIs or semantics change.
- Make backend and execution topology explicit test and benchmark inputs. Use the
  available NVIDIA GeForce RTX 5060 with tinygrad `DEV=NV` for the first recorded
  baseline without making it a product boundary.

**Subphase gate:** Infurnace installs as a source-layout package, its CPU-only tests
run without initializing an accelerator, and NV, model, and slow tests can be
selected explicitly.

### Phase 0B: Checkpoint Identity and Acquisition

- Pin the official Qwen3-0.6B Q8_0 GGUF by URL, revision, byte size, and SHA-256.
- Download to a temporary path, verify identity while streaming, and publish the
  local artifact atomically only after its size and hash match.

**Subphase gate:** A machine-readable manifest and acquisition tool reject missing
identity fields, partial downloads, and artifact mismatches before tinygrad parsing.

### Phase 0C: GGUF Inspection and Tinygrad Contracts

- Record model configuration and tokenizer metadata from the pinned artifact.
- Inspect the complete GGUF tensor inventory through current tinygrad.
- Establish focused contracts for device selection, initialized buffer realization,
  mutation dependencies, TinyJit replay, and compatible input-buffer replacement.

**Subphase gate:** GGUF inspection matches the manifest and reports configuration,
tokenizer metadata, and every tensor. CPU and NV contract tests expose the current
tinygrad behavior on which later phases rely.

### Phase 0D: Upstream Functional Baseline

- Run tinygrad's existing Qwen load, prompt, and decode path with fixed token IDs.
- Establish lazy quantized expressions and fully realized FP16 weights as separate
  execution policies rather than assuming Q8 storage remains resident.

**Subphase gate:** The verified checkpoint loads with current tinygrad on `DEV=NV`,
a fixed token-ID workload produces recorded repeatable greedy output, and the tested
weight-realization policies are identified separately.

### Phase 0E: Benchmark Harness and Report

- Establish separate model, decode, and end-to-end benchmark entry points.
- Measure GGUF loading peak memory, lazy-weight behavior, and fully realized FP16
  weight behavior.
- Record a schema version, checkpoint identity, device and driver information,
  execution settings, raw timing samples, and memory sources.
- Keep raw benchmark output separate from a committed summary of reproducible
  commands, results, and known measurement limitations.

**Subphase gate:** Benchmark entry points synchronize device work correctly, emit
validated structured results, and a committed report covers the required Phase 0
measurements on execution devices without competing workloads.

**Gate:** The current environment produces a documented baseline with recorded
output, latency, generated-token throughput, startup time, and memory measurements.

## Phase 1: Stateless Qwen3 Model Runner

### Phase 1A: Exact Model Contract

- Document dimensions, RoPE, Q/K normalization, biases, tied embeddings, dtypes,
  quantization, and every required GGUF tensor.

**Subphase gate:** An immutable Infurnace model configuration is derived from the
verified GGUF and rejects unsupported or inconsistent metadata without constructing
the model.

### Phase 1B: Strict Weight Mapping

- Copy the artifact into a private snapshot while verifying it, then keep that
  snapshot open through parsing so pathname replacement or in-place mutation cannot
  change the contents tinygrad loads.
- Load GGUF metadata and weights through tinygrad from that verified snapshot.
- Support lazy quantized FP16 expressions and realized FP16 weights. Retain realized
  FP16 as the provisional default from Phase 0, then remeasure both policies with the
  stateless Infurnace forward pass in Phase 1C.

**Subphase gate:** Tinygrad parses the private verified snapshot and every required
model parameter maps to one validated GGUF tensor;
missing, duplicate, incorrectly shaped, and unexpected tensors fail loading.

### Phase 1C: Stateless Eager Forward

- Implement Qwen3 model operations with tinygrad Tensor APIs.
- Return last-token logits and keep token history and KV state outside the model.
- Keep PyTorch restricted to tests and development comparison tools.

**Subphase gate:** Component operations and the assembled eager forward pass run
from fixed token IDs, return logits, and retain no conversation or cache state.

### Phase 1D: Independent Numerical Validation

- Compare configuration, mapped weights, layer intermediates, logits, and greedy
  tokens with a PyTorch implementation using the same dequantized GGUF values.
- Compare final behavior with the current `tinygrad.llm` baseline, accounting for its
  model-owned cache and sampling interface.

**Subphase gate:** Differences are reported from the first failing layer or position
with maximum absolute and relative errors under documented per-dtype tolerances.

**Gate:** Stateless prompt logits and greedy tokens match the independent reference
within documented per-dtype tolerances. Missing and unexpected GGUF tensors fail
loading instead of being silently ignored.

## Phase 2: External Contiguous KV

### Phase 2A: Cache Contract and Allocation

- Select explicit model, accumulation, and KV-cache dtypes.
- Allocate initialized, contiguous, realized KV storage outside the model.
- Document cache shape, axis order, ownership, replacement contract, and memory cost.

**Subphase gate:** Cache allocation is deterministic and initialized, its measured
size matches the contract, and the model cannot allocate or retain conversation KV.

### Phase 2B: Eager Prefill and Decode

- Implement single-request bounded chunked prefill and one-token decode.
- Use a conservative context limit derived from measured per-device and aggregate
  memory budgets for the selected execution topology.
- Represent cache writes and write-before-read dependencies explicitly.
- Validate eager execution before adding TinyJit.

**Subphase gate:** Eager cache writes mutate only caller-assigned locations, chunked
prefill matches unchunked execution, and cached decode matches full recomputation.

### Phase 2C: TinyJit Execution Contracts

- Capture independent prefill and decode contracts only after their input shapes,
  views, and symbolic variables are documented.
- Give each contract its own `TinyJit` instance.
- Use disposable cache state during both warmup and capture.
- Replay with a different cache allocation matching the complete input contract.

**Subphase gate:** Warmup, capture, immediate post-capture execution, and replay have
explicitly validated mutations and work with a compatible replacement cache.

### Phase 2D: Stateful Stress Validation

- Compare cached decode logits with full-sequence recomputation at every position,
  including chunk and cache boundaries.
- Repeat conversations, cache replacement, cancellation-equivalent cleanup, and
  context-boundary workloads without reusing stale state.

**Subphase gate:** Repeated and boundary-heavy traces preserve logits and caller-owned
cache contents without state leaking between conversations.

**Gate:** Single-request cached generation remains correct through eager execution,
warmup, capture, replay, cache replacement, and repeated conversations. The model
owns no conversation history or KV state.

## Phase 3: Minimal Streaming Server

### Phase 3A: Request State Machine

- Implement request IDs, waiting, prefilling, decoding, and terminal outcomes.
- Support rejection, failure, and cancellation from every nonterminal state.
- Add bounded input, output, context, and generated-token limits.

**Subphase gate:** CPU-only transition tests prove that every request reaches at most
one terminal outcome and releases logical resources exactly once.

### Phase 3B: Scheduler and Execution Plans

- Add immutable prefill and decode execution plans.
- Keep prefill single-request and execution serial in this phase. Concurrent clients
  may queue, but the server makes no throughput claim yet.

**Subphase gate:** A fake runner produces deterministic queue order and plans that do
not change after submission; cancellation suppresses in-flight output safely.

### Phase 3C: Offline Engine

- Expose an offline `add_request`, `step`, cancellation, and token-output API.

**Subphase gate:** The complete request and scheduler path works with a deterministic
fake runner, including queueing, completion, failure, rejection, and cancellation.

### Phase 3D: Tokenization and Output Streaming

- Add tokenizer and detokenizer integration, including incremental UTF-8 decoding.
- Bound output queues to enforce backpressure without blocking scheduler progress.
- Use greedy sampling so end-to-end behavior is deterministic.

**Subphase gate:** Offline output never emits malformed UTF-8, terminal output is
emitted once, and slow consumers cannot grow memory without bound.

### Phase 3E: Real Runner Integration

- Connect the external contiguous-KV runner to the same offline engine contracts.

**Subphase gate:** Fixed requests produce the same greedy tokens through direct model
execution and the offline engine, with no cache state leaked on any terminal path.

### Phase 3F: HTTP Adapter

- Put a small documented HTTP completion and streaming adapter over that same engine.
- Cancel disconnected clients and bound output queues to enforce backpressure.
- Add readiness, liveness, and deterministic capacity errors.

**Subphase gate:** Protocol tests prove that HTTP cannot bypass the engine and that
disconnect, backpressure, health, and capacity errors preserve engine invariants.

**Gate:** Concurrent clients can queue, stream, finish, fail, and cancel independently
without leaked cache state or a protocol path that bypasses the engine. Repeated
runs produce the same schedule and greedy output.

**Release:** `v0.1`, constrained tinygrad inference server.

## Phase 4: Fixed Batched Decode

### Phase 4A: Persistent Slot State

- Add persistent request slots and per-request positions and active masks.
- Select an initial slot count and context limit that fit the measured memory budget.
- Test row movement, slot reuse, requests finishing at different lengths, and EOS on
  the first sampled token.

**Subphase gate:** Slot allocation and reuse reset all request-private state and do
not change another request's output under randomized CPU-only lifecycle traces.

### Phase 4B: Fixed Decode Contracts

- Add decode contracts for batch sizes justified by measured workloads, beginning
  with 1, 2, and 4 rather than an unbounded bucket sequence.
- Keep prefill one request per execution batch.
- Use contiguous per-request cache regions as the numerical baseline.
- Mask inactive cache writes or provide a unique dummy write slot per padded row.
- Keep changing metadata in persistent fixed-shape device inputs.

**Subphase gate:** Each supported shape agrees with repeated independent
single-request execution, including inactive rows and position boundaries.

### Phase 4C: Batched Sampling and Engine Integration

- Batch the existing tinygrad greedy sampler and return only selected token IDs to
  the engine.

**Subphase gate:** Requests finish independently and greedy output is unchanged by
batch membership, row movement, inactive padding, or slot reuse.

### Phase 4D: Steady-State Stability

- Verify that steady decode creates no new persistent device buffers and does not
  recapture a contract.

**Subphase gate:** Long decode traces preserve allocation and capture counts after
warmup for every supported batch contract.

**Gate:** Fixed ragged decode batches match independent single-request execution
without cache corruption, inactive-row mutation, recapture, or cross-request output
changes.

**Release:** `v0.2`, fixed batched tinygrad inference server.

## Phase 5: Paged Decode KV

### Phase 5A: Logical Page Allocator

- Choose and benchmark a page size using the selected model and backend.
- Add a logical page pool, request block tables, reference counts, and reserved dummy
  read state.
- Preallocate the physical KV pool as initialized tinygrad tensors.
- Derive pool capacity from measured weight, workspace, and backend reserve.
- Plan multi-page allocations atomically and roll back failed admission.
- Track active-request and in-flight execution ownership separately.
- Test randomized allocate, retain, release, cancel, exhaustion, and stale-ID traces.

**Subphase gate:** CPU-only randomized traces preserve allocator and ownership
invariants, including atomic failure and delayed in-flight reclamation.

### Phase 5B: Indexed KV Store

- Implement a custom UOp indexed KV-store operation with masked inactive writes.

**Subphase gate:** Indexed stores match dense reference updates across page and dtype
boundaries without inactive or aliased dummy writes.

### Phase 5C: Paged Decode Attention

- Implement custom UOp paged one-token decode attention that follows block tables.
- Preserve explicit store-before-attention dependencies.
- Compare every paged result with contiguous dense attention across page boundaries,
  partial pages, GQA head mapping, and supported dtypes.

**Subphase gate:** Paged attention and contiguous attention agree independently of
physical page assignment, partial tails, and GQA head mapping.

### Phase 5D: Engine Integration

- Replace contiguous slot ownership with page ownership without changing request,
  scheduler, execution-plan, or protocol contracts.

**Subphase gate:** Completion, failure, cancellation, and in-flight execution reclaim
pages at the correct time under end-to-end request traces.

**Gate:** Paged and contiguous decode agree numerically, allocator invariants hold
under stress, and pages are reclaimed only after active and in-flight ownership ends.

## Phase 6: Continuous Batching

### Phase 6A: Policy Simulation

- Admit and retire requests between engine iterations.
- Prioritize decode sufficiently to bound inter-token latency.
- Schedule one single-request prefill chunk under a bounded token budget when decode
  policy permits it.
- Keep prefill and decode as separate execution plans.
- Add cache-aware admission, bounded waiting, and explicit fairness metrics.

**Subphase gate:** Deterministic CPU-only workload simulations satisfy documented
admission, decode-priority, bounded-waiting, and fairness invariants.

### Phase 6B: Engine Integration and Replay

- Replay steady, bursty, long-prompt, and mixed prefill/decode workloads.

**Subphase gate:** Real execution preserves simulated scheduling decisions, outputs,
and page ownership across the workload suite.

### Phase 6C: Overlap Evaluation

- Introduce control-plane/device overlap only with an explicit completion mechanism
  that prevents early page reuse.
- Measure a unified token-budget policy only after the separate policy is stable.

**Subphase gate:** Any retained overlap or unified policy improves a named metric
without changing output, page lifetime, fairness, or latency bounds.

**Gate:** Continuous batching improves measured throughput over fixed batching
without output changes, leaked pages, unbounded starvation, or latency outside the
documented policy.

**Release:** `v0.3`, paged continuous-batching inference server.

## Phase 7: Production Sampling

### Phase 7A: Sampling Semantics

- Define sampler behavior for temperature zero, invalid values, ties, NaN and Inf
  logits, and unsupported parameter combinations.

**Subphase gate:** CPU reference tests define every accepted parameter boundary and
failure mode before stochastic device implementation begins.

### Phase 7B: Sampling Primitives

- Add temperature, fixed-contract top-k, top-p, penalties, and requested logprobs.
- Measure built-in tinygrad operations before writing custom sampling UOps.

**Subphase gate:** Each primitive independently matches the CPU reference and reports
requested logprobs under documented numerical tolerances.

### Phase 7C: Request-Local Randomness

- Store explicit random key and counter state per request.
- Keep random streams stable across batch reordering and request-slot reuse.

**Subphase gate:** Seeded output is invariant to unrelated requests, row movement,
batch shape, and slot reuse, and statistical tests accept sampled distributions.

### Phase 7D: Fused Sampling Evaluation

- Decide by profiling whether production contracts combine model execution and
  sampling to avoid materializing full logits between stages.
- Preserve a logits-returning validation path.

**Subphase gate:** Fusion is retained only if it improves measured performance while
matching the unfused logits and sampler validation paths.

**Gate:** Sampling distributions pass statistical tests, seeded requests are
reproducible independently of scheduling, and one request cannot consume another's
random stream or token history.

## Phase 8: Prefix Cache

### Phase 8A: Prefix Identity

- Cache only complete immutable token blocks.
- Namespace entries by checkpoint and all execution settings that affect KV values.
- Verify token identity on a hash match.

**Subphase gate:** Prefix lookup cannot select unrelated tokens under collision and
namespace tests, and partial blocks are never published.

### Phase 8B: Shared Ownership

- Share pages by reference and keep partial active tails private.
- Protect active and in-flight pages from reclamation during cache eviction.

**Subphase gate:** Randomized sharing, cancellation, and in-flight traces preserve
separate active, prefix, and execution references.

### Phase 8C: Capacity and Eviction

- Define capacity, eviction order, and invalidation on model reload.

**Subphase gate:** Deterministic eviction and reload invalidation reclaim only
eligible entries and leave active output numerically unchanged.

**Gate:** Prefix hits preserve numerical output, collisions cannot select unrelated
tokens, and all references remain valid under cancellation, eviction, and concurrent
execution.

## Phase 9: Server Hardening and Isolation

### Phase 9A: Execution Isolation

- Separate API and tokenization from execution worker processes.
- Define a versioned IPC request, cancellation, result, and health protocol.
- Add worker startup failure, crash, timeout, and graceful-shutdown behavior.

**Subphase gate:** Worker lifecycle fault tests terminate or fail every request
without hangs, duplicate terminal output, or orphaned worker state.

### Phase 9B: Compatible Protocol Surface

- Add an OpenAI-compatible chat-completions subset and exact chat-template contract.

**Subphase gate:** Supported requests and streaming events match the documented
subset, while unsupported fields fail deterministically at the protocol boundary.

### Phase 9C: Observability and Deployment Policy

- Add structured metrics for queueing, time to first token, inter-token latency,
  throughput, cache occupancy, rejection, cancellation, and compilation.
- Add authentication and deployment policy only at the protocol boundary.

**Subphase gate:** Metrics reconcile with request outcomes under load, and deployment
policy cannot alter or bypass engine semantics.

**Gate:** Concurrent clients stream independently under backpressure, worker failure
does not hang API requests, and process isolation does not change engine output.

**Release:** `v0.4`, hardened network inference server.

## Phase 10: Batched Prefill Evaluation

### Phase 10A: Workload and Contract Selection

- Profile whether single-request chunked prefill is a measured bottleneck.
- Choose padded batch-by-sequence or packed-token execution based on target workload.
- For padded prefill, define valid-query and key masks and inactive-write behavior.
- For packed prefill, define query offsets, request mapping, absolute positions,
  per-query context, offset causal boundaries, and per-request logits indices.

**Subphase gate:** A measured workload justifies one representation and its complete
input, masking, cache-write, and output contract is documented before implementation.

### Phase 10B: Numerical Implementation

- Compare the candidate with repeated single-request chunked prefill.
- Keep decode scheduling independent of the selected prefill representation.

**Subphase gate:** The candidate matches repeated single-request prefill across mixed
lengths and cached prefixes without cross-request attention or mutation.

### Phase 10C: Performance Decision

- Measure time to first token, throughput, memory, compilation, and scheduler impact
  against the single-request baseline.

**Subphase gate:** Batched prefill is retained only when it improves a target metric
without weakening decode scheduling or correctness contracts.

**Gate:** Batched prefill improves measured time to first token or throughput and
matches single-request prefill without cross-request attention or cache writes.

## Phase 11: Compiled Function Artifacts

### Phase 11A: Artifact Contract

- Represent local execution as versioned model, prefill, decode, and sampler
  contracts.
- Record GGUF hash, tinygrad-defined compatibility metadata, device target, complete
  input contract, workspace, and benchmark metadata.

**Subphase gate:** Artifact identity fully describes the local execution contract and
rejects incomplete or incompatible metadata before loading executable content.

### Phase 11B: Validated Loading

- Load artifacts only if tinygrad defines a validated format for the selected
  backend.
- Verify artifact output before benchmarking or serving it.

**Subphase gate:** A loaded artifact passes the same numerical and mutation tests as
local compilation before it can become ready for traffic.

### Phase 11C: Performance Decision

- Compare artifact load time and performance with local JIT compilation.

**Subphase gate:** Artifact loading is retained only if it improves startup or
deployment behavior without changing runtime contracts or steady performance.

**Gate:** A compatible artifact replaces local compilation without changing model,
scheduler, cache-manager, sampling, or request semantics.

## Phase 12: Scalable Execution Topologies

### Phase 12A: Topology and Placement Contracts

- Represent devices, workers, model replicas, model shards, and communication groups
  as explicit versioned configuration.
- Keep request state, scheduling policy, cache ownership, and protocol behavior
  independent of placement.
- Validate capacity and admission atomically across every resource participating in
  an execution plan.

**Subphase gate:** Equivalent placements expose the same engine contract, and invalid
or partially available topologies fail before accepting traffic.

### Phase 12B: Replicated Serving

- Route requests across model replicas with deterministic capacity and health-aware
  policy.
- Isolate cache, random state, compilation, cancellation, and failures by replica.
- Measure throughput scaling, load balance, and tail latency under mixed workloads.

**Subphase gate:** Replication improves measured capacity without changing request
output or allowing one replica's lifecycle to corrupt another.

### Phase 12C: Sharded Model Execution

- Select tensor, pipeline, or other sharding strategies from model and hardware
  measurements rather than embedding one strategy in the engine.
- Define collective operations, intermediate ownership, synchronization, cache
  placement, and failure semantics through tinygrad execution contracts.
- Compare sharded logits and generated tokens with an unsharded numerical reference.

**Subphase gate:** Sharded execution meets documented numerical tolerances and
preserves cache mutation, sampling, cancellation, and output-ordering invariants.

### Phase 12D: Distributed Fault and Performance Validation

- Extend execution topology across hosts without changing the public request API.
- Define startup, membership, timeout, partial failure, draining, and recovery
  behavior for distributed workers.
- Measure communication cost, scaling efficiency, throughput, and latency against
  local execution topologies.

**Subphase gate:** Distributed execution has deterministic failure behavior, does not
leak or duplicate requests, and is retained only for workloads where it provides a
measured benefit.

**Gate:** Replicated, sharded, and distributed placements preserve engine semantics,
numerical output, cache ownership, and request isolation while providing documented
capacity or performance gains.

## Deferred Scope

- MoE and multimodal models
- Speculative decoding
- LoRA
- Broad compatibility with unrelated decoder architectures
- Remote or tiered KV storage
- A tensor framework or independent device compiler
