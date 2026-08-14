# Architecture

Infurnace is a single-GPU inference server for decoder-only language models. It uses
tinygrad as its only production tensor compiler and device runtime. It does not
define a tensor framework, render device code independently, or wrap
`tinygrad.llm.Transformer.generate()` as a multi-request serving interface.

The first supported system is deliberately narrow: Qwen3-0.6B, one NVIDIA GPU,
`DEV=NV`, one worker, external KV state, and a constrained network API. New model
architectures and serving optimizations are added only after this path is correct.

## Product Boundary

Infurnace owns the inference server:

- protocol validation, tokenization integration, streaming, and backpressure;
- request admission, state transitions, cancellation, and completion;
- prefill and decode scheduling, batching, and fairness policy;
- model execution contracts expressed with tinygrad tensors;
- KV-cache layout, capacity, lifetime, logical allocation, and prefix policy;
- serving-specific custom UOp algorithms such as indexed KV store and paged
  attention;
- sampling policy, per-request random state, and requested logprobs;
- metrics, health reporting, and worker supervision.

Tinygrad owns the compute substrate:

- GGUF parsing and tensor representations;
- Tensor and UOp semantics;
- scheduling, optimization, rendering, and compilation;
- `TinyJit` capture and replay;
- physical buffer allocation and hardware command submission;
- device graph support where the selected backend provides it;
- the NV backend and other hardware runtimes.

Infurnace decides which buffers exist, their shapes and dtypes, and how long they
must live. The tinygrad executor retains the corresponding Tensor objects while
tinygrad allocates and executes their physical backing buffers. Infurnace custom
kernels are defined through tinygrad UOps; Infurnace does not ship a parallel CUDA,
PTX, or device-runtime implementation.

PyTorch may be used only as an independent test reference. It is not a production
dependency or fallback execution path.

## Tinygrad Tracking

Infurnace develops against the current tinygrad codebase. It does not pin tinygrad,
maintain a supported-version matrix, or add compatibility branches for older
tinygrad APIs. A narrow executor boundary and focused contract tests cover the
tinygrad behavior Infurnace relies on: GGUF loading, buffer realization, assignment
dependencies, TinyJit input replacement, replay, output reuse, random primitives,
and custom UOps.

When upstream behavior changes, Infurnace updates to the current contract rather
than retaining the old one. Test and benchmark records may identify the checkout
that produced a result for diagnosis, but that identity is not a compatibility
promise.

## End-to-End Path

The offline and network APIs feed the same engine path:

```text
HTTP or offline caller
          |
          v
protocol validation and tokenization
          |
          v
engine request state and output queue
          |
          v
scheduler and KV-capacity admission
          |
          v
prefill or decode execution plan
          |
          v
tinygrad executor and model runner
          |
          v
TinyJit execution contract
          |
          v
generated programs and external KV storage
          |
          v
tinygrad sampling stage
          |
          v
token result, request update, and streamed output
```

Protocol adapters cannot bypass the engine or call model generation directly. This
keeps offline tests representative of network-serving behavior.

## Request Lifecycle

A request has one nonterminal state and one terminal outcome:

```text
                  +-------------+
                  |             v
waiting -> prefilling -> decoding -> finished
   |           |          |
   +-----------+----------+--------> cancelled
   +-----------+----------+--------> failed

admission failure -----------------> rejected
```

Cancellation is valid from every nonterminal state. A request can finish directly
after prefill when `max_tokens` is zero or when the first sampled token is terminal.
Failures carry an error category suitable for the protocol adapter; internal traces
stay in worker logs.

A request records token IDs, sampling parameters, computed-token count, status,
output-queue state, and logical cache ownership. It does not contain tinygrad
tensors or model-specific weight state.

An execution plan is immutable after submission. It identifies request IDs, token
spans, positions, cache assignments, the exact execution contract, and output
routing. Cancelling an in-flight request suppresses its output immediately, but its
cache cannot be reclaimed until device work that references it has completed. The
initial synchronous worker obtains this guarantee when sampled results are copied
to the control plane. Any later overlap requires explicit in-flight ownership or a
device completion fence before page reuse.

## Model Runner

The serving model is stateless with respect to conversations. Weights and RoPE
tables may be persistent model state; token history and KV contents are explicit
execution inputs. The transformer returns logits and never selects tokens itself.

The first model path has two conceptual operations:

```text
prefill_one(
  input_ids,
  start_position,
  valid_token_count,
  cache_slot,
  kv_storage,
) -> last_token_logits

decode_batch(
  input_ids,
  positions,
  cache_slots,
  active_mask,
  kv_storage,
) -> logits
```

Initial prefill processes one request at a time in bounded chunks. It is not a
packed multi-request operation. Chunk lengths use either exact fixed contracts or a
symbolic range whose views, writes, masks, and replay behavior have dedicated tests.
The implementation must not rely on Python branches changing after TinyJit capture.

Initial decode is one token per active request. Fixed decode shapes may include
inactive rows, but inactive rows must not alter live or shared cache state. The
first implementation can use persistent contiguous request slots and conservative
batch and context limits rather than solve compact scheduling immediately.

The future batched-prefill contract must be specified before implementation. At a
minimum it must encode valid query tokens, request boundaries, each query's absolute
position and available context, causal boundaries for cached prefixes, and the
token index that produces logits for each request. Packed tokens cannot be enabled
by interpreting a total-token bucket as one sequence.

## Contiguous KV Baseline

The first cache is a persistent, external contiguous tensor conceptually shaped as:

```text
kv_storage[
  layer,
  K_or_V,
  request_slot,
  kv_head,
  token_position,
  head_dim
]
```

The exact axis order is selected by measurement, but it is part of the executor
contract once captured. The cache dtype, maximum context, and number of request
slots are explicit server configuration derived from a measured memory budget.

Persistent caches are initialized allocations, for example with `Tensor.zeros`,
made contiguous, and realized before TinyJit capture. `Tensor.empty(...).realize()`
is not treated as proof that a replaceable backing buffer was allocated. A cache
replacement must match the complete TinyJit input contract: argument position,
shape, dtype, device, view structure, and symbolic-variable definition.

The contiguous implementation is intentionally simple and remains the independent
device reference for paged execution. The first server may expose a context limit
below the model maximum so multiple cache slots, model weights, JIT workspaces, and
backend overhead fit in memory.

## Paged KV Target

After the server and fixed batched decode are correct, Infurnace replaces contiguous
request slots with logical pages over a preallocated tinygrad tensor:

```text
kv_pool[
  layer,
  K_or_V,
  physical_page,
  page_offset,
  kv_head,
  head_dim
]

block_table[request_slot, logical_page] = physical_page
slot_mapping[query_token] = physical_page * page_size + page_offset
```

Infurnace owns free-page state, reference counts, block tables, admission rollback,
and page-lifetime rules. The tinygrad executor owns persistent device copies of the
pool and batch metadata.

Paged decode requires two serving-specific UOp operations:

```text
store_kv(kv_pool, slot_mapping, active_mask, new_k, new_v)
paged_attention(query, kv_pool, block_tables, sequence_lengths, active_mask)
```

The store operation must encode a write-before-read dependency consumed by
attention. Inactive rows skip stores. A reserved dummy request and page may provide
valid addresses for inactive reads, but multiple inactive rows never perform
concurrent writes to one dummy location. An alternative implementation may reserve
unique dummy write slots per padded row.

Pages referenced by submitted device work are in flight and cannot return to the
free pool. Prefix-cache ownership, active-request ownership, and in-flight ownership
are accounted for separately.

## TinyJit Contracts

TinyJit is an execution mechanism, not the scheduler. Each prefill or decode
contract has its own `TinyJit` instance. Different fixed shapes use different
instances unless one symbolic contract has been explicitly validated over its full
range.

Every captured contract follows these rules:

- each physical backing buffer appears once in the input tree;
- input argument names, shapes, dtypes, devices, and view structures remain stable;
- persistent full buffers are inputs; position-dependent slices are created inside
  the captured function;
- changing positions, lengths, block tables, masks, and sampling values are Tensor
  inputs or bound integer UOps, not ordinary Python control flow;
- floating sampling values are Tensor inputs;
- closure weights and persistent buffers are realized and stable before capture;
- nested TinyJit capture is not used;
- warmup and capture use disposable KV state or reset every mutated location;
- outputs that must survive the next replay are consumed or copied first;
- replay is tested with different compatible backing buffers, not only different
  values in the capture buffers.

The first call warms up, the second captures and executes, and later calls replay.
All three calls are observable cache mutations. Device graph batching is an optional
backend optimization; correctness and the term "TinyJit replay" do not assume that
an entire contract becomes one device graph.

## Sampling

Sampling is an Infurnace policy implemented as tinygrad operations after model
logits. Greedy `argmax` is the first deterministic path. It exists to validate the
server and is not presented as the recommended quality configuration for Qwen3.

Later sampler contracts accept temperature, top-k, top-p, penalties, token-history
metadata, logprob requests, and explicit per-request random keys and counters.
Infurnace does not use tinygrad's global per-device RNG counter as request identity.
Random state follows the request when batch rows move and is reset when a request
slot is reused.

Production may capture model and sampling together if profiling shows that avoiding
a materialized batch-by-vocabulary logits buffer is important. A logits-returning
debug path remains available for numerical validation.

## Server and Process Model

The first network milestone runs protocol handling, tokenization, engine,
scheduler, and tinygrad executor in one process on one GPU. This is a deployment
constraint, not a coupling between the protocol and model layers. It exposes a
small documented completion API, streaming output, cancellation on disconnect,
bounded output queues, health status, and deterministic capacity errors.

After the single-process server is correct, API and tokenization move to a separate
process:

```text
API and tokenizer process
           |
          IPC
           |
one engine worker per GPU
```

Each worker selects its tinygrad device before creating tensors and owns one model,
one scheduler loop, its KV capacity, and its captured function registry. Multi-GPU
model execution is deferred.

## Memory and Admission

Admission uses memory owned by the server contract, not tinygrad's global allocation
counter as the sole source of truth. The startup budget records:

- realized model and constant buffers;
- contiguous cache slots or paged-pool bytes;
- persistent input, output, and sampler buffers;
- per-contract TinyJit workspace;
- measured backend and allocator reserve.

For a dense attention model, unquantized KV bytes per token are:

```text
layers * 2 * kv_heads * head_dim * cache_dtype_bytes
```

The server rejects requests that exceed the configured per-request context or
available cache capacity before mutating allocator state. Allocation of multiple
pages is atomic from the request's perspective.

## Prefix Cache

Prefix caching is added only after paged ownership is stable. Only complete,
immutable token blocks are shared. Cache identity includes the model checkpoint and
execution configuration that affect KV values, including future adapters and RoPE
settings. Hash matches are verified against token IDs or use an equivalently strong
collision-safe identity.

Dropping a prefix-cache reference does not free a page that still has active or
in-flight owners. Partial blocks remain private, avoiding copy-on-write ambiguity at
the active sequence tail.

## Model Support

Qwen3-0.6B is the first exact model contract. Qwen-specific metadata, weight
mapping, RoPE, Q/K normalization, tied embeddings, and tensor layouts stay in the
model layer. Scheduling, protocol handling, request state, and logical cache
allocation do not depend on Qwen weight names.

Another architecture begins with its own checkpoint manifest and numerical
reference. Compatibility is not inferred only because a checkpoint is a
decoder-only GGUF.

## Compiled Artifacts

Local execution is represented by a versioned set of model, prefill, decode, and
sampler contracts. Artifact loading is deferred until tinygrad provides a validated
format suitable for the selected backend. Any artifact records at least the GGUF
hash, compatibility metadata required by tinygrad, device target, full input
contract, workspace, and validation results. It cannot change scheduler, request,
or KV-ownership semantics.
