# Infurnace

Infurnace is a single-GPU inference server for decoder-only language models. It uses
tinygrad as its only production tensor compiler and device runtime.

Infurnace owns the server: protocol handling, tokenization integration, request
lifecycle, scheduling, batching, KV-cache layout and lifetime, sampling policy,
cancellation, streaming, and metrics. Its model runners and serving kernels are
written with tinygrad Tensor and UOp APIs. Tinygrad owns GGUF parsing, tensor and UOp
semantics, optimization, code generation, compilation, device allocation, and
execution.

Infurnace is not another tensor framework or hardware backend, and it does not use
`tinygrad.llm.Transformer.generate()` as its serving interface. The first model
runner targets Qwen3-0.6B on one NVIDIA GPU with `DEV=NV`, explicit external KV
state, and logits returned to an Infurnace sampling stage.

Infurnace follows current tinygrad development. It does not pin a tinygrad commit,
publish a supported-version matrix, or preserve adapters for old tinygrad APIs.
Contract tests expose upstream changes so Infurnace can move forward with them.

The first useful milestone is a constrained but real streaming server. It uses
single-request chunked prefill, deterministic greedy sampling, conservative context
and concurrency limits, and external contiguous KV storage. Fixed batched decode,
paged KV attention, continuous batching, richer sampling, and prefix caching follow
only after the simpler path is correct and measured.

## Current State

The repository contains an empty implementation scaffold and design documents. No
runtime behavior has been implemented. The documents define phase gates rather than
claiming support for features that have not passed their correctness tests.

## Structure

```text
benchmarks/                    Prefill, decode, and serving benchmarks
docs/                          Architecture, validation, and roadmap
models/                        Checkpoint contracts; weights are not committed
src/infurnace/api/             Protocol and server adapters
src/infurnace/engine/          Request lifecycle and engine loop
src/infurnace/scheduler/       Admission and batch planning
src/infurnace/cache/           Logical KV pages and prefix-cache policy
src/infurnace/executor/        Execution batches, runner contract, and sampling
src/infurnace/executor/tinygrad/  Tinygrad model, buffers, weights, and compilation
src/infurnace/models/          Model configuration and Qwen-specific mapping
src/infurnace/kernels/         Custom UOp KV-store and paged-attention kernels
tests/                         Correctness and invariant tests
tools/                         Development-only comparison utilities
```

See [the architecture](docs/architecture.md) for the server and execution contracts,
[the roadmap](docs/roadmap.md) for implementation gates, and
[numerical validation](docs/numerical_validation.md) for correctness requirements.
