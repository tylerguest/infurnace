# Optimization Log

Performance changes are measured experiments, not assumptions:

1. Define the workload and current baseline.
2. Profile the bottleneck.
3. State one optimization hypothesis.
4. Make one focused change.
5. Revalidate numerical and state correctness.
6. Record latency, throughput, memory, compilation time, and profiler evidence.
7. Keep or revert the change based on the result.

Prefill, decode, sampling, and end-to-end serving results are reported separately.
They have different shapes, bottlenecks, and user-visible effects.

## Experiment Identity

Every result records:

- GGUF source, SHA-256, and quantization;
- weight realization, model dtype, accumulation dtype, and cache dtype;
- current tinygrad checkout, device backend, driver, and GPU;
- context length, prefill chunk, decode bucket, page size, and active-row count;
- eager, warmup, capture, replay, and device-graph state;
- cold or warm compilation and allocator-cache state.

The tinygrad checkout identifies the experiment without declaring a supported
version. New work always targets current tinygrad.

## Model Metrics

Model experiments report at least:

- startup and GGUF loading time;
- peak loading and steady model memory;
- TinyJit warmup and capture time per contract;
- prefill latency and tokens per second by prompt length;
- decode latency per token and tokens per second by active batch size;
- sampler latency and full-logit materialization cost;
- kernel count, memory traffic estimate, and actual device timing where available.

## Server Metrics

Serving experiments also report:

- request arrival pattern and prompt/output length distribution;
- admitted, queued, rejected, cancelled, failed, and completed requests;
- time to first token and inter-token latency percentiles;
- end-to-end latency and total generated-token throughput;
- active sequences, cache occupancy, prefix hit rate when enabled, and page churn;
- scheduler waiting time split between prefill and decode;
- output-queue backpressure and client concurrency.

Steady, bursty, long-prompt, decode-heavy, and mixed workloads are separate results.
An optimization is not accepted if it improves aggregate throughput by violating the
documented latency, fairness, memory, or correctness contract.
