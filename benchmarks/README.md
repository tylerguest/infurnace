# Benchmarks

Benchmark entry points are development tools, not serving APIs. Raw results belong
under the ignored `results/` directory; committed baseline documents summarize the
commands, results, and limitations.

## Upstream Prefill

The first Phase 0E benchmark measures checkpoint verification, model loading, and
steady prefill time to first token through tinygrad's upstream model-owned-cache
path. Run each weight policy in a separate clean process:

```sh
DEV=NV JIT=1 .venv/bin/python benchmarks/benchmark_prefill.py \
  --manifest models/qwen3-0.6b-q8_0.json \
  --artifact artifacts/models/Qwen3-0.6B-Q8_0.gguf \
  --weight-policy lazy \
  --max-context 1024 \
  --chunk-size 32 \
  --prompt-tokens 16 \
  --samples 5 \
  --memory-sample-ms 50 \
  --output results/phase0e-prefill-lazy.json

DEV=NV JIT=1 .venv/bin/python benchmarks/benchmark_prefill.py \
  --manifest models/qwen3-0.6b-q8_0.json \
  --artifact artifacts/models/Qwen3-0.6B-Q8_0.gguf \
  --weight-policy realized-fp16 \
  --max-context 1024 \
  --chunk-size 32 \
  --prompt-tokens 16 \
  --samples 5 \
  --memory-sample-ms 50 \
  --output results/phase0e-prefill-realized-fp16.json
```

The runner verifies the artifact before model-load timing. It then executes two
same-shape prefill calls for TinyJit warmup and capture. The measured samples are
replays with prompts whose first tokens differ, preventing upstream prefix-cache
reuse. Each TTFT sample includes prompt execution, greedy selection of one token,
host retrieval, and device synchronization. Prompt throughput divides prompt tokens
by TTFT and does not count the generated token.

Results use schema version 1 and preserve raw timing samples and generated token IDs.
The validator rejects missing checkpoint, execution, workload, device, timing,
output, or memory fields before publishing the JSON file atomically.

## Memory Sources

The result keeps three non-equivalent memory measurements separate:

- `/proc/self/status` supplies current RSS and process-lifetime peak RSS at phase
  boundaries.
- `tinygrad.GlobalCounters.mem_used_per_device` supplies live requested bytes at
  phase boundaries. It is not a peak and excludes runtime allocations and allocator
  cache retention.
- `nvidia-smi.compute-apps` supplies sampled driver-visible memory for the benchmark
  PID. It is an integer-MiB sample, not an exact high-water mark, and can miss
  transients shorter than the configured interval.

The GPU-wide baseline includes memory used by desktop contexts, while sampled peaks
are scoped to the benchmark process. Runs should not overlap another model or
compute workload.

The recorded comparison is in
[`phase0e-upstream-prefill.md`](../docs/baselines/phase0e-upstream-prefill.md).

## Upstream Decode

The decode entry point uses one generator for a complete conversation. It produces
one cold prefill/TTFT token, two one-token rollout calls for TinyJit warmup and
capture, and then individually times steady replay tokens:

```sh
DEV=NV JIT=1 .venv/bin/python benchmarks/benchmark_decode.py \
  --manifest models/qwen3-0.6b-q8_0.json \
  --artifact artifacts/models/Qwen3-0.6B-Q8_0.gguf \
  --weight-policy lazy \
  --max-context 1024 \
  --chunk-size 32 \
  --prompt-tokens 16 \
  --decode-tokens 16 \
  --memory-sample-ms 50 \
  --output results/phase0e-decode-lazy.json

DEV=NV JIT=1 .venv/bin/python benchmarks/benchmark_decode.py \
  --manifest models/qwen3-0.6b-q8_0.json \
  --artifact artifacts/models/Qwen3-0.6B-Q8_0.gguf \
  --weight-policy realized-fp16 \
  --max-context 1024 \
  --chunk-size 32 \
  --prompt-tokens 16 \
  --decode-tokens 16 \
  --memory-sample-ms 50 \
  --output results/phase0e-decode-realized-fp16.json
```

The cold prefill value includes first-execution and JIT overhead and is not
comparable with the steady prefill replay metric. Decode throughput is one generated
token divided by each synchronized replay latency. See the recorded
[`phase0e-upstream-decode.md`](../docs/baselines/phase0e-upstream-decode.md)
comparison.

## Upstream End-to-End Generation

The roadmap's initial end-to-end entry point remains an upstream generation
benchmark because Infurnace has no server runtime yet. Despite the placeholder
filename, its result identity is `upstream_end_to_end` and its schema explicitly
records that transport, tokenization, scheduling, and server runtime are absent.

Run two setup generations followed by five closed-loop measured generations:

```sh
DEV=NV JIT=1 .venv/bin/python benchmarks/benchmark_serving.py \
  --manifest models/qwen3-0.6b-q8_0.json \
  --artifact artifacts/models/Qwen3-0.6B-Q8_0.gguf \
  --weight-policy lazy \
  --max-context 1024 \
  --chunk-size 32 \
  --prompt-tokens 16 \
  --setup-output-tokens 3 \
  --measured-generations 5 \
  --output-tokens 16 \
  --memory-sample-ms 50 \
  --output results/phase0e-end-to-end-lazy.json

DEV=NV JIT=1 .venv/bin/python benchmarks/benchmark_serving.py \
  --manifest models/qwen3-0.6b-q8_0.json \
  --artifact artifacts/models/Qwen3-0.6B-Q8_0.gguf \
  --weight-policy realized-fp16 \
  --max-context 1024 \
  --chunk-size 32 \
  --prompt-tokens 16 \
  --setup-output-tokens 3 \
  --measured-generations 5 \
  --output-tokens 16 \
  --memory-sample-ms 50 \
  --output results/phase0e-end-to-end-realized-fp16.json
```

Setup must leave both upstream prefill and rollout TinyJit contracts captured.
Measured prompts have distinct first tokens so upstream cannot reuse a cached prefix.
Each generation records one TTFT, 15 inter-token intervals, and exact end-to-end
latency. Aggregate throughput uses the complete closed-loop measurement window,
including generator construction, synchronization, and inter-generation overhead.

See the recorded
[`phase0e-upstream-end-to-end.md`](../docs/baselines/phase0e-upstream-end-to-end.md)
comparison and the consolidated [`phase0e-report.md`](../docs/baselines/phase0e-report.md).
