# Phase 0E Upstream Prefill Baseline

This partial Phase 0E baseline compares lazy Q8_0 expressions with fully realized
FP16 weights through tinygrad's upstream Qwen path. It measures checkpoint
verification, model loading, TinyJit contract setup, and steady 16-token prefill time
to first token. It does not measure steady decode or end-to-end serving.

## Environment

```text
device:         NVIDIA GeForce RTX 5060
device memory:  8,151 MiB
driver:         610.43.02
backend:        DEV=NV
JIT:            enabled
model context:  1,024 tokens
prefill chunk:  32 tokens
prompt:         16 fixed token IDs
samples:        5 TinyJit replay calls
GPU sampling:   nvidia-smi every 50 ms
```

The device had 6,383 MiB free before each run. Desktop graphics contexts accounted
for the remaining baseline usage; no competing model or compute workload was run.
Infurnace follows current tinygrad through contract tests and does not attach source
checkout identity to benchmark results.

## Commands

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

Each policy ran in a clean process. The artifact was held through one stable file
descriptor for verification and loading. Two divergent same-shape prompts performed
TinyJit warmup and capture before five divergent replay prompts were measured. Every
timed sample synchronized the device before and after execution and generated one
TTFT token with no timed decode tokens.

## Results

| Metric | Lazy Q8_0 expressions | Realized FP16 |
| --- | ---: | ---: |
| Artifact verification | 0.338 s | 0.342 s |
| Model loading | 1.206 s | 3.448 s |
| TinyJit setup call 1 | 7.175 s | 5.503 s |
| TinyJit setup call 2 | 3.626 s | 3.141 s |
| Mean prefill TTFT | 105.53 ms | 134.10 ms |
| Minimum prefill TTFT | 105.02 ms | 133.26 ms |
| Maximum prefill TTFT | 106.16 ms | 134.69 ms |
| Mean prompt throughput | 151.62 tok/s | 119.32 tok/s |
| Host peak RSS | 293.66 MiB | 308.31 MiB |
| Tinygrad live bytes after model load | 609.82 MiB | 1,136.88 MiB |
| Tinygrad live bytes after setup | 839.69 MiB | 1,366.74 MiB |
| Sampled process GPU peak during model load | 632 MiB | 1,939 MiB |
| Sampled process GPU peak over complete run | 1,213 MiB | 2,358 MiB |

Lazy TTFT samples in milliseconds were:

```text
[105.506, 105.879, 105.024, 106.157, 105.061]
```

Realized-FP16 TTFT samples in milliseconds were:

```text
[133.259, 134.074, 134.422, 134.688, 134.037]
```

Both policies produced setup tokens `[657, 198]` and measured tokens
`[198, 198, 198, 198, 198]`.

## Interpretation

For this short-prefill workload, lazy Q8_0 expressions loaded 2.86 times faster,
used 1,145 MiB less sampled peak process GPU memory, and delivered 27 percent more
prompt throughput than fully realized FP16 weights. This result favors the lazy
policy for prefill, but the serving policy remains undecided until the dedicated
decode benchmark measures steady token latency and memory under its own contract.

## Measurement Limits

- `VmHWM` is a process-lifetime host RSS high-water mark, not a phase-local peak.
- Tinygrad counters report live requested bytes, not peak physical allocation. They
  exclude runtime allocations and allocator-cache retention.
- Per-process GPU memory is sampled at integer-MiB resolution. A transient shorter
  than 50 ms can be missed.
- A sample whose `nvidia-smi` query window overlaps a phase is conservatively
  attributed to that phase; adjacent phase peaks can therefore share a boundary
  sample.
- The benchmark uses tinygrad's upstream model-owned KV and sampling path. It is a
  Phase 0 reference, not the future Infurnace external-KV model contract.
