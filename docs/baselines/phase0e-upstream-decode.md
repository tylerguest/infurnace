# Phase 0E Upstream Decode Baseline

This partial Phase 0E baseline compares steady one-token decode through tinygrad's
upstream model-owned-cache path. It uses the same checkpoint, execution device, and
weight policies as the upstream prefill baseline.

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
TTFT tokens:    1
setup tokens:   2
decode tokens:  16 timed TinyJit replays
GPU sampling:   nvidia-smi every 50 ms
```

The device had 6,409 MiB free before the lazy run and 6,441 MiB free before the
realized-FP16 run. Desktop graphics contexts accounted for baseline usage; no
competing model or compute workload was run.

## Commands

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

Each policy ran in a clean process. One generator produced all 19 output tokens: one
cold prefill/TTFT token, two rollout setup tokens, and 16 measured decode tokens.
Every generator yield was synchronized immediately before and after execution.

## Results

| Metric | Lazy Q8_0 expressions | Realized FP16 |
| --- | ---: | ---: |
| Artifact verification | 0.338 s | 0.335 s |
| Model loading | 1.202 s | 3.457 s |
| Cold prefill/TTFT | 7.312 s | 5.419 s |
| Rollout setup call 1 | 3.377 s | 1.688 s |
| Rollout setup call 2 | 1.446 s | 1.177 s |
| Mean decode latency | 20.91 ms | 16.61 ms |
| Minimum decode latency | 19.85 ms | 15.66 ms |
| Maximum decode latency | 22.90 ms | 17.61 ms |
| Aggregate decode throughput | 47.82 tok/s | 60.21 tok/s |
| Host peak RSS | 294.61 MiB | 307.83 MiB |
| Tinygrad live bytes after decode | 835.74 MiB | 1,362.79 MiB |
| Sampled process GPU peak | 1,210 MiB | 2,356 MiB |

Lazy decode samples in milliseconds were:

```text
[19.942, 21.171, 19.856, 21.588, 21.242, 19.852, 22.898, 20.096,
 21.374, 21.144, 19.946, 21.461, 21.194, 19.957, 21.580, 21.269]
```

Realized-FP16 decode samples in milliseconds were:

```text
[17.403, 15.846, 17.196, 15.657, 17.020, 15.744, 17.270, 15.785,
 17.440, 15.660, 17.267, 17.611, 15.706, 17.249, 15.705, 17.158]
```

Both policies generated:

```text
TTFT:   [657]
setup:  [11, 714]
decode: [279, 2038, 374, 537, 4396, 13, 220, 5209,
         1492, 752, 311, 5046, 432, 624, 785, 2038]
```

## Interpretation

Realized FP16 reduced mean decode latency by 20.6 percent and increased aggregate
decode throughput by 25.9 percent. It used 1,146 MiB more sampled process GPU memory
and took 2.26 seconds longer to load.

Combined with the steady prefill baseline, the extra realized-FP16 prefill latency is
recovered after approximately seven decoded tokens. The provisional latency-oriented
policy for the constrained single-request milestone is therefore realized FP16.
Lazy Q8_0 remains a viable memory-oriented policy when additional context or cache
capacity is more valuable than decode latency. The end-to-end serving benchmark must
confirm the provisional choice before Phase 0E closes.

## Measurement Limits

- The cold prefill/TTFT value includes first execution, compilation, and JIT setup;
  it is not the steady prefill metric reported by the separate prefill benchmark.
- Decode positions increase across one conversation, so samples are replay timings
  at adjacent positions rather than repeated measurements of one position.
- `VmHWM`, tinygrad live requested bytes, and sampled driver-visible GPU memory retain
  the distinct meanings documented in `benchmarks/README.md`.
- A GPU-memory sample whose query window overlaps a phase is conservatively
  attributed to that phase and can be shared across adjacent boundaries.
- The upstream model owns KV state and returns sampled tokens. This is a Phase 0
  reference rather than the future external-KV Infurnace runner contract.
