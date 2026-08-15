# Phase 0E Upstream End-to-End Baseline

This baseline measures closed-loop sequential generation through
`tinygrad.llm.Transformer.generate()`. It combines steady prefill and decode in one
observable workload, but it is not an Infurnace server benchmark.

## Scope

```text
server runtime:  absent
transport:       none
tokenization:    none; fixed token IDs
scheduling:      none
concurrency:     1
arrival pattern: closed-loop sequential
```

No result here measures HTTP, queueing, admission, streaming, cancellation,
backpressure, external KV, or concurrent clients. Those contracts begin in later
roadmap phases.

## Environment

```text
device:          NVIDIA GeForce RTX 5060
device memory:   8,151 MiB
driver:          610.43.02
backend:         DEV=NV
JIT:             enabled
model context:   1,024 tokens
prefill chunk:   32 tokens
prompt:          16 fixed token IDs
setup workload:  2 generations of 3 tokens
measured work:   5 generations of 16 tokens
GPU sampling:    nvidia-smi every 50 ms
```

The device had 6,317 MiB free before the lazy run and 6,336 MiB free before the
realized-FP16 run. Desktop graphics contexts accounted for baseline usage; no
competing model or compute workload was run.

## Commands

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

Each policy ran in a clean process. Setup used two divergent prompts and left the
upstream prefill and rollout TinyJit contracts captured at counts 2 and 4. Five more
divergent prompts prevented upstream prefix reuse during measurement.

Benchmark-controlled startup begins before manifest loading and stable artifact
opening and ends immediately after JIT setup. It excludes process launch, Python
module import before `main()`, and argument parsing, and it is not server readiness.

## Results

| Metric | Lazy Q8_0 expressions | Realized FP16 |
| --- | ---: | ---: |
| Artifact verification | 0.340 s | 0.340 s |
| Model loading | 1.208 s | 3.520 s |
| JIT contract setup elapsed | 15.973 s | 11.622 s |
| Benchmark startup-to-ready | 17.995 s | 15.971 s |
| Mean TTFT | 106.07 ms | 133.06 ms |
| Mean inter-token latency | 20.49 ms | 16.07 ms |
| Mean 16-token latency | 413.49 ms | 374.04 ms |
| Closed-loop generated throughput | 38.69 tok/s | 42.77 tok/s |
| Host peak RSS | 294.52 MiB | 308.21 MiB |
| Tinygrad live bytes after measurement | 841.10 MiB | 1,368.15 MiB |
| Sampled process GPU peak | 1,220 MiB | 2,365 MiB |

Per-generation end-to-end samples in milliseconds were:

```text
lazy:          [411.818, 406.212, 408.039, 419.931, 421.436]
realized-fp16: [375.990, 373.239, 373.292, 373.655, 374.036]
```

Every measured prompt produced the same output under both policies:

```text
[198, 785, 2038, 498, 3897, 4977, 311, 387,
 264, 6514, 315, 2155, 15459, 323, 10767, 1045]
```

## Interpretation

Realized FP16 reduced mean 16-token generation latency by 9.5 percent and improved
complete closed-loop throughput by 10.5 percent. Its TTFT remained 25.4 percent
higher, and it used 1,145 MiB more sampled process GPU memory. Faster JIT setup made
its benchmark-controlled startup-to-ready 2.02 seconds shorter despite slower model
loading.

This confirms realized FP16 as the provisional latency-oriented policy for the
constrained single-request milestone. Lazy Q8_0 remains the memory-oriented option
when approximately 1.1 GiB of additional cache or context capacity matters more than
steady generation latency.

## Measurement Limits

- Five generations establish a repeatable baseline but not tail-latency percentiles.
- Fixed-length generation ignores EOS and does not implement completion semantics.
- Synchronization and host token retrieval are intentionally included.
- GPU memory is sampled at integer-MiB resolution and can miss short transients.
- A query overlapping a phase boundary is conservatively attributed to both phases;
  only overall sampled peak is used for the policy comparison.
- Current tinygrad behavior is tracked through contracts rather than source-checkout
  identity.
