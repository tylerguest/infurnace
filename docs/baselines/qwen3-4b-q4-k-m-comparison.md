# Qwen3-4B Q4_K_M: Infurnace vs llama.cpp (RTX 5060)

Head-to-head steady decode on the same checkpoint, same card, same greedy
sampling, 16-token prompt. This is the "measure first" bar that gates the
throughput-kernel work in the roadmap's Current Work section.

A single command reproduces the comparison table and writes a validated JSON:

```sh
DEV=NV JIT=1 .venv/bin/python benchmarks/benchmark_comparison.py \
  --manifest models/qwen3-4b-q4_k_m.json \
  --artifact artifacts/models/Qwen3-4B-Q4_K_M.gguf \
  --output results/infurnace-llama-cpp-comparison.json
```

## Reference

- Checkpoint: `Qwen/Qwen3-4B-GGUF` revision `bc640142c66e1fdd12af0bd68f40445458f3869b`,
  `Qwen3-4B-Q4_K_M.gguf`, 2,497,280,256 bytes,
  SHA-256 `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`.
- Device: NVIDIA GeForce RTX 5060, 7,743 MiB, compute capability 12.0.
- Ceiling: ~448 GB/s GDDR7. For the 2.49 GB of model weights, reading every
  weight once per generated token bounds decode at ~180 tok/s.

## llama.cpp (llama-bench, build 7e4c0a968)

`llama-bench -m Qwen3-4B-Q4_K_M.gguf -p 16 -n 200 -ngl 99 -r 5`:

| test | result |
| --- | ---: |
| pp16 (prompt processing) | 1361.39 ± 307.92 tok/s (~12 ms TTFT) |
| tg200 (text generation / decode) | **124.08 ± 1.20 tok/s** |
| model size | 2.32 GiB (4.02 B params) |
| achieved decode bandwidth | ~309 GB/s (~69% of ceiling) |

## Infurnace (benchmark_infurnace, DEV=NV, lazy-fp16)

`DEV=NV JIT=1 .venv/bin/python benchmarks/benchmark_infurnace.py
--manifest models/qwen3-4b-q4_k_m.json
--artifact artifacts/models/Qwen3-4B-Q4_K_M.gguf
--max-context 512 --prompt-tokens 16 --decode-tokens 200`

| metric | result |
| --- | ---: |
| cold TTFT (16-token prefill) | 31,176 ms (dominated by first-execution compile) |
| decode contract capture | 29,446 ms (one-time TinyJit capture) |
| steady decode | **7.93 tok/s** (median 125.4 ms/token) |
| achieved decode bandwidth | 19.8 GB/s (~4.4% of ceiling) |
| peak device memory | 3.62 GiB |
| measured decode tokens | 200 (warm replays) |

Raw validated JSON: `results/infurnace-4b-q4-k-m-decode.json` (ignored).

## Comparison

| metric | llama.cpp | Infurnace | ratio |
| --- | ---: | ---: | ---: |
| decode tok/s | 124.08 | 7.93 | Infurnace 6.4% (15.6x slower) |
| achieved GB/s | ~309 | 19.8 | Infurnace 6.4% |
| vs ~448 GB/s ceiling | 69% | 4.4% | — |

TTFT is not directly comparable: llama-bench's pp16 is a warm, precompiled
kernel measurement (~12 ms), while Infurnace's cold TTFT includes compiling the
36-layer eager prefill graph (~31 s). A warm Infurnace prefill (kernels already
compiled) is the fair comparison and should be measured alongside the first
kernel milestone.

## Reading

The 15.6x decode gap is almost entirely kernel efficiency, not architecture.
llama.cpp's custom CUDA GEMV kernels sustain 69% of the bandwidth ceiling;
tinygrad's generic NV path (lazy Q4_K/Q6_K dequant fused into matmul kernels)
sustains 4.4%. The roadmap's throughput-kernel work (custom fused
Q4_K/Q6_K dequant-GEMV and fused single-token paged attention on tinygrad's NV
renderer) targets exactly this gap, with llama.cpp's 124 tok/s as the bar and
~180 tok/s as the bandwidth ceiling.

The 3.62 GiB peak confirms the lazy-quant resident policy fits the 8 GiB card
with ~4 GiB of headroom for a paged KV pool and workspaces.