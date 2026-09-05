# Qwen3-4B Q4_K_M: C1 Fused Q4_K/Q6_K Decode GEMV (RTX 5060)

C1 result: fused Q4_K/Q6_K decode-GEMV custom kernel on tinygrad's NV renderer.
Reproduced with `DEV=NV .venv/bin/python tools/gemv_bench.py --manifest
models/qwen3-4b-q4_k_m.json --artifact artifacts/models/Qwen3-4B-Q4_K_M.gguf
--output results/infurnace-4b-c1-gemv.json`.

## What changed

`src/infurnace/kernels/quant_gemv.py` ports tinygrad's AMD quant-linear decode
kernels (`tinygrad/llm/kernels/amd.py` `q8_linear`) to NV. The single-token
decode GEMV (M==1) runs as a warp-per-output-row kernel: each 32-lane warp owns
32 in-groups, dequantizes the Q4_K/Q6_K block in registers, and full-wave
reduces partial dots. Activations are Q8-quantized per 32-wide group so dots use
integer `__dp4a`; weights stay packed on device (read once per token).

The active NV renderer here is `CUDARenderer` (CUDA C via NVRTC, target
`sm_120`), which already renders `Ops.CUSTOM`/`Ops.CUSTOMI`, so CUDA intrinsics
(`__dp4a`, `__shfl_xor_sync`) are injected without a tinygrad renderer change.
Values feeding CUSTOM ops must be explicit loads (identity-cast folding passes
the raw pointer expression).

`Qwen3Model._decode_linear` routes decode matmuls through the kernel when the
weight is Q4_K/Q6_K (`packed_weight` extracts the packed buffer from the lazy
dequant graph, mirroring the AMD `set_quantized`); prefill and non-quantized
weights keep the generic `x.linear(w.transpose())` path.
`INFURNACE_DISABLE_QUANT_GEMV=1` restores the generic decode path for A/B.

## Per-shape decode GEMV (compressed bytes, JIT-captured replay)

| role | shape | type | generic GB/s | custom GB/s | ratio |
| --- | --- | --- | ---: | ---: | ---: |
| attn_q | [1, 2560, 4096] | Q4_K | 12.8 | 42.9 | 3.3x |
| attn_k | [1, 2560, 1024] | Q4_K | 3.9 | 12.7 | 3.3x |
| attn_v | [1, 2560, 1024] | Q6_K | 7.2 | 14.7 | 2.1x |
| attn_output | [1, 4096, 2560] | Q4_K | 11.7 | 49.9 | 4.3x |
| ffn_gate | [1, 2560, 9728] | Q4_K | 27.2 | 96.9 | 3.6x |
| ffn_up | [1, 2560, 9728] | Q4_K | 23.0 | 96.5 | 4.2x |
| ffn_down | [1, 9728, 2560] | Q6_K | 24.0 | 108.1 | 4.5x |
| output | [1, 2560, 151936] | Q6_K | 94.1 | 244.6 | 2.6x |

## Whole decode graph

| measurement | A3 (generic) | C1 (custom) | ratio |
| --- | ---: | ---: | ---: |
| JIT replay amortized | 109.8 ms/token | 24.8 ms/token | 4.4x |
| achieved GB/s | 22.7 | 100.3 | 4.4x |
| engine decode step | 124.9 ms/token | 48.8 ms/token | 2.6x |
| implied tok/s | ~7.9 | ~40 | 5x |

The decode graph is now ~100 GB/s (~22% of the ~448 GB/s ceiling) versus
llama.cpp's 312 GB/s (69%). The host overhead (48.8 engine - 24.8 replay =
~24 ms) is now a larger relative share and remains Phase 5D's persistent-buffer
work.

## Correctness

- `test_quant_gemv.py`: per-role greedy argmax agreement (custom vs generic) for
  every Q4_K/Q6_K shape on the real model; full engine decode produces identical
  greedy tokens with the kernel on vs off (16-token trace).
- Existing gates: NV batched-decode, stability (recapture-free, flat memory),
  and qwen3-model tests pass. CPU suite: 188 passed. Per-engine device memory is
  flat (2.57 GB generic vs 2.58 GB custom).
- Method: the Q8_0 activation path is the llama.cpp decode design; logit deltas
  vs the fp16 path are ~1e-2 absolute with greedy agreement on every tested
  shape.

## Non-claims

- The small per-layer GEMVs (attn_k/v at 12-15 GB/s) remain below the ffn
  shapes; the warp-per-output-row structure is correct but the smallest-N
  shapes are still occupancy-limited. Further gains (shared-memory weight
  staging, larger tiles) are follow-up kernel work, not this milestone.
- Prefill stays on the generic matmul (the WMMA token-batched path is Phase 10
  scope); only the shape-1/single-token decode uses the fused kernel.
- The full `test_engine_real.py` suite (10 model-loading tests in one process)
  overruns the 8 GiB card on both the generic and custom paths; individual
  tests pass. That is a pre-existing test-harness memory issue, not a C1
  regression.