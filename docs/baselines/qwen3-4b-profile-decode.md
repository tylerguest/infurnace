# Qwen3-4B Q4_K_M: Decode-Step Profile (RTX 5060)

A3 gap analysis: where the ~125 ms/token decode goes on the serving path.
Reproduced with `DEV=NV JIT=1 .venv/bin/python tools/profile_decode.py
--manifest models/qwen3-4b-q4_k_m.json
--artifact artifacts/models/Qwen3-4B-Q4_K_M.gguf`.

## Decode path per token

| measurement | ms/token | meaning |
| --- | ---: | --- |
| engine decode | 124.9 | serving path: sampler, `.tolist()`, scheduler, JIT replay |
| JIT replay single | 110.3 | one shape-1 TinyJit replay, sync-bracketed |
| JIT replay amortized | 109.8 | N replays queued back-to-back, one trailing sync |
| **host overhead** | **15.1** | engine minus amortized replay (host work per token) |
| JIT replay bandwidth | 22.7 GB/s | 2.49 GB of weights read once per token |

`single ≈ amortized`: the decode step is **device-bound**. The 15.1 ms/token
host overhead is the sampler `.item()` + the per-step `.tolist()` input build
(the Phase 4D note), not the dominant cost. The kernels themselves are the
bottleneck: 22.7 GB/s is ~5% of the ~448 GB/s ceiling.

## Isolated GEMV microbench (relative, not additive)

Per-shape GEMVs measured on the model's quantized weights. Absolute ms are
inflated by per-op eager planning overhead (the whole graph is ~110 ms, not the
sum here); the relative picture is the signal.

| role | shape | isolated GB/s |
| --- | --- | ---: |
| attn_q | [1, 2560, 4096] | 13.3 |
| attn_k | [1, 2560, 1024] | 3.3 |
| attn_v | [1, 2560, 1024] | 3.0 |
| attn_output | [1, 4096, 2560] | 13.2 |
| ffn_gate | [1, 2560, 9728] | 29.7 |
| ffn_up | [1, 2560, 9728] | 32.0 |
| ffn_down | [1, 9728, 2560] | 28.4 |
| **output** | [1, 2560, 151936] | **394.2** |

The small per-layer GEMVs (1-row, M small) run at 3-32 GB/s while the large
output GEMV approaches the ceiling (~394 GB/s). The inefficiency is the
small-M dequant-GEMV tiling, not the dequant math itself.

## Gap list (drives C1)

1. **Decode is device-bound at ~110 ms/token (23 GB/s, ~5% of ceiling).**
   Kernel work, not host work, is the lever.
2. **The small per-layer Q4_K/Q6_K GEMVs are the inefficiency** (3-32 GB/s
   isolated); the large output GEMV is already near bandwidth.
3. Target: the shape-1 GEMV path. llama.cpp sustains 312 GB/s (69%) on the
   same card/model, so the ceiling is achievable on this hardware.
4. Secondary: 15.1 ms/token host overhead (`.tolist()` + sampler) is folded
   into Phase 5D's persistent input buffers, not a blocker for C1.

## Non-claims

- The isolated GEMV numbers are relative (per-op planning overhead inflates
  them); the reliable aggregate is the JIT replay bandwidth (22.7 GB/s).
- tinygrad `PROFILE` timestamps were found unreliable for kernel durations on
  this backend (a 3.3 ms matmul read as ~1 µs), so this profile uses
  sync-bracketed wall-clock timing instead.
- Attention at position 200 is negligible inside the graph; the isolated
  attention measurement is also inflated.