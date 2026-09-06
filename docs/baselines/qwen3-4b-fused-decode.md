# Qwen3-4B Q4_K_M: Phase 5D Fused Decode (in-graph argmax + device token feedback)

Fused-decode result: the single-token decode step runs entirely as one TinyJit
replay — in-graph greedy argmax, persistent on-device token feedback, zero
per-step host roundtrip. Reproduced with `DEV=NV JIT=1 .venv/bin/python
benchmarks/benchmark_comparison.py --manifest models/qwen3-4b-q4_k_m.json
--artifact artifacts/models/Qwen3-4B-Q4_K_M.gguf --output
results/infurnace-llama-cpp-comparison.json`.

## What changed

- `Qwen3Model._decode_step_fused`: the shape-1 decode graph ends with
  `logits.argmax(axis=1)` and an SSA `uop.store` of the sampled token into a
  persistent `[1, 1]` int32 feedback buffer. The graph reads the previous
  sampled token from the paired buffer at the start, so the input write and the
  greedy sampler are captured kernels, not eager ops.
- `Qwen3Runner.decode_fused` / `initialize_fused_decode`: the runner owns two
  `[1, 1]` feedback buffers and two captures with constant read/write roles
  (read A -> write B and read B -> write A), alternating between them per call.
  Prefill seeds both buffers once per request; the first decode re-seeds after
  capture (the capture warmups consume the seed). Warmups run at the last cache
  position so capture never touches the request's real KV prefix.
- `Engine._execute_decode`: single-request plans (the benchmark path) use the
  fused contract; multi-request plans keep the batched `decode_batch` contract.
  `_execute_prefill` seeds the feedback buffers.

## Why this was the bottleneck

Sync-bracketed and host-dispatch probes (`tools/profile_decode_probes.py`)
showed that **every eager (non-JIT) op costs ~13-16 ms once the 2.5 GB model is
resident**, regardless of size — a 1-element add took 12.8 ms, a 100-element
argmax 15.8 ms — while JIT-captured kernels cost ~2 ms each. The old engine
step paid the eager tax twice per token: the input build (~13 ms) and the host
sampler argmax (~16 ms), plus the `.tolist()` roundtrip. Engine decode measured
49.5 ms/step with only ~11.7 ms of device work.

## A subtle NV replay bug (found and fixed during validation)

The first implementation used one `[2, 1]` feedback buffer with a bound
`feedback_idx` variable selecting the read/write row. On the NV HCQ path the
var_vals substitution for variable-indexed slices of a small closure buffer
lags by one replay on first use: the first replay with a new index value
executed with the stale row, silently duplicating tokens in pairs (the greedy
token trace showed `785, 785, 702, 702, ...`). The paired-buffer design removes
every variable-indexed feedback slice (read/write targets are Python constants
in the captured graph), eliminating the stale-substitution path entirely.
Fused greedy tokens now match `decode_batch` exactly on NV (16-token trace) and
on CPU. The first benchmark pass (63.1 tok/s) was measured with this bug active
and is superseded by the corrected 88.5 tok/s.

## Results (RTX 5060, Qwen3-4B Q4_K_M)

| measurement | A3 (generic) | C1 (GEMV) | C1 + fused decode (fixed) |
| --- | ---: | ---: | ---: |
| engine decode step | 124.9 ms | 48.8 ms | **11.3 ms** |
| decode tok/s | ~7.9 | 20.78 | **88.49** |
| achieved GB/s | 19.9 | 51.8 | **220.4** |
| % of llama.cpp decode | 6.4% | 17.2% | **68.4%** |
| vs ceiling (~448 GB/s) | 4.4% | 11.6% | **49.2%** |

The engine step now runs at essentially the raw replay speed (~11.8 ms/token at
position 200); the token readback and engine host work are ~1-2 ms. llama.cpp
sustains 129.29 tok/s (322.1 GB/s) on the same card and checkpoint, so the
remaining gap is 1.5x.

## Correctness

- CPU test: fused decode greedy tokens match `decode_batch` greedy tokens over
  a multi-position trace; the two fused captures are reused across replays.
- NV: 16-token fused trace matches `decode_batch` exactly on the real model;
  all model/engine tests pass with the engine on the fused path.
- The benchmark's 200-token output has no consecutive duplicates.

## Non-claims

- The eager-op tax (~13-16 ms per eager op with the model resident) is a
  backend/execution-path artifact on this box: prefill, batched decode (S >= 2),
  and profile tools measuring `decode_batch` still pay it. The fused contract
  avoids it for the single-request steady decode.
- The device replay (~11.8 ms, ~211 GB/s) is still ~1.45x off llama.cpp's
  ~8.2 ms/step (322 GB/s); the remaining levers are attention fusion (C2 /
  Phase 5C), the small-N GEMV shapes, and the ~1-2 ms of readback and host work
  per token.