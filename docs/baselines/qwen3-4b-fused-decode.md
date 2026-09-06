# Qwen3-4B Q4_K_M: Phase 5D Fused Decode (in-graph argmax + device token feedback)

Fused-decode result: the single-token decode step runs entirely as one TinyJit
replay — in-graph greedy argmax, persistent on-device token feedback, zero
per-step host roundtrip. Reproduced with `DEV=NV JIT=1 .venv/bin/python
benchmarks/benchmark_comparison.py --manifest models/qwen3-4b-q4_k_m.json
--artifact artifacts/models/Qwen3-4B-Q4_K_M.gguf --output
results/infurnace-llama-cpp-comparison.json`.

## What changed

- `Qwen3Model._decode_step_fused`: the shape-1 decode graph now ends with
  `logits.argmax(axis=1)` and an SSA `uop.store` of the sampled token into a
  persistent `[2, 1]` int32 feedback buffer at row `1 - feedback_idx` (bound
  Variable). The graph reads the previous sampled token from row
  `feedback_idx` at the start, so the input write and the greedy sampler are
  captured kernels, not eager ops.
- `Qwen3Runner.decode_fused` / `initialize_fused_decode`: the runner owns the
  feedback buffer, toggles the read/write row per call, and returns the sampled
  token as a host int (one `[1, 1]` tolist per step). Prefill seeds the buffer
  once per request.
- `Engine._execute_decode`: single-request plans (the benchmark path) use the
  fused contract; multi-request plans keep the batched `decode_batch`
  contract. `_execute_prefill` seeds the feedback buffer.

## Why this was the bottleneck

Sync-bracketed and host-dispatch probes (`tools/profile_decode_probes.py`)
showed that **every eager (non-JIT) op costs ~13-16 ms once the 2.5 GB model is
resident**, regardless of size — a 1-element add took 12.8 ms, a 100-element
argmax 15.8 ms — while JIT-captured kernels cost ~2 ms each (6 kernels = 11.7
ms total replay). The old engine step paid the eager tax twice per token: the
input build (`Tensor([[tid]]).realize()`, ~13 ms) and the host sampler
(`logits.argmax().tolist()`, ~16 ms), plus the `.tolist()` roundtrip. Engine
decode measured 49.5 ms/step with only ~11.7 ms of device work.

## Results (RTX 5060, Qwen3-4B Q4_K_M)

| measurement | A3 (generic) | C1 (GEMV) | C1 + fused decode |
| --- | ---: | ---: | ---: |
| engine decode step | 124.9 ms | 48.8 ms | **15.2 ms** |
| decode tok/s | ~7.9 | 20.78 | **63.12** |
| achieved GB/s | 19.9 | 51.8 | **157.3** |
| % of llama.cpp decode | 6.4% | 17.2% | **51.3%** |
| vs ceiling (~448 GB/s) | 4.4% | 11.6% | **35.1%** |

Fused decode replay is 11.78 ms/token (device, amortized); the token readback
and engine host work add ~4 ms. The C1-vs-generic kernel gain is unchanged
(replay 95.7 -> 11.8 ms). llama.cpp sustains 123.06 tok/s (306.6 GB/s) on the
same card and checkpoint, so the remaining gap is 1.9x.

## Correctness

- New CPU test: fused decode greedy tokens match `decode_batch` greedy tokens
  over a multi-position trace; the fused path reuses one capture.
- All NV model/engine tests pass with the engine on the fused path
  (`test_engine_real.py` traces generate identical greedy tokens).

## Non-claims

- The eager-op tax (~13-16 ms per eager op with the model resident) is a
  backend/execution-path artifact on this box, not a model property: prefill,
  batched decode (S >= 2), and the profile tools' `decode_batch` replay still
  pay it. The fused contract avoids it for the single-request steady decode.
- The device replay (11.78 ms, ~211 GB/s) is still ~1.45x off llama.cpp's
  8.13 ms/step (306.6 GB/s); the remaining levers are attention fusion (C2 /
  Phase 5C) and the small-N GEMV shapes, then the ~4 ms of readback and host
  work per token.