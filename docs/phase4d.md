# Phase 4D: Steady-State Stability

Phase 4D verifies that fixed-shape batched decode is stable under long, ragged
decode traces: steady `decode_batch` replays allocate no new persistent device
buffers and never re-capture a TinyJit contract. It completes the Phase 4 gate
(fixed ragged decode batches, `v0.2`).

## Contracts under test

Batched decode pads each call to the next supported shape (1, 2, 4). With
`num_slots=4` the reserved dummy slot is physical slot 4, so the distinct
captured contracts are:

| Active rows B | Padded shape | Captured key |
| --- | --- | --- |
| 1 | 1 | `(0,)` |
| 2 | 2 | `(0, 1)` |
| 3 | 4 (one dummy row) | `(0, 1, 2, 4)` |
| 4 | 4 | `(0, 1, 2, 3)` |

Each contract is captured lazily on first use (`Qwen3Runner._decode_batch_jit`)
and replayed afterward.

## Measurement

- **Recapture detection.** `TinyJit.cnt` is 0 (ignore), 1 (capture), >= 2 (replay);
  capture runs once and a shape mismatch raises `JitError` rather than re-capturing.
  Tests snapshot `jit.captured` (identity) and `jit.cnt` after warmup and assert
  every steady step increments `cnt` by exactly one with an unchanged `captured`.
- **Allocation.** `GlobalCounters.mem_used` (CPU) and `mem_used_per_device["NV"]`
  (NV) count live requested bytes; deallocation subtracts, so a flat value after
  warmup means no per-step persistent growth. Capture workspaces are allocated
  once at warmup and reused by every replay (`CapturedJit.__call__` runs the
  memory-planned `linear` with `jit=True`, so replays reuse planned buffers).

## Comparison with production servers

The steady-state decode model matches vLLM, SGLang, and llama.cpp:

- **Memory is pre-allocated once.** KV pool (`ContiguousKVCache`), like vLLM's
  `BlockPool`/`init_kv_cache`, SGLang's `MemoryPool`, and llama.cpp's KV tensors
  and reserved compute buffers. No per-step growth.
- **Capture once, replay per step.** One TinyJit per supported shape, like
  vLLM's per-descriptor CUDA graphs and SGLang's per-bucket graphs; a shape
  mismatch raises rather than re-capturing, like vLLM's `CompilationCounter`
  and llama.cpp's graph reuse guard.
- **Finished requests are dropped before each decode step** (`filter_batch`,
  scheduler `remove_all`, `i_batch[i] = -1`), and the batch is padded to a
  captured shape (SGLang `_pad_to_bucket`, vLLM dispatch padding, llama.cpp
  `n_ubatch` slicing).
- **Sampling runs once per decode step** over the whole batch.

Known divergence, deferred: vLLM/SGLang copy live inputs into pre-allocated
static buffers to avoid any per-step input allocation; `decode_batch` currently
builds a fresh `padded_ids` input tensor (including a host roundtrip via
`.tolist()`) each step. The buffer is transient and freed, so steady-state
memory stays flat, but replacing it with a persistent input buffer is a future
optimization rather than a stability fix.

## Test coverage

### CPU suite (tiny 2-layer model, `tests/test_stability.py`)

`TestSteadyStateStability`:

| Test | Validates |
|------|-----------|
| `test_steady_shapes_flat_and_no_recapture` | 30 steady steps at every shape: memory returns to post-warmup baseline, `cnt` grows by exactly 30 per contract, `captured` identity stable |
| `test_ragged_trajectory_no_new_contracts` | Batch shrinks 4 -> 3 -> 2 -> 1: no new contracts, captures stable, memory flat, and every step matches independent eager single decode |

### NV tiny-model suite

`TestSteadyStateStabilityNV` runs the same invariants against
`mem_used_per_device["NV"]` on real device buffers (20 steady steps per shape).

### NV full-model smoke (Qwen3-0.6B Q8_0)

`test_real_runner_ragged_batch_stable` (`test_engine_real.py`) drives four
concurrent requests through the engine: the decode batch reaches shape 4, shrinks
ragged to 1, uses only the four known contracts, and each request's greedy output
matches the same request run alone.

## Results

All gates ran on the RTX 5060 (8 GB) with the current tinygrad checkout.

- **CPU** (`TestSteadyStateStability`): `test_steady_shapes_flat_and_no_recapture`
  and `test_ragged_trajectory_no_new_contracts` pass. Over 30 steady steps at
  every supported shape, `GlobalCounters.mem_used` returns to its post-warmup
  baseline, each contract's `cnt` grows by exactly 30 with an unchanged
  `captured`, and the ragged 4 -> 3 -> 2 -> 1 trajectory creates no new
  contracts and matches independent eager single decode.
- **NV tiny-model** (`TestSteadyStateStabilityNV`): the same invariants hold on
  real device buffers over 20 steady steps per shape; `mem_used_per_device["NV"]`
  is flat after warmup.
- **NV full-model smoke** (`test_real_runner_ragged_batch_stable`): four
  concurrent Qwen3-0.6B requests decode in a batch of 4 that shrinks ragged to 1;
  the shape-4 contract `(0, 1, 2, 3)` is reached, only the four known contracts
  exist, and each request's greedy output matches the same request run alone.
  The smoke uses lazy FP16 weights to keep five resident models within the 8 GB
  device.
- **Capture cost measured** (realized FP16): model+load 1.19 GB, +runner build
  1.34 GB, +all four captures 1.35 GB. Lazy FP16: 0.64 GB -> 0.79 GB -> 0.80 GB.
  Captures add under 20 MB; replay adds nothing.

## Phase 4 gate

> Fixed ragged decode batches match independent single-request execution without
> cache corruption, inactive-row mutation, recapture, or cross-request output
> changes.

**Status: PASS**

- Fixed ragged batches match single-request execution: 4B batch contracts and the
  4D ragged trajectory compare against independent eager single decode.
- No cache corruption / inactive-row mutation: 4B dummy-row tests; the 4D ragged
  trajectory keeps every active row correct while rows are dropped.
- No recapture: 4D asserts `TinyJit.captured` identity and `cnt` deltas.
- No cross-request output changes: 4C engine tests and the NV batch-membership
  smoke compare batched and single-request output.