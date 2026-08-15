# Phase 2C: TinyJit Decode Contract

> **Note:** This document describes the original Phase 2C mask-based approach,
> which has been **superseded by Phase 2E** (SSA cache writes + symbolic
> `Variable` position). Phase 2E reduces decode latency from ~900ms/token to
> ~3ms/token. See `docs/phase2e.md` for the current implementation.

Phase 2C captures a fixed-shape TinyJit decode contract so a single compiled program
replays for every decode position without recompilation.

## Design

The decode step reads the **full** pre-allocated KV cache (fixed shape) and applies a
float attention mask to ignore unpopulated positions. The position-dependent inputs
(mask and rope slice) are passed as realized Tensor arguments; the new K/V stores are
returned from the JIT function and written to the cache **outside** the JIT. This
keeps the UOp graph fixed-shape and lets TinyJit reuse one captured program for all
positions.

### Why not symbolic slicing

The original plan used a bound `Variable("pos", 0, max_context-1)` to slice the KV
cache (`kv[:, :position, ...]`) inside the JIT, matching the upstream LLaMA pattern.
This worked on CPU and on NV with a tiny 2-layer model but **hung during JIT capture**
on the full 28-layer Qwen3-0.6B model on NV (timeout after >60s). `DEBUG=2` showed
thousands of kernels with `(pos+1)` symbolic dims being compiled before the process
was killed.

Two root causes were investigated and ruled out:
- KV cache as external JIT input vs model attribute (both hang).
- Per-layer `Tensor.realize` of KV writes vs deferred realize (both hang).

The mask-based approach avoids `Variable` entirely and completes capture in ~13s.

### Architecture

```
Qwen3Runner
├── __init__: sets model.kv_cache, captures decode JIT per slot (BEAM=0)
├── prefill:  eager forward (Phase 2B path, external kv argument)
├── decode:   builds mask + rope slice, calls JIT, writes stores to cache
└── _decode_jit[slot]: TinyJit wrapping model._decode_step

Qwen3Model._decode_step(input_ids, attn_mask, rope, slot) -> (logits, k_stores, v_stores)
├── reads full cache via self.kv_cache.kv  (closure buffer, not JIT input)
├── cats new token K/V at position max_context (end of cache)
├── applies attn_mask: 0 for valid positions, -inf for unpopulated
├── returns logits + stacked k_stores/v_stores
└── caller writes stores to kv[..., position:position+1, ...] outside JIT
```

### Attention mask construction

The mask has shape `[1, 1, 1, max_context + 1]`. The `+1` accounts for the new token
appended at the end of the full cache via `.cat()`:
- Positions `[0, position)`: `0.0` (valid cached tokens)
- Position `position`: `-inf` (not yet populated in the cache)
- Positions `(position, max_context)`: `-inf` (unpopulated cache slots)
- Position `max_context`: `0.0` (the new token just appended)

The mask is built with `Tensor.where` on a boolean index arange, then
`.contiguous().realize()` to produce a physical buffer that TinyJit accepts.

### JIT capture contract

- Warmup and capture run on the production cache with `Context(BEAM=0)` to skip
  kernel search and reduce one-time capture cost.
- The cache is a model attribute (`model.kv_cache`), so it is a closure buffer in
  the captured graph — not a JIT input. This matches the upstream LLaMA pattern and
  avoids the external-input mutation issue documented in the tinygrad JIT.
- Cache replacement requires creating a new `Qwen3Runner` (which recaptures the
  contract bound to the new cache buffer).

## Known tradeoff

Decode is **O(max_context)** per token because it reads all `max_context` cache slots
every step. For validation (4–8 tokens at max_context=1024) this is acceptable:
~900ms/token steady state vs the original Phase 2B eager ~5min for 4 tokens.

For production throughput, this should be replaced with position-bounded reads either
via the upstream `Variable` path (after investigating the NV hang at 28-layer scale)
or via bucketed fixed-shape windows.

## Measured results (NV, Qwen3-0.6B Q8_0, realized-fp16, max_context=1024)

| Phase            | Time    |
|------------------|---------|
| JIT capture      | ~13s    |
| Prefill (16 tok) | ~6s     |
| First decode     | ~925ms  |
| Avg decode       | ~900ms  |

Greedy output for the fixed prompt matches Phase 2B eager: `[657, 11, 714, 279]`.

## Test coverage

- `tests/test_cached_forward.py`: Phase 2B eager prefill/decode correctness (88 tests)
- `tests/test_runner.py`: Phase 2C JIT decode replay, no-recompile, independent
  runners, input validation, model-rejects-without-cache (6 tests)
- Full CPU suite: 94 passed, 11 deselected
- NV tiny-model suite: 6 passed
- NV full-model harness: confirmed working with timing output