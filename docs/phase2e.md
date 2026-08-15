# Phase 2E: SSA Cache Pattern with Symbolic Position

Phase 2E replaces Phase 2C's fixed-shape mask approach with SSA cache writes
and a symbolic `Variable("position")` for cache slicing. This achieves
**O(position) decode** instead of O(max_context), reducing decode latency from
~900ms/token to ~3ms/token (280× speedup) on the full Qwen3-0.6B model.

## Design

### SSA store / after

Cache updates use the SSA `uop.store` / `uop.after` pattern from
`tinygrad/llm/model.py` instead of imperative `.assign().realize()`:

```python
# Store new k/v at the symbolic position:
store_uop = kv[i, :, slot, position:position+1, :, :].uop.store(
    Tensor.stack(k_for_store, v_for_store).uop
)
# Observe cache after the store and read [0:position+1]:
assigned = Tensor(kv[i, :, slot].uop.after(store_uop))
cached_k = assigned[0, 0:position+1, :, :].permute(1, 0, 2).unsqueeze(0).float()
```

The `Ops.STORE` is a pure graph node (no eager side effect). `Ops.AFTER`
declares "the buffer as observed after these stores complete." The cache
buffer is captured by `TinyJit` as a closure buffer (like model weights), so
its mutations persist across JIT replays without being defensively copied by
`_copy_input`.

### Symbolic position via Variable

The JIT contract is captured once with `Variable("position", 0,
max_context-1).bind(0)` during warmup. On every decode position, a new bound
value is passed; the cached linear substitutes the new value via `var_vals`,
reshaping the symbolic slices `position:position+1` and `0:position+1` to
the actual value. One compiled program replays for every position without
recompilation.

### Why @function was not used

The initial plan was to use `@function(allow_implicit=True,
precompile=True)` from `tinygrad.function` to gather weights, cache, and rope
as implicit `Ops.PARAM` inputs — matching `tinygrad/llm/model.py`.

Empirically, `@function(precompile=True)` baked the body to the warmup
binding (position=0), and replays at other positions returned stale
(identical) outputs. The precompiled body was not properly specializing for
variable bindings.

The fix: skip `@function`, let `TinyJit` capture the full graph directly.
The cache buffer naturally enters the graph as a closure buffer (same path
as model weights), and the symbolic Variable threads through the JIT's
`var_vals` mechanism correctly.

### Why the earlier Variable attempt hung

In Phase 2C, the first attempt at symbolic position used the
`.assign().realize()` imperative store pattern inside `_decode_step`. This
hung during JIT capture on NV at the full 28-layer scale (timeout >60s). The
root cause was likely the imperative `.assign().realize()` creating eager
side effects during graph construction that interacted badly with symbolic
shape handling at scale.

The SSA `uop.store/.after` pattern is purely graph-based and completes
capture in ~13s (with `BEAM=0`).

## Measured results (NV, Qwen3-0.6B Q8_0, realized-fp16, max_context=1024)

| Phase            | Time            |
|------------------|-----------------|
| JIT capture      | ~13s (BEAM=0)   |
| Prefill (16 tok) | ~6s (eager)     |
| First decode     | ~3ms            |
| Avg decode       | ~3.2ms/token    |

Decode is now **O(position)** per token. At position 16, decode reads 17 cache
positions instead of 1024 — a 60× reduction in cache access for the first
step, and the savings compound with longer contexts.

Greedy output for the fixed prompt matches Phase 2B/C: `[657, 11, 714, 279]`.

## Compared to Phase 2C

| Aspect              | Phase 2C (mask)         | Phase 2E (SSA + Var)     |
|---------------------|-------------------------|--------------------------|
| Decode complexity   | O(max_context)          | O(position)              |
| Decode latency      | ~900ms/token            | ~3ms/token               |
| Cache writes        | Outside JIT (returned)  | Inside graph (SSA)       |
| Attention masking   | Float mask (-inf/0)     | Symbolic slice (no mask) |
| Symbolic position   | None (fixed shapes)     | `Variable("position")`   |
| JIT replay          | All same graph          | Reshaped per position    |
| Capture time        | ~13s                    | ~13s                     |

## Test coverage

All Phase 2C and Phase 2D tests pass unchanged — they call `runner.decode()`
(public API unchanged). Numerical correctness is now within ~1e-6 relative
tolerance against eager decode (compared to Phase 2C's ~5% divergence due to
the mask's float rounding and unused positions).

- CPU: 102 passed, 11 deselected
- NV tiny-model: 14 passed
- NV full-model stress: 4/4 pass (correctness, clear+reuse, cancellation, replacement)