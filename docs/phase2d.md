# Phase 2D: Stateful Stress Validation

Phase 2D validates that cached decode remains correct under stress: repeated
conversations, cache replacement, cancellation cleanup, context-boundary
workloads, and multi-slot isolation — all through the JIT decode contract
captured in Phase 2C.

## Design

### Slot clearing

`ContiguousKVCache.clear_slot(slot)` zeros out all KV entries for a slot,
returning it to the same state as a freshly allocated cache. This is the
mechanism for conversation reuse, cancellation cleanup, and slot recycling.

The caller is responsible for re-prefilling after clearing. The JIT decode
contract is unaffected: the mask is position-based (not content-based), so
zeroed positions are correctly hidden by the mask until repopulated.

### No changes to runner or model

Phase 2D requires no changes to `Qwen3Runner` or `Qwen3Model`. The existing
`_decode_step` / `decode` / `prefill` APIs are sufficient. The only code
change is `clear_slot` on `ContiguousKVCache`.

## Test coverage

### CPU suite (tiny 2-layer model, `tests/test_runner.py`)

`TestQwen3RunnerStress` adds 8 tests:

| Test | Validates |
|------|-----------|
| `test_decode_matches_recompute_every_position` | 16-token sequence, logits at every position match full recompute |
| `test_context_boundary_decode` | Decode at position max_context-1 is correct; position max_context is rejected |
| `test_chunked_prefill_then_jit_decode` | Chunked eager prefill followed by JIT decode matches recompute |
| `test_repeated_conversation_no_leak` | Conversation A → clear → conversation B matches fresh runner |
| `test_cancellation_cleanup` | Partial conversation → cancel → new conversation matches fresh runner |
| `test_clear_slot_rejects_out_of_range` | clear_slot validates slot bounds |
| `test_cache_replacement_same_output` | Two runners with separate caches produce identical output |
| `test_multi_slot_runner_isolation` | Runner with 2 slots, interleaved decodes, independent output |

CPU results: 102 passed, 11 deselected, 90 subtests passed.

### NV tiny-model suite

14 passed (6 Phase 2C + 8 Phase 2D).

### NV full-model stress (Qwen3-0.6B Q8_0, realized-fp16, max_context=1024)

| Test | Result |
|------|--------|
| Multi-token decode correctness | `[657, 11, 714, 279]` matches Phase 2C |
| Repeated conversation (clear + reuse) | Matches fresh runner, no stale state |
| Cancellation cleanup | Matches fresh runner, no stale state |
| Cache replacement (two runners) | Identical output from independent runners |

## Phase 2 gate

> Single-request cached generation remains correct through eager execution,
> warmup, capture, replay, cache replacement, and repeated conversations.
> The model owns no conversation history or KV state.

**Status: PASS**

- Eager execution: Phase 2B tests validate cached decode vs full recompute.
- Warmup and capture: Phase 2C tests validate JIT capture on CPU and NV.
- Replay: Phase 2C `test_decode_does_not_recompile_per_position` confirms single
  contract replays across positions; Phase 2D `test_decode_matches_recompute_every_position`
  extends this to 16 positions with per-position logits comparison.
- Cache replacement: Phase 2D `test_cache_replacement_same_output` confirms
  independent runners with separate caches produce identical output.
- Repeated conversations: Phase 2D `test_repeated_conversation_no_leak` and
  `test_cancellation_cleanup` confirm no stale state leaks between conversations
  after `clear_slot`.
- The model owns no conversation history or KV state: `Qwen3Model` has no
  conversation state; `kv_cache` is set by the runner and is external to the model
  constructor. `clear_slot` is on the cache, not the model.