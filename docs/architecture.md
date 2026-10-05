# Architecture

Infurnace is a single-user local LLM runner on tinygrad. It exists to run one
model on one GPU, for you, as fast as possible — deliberately in the shape of
tinygrad's `tinygrad/llm` package and `examples/llama.py`.

This is the target design, not the current implementation status. See the
[roadmap](roadmap.md) for completed work and verification gates.

## Goal

- **Single user, single GPU.** No server, no batching, no scheduling.
- **Few files.** Every file has one job.
- **Fast decode.** The quantized GEMV kernel stays; serving infrastructure does not.
- **Built on tinygrad.** Infurnace owns GGUF parsing and model execution;
  tokenization reuses `tinygrad.llm.cli.SimpleTokenizer` through a thin validated
  Qwen2 adapter. No duplicate BPE engine.

## Module responsibilities

| File | Job |
|---|---|
| `gguf.py` | Validate GGUF, stage packed bytes on the selected device, return metadata and lazy dequant views |
| `tokenizer.py` | Tinygrad tokenizer adapter + metadata validation; bundled Qwen3 text-only chat rendering |
| `model.py` | `Qwen3Config` + `Qwen3Model`: prefill (`forward`), decode, and the `generate` loop; owns the KV cache |
| `kernels/quant_gemv.py` | NV decode GEMV over packed Q4_K/Q6_K weights (self-contained, device-specific) |
| `cli.py` | Single-shot / chat / benchmark driver |

## Data flow

```
GGUF file
  │  gguf.load_gguf()
  ▼
(metadata, state_dict)
  │              │
  │              ├─ model.config_from_gguf()  → Qwen3Config
  │              └─ Qwen3Model (linear weights = lazy fp16 views over resident packed bytes)
  │
  ▼
tokenizer.Tokenizer.from_gguf(metadata)
  │
  ▼
cli: tokenizer.encode(prompt) → model.generate(ids) → tokenizer.stream_decoder() → stdout
```

## Key decisions

### GGUF: packed views, not dequantized weights

Parse bounded DISK byte views, then copy packed bytes to the selected compute
device before constructing lazy dequant expressions. DISK is storage, not a
dequant execution backend. Keep native F32 at the loader boundary; the model
selects compute dtypes. Packed bytes may be realized once, decoded model-wide
weights must not be.

The NV decode path should read the same resident bytes through validated
`quant_gemv.packed_weight` extraction or explicit packed descriptors. This
avoids dequant → fp16 → re-extract. Laziness does **not** guarantee fusion:
measure embedding/prefill schedules and peak memory, bounding per-layer/tiled
dequantization if needed. Token chunking alone cannot fix retained full weights.

Supported scope is F32, Q4_K, and Q6_K; implementation is tracked in the
[GGUF roadmap](gguf-roadmap.md).

### KV cache lives inside the model

A contiguous fp16 tensor `[layers, 2, context, kv_heads, head_dim]`, allocated
by the model. No slots, no paging, no pool, no `clear_slot`/`move_slot`.
Cap context before allocation. Start with correct eager writes and reads;
then capture decode with SSA `uop.store`/`uop.after` and a bound
`UOp.variable("position", ...)`, so one graph replays across positions.
Track only valid cache entries; exact-prefix reuse comes after reset-based
generation is correct.

### Sampling is inline

Greedy argmax at temperature 0, Gumbel-max otherwise — inside `generate`, not a
`Sampler` abstraction.

### What was removed

The old serving layer is gone: `engine/`, `scheduler/`, `cache/block_pool.py`
(paged KV), `executor/` (abstract Runner + tinygrad executor), `models/`
(manifest/pinning), `sampler/`, and the tokenizer wrapper modules. Removing the
manifest means no strict artifact pinning — `gguf.py` parses whatever the file
says (tinygrad-style flexibility).

## Dependencies

`tinygrad` only at runtime. Parse header/offset bookkeeping in plain Python;
stage packed tensor data from tinygrad `disk:` tensors to the compute device.
Reference LLM implementations and tokenizer/template tooling are development
oracles, not runtime dependencies.