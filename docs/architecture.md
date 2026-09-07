# Architecture

Infurnace is a single-user local LLM runner on tinygrad. It exists to run one
model on one GPU, for you, as fast as possible — deliberately in the shape of
tinygrad's `tinygrad/llm` package and `examples/llama.py`.

## Goal

- **Single user, single GPU.** No server, no batching, no scheduling.
- **Few files.** Every file has one job.
- **Fast decode.** The quantized GEMV kernel stays; serving infrastructure does not.
- **Standalone on tinygrad.** Only `tinygrad.Tensor` (and friends) are imported;
  tinygrad's own LLM helpers (`tinygrad.llm.gguf`, `tinygrad.llm.cli.SimpleTokenizer`)
  are *not* used — infurnace owns its parser and tokenizer.

## Module responsibilities

| File | Job |
|---|---|
| `gguf.py` | Parse the GGUF container; return `(metadata, state_dict)` with quantized weights as lazy packed views |
| `tokenizer.py` | BPE encode/decode + streaming decode + chat template, built from GGUF vocab metadata |
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
  │              └─ Qwen3Model (weights = lazy fp16 views over packed bytes)
  │
  ▼
tokenizer.Tokenizer.from_gguf(metadata)
  │
  ▼
cli: tokenizer.encode(prompt) → model.generate(ids) → tokenizer.stream_decoder() → stdout
```

## Key decisions

### GGUF: packed views, not dequantized weights

The Q4_K/Q6_K linears are kept as *lazy packed byte views* over the file. The
dequant expression is fused into the consuming kernel (embedding lookup, fp16
matmul fallback), so nothing materializes 2x-the-file fp16. The decode fast
path reads the same packed view directly via `quant_gemv.packed_weight`. This
avoids the old pipeline's dequant → fp16 → re-extract round trip.

Only the quant types the model needs are implemented: F32, Q4_K, Q6_K.

### KV cache lives inside the model

A contiguous fp16 tensor `[layers, 2, context, kv_heads, head_dim]`, allocated
by the model. No slots, no paging, no pool, no `clear_slot`/`move_slot`.
Decode writes K/V with SSA `uop.store`/`uop.after` at a symbolic `Variable("position")`
inside a TinyJit capture, so one compiled program replays for every position.

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

`tinygrad` only. The parser reads the file header/offset bookkeeping in plain
Python; tensor data is read through tinygrad `disk:` device tensors.