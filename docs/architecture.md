# Architecture

Infurnace is a single-user **Qwen3.5-4B application on tinygrad**, with its DFlash
speculative draft. It is not a second inference engine or general model runner.

## Ownership

| Module | Responsibility |
|---|---|
| `loader.py` | One supported preset, first-use downloads, architecture validation and device dispatch |
| `gguf.py` | Tinygrad's GGUF parser, truncation checks, packed DISK views and lazy decoding |
| `models/qwen35.py` | Upstream model loading, runner-facing config, bounded generation, packed loading and parameter cache |
| `models/linear.py` | Packed NV projection adapter shared by target and draft |
| `tokenizer.py` | Qwen3.5 BPE/NFC encoding, byte-stream decoding and text-only chat |
| `cli.py` | Chat, raw prompts, warmup, context rollover and per-step benchmarks |
| `serve.py` | Device/lifecycle adapter and GGUF template for tinygrad's HTTP server |
| `dflash/` | Draft features/cache, request ownership, adaptive drafting and parallel verification |
| `kernels/` | Packed GEMV/GEMM, embedding lookup and fused GDN operations |

Tinygrad owns hybrid layer configuration, ordinary transformer execution, cache
allocation/prefix reuse, RoPE, sampling and JIT. Infurnace dispatches packed
projections and supported one-row GDN work to custom tinygrad UOp kernels.
DFlash's multirow verifier implements speculative execution and commits only
accepted recurrent-state rows; see [DFlash](dflash.md).

## Loading

```text
qwen3.5:4b or local GGUF
  → resolve/download target
  → validate dense Qwen3.5 architecture
  → build Qwen3.5 tokenizer and upstream Transformer
  → on eligible NV, stage packed weights and install packed projections
  → realize remaining small parameters, using a source/model-keyed disk cache
```

The tied output head and embedding share packed storage. An explicit untied
output head must not replace the embedding table. Multirow embeddings use a
packed gather for tied weights or a host FP16 table otherwise.

Other architectures are rejected; there is no alternate model-family fallback.
The supported preset is 4B, though small Qwen3.5 fixtures exercise CPU behavior.

## Generation and state

`Qwen35Model.generate()` validates IDs and budgets, copies the prompt, bounds
output by context, and delegates to upstream generation. Its device context is
active only while advancing upstream, not while yielding application tokens.
Only one stream may own the model; callers must close abandoned streams.
Reset invalidates reusable prefix tokens without reallocating caches or JITs.

DFlash holds the same exclusive model ownership, resets its transient state per
request, and invalidates ordinary-generation prefix reuse. Sequential verification
steps one row at a time; NV defaults to fixed-width parallel verification.

The CLI handles EOS and UTF-8 flushes, rejects oversized individual messages,
and drops oldest complete turns with a visible notice. Thinking is enabled by
default. HTTP serving uses the GGUF Jinja template and upstream tool parser;
it does not use DFlash or the basic CLI chat formatter.

## Performance boundary

Packed GEMV uses Q8-rounded activations; multirow tensor-core GEMM rounds to
FP16 and accumulates in FP32. These differ numerically from generic operations.
Use `loader.load_model(..., nv_quant=False)` or benchmark `--no-nv-quant` for
the generic baseline. Plain NV generation still prefills recurrent models one
row at a time; DFlash's prefill is multirow.

Timing counters estimate traffic and work, not measured DRAM bandwidth or GPU
peak memory. See [benchmarks](../benchmarks/README.md) and [roadmap](roadmap.md).
