# infurnace

Run LLMs on your GPU with tinygrad. No server, no batching, no scheduler —
load a GGUF, generate tokens, stream them out. A few files, one dependency.

## Install

```sh
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Planned CLI

Inference is not implemented yet. Once ready, put a GGUF in `models/`, then:

```sh
.venv/bin/infurnace --prompt "Hello"      # single-shot
.venv/bin/infurnace --chat                # chat
.venv/bin/infurnace --benchmark 20        # tokens/sec
```

`--model` overrides the default `models/Qwen3-4B-Q4_K_M.gguf`.

## Status

Phase 2: F32/Q4_K/Q6_K GGUF loading works on CPU with lazy dequantization
and explicit packed-byte staging. GPU integration still needs verification.
Qwen3 config/weight validation, a tinygrad tokenizer adapter, and text-only
Qwen3 chat rendering are implemented. Sampled tokenizer IDs match llama.cpp;
chat output matches the bundled template. Model execution and CLI modes remain stubs. NV decode kernel eager/JIT tests
pass; model integration is pending. See the [roadmap](docs/roadmap.md)
for ordered tasks and verification gates.

## Layout

```
infurnace/
├── gguf.py            # GGUF parser + quantized dequant
├── tokenizer.py       # tinygrad tokenizer adapter + Qwen3 chat rendering
├── model.py           # forward / decode / generate + KV cache
├── kernels/
│   └── quant_gemv.py  # NV Q4_K/Q6_K decode kernel
└── cli.py             # the CLI above

docs/
├── architecture.md
├── roadmap.md
└── gguf-roadmap.md
```