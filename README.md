# infurnace

Run LLMs on your GPU with tinygrad. No server, no batching, no scheduler —
load a GGUF, generate tokens, stream them out. A few files, one dependency.

## Install

```sh
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Run

Drop a GGUF in `models/`, then:

```sh
.venv/bin/infurnace --prompt "Hello"      # single-shot
.venv/bin/infurnace --chat                # chat
.venv/bin/infurnace --benchmark 20        # tokens/sec
```

`--model` overrides the default `models/Qwen3-4B-Q4_K_M.gguf`.

## Status

Scaffold. It installs and imports; the parser, tokenizer, and model aren't
written yet — order of work is in `docs/roadmap.md`.

## Layout

```
infurnace/
├── gguf.py            # GGUF parser + quantized dequant
├── tokenizer.py       # BPE tokenizer + chat template
├── model.py           # forward / decode / generate + KV cache
├── kernels/
│   └── quant_gemv.py  # NV Q4_K/Q6_K decode kernel
└── cli.py             # the CLI above

docs/
├── architecture.md
└── roadmap.md
```