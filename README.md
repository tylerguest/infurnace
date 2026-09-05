# Infurnace

Infurnace is an inference server for decoder-only language models. It runs on
tinygrad, which handles all tensor compilation and device execution, while
Infurnace owns the serving side: scheduling, batching, KV-cache management,
sampling, streaming, and metrics.

The current target is Qwen3-4B (Q4_K_M) on NVIDIA GPUs. Infurnace tracks current
tinygrad rather than pinning a version.

## Install

Requires Python 3.11+. Clone tinygrad into an adjacent directory, then:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e ../tinygrad
.venv/bin/python -m pip install -e ".[test]"
```

## Quick example

Acquire the pinned Qwen3-4B GGUF, then run:

```sh
.venv/bin/python tools/fetch_model.py \
  --manifest models/qwen3-4b-q4_k_m.json \
  --output artifacts/models/Qwen3-4B-Q4_K_M.gguf

.venv/bin/python -m infurnace.cli \
  --artifact artifacts/models/Qwen3-4B-Q4_K_M.gguf \
  --prompt "Hello, world" --max-tokens 32
```

## Tests

```sh
.venv/bin/python -m pytest
```

Hardware and model suites are opt-in and require a GPU and model artifact:

```sh
DEV=NV .venv/bin/python -m pytest -m nv
DEV=NV .venv/bin/python -m pytest -m "nv and model and slow"
```

## Learn more

See the [architecture](docs/architecture.md) and [roadmap](docs/roadmap.md)
for design and implementation gates, and [benchmarks](benchmarks/README.md)
for measurement methodology.