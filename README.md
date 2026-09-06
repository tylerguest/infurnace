# Infurnace

Infurnace is an inference server for decoder-only language models. It runs on
tinygrad, which handles all tensor compilation and device execution, while
Infurnace owns the serving side: scheduling, batching, KV-cache management,
sampling, streaming, and metrics.

**Status:** Development target is v0.3 — a paged continuous-batching
inference server on one GPU with a minimal HTTP surface. See the
[roadmap](docs/roadmap.md) for the end-state gates.

## Install

Requires Python 3.11+. Clone tinygrad into an adjacent directory, then:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e ../tinygrad -e ".[test]"
```

## Quick example

Fetch the pinned Qwen3-4B GGUF, then run the CLI:

```sh
.venv/bin/python tools/fetch_model.py \
  --manifest models/qwen3-4b-q4_k_m.json \
  --output artifacts/models/Qwen3-4B-Q4_K_M.gguf

.venv/bin/infurnace --prompt "Hello, world"
```

Add `--chat` for interactive multi-turn chat. The CLI defaults to the fetched
artifact and manifest; see `--help` for all options.

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