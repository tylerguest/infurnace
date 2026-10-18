# infurnace

Fast local inference for modest GPUs: **Qwen3.5-4B** on [tinygrad](https://github.com/tinygrad/tinygrad),
with [DFlash](docs/dflash.md) speculative decoding. Chat in the terminal or serve an OpenAI-compatible
API for coding agents. One user and one model; no batching, scheduler or multi-GPU.

On an 8 GB RTX 5060:

| | |
|---|---|
| Plain decoding | ~105 tok/s up to 8k tokens of context, ~95 tok/s at 16k |
| DFlash | ~340 tok/s on code, ~200–255 tok/s on prose; ~180 tok/s at 16k |
| Reading a prompt | ~1,850–2,300 tok/s from 1k to 16k tokens (~1,400–1,750 with DFlash): 0.55 s to first token at 1k, 7.4 s at 16k |
| Context | 8,192 tokens by default; 32,768 fits in 8 GB |
| Later turns | resume from a saved prompt state: ~0.6 s to first token in a 3k-token chat |

Only this card is validated. The fast path needs an NVIDIA GPU (RTX 30-series or newer for DFlash);
everything else runs tinygrad's generic code, which is far too slow to use. See the [roadmap](docs/roadmap.md).

## Install

Requires Python 3.11+, Git, an NVIDIA driver, and NVIDIA's NVRTC compiler with the CUDA headers, which
compile the kernels. The **CUDA toolkit** provides both. If you'd rather not install it, point tinygrad at any
other copy, for example NVIDIA's pip packages:

```sh
pip install nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12
export NVRTC_PATH=$(python -c "import nvidia.cuda_nvrtc, glob; print(glob.glob(nvidia.cuda_nvrtc.__path__[0] + '/lib/libnvrtc.so.*')[0])")
export CUDA_PATH=$(python -c "import nvidia.cuda_runtime; print(nvidia.cuda_runtime.__path__[0])")
```

When either is missing, infurnace stops and says which.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .              # '.[server]' for the HTTP server, '.[dev]' for tests
```

The tinygrad dependency is pinned to a fork with one fix for back-to-back GPU submissions. After
pulling changes, reinstall; if the pin changed, force-reinstall tinygrad.

## Run

```sh
infurnace                                    # chat with the default model (starts the server if needed)
infurnace run "Explain RAII in one paragraph"  # one answer, then exit
cat notes.txt | infurnace run "Summarize this" --nothink
infurnace serve                              # start the server yourself (keeps the model warm on the GPU)
infurnace ps                                 # what the server is running
infurnace stop                               # stop it and free the GPU
infurnace launch pi                          # point pi at the server and start it
infurnace pull qwen3.5:4b                    # download a model (and its DFlash draft); also: list, rm
```

- **The server:** `infurnace serve` loads a model once and keeps it warm; `run`, `ps`, `stop` and `launch` talk
  to it, and `run` starts one in the background when none answers (log: `~/.local/state/infurnace/serve.log`).
  It listens right away and warms up meanwhile: about a minute on a fresh start, then replies start in a
  fraction of a second. `INFURNACE_HOST` picks the address (default `127.0.0.1:8000`). It serves an
  OpenAI-compatible API at `/v1` and infurnace's own at `/api` (chat, generate, tags, ps, show, status).
- **Chat commands:** `/set think`, `/set nothink`, `/set temperature T`, `/clear`, `/bye`, and `"""` for
  multi-line messages.
- **Models:** `qwen3.5:4b` (default) downloads the target (2.7 GB) and draft (0.4 GB) into
  `~/.local/share/infurnace/models/`. `serve --model PATH` runs a local Qwen3.5 GGUF. See [Qwen3.5](docs/qwen35.md).
- **Context:** the most that fits in free GPU memory, in steps of 8,192 (32,768 on an 8 GB card next to a
  desktop); `--max-context` overrides. Memory with DFlash is about 4.6 GB plus 36 KB per token of context.
- **Device:** NVIDIA through tinygrad's `NV` backend (Linux, probed first) or `CUDA`; `--device` or `DEV` overrides.
- **Thinking:** on by default; `--nothink` (run) or `serve --no-thinking` turns it off. Replies run until the
  model stops or the context fills, unless capped with `--max-tokens` (which counts thinking).
- **Sampling:** greedy by default; `--temperature T` samples. Qwen suggests about 0.6 with thinking, where
  greedy can repeat.
- **Chat history:** when a conversation no longer fits, the oldest turns are dropped with a notice.
- **Without the server:** the original flags still run in-process: `infurnace --chat`, `--prompt "..."`
  (a raw prompt, no chat template), `--serve PORT`, `--benchmark 512,32`, `--no-dflash`.
- **Serving details and pi:** [serving](docs/serve.md).

## Docs

- [Architecture](docs/architecture.md): modules, kernels, execution paths, snapshots
- [DFlash](docs/dflash.md): how speculative decoding works here, and measurements
- [Serving](docs/serve.md): the HTTP server and agent setup
- [Roadmap](docs/roadmap.md): status, priorities, validation, what was tried
- [Benchmarks](benchmarks/README.md)

## Layout

```text
infurnace/
├── cli.py          # chat, raw prompt, server, benchmark
├── serve.py        # tinygrad's HTTP server with DFlash and per-request thinking
├── loader.py       # named models, downloads, device dispatch
├── device.py       # device probing, NVIDIA fast-path check
├── gguf.py         # GGUF parsing and metadata
├── tokenizer.py    # Qwen3.5 tokenizer and chat formatting
├── models/         # Qwen3.5 adapter (packed weights, fused layers, snapshots), packed projections
├── dflash/         # draft, verification, adaptive drafting
└── kernels/        # projections, recurrent layers, attention, sampling, embedding
```
