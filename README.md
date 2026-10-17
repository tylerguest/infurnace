# infurnace

Fast local inference for modest GPUs: **Qwen3.5-4B** on [tinygrad](https://github.com/tinygrad/tinygrad),
with [DFlash](docs/dflash.md) speculative decoding. Chat in the terminal or serve an OpenAI-compatible
API for coding agents. One user and one model; no batching, scheduler or multi-GPU.

On an 8 GB RTX 5060:

| | |
|---|---|
| Plain decoding | ~110 tok/s up to 8k tokens of context, ~89 tok/s at 16k |
| DFlash | ~340 tok/s on code, ~140–230 tok/s on prose; ~100 tok/s at 16k |
| Reading a prompt | ~600 tok/s from 1k to 16k tokens: 1.7 s to first token at 1k, 30 s at 16k |
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
infurnace                                    # chat; downloads ~3.1 GB on first use, warms up while you type
infurnace --no-thinking
infurnace --temperature 0.6                  # sample instead of greedy (works with DFlash)
infurnace --prompt "Hello" --max-tokens 64   # one raw prompt, no chat template
infurnace --serve 8000 --max-context 32768   # OpenAI-compatible API, e.g. for pi
infurnace --benchmark 512,32                 # per-step plain decode timing
infurnace --no-dflash                        # plain decoding
```

- **Models:** `--model qwen3.5:4b` (default) downloads the target (2.7 GB) and draft (0.4 GB) into
  `~/.local/share/infurnace/models/`. `--model PATH` runs a local Qwen3.5 GGUF. See [Qwen3.5](docs/qwen35.md).
- **Device:** NVIDIA through tinygrad's `NV` backend (Linux, probed first) or `CUDA`; `--device` or
  `DEV` overrides. Startup takes about a minute (loading plus capturing GPU graphs).
- **Context:** `--max-context` (default 8,192). Memory with DFlash is about 5.2 GB at 8k and
  5.5 GB at 32k, plus whatever your desktop uses.
- **Thinking:** on by default in chat and serving; `--no-thinking` turns it off. Replies run until
  the model stops or the context fills, unless capped with `--max-tokens` (which counts thinking).
- **Sampling:** greedy by default; `--temperature T` samples. Qwen suggests about 0.6 with thinking,
  where greedy can repeat.
- **Chat history:** when a new message doesn't fit, the oldest turns are dropped with a notice.
- **Serving:** [server setup and pi configuration](docs/serve.md).

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
