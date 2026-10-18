# infurnace

Fast small language models for the GPU poverty stricken.
Pure Python on [tinygrad](https://github.com/tinygrad/tinygrad): one pip install, then one command.

On an 8 GB RTX 5060: ~110 tokens/s, 200–340 with speculative decoding, prompts read at ~2,000 tokens/s.

## Install

```sh
git clone https://github.com/tylerguest/infurnace.git
cd infurnace
python3 -m pip install -e .
```

You need an NVIDIA GPU and Python 3.11+. If you don't have the CUDA toolkit, add NVIDIA's compiler from pip:

```sh
python3 -m pip install nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12
```

## Use

```sh
infurnace                       # chat
infurnace run "your question"   # one answer
infurnace launch pi             # use it in the pi coding agent
infurnace stop                  # free the GPU
```

The first run downloads the model (~3 GB) and warms up for about a minute. After that, replies start instantly.

Other apps can use it as an OpenAI-compatible API at `http://127.0.0.1:8000/v1`.

## Models

| Model | Name | Speed (RTX 5060) |
|---|---|---|
| Qwen3.5-4B | `qwen3.5:4b` (default) | ~110 tokens/s, 200–340 with DFlash |
| Qwen3-4B | `qwen3:4b` | ~108 tokens/s |

`infurnace run qwen3:4b` runs another one; `infurnace list` shows them all. More small models are coming.

## Learn more

[Commands and serving](docs/serve.md) · [How it works](docs/architecture.md) · [Roadmap](docs/roadmap.md) · [Benchmarks](benchmarks/README.md)
