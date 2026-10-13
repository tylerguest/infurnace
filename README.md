# infurnace

A small, single-user **Qwen3.5-4B runner on tinygrad**, with its optional
[DFlash draft](docs/dflash.md). Chat locally or serve an OpenAI-compatible API;
no batching, scheduler, multi-GPU or other model-family support.

The tested setup is **Qwen3.5-4B Q4_K_M + DFlash on an 8 GB RTX 5060**.
Tinygrad owns the transformer, caches, sampling and JIT. Infurnace adds GGUF
validation, a local chat CLI, packed NVIDIA projections and speculative decoding.
Other GPUs and Qwen3.5 variants are not validated.

Local measurements put plain decoding around 110 tok/s. DFlash can improve code
answers by 1.4–2.2×, but falls back when drafting loses. These are workload-dependent
measurements, not quality parity or a comparison against another inference engine.

## Install

Requires Python 3.11+ and Git:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[qwen35]'
```

For development/tests, use `pip install -e '.[qwen35,dev]'`. The tinygrad dependency
is pinned. After pulling changes, reinstall; if the pin changes, force reinstall
that dependency because pip can retain an older same-version wheel.

## Models

`--model` / `-m` accepts `qwen3.5:4b` (the default) or a local Qwen3.5 GGUF path.
Named files are downloaded on first use into `~/.local/share/infurnace/models/`
(`XDG_DATA_HOME` respected). Existing copies there or under checkout `models/`
are reused. Model weights are not shipped in the package.

- Target: [Unsloth Qwen3.5-4B Q4_K_M](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF), about 2.7 GB.
- Draft: [Qwen3.5-4B DFlash Q4_K_M](https://huggingface.co/Anbeeld/Qwen3.5-4B-DFlash-GGUF), about 0.4 GB;
  downloaded only when automatic DFlash is used on NV.

```sh
infurnace                         # download if needed, warm up and chat
infurnace -m /path/to/Qwen3.5-4B-Q4_K_M.gguf
infurnace --no-dflash              # plain decoding
```

Local targets use an installed draft only. `--dflash PATH` or
`INFURNACE_DFLASH_DRAFT` overrides the draft. Uninstalling retains downloaded files.
Details and the selected model revision: [Qwen3.5 setup](docs/qwen35.md).

## Run

```sh
infurnace --no-thinking
infurnace --prompt "Hello" --max-tokens 64   # raw text, no chat template
infurnace --serve                          # 127.0.0.1:8000, plain decoding
infurnace --benchmark 512,32                # per-step timing
infurnace --device CPU
```

- **Hardware:** probes usable GPUs, falling back to CPU; `--device` or `DEV`
  overrides selection. NV/CUDA-C uses packed projections automatically.
- **Startup:** chat and serving warm up first; `--no-warmup` skips this.
  Single-shot generation skips automatic warmup.
- **Thinking:** enabled in chat; `--no-thinking` disables it.
- **Context:** defaults to 8,192 tokens; `--max-context` changes the window. At the default, infurnace
  uses about 5.3 GB of GPU memory with DFlash (RTX 5060). Each extra 8k tokens adds roughly half a GB
  of cache, so 16,384 fits on an 8 GB card if other apps leave room.
  Chat drops oldest complete turns with a notice and rejects oversized messages.
- **Replies:** run until EOS or context exhaustion unless capped with
  `--max-tokens` (includes thinking). Raw prompts default to 64 output tokens.
- **Serving:** install `.[qwen35,server]`; tool formatting/parsing reuses the GGUF
  template and tinygrad server. Real-model tool-use reliability is unvalidated.
  See [server setup](docs/serve.md).

`infurnace --help` lists all flags. Benchmark scripts can disable packed execution
with `--no-nv-quant`; programmatic loading accepts `nv_quant=False`.

## Acceleration and limits

Q4_K/Q5_K/Q6_K projections stay packed. Single-row GEMV uses Q8-rounded
activations; eligible multirow work uses FP16 tensor-core GEMM with FP32
accumulation. The one-row GDN path also has an NV fused kernel. Norms, RoPE,
attention, sampling and ordinary generation remain upstream.

Plain NV generation currently prefills one row at a time because of upstream's
recurrent-model generation path. DFlash uses fixed-width multirow verification
and prefill, plus an adaptive fallback to one-row decoding.

Acceleration is experimental. Full-model quality comparisons, long-session
reliability and GPU peak-memory measurements remain open.

- [Architecture](docs/architecture.md)
- [Roadmap](docs/roadmap.md)
- [Measurement limits](docs/baseline.md)
- [Benchmarks](benchmarks/README.md)

```sh
python benchmarks/decode.py --tokens 32
python benchmarks/compare.py --model qwen3.5:4b --tokens 32 --runs 1 \
  --output /tmp/infurnace-vs-tinygrad.json
```

## Layout

```text
infurnace/
├── cli.py              # chat, streaming and per-step benchmarks
├── loader.py           # Qwen3.5 preset, downloads and device dispatch
├── device.py           # backend probing and NV eligibility
├── gguf.py             # validated packed storage and lazy decoding
├── tokenizer.py        # Qwen3.5 BPE, streaming decode and text chat
├── serve.py            # tinygrad HTTP server adapter
├── models/
│   ├── qwen35.py       # bounded upstream runner, packed loading and GDN dispatch
│   └── linear.py       # packed NV adapter shared with DFlash
├── dflash/             # draft, target observation and multirow verification
└── kernels/            # packed GEMV/GEMM, embedding lookup and GDN kernels
```
