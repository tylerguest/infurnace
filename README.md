# infurnace

A small, single-user **local LLM runner on tinygrad**. Load one GGUF, warm up
and chat, or serve a single-user OpenAI-compatible API—no batching or scheduler.

Tinygrad provides the transformer, tokenizer, KV cache and JIT. Infurnace adds
strict GGUF validation, a local chat CLI and custom NVIDIA kernels. The tested setup
is **Qwen3.5-4B Q4_K_M with its DFlash draft on an 8 GB RTX 5060** (Qwen3-4B also
works); other models and GPUs are not guaranteed to work.

Plain decoding runs at ~110 tok/s. [DFlash speculative decoding](docs/dflash.md)
speeds that up as far as the draft predicts the text: 1.4–2.2× on code answers,
roughly break-even on open-ended English (it falls back to plain decoding when
drafting loses). Qwen3.5 setup: [docs/qwen35.md](docs/qwen35.md).

## Install

Requires Python 3.11+ and Git. From your clone:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then run `infurnace` from any directory while the environment is active.
For development/tests, install with `pip install -e '.[dev]'` instead.
The runtime dependency is a pinned tinygrad commit.

Choose a model with `--model` / `-m`: a name from the list below, or a path to a local GGUF file.
A named model is downloaded on first use into `~/.local/share/infurnace/models/`
(`XDG_DATA_HOME` respected) and reused after that, like tinygrad's `--model`.

| Name | Files |
|---|---|
| `qwen3.5:4b` (default) | [Unsloth Qwen3.5-4B Q4_K_M](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF), 2.7 GB, plus its [DFlash draft](https://huggingface.co/Anbeeld/Qwen3.5-4B-DFlash-GGUF), 0.4 GB (downloaded only when DFlash runs, i.e. on NV) |
| `qwen3:4b` | [Qwen3-4B Q4_K_M](https://huggingface.co/Qwen/Qwen3-4B-GGUF), 2.5 GB |

```sh
pip install -e '.[qwen35]'   # Qwen3.5 tokenizer
infurnace                    # downloads qwen3.5:4b on first run, then chats
infurnace -m qwen3:4b
infurnace -m /path/to/model.gguf
```

`--no-dflash` decodes without the draft. A local Qwen3.5 file uses the Qwen3.5-4B draft only if
it is already installed; `--dflash PATH` or `INFURNACE_DFLASH_DRAFT` names one explicitly.

After pulling changes, rerun `pip install -e .`. If the tinygrad pin changes,
force reinstall that exact dependency: pip can retain an older same-version
wheel. Uninstall with `pip uninstall infurnace`; model files are retained.

## Run

```sh
infurnace                                   # warm up and chat (DFlash on for qwen3.5:4b on NV)
infurnace --no-thinking                     # chat without thinking
infurnace --prompt "Hello" --max-tokens 64  # one raw prompt, no chat template
infurnace --serve                           # OpenAI-compatible server on 127.0.0.1:8000
infurnace --no-dflash                       # plain decoding
infurnace --benchmark 512,32                # per-step timing: 512-token prompt, 32 outputs
infurnace --device CPU                      # explicit device override
```

For a cold/warm JSON report: `.venv/bin/python benchmarks/decode.py --tokens 32`.

`infurnace --help` lists every option.

- **Hardware:** probes for a usable GPU, falling back to CPU. NV/CUDA-C devices
  automatically use packed quantized kernels and, where supported, tensor-core
  prefill. `--device` or `DEV` overrides selection. (Benchmark scripts can force
  individual accelerations through `infurnace.loader.load_model`.)
- **Startup:** interactive chat warms up before the first prompt and shows
  progress. `--no-warmup` skips it. Single-shot generation skips automatic
  warmup; benchmarks measure cold and warm passes separately.
- **Thinking:** enabled by default in chat. `--no-thinking` disables it.
  `--prompt` uses raw text, not chat formatting.
- **Context:** defaults to 2048 tokens; change with `--max-context`. Larger
  windows use more memory. Chat drops oldest complete turns with a visible
  notice when necessary—continued sessions, not unlimited memory.
- **Responses:** chat has no default token cap beyond remaining context.
  `--max-tokens` sets a cap including thinking; raw single-shot defaults to 64.
  Token/context stops are reported. Oversized individual messages are rejected.

## Current status

Local generation, chat, warmup and benchmarks work on the tested model/GPU.
The custom fast path keeps Q4_K/Q6_K weights packed and uses Q8-rounded
activations for single-token linear projections. Multi-token prefill automatically
uses packed Q4/Q6 FP16 tensor-core GEMM on eligible NV devices, with generic
fallback elsewhere: activations round to FP16, accumulation is FP32,
and no persistent dense weight copy is retained. Eligible NV runs automatically
use shared-KV, split-context fused GQA decode attention with FP32 online softmax
and a merge launch; unsupported shapes retain upstream attention.
Prefill is unchanged. Norms, RoPE, cache allocation and sampling remain upstream.

With a 512-token prompt and 32 generated tokens, local warm tests measured
**~81 decode tok/s**. Experimental packed prefill reduced warm first-token latency
from **~16.66 s to ~0.98 s** on the tested GPU. These are separate local runs,
not full-model quality parity or a comparison against llama.cpp. See benchmark
notes for settings and limitations.

Acceleration remains **experimental**: isolated kernel/JIT tests and real-model
smoke tests pass, but full-model quality comparisons and long-session memory/
reliability testing remain open. Use `--no-nv-quant` for the generic baseline.
HTTP serving and template-based tool calling reuse tinygrad's server; real-model tool-use reliability remains unvalidated. See [Pi setup](docs/serve.md).

Next: full-model quality checks, prefill tuning and long-context attention—not a
second inference engine.

- [Roadmap](docs/roadmap.md): priorities and validation gates
- [Architecture](docs/architecture.md): ownership and execution paths
- [Baseline](docs/baseline.md): measurements, settings and limitations
- [Benchmarks](benchmarks/README.md): independent comparison with tinygrad;
  defaults to 512 prompt tokens, configurable with length or a raw prompt file

```sh
python benchmarks/compare.py --model models/Qwen3-4B-Q4_K_M.gguf \
  --tokens 256 --output /tmp/infurnace-vs-tinygrad.json
```

## Layout

```text
infurnace/
├── cli.py             # local chat, streaming and benchmarks
├── device.py          # backend probing and NV eligibility
├── gguf.py            # validated packed storage and lazy decoding
├── model.py           # tinygrad Transformer adapter and NV linear dispatch
├── tokenizer.py       # tokenizer adapter and Qwen3 chat formatting
└── kernels/
    ├── quant_gemv.py  # Q8 activation packing and NV Q4_K/Q6_K GEMV
    ├── quant_gemm.py  # packed FP16 tensor-core prefill
    └── attention.py   # opt-in parallel-softmax and fused GQA NV decode
```
