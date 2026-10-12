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

Model weights are **not shipped or downloaded automatically**. Place the GGUFs in
the checkout's `models/` directory, or in user storage:

```sh
pip install -e '.[qwen35]'   # Qwen3.5 tokenizer
mkdir -p ~/.local/share/infurnace/models
cp /path/to/Qwen3.5-4B-Q4_K_M.gguf /path/to/qwen35-4b-dflash-Q4_K_M.gguf ~/.local/share/infurnace/models/
infurnace
```

Bare `infurnace` chats with Qwen3.5-4B and, on NV, its [DFlash](docs/dflash.md) draft;
`--no-dflash` decodes without it. Without a Qwen3.5 GGUF it falls back to Qwen3-4B
(`Qwen3-4B-Q4_K_M.gguf`).

Model lookup: `INFURNACE_MODEL` → user data directory → checkout `models/`; the draft
is found the same way (`INFURNACE_DFLASH_DRAFT`). User storage respects
`XDG_DATA_HOME`. Override with `infurnace --model /absolute/path/to/model.gguf`.

After pulling changes, rerun `pip install -e .`. If the tinygrad pin changes,
force reinstall that exact dependency: pip can retain an older same-version
wheel. Uninstall with `pip uninstall infurnace`; model files are retained.

## Run

```sh
infurnace                                   # warm up and chat (DFlash on when its draft is installed)
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

## Experimental EAGLE-3

The local AngelSlim Qwen3-4B EAGLE-3 checkpoint can now be used in chat or raw
single-shot generation:

```sh
infurnace --model models/Qwen3-4B-Q4_K_M.gguf --eagle3 models/eagle3-qwen3-4b
```

Requires the matching checkpoint directory (`config.json`, `model.safetensors`),
NV packed linears and fused attention. The CLI never downloads or substitutes a
target model automatically. Only target-verified greedy tokens are streamed; EOS, response
limits and cancellation release both target and draft state. Kernel captures
are reused across requests, but speculative chat currently repeats prefill
rather than reusing conversation prefixes. First-request compilation/capture
can take a minute or more; ordinary warmup does not prime all EAGLE shapes.

`--eagle3` drafts **one token per round with the fast profile**: fused draft
observation/proposal, owned host-mapped token/decision/proposal buffers, parallel
exact-score sampling, zero-copy feature handoff, and looped projections with exact
fused reductions on validated shapes. Ordinary generation without `--eagle3` is unchanged.

**This remains experimental, not a guaranteed speedup or reliability fix.**
The calibrated-head experimental combination reached about 132 tok/s versus
107 tok/s baseline on one 256-output request. That head is **not loaded
automatically**: CLI inference uses the explicitly supplied checkpoint unchanged.
NV timeouts remain possible, including during model loading. The runtime pin
includes the upstream submission-signal reinitialization fix, but does not fix
all observed failures. Benchmark your workload before relying on a speedup. See [implementation notes](docs/eagle3-plan.md)
and [the EAGLE benchmark](benchmarks/README.md#experimental-eagle-3-comparison).

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
