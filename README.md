# infurnace

A small, single-user **local Qwen3 runner on tinygrad**. Load one GGUF, warm up
and chat, or serve a single-user OpenAI-compatible API—no batching or scheduler.

Tinygrad provides the transformer, tokenizer, KV cache and JIT. Infurnace adds
strict GGUF validation, a local chat CLI and custom NVIDIA quantized decode
kernels. The tested target is **Qwen3-4B Q4_K_M on an 8 GB RTX 5060**; other
models and GPUs are not guaranteed to work.

**Text-only Qwen3.5 support** uses tinygrad's existing Unsloth GGUF model
implementation with Infurnace's packed NV projections; see [setup](docs/qwen35.md).
[DFlash speculative decoding](docs/dflash.md) with fused multirow rounds speeds up
decoding only as far as the draft predicts the text: about 1.4× over ~88 tok/s plain
decoding on 512-token code answers, more on highly predictable code, and slower than
plain decoding on open-ended English (RTX 5060). A fuller benchmark is pending.

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
infurnace                                  # select hardware, warm up, chat
infurnace --no-thinking                     # chat without thinking
infurnace --prompt "Hello" --max-tokens 64   # raw single-shot generation
infurnace --benchmark 512,32 --warmup       # per-token timing, 512-token prompt
infurnace --benchmark 32 --benchmark-format json # cold/warm JSON report
infurnace --no-nv-prefill                   # generic prefill, keeping fast NV decode
infurnace --nv-attention                    # experimental parallel-softmax decode
infurnace --no-nv-fused-attention           # disable automatic fused GQA decode
infurnace --no-nv-prepared-weights          # disable automatic tiled/swizzled Q4/Q6 layouts
infurnace --no-nv-quant                     # generic tinygrad baseline
infurnace --device CPU                      # explicit device override
```

- **Hardware:** probes for a usable GPU, falling back to CPU. Supported
  NV/CUDA-C devices automatically use quantized decode and, where supported,
  packed tensor-core prefill. `--no-nv-prefill` disables only prefill acceleration;
  `--no-nv-quant` disables both. This is backend
  preference, not a fastest-device benchmark or a guarantee the model fits.
  `--device` or `DEV` overrides selection.
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
infurnace --device NV --eagle3 models/eagle3-qwen3-4b
# Diagnostic opt-out: retain EAGLE, disable the fast optimization profile.
infurnace --device NV --eagle3 models/eagle3-qwen3-4b --no-eagle-fast
```

Requires the matching checkpoint directory (`config.json`, `model.safetensors`),
NV packed linears and fused attention. The CLI never downloads or substitutes a
target model automatically. Only target-verified greedy tokens are streamed; EOS, response
limits and cancellation release both target and draft state. Kernel captures
are reused across requests, but speculative chat currently repeats prefill
rather than reusing conversation prefixes. First-request compilation/capture
can take a minute or more; ordinary warmup does not prime all EAGLE shapes.

`--eagle3` now selects **one draft token and the fast profile by default**:
fused draft observation/proposal, owned host-mapped token/decision/proposal
buffers, parallel exact-score sampling, zero-copy feature handoff, and looped
projections with exact fused reductions on validated shapes. These are integrated
implementations, not process-wide benchmark monkeypatches. `--eagle-draft-tokens`
overrides the proposal count; `--no-eagle-fast` disables the profile. Ordinary
generation without `--eagle3` is unchanged.

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
`--no-nv-fused-attention` disables it, while `--nv-attention` selects the parallel
masked-softmax alternative. `--no-nv-quant` disables automatic NV acceleration.
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
