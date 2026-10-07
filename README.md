# infurnace

A small, single-user **local Qwen3 runner on tinygrad**. Load one GGUF, warm up
and chat—no server, batching or scheduler.

Tinygrad provides the transformer, tokenizer, KV cache and JIT. Infurnace adds
strict GGUF validation, a local chat CLI and custom NVIDIA quantized decode
kernels. The tested target is **Qwen3-4B Q4_K_M on an 8 GB RTX 5060**; other
models and GPUs are not guaranteed to work.

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

Model weights are **not shipped or downloaded automatically**. Place the tested
GGUF in the checkout's `models/` directory, or in user storage:

```sh
mkdir -p ~/.local/share/infurnace/models
cp /path/to/Qwen3-4B-Q4_K_M.gguf ~/.local/share/infurnace/models/
infurnace
```

Model lookup: `INFURNACE_MODEL` → user data directory → checkout `models/`.
User storage respects `XDG_DATA_HOME`. Override with
`infurnace --model /absolute/path/to/model.gguf`.

After pulling changes, rerun `pip install -e .`. If the tinygrad pin changes,
force reinstall that exact dependency: pip can retain an older same-version
wheel. Uninstall with `pip uninstall infurnace`; model files are retained.

## Run

```sh
infurnace                                  # select hardware, warm up, chat
infurnace --no-thinking                     # chat without thinking
infurnace --prompt "Hello" --max-tokens 64   # raw single-shot generation
infurnace --benchmark 32                    # cold/warm benchmark, 512-token prompt
infurnace --no-nv-prefill                   # generic prefill, keeping fast NV decode
infurnace --nv-attention                    # experimental parallel-softmax decode
infurnace --nv-fused-attention              # experimental fused GQA decode
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

## Current status

Local generation, chat, warmup and benchmarks work on the tested model/GPU.
The custom fast path keeps Q4_K/Q6_K weights packed and uses Q8-rounded
activations for single-token linear projections. Multi-token prefill automatically
uses packed Q4/Q6 FP16 tensor-core GEMM on eligible NV devices, with generic
fallback elsewhere: activations round to FP16, accumulation is FP32,
and no persistent dense weight copy is retained. Attention defaults to upstream;
`--nv-attention` opts into parallel masked softmax for supported single-token NV
decode. `--nv-fused-attention` instead enables shared-KV, split-context fused GQA
attention with FP32 online softmax and a merge launch. Both remain opt-in and
leave prefill unchanged. Norms, RoPE, cache allocation and sampling remain upstream.

With a 512-token prompt and 32 generated tokens, local warm tests measured
**~81 decode tok/s**. Experimental packed prefill reduced warm first-token latency
from **~16.66 s to ~0.98 s** on the tested GPU. These are separate local runs,
not full-model quality parity or a comparison against llama.cpp. See benchmark
notes for settings and limitations.

Acceleration remains **experimental**: isolated kernel/JIT tests and real-model
smoke tests pass, but full-model quality comparisons and long-session memory/
reliability testing remain open. Use `--no-nv-quant` for the generic baseline.
No HTTP API or tool calling yet.

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
