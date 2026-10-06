# infurnace

Run LLMs on your GPU with tinygrad. No server, no batching, no scheduler —
load a GGUF, generate tokens, stream them out. A few files, one dependency.

## Install

Requires Python 3.11+, Git, and [pipx](https://pipx.pypa.io/stable/installation/).
On Ubuntu/Debian, install pipx with `sudo apt install pipx`.

From your clone:

```sh
pipx ensurepath
pipx install .
```

Open a new terminal after `ensurepath`. You can now run `infurnace` from any
folder. pipx creates an isolated environment; no shell alias or custom launcher
is needed. NVIDIA hardware is the tested GPU target; CPU works but is slower.

Model weights are **not shipped with the repo or downloaded automatically**.
Obtain a compatible dense Qwen3-4B Q4_K_M GGUF and place it here:

```sh
mkdir -p ~/.local/share/infurnace/models
cp /path/to/Qwen3-4B-Q4_K_M.gguf ~/.local/share/infurnace/models/
infurnace
```

The user data directory respects `XDG_DATA_HOME`. Override with
`infurnace --model /absolute/path/to/model.gguf`, or set `INFURNACE_MODEL` in
your shell profile. Existing editable checkouts also find their `models/` file
regardless of the working directory. Different GGUF architectures/formats are
not supported just because the filename matches.

Update after pulling changes: `pipx reinstall infurnace`. Uninstall with
`pipx uninstall infurnace` (model weights are retained).

### Development install

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/infurnace
```

If upgrading tinygrad's commit in an existing environment, force reinstall the
exact VCS dependency: pip may otherwise keep its same-version old wheel.

## Run

After installation and model setup:

```sh
infurnace                              # interactive chat
infurnace --no-thinking                 # chat without thinking
infurnace --prompt "Hello" --max-tokens 64
infurnace --benchmark 20
```

`--model` overrides the configured/default model path.
`infurnace` checks for a usable GPU, falling back to CPU. Supported NV devices
use quantized acceleration automatically; `--no-nv-quant` uses generic linears.
Interactive chat warms up prefill/decode before displaying the first prompt,
with a visible startup status. Use `--no-warmup` to skip it. Benchmark and
single-shot modes do not automatically warm up; benchmarks retain cold timings.
New prompt shapes or longer contexts can still incur additional work.
This hardware preference is not a benchmark across all installed devices.
`--device NV` / `--device CPU` override selection; `--max-context` defaults to
2048. `--prompt` is raw text; chat formats turns and supports `--no-thinking`.
Chat has no default response-token cap beyond available context; set
`--max-tokens` if wanted. When space gets tight, chat drops the oldest complete
turns with a notice to reserve room for the next answer. This enables long
sessions, not unlimited memory: earlier details are forgotten. Individual
answers can still reach the context limit, which is reported explicitly.
See [baseline measurements](docs/baseline.md) for current performance/limitations.

## Status

Phase 2: F32/Q4_K/Q6_K GGUF loading works on CPU with lazy dequantization
and explicit packed-byte staging.
Local generation, chat, and benchmark modes wrap tinygrad's Transformer.
The bundled Qwen3-4B runs on the RTX 5060 using generic tinygrad operations.
Supported NV devices automatically enable the experimental decode fast path.
Short local runs show ~93–94 decode tok/s vs ~10.75 generic; first-token latency
is similar and loading is slower. It adds Q8 activation rounding and remains
experimental pending broader quality/context tests; disable with `--no-nv-quant`. See the [roadmap](docs/roadmap.md)
for ordered tasks and verification gates.

## Layout

```
infurnace/
├── gguf.py            # GGUF parser + quantized dequant
├── tokenizer.py       # tinygrad tokenizer adapter + Qwen3 chat rendering
├── model.py           # validated tinygrad Transformer adapter + generation limits
├── kernels/
│   └── quant_gemv.py  # NV Q4_K/Q6_K decode kernel
└── cli.py             # the CLI above

docs/
├── architecture.md
├── roadmap.md
└── baseline.md
```