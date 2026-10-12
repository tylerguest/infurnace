# Architecture

Infurnace is a single-user local Qwen3 application **on tinygrad**. It validates
local GGUF input, adapts tinygrad's transformer/tokenizer, and provides a small
CLI. It is not a second transformer implementation or a serving framework.

## Ownership

| Module | Responsibility |
|---|---|
| `gguf.py` | Preflight validation, bounded DISK descriptors, device staging and lazy decoding |
| `model.py` | Qwen3 metadata/shape contract, tinygrad config/state loading, bounded generation |
| `tokenizer.py` | SimpleTokenizer adapter and bundled text-only Qwen3 chat rendering |
| `cli.py` | Single-shot/chat, UTF-8 streaming, interruption and baseline benchmark reporting |
| `kernels/quant_gemv.py` | NV packed projections; generic fallback and shared packed storage |
| `eagle/draft.py` | EAGLE checkpoint loading, observed draft state and proposals |
| `eagle/generate.py` | Verified generation, ownership, cancellation and fast-profile defaults |
| `eagle/reference.py` | Sequential verification reference |
| `eagle/verify.py` | NV multi-row target verification and reusable captures |
| `eagle/sampling.py` | Exact sampling expression and prefix RNG commitment |
| `eagle/io.py` | Owned host-mapped NV scratch; synchronized tiny uploads/readbacks |
| `kernels/parallel_argmax.py` | Two-pass first-index reduction over unchanged FP32 scores |

Tinygrad owns transformer blocks, Q/K normalization, RoPE, attention, FFN,
KV-cache writes, prefix reuse, chunked prefill, sampling and TinyJit. Infurnace
uses the pinned `Transformer` and its state loader rather than reproducing math.

## Loading

```text
GGUF → parse/validate metadata and descriptors
     → validate Qwen3 config and weight shapes
     → transfer packed bytes once to explicit CPU/NV device
     → lazy decoded views, FP16 casts (same cast for tied weights)
     → tinygrad TransformerConfig + Transformer + load_state_dict(realize=False)
```

No whole-model FP16 realization is requested. This does not promise that every
consumer fuses decoding: measure schedules and memory. DISK is storage only.
The custom loader remains while its stricter validation/storage control is
useful; replacing it with preflight checks plus upstream loading is an option.

## Generation and state

`Qwen3Model.generate()` validates inputs, copies the prompt list, bounds output
by token/context limits, and delegates to upstream generation. Upstream samples
and synchronizes each token before returning its Python integer. Temperature
zero follows upstream sampling semantics (temperature is clamped internally).

The device Context is active only while advancing upstream generation, not
while yielding to application code. Only one stream may be active per model;
callers must close abandoned generators. Reset invalidates upstream's cached
prefix, not the allocated KV tensors or captured programs. Prefix reuse is
upstream token-based behavior; changed chat formatting may reduce reuse.

The CLI checks EOS, closes streams and flushes incremental UTF-8 decoding at all
exit paths. Chat reserves response space by dropping oldest complete turns
with a notice; it rejects an individual oversized message without changing
history. Default chat generation runs until EOS or remaining context rather
than an artificial response cap. Explicit --max-tokens remains available. `--prompt` is raw text; `--chat` uses the
bundled Qwen3 template. Thinking is on by default, configurable with
`--no-thinking`. Tools and non-text content are unsupported.

## Baseline and optimization

The baseline disables custom quantized linears and uses generic tinygrad ops on
NV. Packed quantized acceleration (automatic on NV) installs linear adapters using explicit GGUF descriptors:
concrete single-token inputs use GEMV, multi-token inputs use the original
linear. The final output projection may use GEMV during prefill because it
projects only the last token. Packed typed storage is prepared once and shared
with the generic dequant expression. Activation Q8 changes numerical behavior;
this remains experimental pending full-model quality and longer-context evaluation.
The CLI now selects a usable GPU by preference (NV, AMD, Metal, CUDA, CL), then
CPU, respecting explicit --device/DEV overrides. Supported NV/CUDA-C devices
use the fast path automatically; --no-nv-quant restores generic operations.
This is not an exhaustive fastest-device benchmark or model-fit guarantee.

Benchmarks report cold/warm first-token and decode timings, actual counts, host
peak RSS and tinygrad current allocated memory. First-token time includes
prefill/output work and cold compilation; the derived prefill rate is not a
pure transformer-prefill measurement. GPU peak polling and comparisons with
upstream/llama.cpp are separate measurements, not implied by the CLI counters.

## EAGLE fast profile

Ordinary decode does not use EAGLE. `--eagle3 CHECKPOINT` creates an
`EagleGenerator(..., fast=True)` with one proposal by default. Its profile uses
fused drafting, mapped scratch I/O, parallel sampling, zero-copy features and
validated fused projections; `fast=False` retains the basic batched path.
The lower-level Python APIs retain their explicit/reference defaults.

Mapped scratch belongs to the draft/verifier capture owner, never a global
allocator patch. Uploads synchronize previous GPU readers before CPU writes;
readbacks use tinygrad's synchronized host access. Outputs stay alive through
capture reuse. Verification cache identity includes I/O and sampling modes.
Draft I/O is configured before its first observation; replacing storage beneath
existing captures is rejected. Calibration artifacts are not auto-loaded.

The package is organized by responsibility, not model/GPU name. EAGLE is a
decoding strategy; GPU kernels remain in `kernels/` and target-model handling
remains in `model.py`.

## Scope

Runtime dependency: pinned tinygrad. Development tests use Jinja as the actual
GGUF-template oracle and optionally llama.cpp for tokenizer parity. No server,
scheduler, batching or multi-GPU. Evaluate the upstream server before adding a
thin localhost adapter for Pi. See [roadmap](roadmap.md) for gates and status.
