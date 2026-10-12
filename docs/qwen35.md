# Qwen3.5-4B target

Infurnace uses **tinygrad's existing Qwen3.5 model implementation** through
`tinygrad.llm.model.Transformer.from_gguf()`. The adapter in
`infurnace/models/qwen35.py` provides bounded generation, packed NV loading and
supported one-row GDN dispatch. There is no alternate model-family path.

## Selected files

Target: [Unsloth Qwen3.5-4B Q4_K_M](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF).

- Revision: `e87f176479d0855a907a41277aca2f8ee7a09523`
- File: `Qwen3.5-4B-Q4_K_M.gguf`, 2,740,937,888 bytes
- Verified SHA-256: `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`

Draft: [Anbeeld Qwen3.5-4B DFlash](https://huggingface.co/Anbeeld/Qwen3.5-4B-DFlash-GGUF),
`qwen35-4b-dflash-Q4_K_M.gguf`; provenance and design in [DFlash](dflash.md).

```sh
pip install -e '.[qwen35]'
infurnace -m qwen3.5:4b             # target/draft download on first use
infurnace --no-dflash --no-thinking
infurnace -m /path/to/Qwen3.5-4B-Q4_K_M.gguf \
  --no-dflash --no-warmup --prompt 'The capital of France is' --max-tokens 16
```

The named preset resolves pinned download URLs. Local files are accepted only
for dense Qwen3.5 architecture; other sizes/quantizations are not validated.

## Responsibility boundary

Tinygrad owns hybrid layer configuration, ordinary transformer blocks, cache
allocation/reset/prefix reuse, RoPE, sampling and JIT. Infurnace's NV fast path
keeps eligible Q4_K/Q5_K/Q6_K projection weights packed. The shared adapter is
`models/linear.py::_NVLinear`: Q8-activation GEMV for one row and eligible FP16
tensor-core GEMM for multiple rows. Supported one-row GDN work has a fused NV
kernel; full attention remains upstream.

Tied output/embedding weights share packed storage. Untied output heads do not
replace embeddings. Multirow embedding lookup uses packed gather for tied
weights, or a host FP16 copy otherwise. A source/model/device-keyed parameter
cache stores the remaining realized parameters.

Ordinary NV generation currently forces one-row prefill upstream. DFlash uses
fixed-width multirow prefill and verification, committing only accepted recurrent
state. Sequential verification remains the one-row reference; NV defaults to
parallel verification with adaptive plain-step fallback.

The tokenizer is Qwen3.5-specific: GGUF vocabulary/merges, NFC normalization,
Qwen3.5 pretokenization, byte streaming and text-only chat. `tokenizers` is a
text codec dependency, not a model runtime. HTTP serving uses the embedded GGUF
Jinja template and tinygrad tool parser; see [server setup](serve.md).

## Validation boundaries

CPU fixtures compare bounded generation to direct upstream output/RNG behavior,
check tied/untied embeddings, cache reset/divergence, abandonment, context limits,
and DFlash state/verification behavior. NV kernel and real-model tests are opt-in.
A CPU pass does not establish real-model GPU quality or reliability.

Earlier DFlash audit results are retained as explicitly historical
[experiment reports](../benchmarks/experiments/dflash/README.md). Current
measurement commands and limits are in [benchmarks](../benchmarks/README.md).
Full-model quality, tool-use reliability, long-session behavior and peak-memory
measurements remain open gates.
