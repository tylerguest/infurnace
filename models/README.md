# Models

Model weights and GGUF files are not committed to this repository. A checkpoint is
supported only after its manifest records:

- source repository, exact model revision, file URL, byte size, and SHA-256;
- license and any redistribution constraints;
- GGUF version, architecture, quantization, and typed metadata inventory;
- tokenizer vocabulary and merge hashes, special token IDs, chat template, and EOS
  behavior;
- every required tensor name, shape, GGUF dtype, logical dtype, and layout;
- model, accumulation, and KV-cache dtypes used by Infurnace;
- lazy quantized or realized weight policy and measured peak loading memory;
- model context limit and the lower server limit, if any.

Checkpoint identity is exact even though Infurnace follows current tinygrad rather
than pinning tinygrad itself.

## Target

The serving target is the official Qwen3-4B Q4_K_M GGUF:

```text
https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/bc640142c66e1fdd12af0bd68f40445458f3869b/Qwen3-4B-Q4_K_M.gguf
```

Its identity is pinned in
[`qwen3-4b-q4_k_m.json`](qwen3-4b-q4_k_m.json):

```text
revision:  bc640142c66e1fdd12af0bd68f40445458f3869b
size:      2497280256 bytes
SHA-256:   7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5
license:   Apache-2.0
```

The manifest establishes artifact identity and acquisition. GGUF metadata, tokenizer
data, and the complete tensor inventory are recorded in
[`qwen3-4b-q4_k_m.inspection.json`](qwen3-4b-q4_k_m.inspection.json), generated only
after Phase 0C inspection validates them from this exact artifact through current
tinygrad.

Acquire the model into ignored local artifact storage with:

```sh
.venv/bin/python tools/fetch_model.py \
  --manifest models/qwen3-4b-q4_k_m.json \
  --output artifacts/models/Qwen3-4B-Q4_K_M.gguf
```

The tool streams into a temporary file, verifies the exact byte count and SHA-256,
and atomically publishes the destination. It reuses a matching destination and
refuses to overwrite a mismatched one.

Generate or verify the canonical inspection report on CPU with:

```sh
DEV=CPU .venv/bin/python tools/inspect_artifact.py \
  --manifest models/qwen3-4b-q4_k_m.json \
  --artifact artifacts/models/Qwen3-4B-Q4_K_M.gguf \
  --check models/qwen3-4b-q4_k_m.inspection.json
```

The report records all 398 tensor descriptors: 145 F32 normalization tensors,
216 Q4_K matrix tensors, and 37 Q6_K tensors (token embedding, value projections,
and feed-forward-down projections, with the value/FFN-down layers split by the
file's imatrix between Q4_K and Q6_K). Large tokenizer arrays are represented by
their element count and a canonical SHA-256 rather than copied verbatim into the
repository.

Expected Qwen3-4B values used to validate the GGUF metadata are:

```text
layers:                 36
hidden size:            2560
intermediate size:      9728
attention heads:        32
KV heads:               8
head dimension:         128
vocabulary size:        151936
RoPE theta:             1000000
RMS normalization eps:  1e-6
attention bias:         false
tied token embeddings:  true
GGUF file type:         15 (Q4_K_M)
```

Phase 1A implements this as a frozen, tinygrad-independent `Qwen3Config` derived
from live GGUF metadata. The exact contract requires full 128-dimensional RoPE,
per-head Q/K RMS normalization, bias-free attention and MLP projections, a dense
SwiGlu MLP, and an output projection tied to `token_embd.weight`. It generates all
398 required tensor names, logical shapes, and storage dtypes.

The committed inspection report is audit and test evidence, not a production
configuration input. Phase 1B loads the verified artifact through tinygrad and
compares its actual tensor mapping with the Phase 1A contract before constructing a
model. GGUF storage and tinygrad's logical float values are part of Phase 1A;
serving weight, accumulation, and KV-cache dtypes remain separate execution-policy
decisions.

The Phase 1B loader copies the artifact into a private temporary snapshot while
verifying its identity, then keeps that snapshot open through tinygrad parsing. This
prevents pathname replacement or in-place source mutation from changing the parsed
contents. The loader accepts only the pinned identity, whose hash connects those
contents to Phase 0's unique tensor descriptors and storage dtypes. It additionally
requires the exact 398 live tensor names, logical shapes, and tinygrad dtypes before
Infurnace's policy cast, then adds `output.weight` as the same object as
`token_embd.weight`.

Weights are cast to FP16 lazily: the Q4_K/Q6_K dequantization and cast expressions
are fused into execution kernels, so the resident artifact stays at its packed size.
Realized FP16 is not a feasible resident policy for a 4B checkpoint on the 8 GiB
RTX 5060 (FP16 weights alone are ~8 GB), so lazy quantized execution is the serving
policy.

The pinned GGUF metadata reports 40,960 maximum positions, resolving the artifact
contract even though the Qwen3-4B model card describes a 32,768 native context
length. The server may expose a lower limit based on measured memory budgets for
its configured execution topology.

Qwen3-4B exercises RMSNorm, Q/K normalization, RoPE, grouped-query attention,
SwiGLU, and tied embeddings while fitting the 8 GiB device at Q4_K_M.

## KV Capacity

For 36 layers, 8 KV heads, head dimension 128, and a two-byte cache dtype, KV
storage costs:

```text
36 * 2 * 8 * 128 * 2 = 294,912 bytes per token
```

Approximate per-request capacity is therefore:

```text
4,096 tokens:   1.125 GiB
16,384 tokens:  4.5 GiB
32,768 tokens:  9.0 GiB
```

A full 32,768-token cache exceeds the device alone; the 8 GiB card cannot hold the
model, a large contiguous KV pool, and workspaces at the same time. This is why
paged KV (Phase 5) with a bounded total token budget, not the model's advertised
maximum, governs the served context and concurrency limits.

## Integration Rule

Infurnace consumes GGUF through current tinygrad rather than defining another weight
format. The serving model uses tinygrad Tensor operations but owns an external-cache,
logits-returning contract instead of calling `Transformer.generate()`.

A larger dense checkpoint or another decoder architecture can be added only after
the first contract works end to end. Model differences stay in the model layer and
cannot introduce model-specific behavior into protocol handling, request state,
scheduling, or logical cache allocation.