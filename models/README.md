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

## First Target

The first development target is the official Qwen3-0.6B Q8_0 GGUF:

```text
https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf
```

Its identity is pinned in
[`qwen3-0.6b-q8_0.json`](qwen3-0.6b-q8_0.json):

```text
revision:  23749fefcc72300e3a2ad315e1317431b06b590a
size:      639446688 bytes
SHA-256:   9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031
license:   Apache-2.0
```

The manifest currently establishes artifact identity and acquisition. GGUF metadata,
tokenizer data, and the complete tensor inventory are recorded in
[`qwen3-0.6b-q8_0.inspection.json`](qwen3-0.6b-q8_0.inspection.json), generated only
after Phase 0C inspection validates them from this exact artifact through current
tinygrad.

Acquire the model into ignored local artifact storage with:

```sh
.venv/bin/python tools/fetch_model.py \
  --manifest models/qwen3-0.6b-q8_0.json \
  --output artifacts/models/Qwen3-0.6B-Q8_0.gguf
```

The tool streams into a temporary file, verifies the exact byte count and SHA-256,
and atomically publishes the destination. It reuses a matching destination and
refuses to overwrite a mismatched one.

Generate or verify the canonical inspection report on CPU with:

```sh
DEV=CPU .venv/bin/python tools/inspect_artifact.py \
  --manifest models/qwen3-0.6b-q8_0.json \
  --artifact artifacts/models/Qwen3-0.6B-Q8_0.gguf \
  --check models/qwen3-0.6b-q8_0.inspection.json
```

The report records GGUF version 3, 28 metadata entries, and all 310 tensor
descriptors. It records 197 Q8_0 tensors and 113 F32 tensors. Large tokenizer arrays
are represented by their element count and a canonical SHA-256 rather than copied
verbatim into the repository.

Expected Qwen3-0.6B values used to validate the GGUF metadata are:

```text
layers:                 28
hidden size:            1024
intermediate size:      3072
attention heads:        16
KV heads:               8
head dimension:         128
vocabulary size:        151936
RoPE theta:             1000000
RMS normalization eps:  1e-6
attention bias:         false
tied token embeddings:  true
source model dtype:     bfloat16
```

Phase 1A implements this as a frozen, tinygrad-independent `Qwen3Config` derived
from live GGUF metadata. The exact contract requires full 128-dimensional RoPE,
per-head Q/K RMS normalization, bias-free attention and MLP projections, a dense
SwiGLU MLP, and an output projection tied to `token_embd.weight`. It generates all
310 required tensor names, logical shapes, and storage dtypes: 113 normalization
tensors stored as F32 and 197 matrix or embedding tensors stored as Q8_0.

The committed inspection report is audit and test evidence, not a production
configuration input. Phase 1B loads the verified artifact through tinygrad and
compares its actual tensor mapping with the Phase 1A contract before constructing a
model. GGUF storage and tinygrad's logical float values are part of Phase 1A;
serving weight, accumulation, and KV-cache dtypes remain separate execution-policy
decisions.

The pinned GGUF metadata reports 40,960 maximum positions, resolving the artifact
contract even though the GGUF model card describes a 32,768 context length. The
server may expose a lower limit based on measured memory budgets for its configured
execution topology.

Qwen3-0.6B exercises RMSNorm, Q/K normalization, RoPE, grouped-query attention,
SwiGLU, and tied embeddings while remaining small enough for rapid iteration.

## KV Capacity

For 28 layers, 8 KV heads, head dimension 128, and a two-byte cache dtype, KV storage
costs:

```text
28 * 2 * 8 * 128 * 2 = 114,688 bytes per token
```

Approximate per-request capacity is therefore:

```text
4,096 tokens:   448 MiB
32,768 tokens:  3.5 GiB
40,960 tokens:  4.375 GiB
```

These values exclude model weights, TinyJit workspaces, input and output buffers,
sampling buffers, and backend reserve. The first server context and concurrency
limits must come from measured total memory, not only the model's advertised maximum.

## Integration Rule

Infurnace consumes GGUF through current tinygrad rather than defining another weight
format. The serving model uses tinygrad Tensor operations but owns an external-cache,
logits-returning contract instead of calling `Transformer.generate()`.

A larger dense checkpoint or another decoder architecture can be added only after
the first contract works end to end. Model differences stay in the model layer and
cannot introduce model-specific behavior into protocol handling, request state,
scheduling, or logical cache allocation.
