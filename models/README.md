# Models

Model weights and GGUF files are not committed to this repository. A checkpoint is
supported only after its manifest records:

- source repository, exact model revision, file URL, byte size, and SHA-256;
- license and any redistribution constraints;
- GGUF version, architecture, quantization, and complete metadata dump;
- tokenizer vocabulary, special token IDs, chat template, and EOS behavior;
- every required tensor name, shape, GGUF dtype, logical dtype, and layout;
- model, accumulation, and KV-cache dtypes used by Infurnace;
- lazy quantized or realized weight policy and measured peak loading memory;
- model context limit and the lower server limit, if any.

Checkpoint identity is exact even though Infurnace follows current tinygrad rather
than pinning tinygrad itself.

## First Target

The first development target is the official Qwen3-0.6B Q8_0 GGUF:

```text
https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf
```

The artifact revision and SHA-256 must be added to its manifest before integration.
The upstream model is Apache-2.0 licensed.

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

The upstream Transformers configuration reports 40,960 maximum positions, while
the GGUF model card describes a 32,768 context length. Infurnace does not silently
choose between them. The pinned GGUF metadata, numerical tests, and measured memory
budget determine the model contract and the server may expose a lower limit.

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
