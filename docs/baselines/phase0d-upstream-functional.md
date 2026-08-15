# Phase 0D Upstream Functional Baseline

This baseline proves that the pinned Qwen3-0.6B checkpoint runs through current
tinygrad on the NV backend. It uses tinygrad's model-owned KV cache and sampling
interface; it is not the Infurnace model-runner contract.

## Environment

```text
device:       NVIDIA GeForce RTX 5060
driver:       610.43.02
backend:      DEV=NV
JIT:          enabled
context:      1024 tokens
chunk size:   32 tokens
weight path:  lazy Q8_0 expressions with logical FP16 model weights
```

The run started with approximately 6.4 GiB of device memory free and no competing
model process.

## Model Smoke Test

```sh
DEV=NV .venv/bin/python ../tinygrad/extra/benchmark_llm.py \
  --model artifacts/models/Qwen3-0.6B-Q8_0.gguf \
  --max-context 1024 \
  --prompt-tokens 16 \
  --decode-tokens 4 \
  --chunk-size 32
```

Observed output:

```text
load 1.633s
warm 20.029s
prefill 151.513 tok/s
decode 46.855 tok/s output [657, 198, 9, 0, 0]
```

These timings establish functional execution only. Phase 0E defines controlled
performance and memory measurements.

## Deterministic Runner

```sh
DEV=NV JIT=1 .venv/bin/python tools/run_upstream_model.py \
  --manifest models/qwen3-0.6b-q8_0.json \
  --artifact artifacts/models/Qwen3-0.6B-Q8_0.gguf \
  --max-context 1024 \
  --fixed-output-tokens 4 \
  --text-output-tokens 16
```

The fixed token prompt generated:

```text
[657, 198, 9, 0]
```

Two clean-process runs produced identical fixed-token and text results. Repeating
the same prompt on one upstream `Transformer` instance did not provide fresh,
independent conversation semantics because the instance retains model-owned prefix
and KV state. Infurnace therefore validates this baseline across clean processes and
does not adopt the stateful interface as its serving model contract.

The bounded text prompt decoded successfully as:

```text
<think>
</think>

السلام عليكم ورحمة الله وبركاته
```

This is a functional tokenizer and decoding smoke test, not a language-quality
assertion or the final Infurnace chat-template contract.
