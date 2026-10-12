# Initial local baseline

This is a short smoke baseline, **not a speedup claim** or a comprehensive
benchmark. Runtime pin: `ce9c84605e94de9667fb578c1ff3fa8678725a87`, Python 3.14,
RTX 5060 (8151 MiB), NV/CUDARenderer. Model:
`models/Qwen3-4B-Q4_K_M.gguf` (2,497,280,256 bytes). Generic tinygrad linears;
Infurnace's custom NV GEMV is not enabled.

## Verified behavior

- Raw `Hello` produced ` which is` with two generated tokens.
- Chat with `--no-thinking`: "Reply with just the word hello." produced `hello`
  and stopped at EOS.
- Tiny CPU fixtures match direct upstream generation and cover tied storage,
  chunked prefill, prefix reset/divergence, abandoned streams and context limits.

## Measurements

Command (from project root):

```sh
DEV=NV .venv/bin/infurnace --device NV --no-nv-quant --max-context 2048 --benchmark 4
```

Prompt: raw `Hello, explain what a GPU does.` (8 IDs); temperature 0,
upstream chunk size 32, 4 generated tokens. EOS ignored for timing. The runner
resets prefix reuse per pass. Cold pass plus one additional pass warm both JIT
paths, then three measured warm passes.

| Metric | Result |
|---|---:|
| Load + tokenizer construction | 8.52 s |
| Cold first token (includes compilation) | 16.94 s |
| Warm first token, three runs | 0.368 / 0.371 / 0.371 s |
| Warm decode (only 3 decode steps/run) | 10.63 / 10.64 / 10.75 tok/s |
| Host peak RSS (whole process) | 2,809,864,192 bytes |
| Tinygrad current allocation at end | 3,682,310,080 bytes |

External nvidia-smi polling at ~250 ms observed **4980 MiB total GPU memory**,
with ~1481 MiB baseline including desktop/other processes. This is sampled
whole-device usage, not an exact process peak; brief spikes can be missed.
The CLI itself does not measure GPU peak.

Derived "prefill tokens/sec" divides prompt tokens by first-token latency;
it includes output projection/sampling and is **not pure prefill throughput**.
Cold results depend on compiler caches. Four-token runs are too short for
strong conclusions; collect longer prompts/generations and repeated processes.

## Direct upstream comparison

```sh
DEV=NV .venv/bin/python benchmarks/compare.py --device NV --tokens 4
```

A separate paired run on the **same loaded upstream Transformer** gave:

| Metric | Infurnace wrapper | Direct upstream generation |
|---|---|---|
| Warm first token | 0.3668–0.3672 s | 0.3673–0.3689 s |
| Decode | 10.76–10.82 tok/s | 10.78–10.81 tok/s |

No meaningful difference: both execute the same engine. This only checks wrapper
overhead, not independent upstream loading, memory savings, or llama.cpp parity.
A full upstream CLI/loader and llama.cpp comparison remains future work.

## Experimental NV decode integration

Enable explicitly with `--device NV --nv-quant`. Generic multi-token prefill
remains; single-token linears use Q4_K/Q6_K GEMV with Q8-rounded activations.
The final output projection also uses GEMV when its input is one token (including
prefill's last-token logits). 253 linears are eligible in the bundled model.
Resident descriptors identify types/shapes, and generic/fast paths share packed
storage. No graph-size guessing or second persistent Q4 weight copy.

Separate 20-token runs at context 2048, same 8-token prompt/settings:

| Metric | Generic | NV quantized |
|---|---:|---:|
| Load + tokenizer | 8.80 s | 14.50 s |
| Warm first token (3 runs) | 0.368 s | 0.364 s |
| Warm decode (3 runs) | 10.74–10.76 tok/s | 92.92–94.20 tok/s |
| Current tinygrad allocation | 3,682,310,080 bytes | 3,684,847,096 bytes |

This is roughly **8.7x warm decode throughput** in this short local comparison,
not an overall speedup or a comparison against llama.cpp. Cold compilation is
still costly and loading is slower. Larger contexts/prompts and repeated cold
processes remain to be measured. Generic and quantized activation math differ;
full-model logit/output-quality evaluation is pending. The CLI now enables it
by default on supported NV devices; use --no-nv-quant for the generic baseline.
The no-thinking hello chat smoke passes on both paths. Isolated GPU tests cover
Q8-reference numerical parity, JIT replay and generic multi-token fallback.

```sh
infurnace --device NV --no-thinking
infurnace --device NV --benchmark 20
```

## Repeatable verification

```sh
DEV=CPU .venv/bin/python -m pytest -q
DEV=NV INFURNACE_TEST_NV=1 .venv/bin/python -m pytest -q tests/test_tinygrad_compat.py
INFURNACE_TEST_MODEL=models/Qwen3-4B-Q4_K_M.gguf INFURNACE_TEST_GENERATION=1 \
  INFURNACE_GENERATION_DEVICE=NV .venv/bin/python -m pytest -q tests/test_runner.py::test_local_generation
```

Do not silently truncate history or increase context to the model's full 40960
without checking memory. At the CLI default 2048, the FP16 KV estimate is 288 MiB.
