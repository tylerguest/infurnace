# Qwen3.5 benchmarks

Run from the checkout after `pip install -e .`. Stop other model processes first.
The only supported target is Qwen3.5-4B; named targets download on first use.

## Plain decoding

```sh
python benchmarks/decode.py --model qwen3.5:4b --tokens 32
infurnace --benchmark 512,32
```

`decode.py` reports cold/warm first-token latency, decode throughput, host peak RSS
and current tinygrad allocation as JSON. Each pass resets prefix reuse. Synthetic
prompts are exact-length repeated prose (default 512 tokens); `--prompt-file PATH`
uses untruncated raw UTF-8 text. The CLI per-step benchmark instead uses a numbers
prompt and prints tinygrad traffic estimates.

Plain decoding reads prompts 16 rows at a time on NVIDIA GPUs
(`INFURNACE_MULTIROW_PREFILL=0` restores one row at a time).

## Independent upstream comparison

```sh
python benchmarks/compare.py --model qwen3.5:4b --device NV \
  --context 2048 --tokens 32 --runs 1 --output /tmp/infurnace-vs-tinygrad.json
python benchmarks/compare.py --model qwen3.5:4b --device NV \
  --no-nv-quant --tokens 32 --runs 1 --output /tmp/generic-vs-tinygrad.json
```

Workers run sequentially in separate subprocesses and load transformers
independently. They share the Qwen3.5 codec because the upstream SimpleTokenizer
encoder does not implement this model's pretokenization. Prompt IDs and context
capacity must match before a report is accepted. Neither worker uses DFlash.

Infurnace automatically selects packed projections on NVIDIA GPUs (`NV` or `CUDA`).
`--no-nv-quant` forces generic execution; `--no-nv-prefill` disables packed
multirow GEMM while retaining packed decode. On CPU:

```sh
python benchmarks/compare.py --device CPU --no-nv-quant --tokens 4 --runs 1
```

`--prompt-tokens N` controls synthetic active length; `--prompt-file PATH` uses
raw text without chat formatting. Prompt plus requested output must fit context.
The defaults are 512 prompt tokens, 256 outputs and three measured warm passes.
Use shorter outputs and `--runs 1` for a quicker check.

Measurement uses tinygrad CLI-style `GlobalCounters.reset()` and `Timing` around
each synchronized `next(generate)`. EOS is ignored. Warm decode excludes the
first token; aggregate rate is token count divided by total measured time.
JSON includes per-step tokens/timing, estimated bytes/ops/kernel counts, model
SHA-256 and runtime provenance. This is not an HTTP benchmark or quality test.

`wrapper.py` compares the bounded wrapper to direct generation on the **same**
loaded generic transformer, measuring lifecycle overhead rather than independent
loaders or different inference engines:

```sh
python benchmarks/wrapper.py --model qwen3.5:4b --device NV --tokens 20
```

## DFlash protocol

```sh
python benchmarks/dflash.py \
  models/qwen35-validation/Qwen3.5-4B-Q4_K_M.gguf \
  models/dflash-qwen35/qwen35-4b-dflash-Q4_K_M.gguf \
  --device NV --context 512 --output 64 --block-size 16 --verifier parallel \
  --json /tmp/dflash.json
```

Baseline/DFlash/baseline requests are synchronized and bracketed, with running,
aborted or complete JSON reports. The protocol checks exact tokens and RNG
advancement; a failure aborts rather than publishing a speedup. Parallel greedy
verification does not consume RNG per verified row, so this strict protocol can
reject an otherwise token-equivalent parallel run on RNG parity. Sequential
verification is the reference for upstream per-row RNG behavior.

Performance depends on draft acceptance and workload. See [DFlash design and
measurements](../docs/dflash.md) and the explicitly historical
[experiment reports](experiments/dflash/README.md).

## Output quality

```sh
python benchmarks/quality.py --llama-perplexity /path/to/llama.cpp/build/bin/llama-perplexity
```

Perplexity of the one-row, 16-row and DFlash verifier paths with llama.cpp's protocol, and llama.cpp's own
number on the same GGUF and tokens. `--text quality/multilingual.txt` scores the multilingual text, `--ctx`
sets the chunk length. Reference values: `quality/reference.json`; the opt-in
`tests/core/test_nv_quality.py` checks all three paths on both texts stay within 1% of llama.cpp.

## Interpretation limits

- TTFT includes prefill and first-token work; cold runs also include JIT/compilation.
- Compiler caches, GPU clocks and other workloads are not automatically controlled.
- Counters estimate traffic, not measured DRAM bandwidth; custom estimates may be incomplete.
- Current allocation and host RSS are not GPU process peak memory.
- Packed Q8/FP16 arithmetic differs from generic math; throughput does not establish quality parity.
- Allocated capacity and active cache length are separate variables.

GPU tests require `INFURNACE_TEST_NV=1`. Real-model tests need model paths and
flags; see [measuring and validating](../docs/roadmap.md#measuring-and-validating).
