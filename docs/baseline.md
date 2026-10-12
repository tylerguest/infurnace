# Qwen3.5 measurement baseline

The current target is **Qwen3.5-4B Q4_K_M** with its optional Q4_K_M DFlash draft,
on an 8 GB RTX 5060. Model provenance is in [Qwen3.5 setup](qwen35.md).
The runtime revision is pinned in `pyproject.toml`.

Local measurements cited in the README put plain decode around 110 tok/s and
DFlash code-answer throughput around 1.4–2.2× that baseline. Acceptance and
performance depend strongly on prompt/output content; these are not a guaranteed
speedup, full-model quality comparison or long-session reliability result.
See [DFlash measurements](dflash.md) for settings and limits.

## Repeatable measurements

```sh
python benchmarks/decode.py --model qwen3.5:4b --tokens 32
python benchmarks/compare.py --model qwen3.5:4b --device NV \
  --tokens 32 --runs 1 --output /tmp/packed-vs-upstream.json
python benchmarks/compare.py --model qwen3.5:4b --device NV \
  --no-nv-quant --tokens 32 --runs 1 --output /tmp/generic-vs-upstream.json
```

Both comparison workers share the Qwen3.5 tokenizer but load their transformers
independently in sequential subprocesses. Neither uses DFlash. Ordinary NV
prefill is one-row upstream execution even when packed GEMM is available.

- EOS is ignored for timing; prompt plus requested outputs must fit context.
- First-token latency includes prefill/output work and, on cold runs, compilation.
- Warm decode excludes the first output token.
- Reset prevents prefix reuse between passes.
- Tinygrad counters estimate traffic/ops; they do not measure DRAM bandwidth.
- Current allocated memory and host RSS are not GPU process peak memory.
- GPU clocks, compiler caches and other workloads need separate control.
- Q8-rounded packed decoding and generic math require separate quality validation.

## Validation

```sh
.venv/bin/pytest -q
INFURNACE_TEST_NV=1 .venv/bin/pytest -q
INFURNACE_TEST_MODEL=/path/to/Qwen3.5-4B-Q4_K_M.gguf \
  INFURNACE_TEST_GENERATION=1 INFURNACE_GENERATION_DEVICE=NV \
  .venv/bin/pytest -q tests/core/test_runner.py::test_local_generation
```

CPU fixtures cover bounded generation, upstream output/RNG parity, tied/untied
embedding aliasing, cache reset/divergence, abandoned streams and DFlash behavior.
GPU tests are opt-in; a CPU-suite pass does not establish actual NV execution or
real-model parity. Long contexts and larger windows still need peak-memory and
quality measurements before default changes.
