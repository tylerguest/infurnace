# Tools

Development-only utilities belong here, including reference fixture generation,
intermediate-output comparison, compiled-function inspection, and benchmark result
processing. These tools are not part of the serving runtime.

Infurnace consumes GGUF directly through tinygrad, so the tools directory does not
define a separate production weight format.

## Checkpoint Acquisition

Acquire the pinned development checkpoint with:

```sh
.venv/bin/python tools/fetch_model.py \
  --manifest models/qwen3-0.6b-q8_0.json \
  --output artifacts/models/Qwen3-0.6B-Q8_0.gguf
```

Both paths are explicit so invoking the tool cannot silently choose a location for a
large download. The destination is published only after its size and SHA-256 match
the manifest. An existing matching artifact is reused without a network request; an
existing mismatch is left untouched and reported as an error. Model artifacts under
`artifacts/` and all `*.gguf` files are ignored by Git.

## GGUF Inspection

Verify the committed inspection report with current tinygrad on CPU:

```sh
DEV=CPU .venv/bin/python tools/inspect_artifact.py \
  --manifest models/qwen3-0.6b-q8_0.json \
  --artifact artifacts/models/Qwen3-0.6B-Q8_0.gguf \
  --check models/qwen3-0.6b-q8_0.inspection.json
```

Use `--output PATH` to atomically generate a report or omit both `--check` and
`--output` to write it to standard output. Inspection first verifies artifact
identity, then a development-only descriptor scanner records serialized GGUF types,
dimensions, offsets, and sizes. Current tinygrad independently loads the artifact;
the tool requires its metadata, tensor names, logical shapes, and logical dtypes to
agree with the descriptor scan. Production model loading continues to use tinygrad
exclusively.

## Upstream Model Smoke Test

Run the pinned checkpoint through tinygrad's current model-owned-cache generation
path with bounded fixed-token and text workloads:

```sh
DEV=NV JIT=1 .venv/bin/python tools/run_upstream_model.py \
  --manifest models/qwen3-0.6b-q8_0.json \
  --artifact artifacts/models/Qwen3-0.6B-Q8_0.gguf \
  --max-context 1024 \
  --fixed-output-tokens 4 \
  --text-output-tokens 16
```

The runner uses greedy sampling and emits JSON. Determinism is checked across clean
processes because tinygrad's upstream `Transformer` intentionally retains prefix and
KV state between calls. See
[`phase0d-upstream-functional.md`](../docs/baselines/phase0d-upstream-functional.md)
for the recorded functional result.
