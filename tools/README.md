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
