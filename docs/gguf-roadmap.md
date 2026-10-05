# GGUF milestone

Phase 2 of the [roadmap](roadmap.md). Scope: one local GGUF v2/v3 containing
F32, Q4_K, and Q6_K tensors. Reject other formats and split files explicitly;
no architecture expansion or checkpoint-manifest system.

## Current implementation

- [x] Header, metadata scalars/arrays, tensor descriptors, aligned data start.
- [x] F32 bitcasts and reversed logical shapes.
- [x] Q4_K/Q6_K packed-byte counts and whole-block count checks.
- [x] Duplicate tensor names and tensor end-of-file bounds checks.
- [x] Exact reads; metadata-type, rank/dimension, row-block, alignment, range,
  overlap, duplicate-key/name, and split-file checks before tensor allocation.
- [x] 95 CPU tests, including malformed input, nested arrays, and v2/v3 loading.
- [x] All 398 bundled model descriptors pass validation without tensor allocation.
- [x] Full lazy Q4_K/Q6_K FP32 dequantization.
- [x] Explicit device staging and `PackedTensor` descriptors via `parse_gguf()`.
- [x] Quantized numerical/storage tests and opt-in CPU real-model loading.

The bundled Qwen3 file loads all 398 tensors on CPU without realized floating
weight buffers. The opt-in test completed in 8.46 s wall time with 4,994,912 KB
peak RSS (includes file-backed and resident packed storage, not an isolated
loader benchmark). GPU staging and kernel integration remain unverified.

Run: `DEV=CPU INFURNACE_TEST_MODEL=models/Qwen3-4B-Q4_K_M.gguf
.venv/bin/python -m pytest -q tests/test_gguf_quant.py::test_local_model`
(on one shell line).

## 1 — Validate before constructing tensors

- Read little-endian scalars and length-prefixed UTF-8 strings exactly; report
  truncated headers, metadata, arrays, and descriptors with useful context.
- Reject unknown metadata/array element types rather than leaking `KeyError`;
  retain all supported scalar values and nested arrays without coercion.
- Validate tensor rank and positive dimensions, supported GGML type, alignment,
  aligned relative offsets, and complete byte ranges within the file.
- Check quantized **row width**, not just total element count, is divisible by
  256. Reject duplicate names and invalid/overlapping tensor ranges.
- Reject unsupported split files. Validate descriptors before copying weights
  so malformed input cannot trigger partial model allocation.

## 2 — Make storage and device ownership explicit

Keep the public `(metadata, state_dict)` result with original names and
reversed GGUF shapes. Add explicit device selection for execution; retain
small internal packed descriptors (byte view, shape, GGML type/offset).

```text
Python header parsing → bounded DISK byte views
                     → copy/realize packed bytes on CPU or NV once
                     → lazy dequant expression → consuming operation
```

- DISK views are storage, not executable dequant tensors. A direct Q4_K
  dequantization probe on DISK failed; transferring packed bytes to CPU first
  succeeded. Do not dequantize on DISK and then copy floating-point weights.
- F32 stays native F32 at the loader boundary. Dequant arithmetic uses FP32;
  the model applies its compute-dtype policy (including lazy FP16 linear views).
- Realizing **packed bytes** is allowed; realizing the whole decoded model is
  not. Preserve storage lifetime, exact byte offsets, and tied-weight sharing.
- Prove that `packed_weight()` can recover the unchanged resident bytes after
  reshape/cast, or wire the explicit descriptors into decode. Byte-count-only
  graph matching is insufficient for transformed/permuted weights.
- Profile embedding and matmul consumers: a lazy graph alone does not prove
  fusion or bounded memory. Avoid repeating transfers/dequant materialization
  on every token; do not silently retain all decoded weights.

## 3 — Implement and verify block math

| Type | Values/block | Bytes/block | Layout |
|---|---:|---:|---|
| F32 | 1 | 4 | Little-endian native float |
| Q4_K | 256 | 144 | `d:2`, `dmin:2`, scales/mins:12, quants:128 |
| Q6_K | 256 | 210 | low bits:128, high bits:64, signed scales:16, `d:2` |

- Q4_K: unpack eight 32-value groups, reconstruct six-bit scales/minimums,
  then compute `d * scale * q - dmin * minimum`. Low/high nibble ordering is
  per 32-byte group, not all low nibbles followed by all high nibbles.
- Q6_K: combine low/high bits in the correct 128-value halves, subtract 32,
  and apply signed int8 scales for each 16-value group plus the FP16 block scale.
- Test all subgroup boundaries, multiple blocks/rows, negative scales,
  nonzero minimums, and zero blocks; one-byte nibble tests are not sufficient.

## 4 — Completion gate

Default tests are small, offline, and CPU-only:

- [ ] Synthetic v2/v3 files: all metadata types, nested/empty arrays, multiple
  tensors, non-square shapes, custom alignment, and nonzero relative offsets.
- [ ] Rejections: truncation, bad magic/version/type/alignment/dimensions,
  duplicate names, partial rows/blocks, overlapping/out-of-bounds ranges, splits.
- [x] F32/Q4_K/Q6_K values agree with independent scalar fixtures and pinned
  tinygrad `ggml_data_to_tensor` / `gguf_load`, with explicit tolerances.
- [x] Load/staging preserves packed bytes and creates no complete decoded-weight
  buffers; selected slices and downstream consumers execute on CPU.

An opt-in test uses `models/Qwen3-4B-Q4_K_M.gguf`:

- [ ] All 398 names, shapes, types, offsets, and packed lengths match the file;
  metadata supports the Qwen3 config and tokenizer.
- [x] Small sampled blocks of both quant types match the reference; do not
  materialize the full model merely to compare values.
- [ ] Report load/transfer time and host/GPU memory separately. GPU consumer
  and packed-kernel checks belong to Phase 5, not the CPU test gate.

Use tinygrad's GGUF code as a **test oracle only**. The pinned reader copies the
whole file to the default device; newer `gguf_parse()` / `gguf_load()` separate
DISK parsing from packed device staging. Follow that separation without adding
a runtime dependency on `tinygrad.llm` or assuming newer APIs exist at our pin.

**Next:** implement `config_from_gguf()` and validate the Qwen3 contract before
allocating the KV cache or implementing transformer execution.
