# Roadmap

## Focus and current state

Build a reliable **single-user, single-GPU local Qwen3 runner** first. Target:
`models/Qwen3-4B-Q4_K_M.gguf` on the 8 GB RTX 5060; CPU is the correctness
fallback. No server, scheduler, batching, multi-GPU, or additional model families.

| Phase | Status |
|---|---|
| 1. Small module layout / removal of serving infrastructure | Done |
| 2. GGUF loading | CPU quantized loading works; remaining fixture coverage and GPU/storage integration tracked below |
| 3. Qwen3 configuration and tokenizer | Config/weights, tokenizer adapter, and bundled text-only chat rendering implemented |
| 4. Correct eager inference | Stubs |
| 5. Validated NV kernels and JIT decode | Q4_K/Q6_K eager/JIT kernel tests pass on NV; model integration pending |
| 6. Usable CLI and measured local performance | Argument parsing only |

Current verification: **162 CPU tests pass; model/reference and NV tests are opt-in; Ruff
passes.** All 398 bundled tensors load as lazy FP32 views on CPU, with sampled
Q4_K/Q6_K parity. The opt-in test took 8.46 s wall time with ~4.8 GiB peak RSS
(including file-backed storage and resident packed bytes); this is not an
isolated load benchmark. Model execution is still unimplemented.
After upgrading tinygrad and fixing custom-UOp syntax, typed operands, and
NV offset-buffer handling, all four NV eager/JIT parity tests pass on the
RTX 5060. Run `DEV=NV INFURNACE_TEST_NV=1 .venv/bin/python -m pytest -q
 tests/test_tinygrad_compat.py` on one line. No end-to-end inference works yet.

## 2 — Finish GGUF loading (next)

- [x] V2/v3 header, typed metadata/arrays, descriptors, aligned data start.
- [x] F32 bitcast/views, reversed logical shapes, packed-size calculation.
- [x] Duplicate tensor-name and end-of-file range checks.
- [x] Exact reads and validation of metadata types, dimensions, row block
  geometry, alignment, tensor ranges/overlaps, and unsupported split files.
  All 398 bundled descriptors pass validation before tensor allocation.
- [x] Q4_K/Q6_K lazy dequantization, scalar/reference parity, and packed-size tests.
- [x] Explicit storage path: `parse_gguf()` returns bounded DISK `PackedTensor`
  descriptors; `load_gguf(path, device=...)` realizes packed bytes, not weights.
- [x] Explicit packed type/shape and bounded byte view for future decode wiring;
  kernel integration remains Phase 5 (do not infer identity from byte count).
- [ ] Complete scalar-metadata/custom-alignment/truncation fixture coverage and
  separate load/transfer memory measurements; GPU staging is not verified yet.

**Gate:** load all 398 bundled tensors (145 F32, 216 Q4_K, 37 Q6_K), match
reference metadata/shapes/decoded samples, and show no whole-model float
materialization. Implementation details: [GGUF roadmap](gguf-roadmap.md).

## 3 — Lock down the Qwen3 contract

- [x] Implement `config_from_gguf()`; reject unsupported architectures/features
  and inconsistent dimensions before allocation.
- [x] Use explicit key/value lengths, not `embedding_length / head_count`.
  This file has embedding 2560, 32 query heads, 8 KV heads, **head size 128**;
  query projection width is 4096. Default missing RoPE dimension to head size.
- [x] Validate required tensor names/shapes, including per-head Q/K norms.
  When `output.weight` is absent, share `token_embd.weight` (this file is tied).
- [x] Honor `max_context`: positive, capped by model context; estimate KV bytes
  before allocation. At 2048 tokens this model's FP16 KV cache is 288 MiB;
  the advertised 40960-token context alone needs 5.625 GiB.
- [x] Reuse tinygrad `SimpleTokenizer` behind a thin adapter. Packaging is fixed;
  all 14 sampled bundled-Qwen3 prompts match llama.cpp exactly (including digits,
  Unicode, code, whitespace, and special tokens). This is sampled parity, not
  proof of compatibility with every Qwen2 vocabulary or input.
- [x] Adapter: metadata/vocab validation, encode/decode, independent incremental
  UTF-8 decoders, special tokens, BOS policy, EOS/EOT, and preset. `encode()`
  returns raw IDs without automatically prepending BOS; callers use `bos_id`.
  Run tokenizer opt-in fixtures with `INFURNACE_TEST_MODEL`; additionally set
  `INFURNACE_LLAMA_TOKENIZE` to a llama-tokenize binary for live reference checks.
- [x] Implement the bundled Qwen3 text-only chat format: system/user/assistant,
  turn delimiters and optional generation prefix. Thinking defaults on;
  `enable_thinking=False` appends the template's empty think block. Historical
  assistant reasoning is omitted before the latest user turn. Seven histories
  across both thinking/prefix settings match actual GGUF Jinja output (28 cases).
  Tools/non-text messages are rejected; Jinja is a development oracle only.

Config gate passed: synthetic invalid-input tests and the opt-in bundled-model
shape check run before weight staging; factory preserves tied storage. Model
construction/cache allocation remain stubs. Tokenization reuses tinygrad at
runtime; own GGUF parsing and model execution remain independent.

**Gate:** config/shape tests cover this file and invalid inputs; tokenizer IDs
match an authoritative Qwen/llama.cpp reference on ASCII, Unicode, whitespace,
digits, and special tokens. Streaming plus final flush equals full decode;
chat rendering has exact fixtures. Tinygrad's simplified tokenizer is a
secondary reference, not the sole correctness oracle.

## 4 — Make eager inference correct before optimizing

- [ ] Implement embedding, attention/FFN RMSNorm, **per-head Q/K RMSNorm**,
  half-split RoPE, grouped-query attention, residuals, and SwiGLU.
  Do not apply Llama's GGUF Q/K weight permutation to Qwen3.
- [ ] Make precision explicit: lazy FP16 linear weights, FP16 KV storage,
  FP32-sensitive norm/softmax/reduction math. Expose logits for testing; apply
  final norm/output projection only to the last prompt token.
- [ ] Own one contiguous KV cache; writes must precede reads. For prefill at
  nonzero positions use an offset-aware causal mask, not plain `is_causal=True`.
- [ ] Implement eager single-token decode and bounded chunked prefill using
  the same block math; never attend to unwritten cache entries.
- [ ] Implement generation: exact greedy argmax at temperature 0, seeded
  Gumbel-max otherwise. Validate tokens/temperature/limits, preserve the input
  list, bound context writes, and stop cleanly on token/context limits.
  Callers handle EOS and cancellation; start with cache reset per generation.

**Gate:** on a small deterministic Qwen3-shaped fixture, compare logits and
valid KV entries to the reference within documented tolerances. Full prefill,
chunked prefill (including a one-token tail), and token-by-token execution
agree; repeated requests and context boundaries do not leak stale state.
CPU tests must work without a GPU or downloading a large checkpoint.

## 5 — Validate and integrate fast NV decode

- [x] Fix NV GEMV runtime/API failures; add Q4_K/Q6_K GPU parity tests with
  Q8-rounded activations, partial final warps, offset views, and TinyJit replay.
- [ ] Add isolated Q8 quantization tests and broader kernel contract checks.
- [ ] Guard dispatch by backend/renderer and kernel contract: one-token decode,
  matching input shape, 256-value weight blocks, packed dtype/length/alignment,
  and same-device resident buffers. Unsupported cases use the generic path.
- [ ] Harden `packed_weight()` or pass explicit packed descriptors; reject
  transformed/permuted weights. Test nonzero offsets, exact-sized buffers,
  tied weights, negative scales, zero activations, and partial final warps.
- [ ] Compare kernels against dequantized matmul with **Q8-rounded activations**;
  separately measure error versus unrounded activations. This fast path adds
  activation quantization, so bit-identical FP16 results are not the gate.
- [ ] Capture single-token decode with TinyJit and a bound position UOp;
  use SSA `store`/`after` for KV writes. Prepare packed buffers outside capture.
- [ ] Inspect schedules and peak memory for embedding, prefill, and decode.
  Laziness does not guarantee fusion; chunking tokens does not fix persistent
  full-weight materialization. Use bounded per-layer/tiled dequantization if
  needed before adding a custom prefill kernel.

**Gate:** eager/JIT and generic/fast-path comparisons pass; replay at changing
positions and after cache reset needs no per-position compilation or repeated
weight transfers. The bundled model runs within 8 GB without retaining a
whole-model FP16 copy. GPU tests are explicitly marked/skipped on CPU-only hosts.

## 6 — Deliver useful local inference, then measure improvements

- [ ] Fix `_build()` to load once and share metadata/weights. Make `--device`
  effective via tinygrad's device/context API: setting `os.environ['DEV']`
  after tinygrad import does not update its already-initialized setting.
- [ ] Implement single-shot streaming and multi-turn chat; flush UTF-8 on EOS,
  token/context limit, and interruption. Validate conflicting/invalid flags,
  handle EOF/Ctrl-C, and report context exhaustion without silent truncation.
- [ ] After reset-based chat is correct, reuse only the exact cached token
  prefix. Track tokens actually written to KV, not merely emitted tokens;
  test changed prompts, thinking-template rewrites, and abandoned generation.
- [ ] Implement reproducible benchmarks: load time, cold/warm time to first
  token, prefill tokens/sec, steady decode tokens/sec, and peak host/GPU memory.
  Separate JIT warmup, synchronize timings, and report actual token counts.
- [ ] Compare against pinned tinygrad and llama.cpp on the same hardware,
  GGUF, prompt IDs, context, and generation settings. Record versions and
  repeated runs; optimize measured bottlenecks rather than claim a speedup.

**Gate:** CLI integration tests and real-model single-shot/chat/benchmark runs
pass. Publish the benchmark baseline and tradeoffs. "Better" means demonstrated
local latency/throughput/memory improvements **without losing correctness or
reliability**, not more server infrastructure.

## Reference and scope rules

- Runtime dependency stays tinygrad only. Reuse `llm/cli.SimpleTokenizer` for
  tokenization. GGUF/model/kernel code is a development oracle, not a runtime
  implementation dependency (the upstream tokenizer module transitively imports
  its model and server). Keep pinned-import regression tests.
- Current pin: `ce9c84605e94de9667fb578c1ff3fa8678725a87`, including the wheel
  packaging fix for `tinygrad.llm.kernels`. `Transformer`/`SimpleTokenizer`
  imports are now covered by a regression test. Tokenizer correctness is still
  pending; successful import does not establish Qwen3 parity.
- When upgrading an existing environment, pip may retain tinygrad's same-version
  wheel. Force reinstall the exact VCS dependency from `pyproject.toml`, then
  rerun CPU and opt-in NV/JIT tests. Fresh installs use the declared pin.
- Packed extraction copies compressed bytes to aligned storage once (cache the
  result outside JIT). Direct offset Buffer views hung on the tested NV runtime;
  no decoded floating-point weights are copied.
- **Server deferred until Phase 6 passes.** If needed later, add a thin local
  adapter over the same generation path, with request validation, cancellation,
  context limits, and correct streaming. No return to the old serving stack.
