# Roadmap

Implementation phases for the new single-user architecture. Each phase has a
verification gate before moving on. Status: **Phase 1 complete**, Phases 2–5
are the implementation work, Phase 6 is optional.

## Phase 1 — Structure (done)

- [x] New module layout (`gguf.py`, `tokenizer.py`, `model.py`, `cli.py`, `kernels/`)
- [x] Old serving layer removed (engine, scheduler, cache, executor, manifest, sampler, tests, tools, docs)
- [x] `kernels/quant_gemv.py` kept intact (self-contained, already implemented)
- [x] `README.md`, `docs/architecture.md`, `docs/roadmap.md`
- [x] Package imports cleanly (`import infurnace`)

## Phase 2 — Own GGUF parser (`gguf.py`)

- [ ] Binary reader: header, typed KV values, tensor info, aligned data offset
- [ ] F32 native tensors (bitcast)
- [ ] Q4_K / Q6_K block dequant as lazy tinygrad ops
- [ ] `load_gguf(path)` returns `(metadata, state_dict)` with fp16 lazy views
- **Gate:** parse the pinned Qwen3-4B Q4_K_M file — metadata keys and tensor
  names/shapes match the GGUF; quantized tensors report the right packed byte
  counts; results agree with `tinygrad.llm.gguf.gguf_load` on a small file.

## Phase 3 — Own tokenizer (`tokenizer.py`)

- [ ] Byte decoder + GPT-2-style split regex + greedy byte-pair merge
- [ ] `encode` / `decode` / `stream_decoder` / `is_end`
- [ ] `render_chat` (qwen2 `<|im_start|>...<|im_end|>` template)
- **Gate:** round-trip and streaming parity with the previous
  `SimpleTokenizer.from_gguf_kv` on a few sample prompts (same token ids).

## Phase 4 — Model (`model.py`)

- [ ] `config_from_gguf` → `Qwen3Config`
- [ ] `forward`: prefill with RMSNorm, RoPE, scaled-dot-product attention, KV writes
- [ ] `decode`: TinyJit-captured single-token decode with symbolic position + SSA KV writes
- [ ] `_decode_linear`: `quant_gemv` fast path with fp16 matmul fallback
- [ ] `generate`: prefill → sample → decode loop
- **Gate:** greedy output for a tiny prompt on CPU matches the old model path on
  the same weights (or the reference tinygrad `Transformer` for a small model).

## Phase 5 — CLI (`cli.py`)

- [ ] `_run_single_shot`: stream generated tokens, stop at EOS
- [ ] `_run_chat`: interactive multi-turn chat with history
- [ ] `_run_benchmark`: tokens/sec
- **Gate:** run the Qwen3-4B Q4_K_M GGUF on the GPU: correct chat output,
  fast decode (quant_gemv path), no fp16 OOM during prefill.

## Phase 6 — Optional: thin server

Only if a shared/local web UI is wanted. A ~150-line `serve.py` like
`tinygrad.llm.serve.py` (OpenAI-compatible `/v1/chat/completions`, streaming),
no batching or scheduling. Deferred until the CLI path is solid.

## Notes

- tinygrad is pinned by commit in `pyproject.toml` (installed automatically by
  `pip install -e .`). When advancing your tinygrad checkout, update the SHA in
  the same change — the kernels reach into tinygrad internals that move fast.
- Prefill uses the generic fp16 matmul over lazy dequant views; if that OOMs on
  8 GB for the 4B model, fall back to chunked prefill before adding kernels.
- Quant types beyond Q4_K/Q6_K are intentionally out of scope; add them to
  `gguf.py` only when a model needs them.