# Roadmap

**Goal:** fast, easy local inference for people with modest GPUs. One user, one model (Qwen3.5-4B
Q4_K_M with its DFlash draft), on tinygrad. No batching, scheduler, multi-GPU or other model families.

Every change needs three checks: a correctness check (tokens or logits), an end-to-end speed gain
(not only a kernel microbenchmark), and a memory check on the 8 GB card.

## Where it stands

Measured on an RTX 5060 8 GB (30 SMs, 448 GB/s), Qwen3.5-4B Q4_K_M, October 2026.

| | |
|---|---|
| Plain decode | 112–114 tok/s after a short prompt; ~96 tok/s at a 2.5k-token context |
| DFlash decode | ~340 tok/s on code (7.7 tokens per 23 ms round); ~160 tok/s on prose with thinking; 1.6–2.0× plain at temperature 0.6 |
| Prompt reading | 16 rows at a time: ~550–640 tok/s with DFlash, ~400 tok/s plain |
| Later chat or agent turns | resume from a saved prompt state: ~0.6 s to first token instead of ~6 s for a 3k-token chat |
| Context | 8,192 by default; 32,768 fits on the 8 GB card (peak 7.35 GB with ~2 GB of desktop apps) |
| Agent use (pi) | read-and-answer 4.1–4.4 s, read-and-fix 8.9–9.7 s |
| Fast path | NVIDIA GPUs through tinygrad's `NV` (Linux) or `CUDA` backend, with NVRTC and the CUDA headers (stops with instructions when they are missing) |

How it works: [architecture](architecture.md), [DFlash](dflash.md), [serving](serve.md).

## Priorities

In order. Each item says what is wrong, the evidence, and how to check the fix.

### 1. Output quality: measured against llama.cpp

`benchmarks/quality.py` computes perplexity with llama.cpp's protocol (chunks of `--ctx` tokens, fresh state
each, the second half scored), teacher-forcing infurnace's real kernels on three paths: one-row (decoding),
16-row (prompt reading) and DFlash's verifier (`ParallelVerifier.logits`: `gdn_chunk`, the attention kernel at
16 rows, and the closed-form commit that advances the recurrent state between blocks). The reference is
llama.cpp's `llama-perplexity` on the same GGUF; both tokenizers produce identical tokens. RTX 5060:

| Text, context | llama.cpp | one-row | 16-row | verifier |
|---|---|---|---|---|
| `eval.txt` (docs and code, 10,436 tokens), 512 × 8 | 17.032 | 16.997 (−0.2%) | 16.985 (−0.3%) | 16.986 (−0.3%) |
| `eval.txt`, 4,096 × 2 | 6.518 | 6.515 (−0.0%) | 6.511 (−0.1%) | 6.511 (−0.1%) |
| `multilingual.txt` (9 languages, 3,057 tokens, 43% above id 100k), 512 × 5 | 6.930 | 6.906 (−0.4%) | 6.893 (−0.5%) | 6.893 (−0.5%) |

Before the Marlin-style kernels the one-row and 16-row paths gave 16.9965 and 16.9845 on the first row: no
measurable change. `tests/core/test_nv_quality.py` (opt-in, ~2.5 minutes) fails if any path on either text
drifts more than 1% from llama.cpp. (tinygrad's generic path can't be the reference: it dequantizes to FP16,
about 8 GB.) `multilingual.txt` was written for this harness, not sampled from natural text: it is for
comparing engines on the same tokens, not for absolute perplexity.

- [x] Perplexity of the one-row and 16-row paths against llama.cpp, as a test.
- [x] The DFlash verifier's own path, through a logits method on the verifier.
- [x] A multilingual text (exercises the embedding and output rows CJK, Cyrillic, Arabic and Hindi use) and
  a 4,096-token context.
- [ ] A chat-formatted text with special tokens: `llama-perplexity` always tokenizes them as plain text,
  so llama.cpp can't score it as-is.

### 2. Fit any card without flags

The default context is a fixed 8,192 tokens (about 5.2 GB with DFlash). A 6 GB card with a desktop
running runs out of memory, and a 12 GB card wastes room.

- [ ] Choose `--max-context` from free VRAM at startup; turn DFlash off when memory is tight.
- [ ] Publish a measured "what fits" table (card size × context × DFlash).
- [ ] If smaller Qwen3.5 GGUFs exist (with or without drafts), add presets for 4–6 GB cards.

Memory at 32k context, for reference: weights 2.64 GB, attention cache 1.0 GB (32 KiB per token),
draft weights 0.36 GB, draft caches 0.21 GB, recurrent state 0.05 GB, captured-graph scratch ~0.35 GB,
tinygrad's runtime ~0.39 GB, and ~0.47 GB of CUDA context outside tinygrad.

### 3. Startup takes about a minute

Every launch spends ~20 s loading and 40–70 s warming up (`NV`; `CUDA` takes ~20 s more). Chat hides
part of it by warming up while the user types; the server does not. Most of the warmup is capturing a
separate draft graph for each proposal count from 1 to 15 (`_denoise_graph` in `dflash/draft.py`).

- [ ] One denoise graph padded to the full block; time warmup before and after; outputs unchanged.

### 4. Sensible sampling defaults

Decoding is greedy unless asked otherwise, and pi sends no temperature, so agents get greedy decoding,
which Qwen warns repeats in thinking mode. The fused sampler has no top-k or top-p, which Qwen's
recommended settings (0.6, top-p 0.95, top-k 20) need.

- [ ] Top-k/top-p in `kernels/sampling.py`; Qwen's settings as the default with thinking.
- [ ] A real tool-use reliability check with pi.

### 5. Agent latency: fixed cost per request

Each tool step reads its new tokens at only 250–380 tok/s against ~600 for long prompts: about
0.3–0.4 s of fixed cost per request. Saving and restoring prompt snapshots accounts for ~0.1 s
(50 MB each way to host memory).

- [ ] Profile one short follow-up request; cut the fixed cost.

### 6. Raw speed

A DFlash round was 23 ms: the 16-row target verify 15.8 ms (its matmuls 11 ms, now 8.6 ms), the draft
4.1 ms, the rest ~1.6 ms. Plain decode is 10.3 ms per token (its matmuls 7.6 ms).

- [x] **16-row Q4_K matmuls, Marlin-style** (done; see Done). All Q4_K projections 5.38 → 4.48 ms; with
  weights in L2 the kernel runs at 427 GB/s (95% of peak), so what remains is DRAM access.
- [x] **The same design for Q6_K** (done; see Done): all Q6_K projections 3.29 → 2.35 ms.
- [x] **Q5_K** (done; see Done): all Q5_K projections 2.28 → 1.78 ms. All 16-row matmuls together:
  10.95 → 8.61 ms per round.
- [ ] **Draft vocabulary.** The draft scores all 248k tokens (1.9 ms of its 4.1 ms); scoring only the
  most frequent ~32k, as FR-Spec does, keeps output exact (the target still verifies). About +7%.
- [ ] **Per-GPU tuning tables** (after ExLlamaV3's autotuner): replace hand-picked kernel constants
  (`quant_gemm._schedule`, `quant_gemv._decode_warps`, `PREFILL_ROWS`) with values measured per GPU and
  checked in, with launch sizes derived from the SM count as the default for unknown cards.
- [ ] **Bundled projections:** one launch for MLP gate+up and for GDN qkv+gate. Shared Q8 quantization
  is already done; bundling must win end to end, not only in launch count.
- [ ] **Cheaper probing:** prose with thinking off runs 15–21% slower with DFlash, because the drafting
  policy keeps paying for probe rounds; back off sooner when a probe accepts almost nothing.
- [ ] **Recurrent-layer chunk on tensor cores** (`gdn_chunk`): part of the ~4.8 ms of non-matmul work
  in each verify.

### 7. Hardware beyond current NVIDIA cards

- RTX 20 and GTX 16 (sm_75): tinygrad offers only the m16n8k8 tensor-core shape there, so no 16-row
  path and no parallel DFlash. Plain decode should work; unmeasured.
- AMD, Apple: the kernels use CUDA-only intrinsics (`dp4a`, `mma.sync`, warp shuffles), so these are
  ports, not flags. tinygrad's own RDNA3/4 kernels may already make plain decoding usable on AMD.
- Windows: tinygrad's `CUDA` backend loads `nvcuda.dll`, so the fast path should work there with
  NVRTC; untested.

## Reliability gates

- [ ] Output quality per path (priority 1).
- [ ] DFlash output parity across prompt types, EOS positions, tail blocks, repeated requests and
  context exhaustion.
- [ ] Long sessions: resets, snapshot restores after the draft's ring cache wrapped, interrupts,
  device timeouts.
- [ ] Real-model HTTP tool use and truncated responses.
- [ ] GPU tests runnable in one process (kernel tests run out of memory next to the DFlash model fixture).
- [ ] Direct `_DraftPolicy` tests with fake timings (`dflash/generate.py`).

## Measuring and validating

```sh
.venv/bin/pytest -q                              # CPU suite
INFURNACE_TEST_NV=1 .venv/bin/pytest -q tests/kernels/test_nv_attention.py   # GPU tests, one file at a time
INFURNACE_TEST_MODEL=/path/to/Qwen3.5-4B-Q4_K_M.gguf INFURNACE_TEST_GENERATION=1 \
  .venv/bin/pytest -q tests/core/test_runner.py::test_local_generation
python benchmarks/quality.py --llama-perplexity ~/repos/llama.cpp/build/bin/llama-perplexity   # quality vs llama.cpp
infurnace --benchmark 512,32                     # per-step plain decode timing
```

More in [benchmarks](../benchmarks/README.md). Rules that have caught mistakes before:

- Benchmark projections with distinct weights per call: repeating one weight lets the 24 MB L2 serve it.
- Time whole runs, not single kernels in `DEBUG=2` mode (it adds ~15 µs to every kernel).
- `DEBUG=7` prints SASS; `PROFILE=1` records GPU timestamps. Avoid `VIZ=2`: it enables the PMA
  counters, which froze this desktop before.
- Keep cold and warm runs apart, and separate allocated capacity from active length.
- Multirow and one-row kernels sum in different orders, so near-ties can flip a greedy token; compare
  outputs up to the first near-tie.

## Investigated and set aside

- **int8 tensor cores** (llama.cpp's MMQ approach): tinygrad's CUDA backend has FP16, BF16, TF32 and FP8
  shapes but no int8, and we don't change tinygrad.
- **Building FP16 fragments from scalar halves** (the first 16-row rewrite): nvcc repacks them with
  `MOV`/`PRMT`, a quarter of the loop. Packed `half2` math cut to halves only at the MMA avoids it; a
  vector bitcast or a vector-shaped custom op instead sends tinygrad's devectorizer into an infinite loop.
- **FP16 accumulation** in the 16-row matmuls (ExLlamaV3 uses it on consumer GPUs): slower on the 5060
  and 1,000× more error.
- **More split-K, smaller tiles or blocks** for the 16-row matmuls: no gain once measured without L2 reuse.
- **Confidence-calibrated draft windows** (ExLlamaV3): pointless while verification always runs a
  full 16-row block.
- **Quantized attention cache** (FP8): ~575 MB saved at 32k context; revisit if priority 2 needs it.
- **`CUDA` as the default backend:** same speed and output as `NV`, slower warmup; `NV` stays first on
  Linux until `CUDA` has peak-memory and long-session data.
- **EAGLE-3 and Qwen3 support:** removed to keep one model and one draft.
- **EXL3 weights, batching, paged scheduling, multi-GPU:** outside the single-user GGUF scope.

## Done

- 2026-10-15: quality harness covers the DFlash verifier (`ParallelVerifier.logits`), a multilingual text and
  4,096-token chunks; all paths within 0.6% of llama.cpp.
- 2026-10-15: NVIDIA GPUs without NVRTC or the CUDA headers no longer run the generic path silently: loading
  compiles one FP16 kernel and stops with what is missing and how to provide it (the CUDA toolkit, or
  `NVRTC_PATH`/`CUDA_PATH` pointing at a copy such as NVIDIA's pip wheels). No new dependency: the user
  chooses. `nv_quant=False` still runs the generic path on request.
- 2026-10-15: Q5_K through the same 16-row kernel (fifth bits merged under the FP16 trick) and tiled like
  Q4_K: all Q5_K projections 2.28 → 1.78 ms; DFlash +3%; plain decode unchanged. Over the day, Q4_K + Q5_K +
  Q6_K: all 16-row matmuls 10.95 → 8.61 ms per round, DFlash code 290 → ~316–323 tok/s, prose 163 → ~184–186.
- 2026-10-15: Marlin-style 16-row Q6_K kernel and a tiled Q6_K layout (replacing the swizzled one). All
  Q6_K projections at 16 rows 3.29 → 2.35 ms (vocabulary layer 1.83 → 1.33 ms, 403 GB/s); DFlash code
  303 → 323 tok/s, prose 172 → 184; plain decode unchanged within noise.
- 2026-10-15: Marlin-style 16-row Q4_K kernel and a tiled Q4_K layout shared with the one-row kernel.
  All Q4_K projections at 16 rows 5.38 → 4.48 ms; DFlash code 290 → 301–304 tok/s, prose 163 → 168–171;
  plain decode within noise (−1.5%, with 8 warps per block for one-row Q4_K). The kernel rounds d·scale to
  FP16 as Marlin does: output error 5.2e-5 mean against 3.7e-5 before, relative to the largest output.
- 2026-10-15: fast kernels on tinygrad's `CUDA` backend too (identical output to `NV`).
- 2026-10-13:
  - Ring caches for the draft's five sliding-window layers (~520 MB less at 32k); window attention
    skips segments before the window.
  - Serving with DFlash and per-request thinking and temperature.
  - Up to 8 prompt snapshots in host memory, keeping the last user message's, so an agent's new
    message doesn't re-read the conversation.
  - Fused sampling kernel: one verify graph for every temperature.
- 2026-10-12:
  - 16-row prompt reading for plain decoding (`gdn_prefill`): first token for a 2.5k-token prompt 26.9 → 6.4 s.
  - Active-length attention kernel (after vLLM's unified attention): DFlash at a 16k window 91 → 332 tok/s;
    plain decode at 2.5k 74 → 96 tok/s.
  - Prompt snapshots.
  - Temperature sampling for plain decoding and DFlash.
  - Shared Q8 activations for projections with the same input; Q6 output head retuned.
  - Models by name with first-use download; tokenizers in the base install.
