# Roadmap

**Goal:** fast, easy local inference for people with modest GPUs. One user, one model (Qwen3.5-4B
Q4_K_M with its DFlash draft), on tinygrad. No batching, scheduler, multi-GPU or other model families.

Every change needs three checks: a correctness check (tokens or logits), an end-to-end speed gain
(not only a kernel microbenchmark), and a memory check on the 8 GB card.

## Where it stands

Measured on an RTX 5060 8 GB (30 SMs, 448 GB/s), Qwen3.5-4B Q4_K_M, October 2026.

| | |
|---|---|
| Plain decode | 103–110 tok/s up to 8k tokens of context; 95 tok/s at 16k |
| DFlash decode | ~340 tok/s on code; 200–255 tok/s summarizing docs; ~180 tok/s at 16k; 1.6–2.0× plain at temperature 0.6 |
| Prompt reading | 64 rows at a time: 1,850–2,300 tok/s plain, 1,400–1,750 with DFlash, from 1k to 16k tokens (0.55 s to first token at 1k, 7.4 s at 16k) |

Prompt-length sweep (`benchmarks/sweep.py`: a chat message holding `eval.txt` cut to length, then "Summarize
the text above", 128 tokens out, after warmup):

| Prompt tokens | TTFT | Prompt tok/s | Plain decode tok/s | DFlash decode tok/s |
|---|---|---|---|---|
| 25 | 0.20 s | – | 111.6 | 160.7 |
| 124 | 0.22 s | 561 | 111.8 | 267.0 |
| 524 | 0.42 s | 1,263 | 109.2 | 164.2 |
| 1,024 | 0.63 s | 1,631 | 101.0 | 182.6 |
| 2,024 | 1.1 s | 1,876 | 105.6 | 190.8 |
| 4,024 | 2.0 s | 2,041 | 108.1 | 197.7 |
| 8,024 | 3.9 s | 2,051 | 103.0 | 189.6 |
| 16,024 | 8.3 s | 1,934 | 92.9 | 183.8 |

TTFT and prompt rate are plain decoding's; with DFlash the prompt reads at 1,330 (1k), 1,574 (4k) and 1,561
(16k) tok/s, 10.3 s to first token at 16k.

Run-to-run noise is about ±5 tok/s (other apps drawing on the GPU cost up to 10%); DFlash depends on what it
generates.
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

`serve` picks the context from free VRAM (`loader.fit_context`: weights + 36 KB a token + ~1.45 GB of graphs and
buffers + a 0.25 GB margin; measured 4.68 GB at 8k, 5.57 GB at 32k, 5.83 GB at 40k after warmup, ~0.2 GB more
after the prompt graph). On the 8 GB card next to a desktop that is 24-32k.

- [x] Choose the context from free VRAM at startup.
- [ ] Turn DFlash off when memory is tight (below ~5 GB free the 4B model with its draft does not fit at all).
- [ ] Publish a measured "what fits" table (card size × context × DFlash).
- [ ] If smaller Qwen3.5 GGUFs exist (with or without drafts), add presets for 4–6 GB cards.

Memory at 32k context, for reference: weights 2.64 GB, attention cache 1.0 GB (32 KiB per token),
draft weights 0.36 GB, draft caches 0.21 GB, recurrent state 0.05 GB, captured-graph scratch ~0.35 GB,
tinygrad's runtime ~0.39 GB, and ~0.47 GB of CUDA context outside tinygrad.

### 3. Startup takes about a minute

Load (~11 s) plus warmup: ~47 s plain, ~75 s with DFlash (`NV`; `CUDA` takes ~20 s more). Nearly all of the
warmup is capturing graphs (DFlash: prompt chunk 25 s, verify 19 s, one-row target 7 s, draft 6 s, the rest
~4 s), and most of a capture is tinygrad's scheduler, redone on every start. Chat warms up while the user
types; the server listens after ~50 s and captures the prompt graph in its first idle moment.

- [x] One denoise graph padded to the full block (drafting reuses it for any proposal count).
- [x] No eager pass before the one-row target capture; unfused proposals share the fused round's draft graph.
- [x] The server captures the prompt graph after it starts listening (`capture_prompt_graph`).
- [ ] Save captured graphs between runs (without the weights they reference), so a start skips capturing.

### 4. Sensible sampling defaults

Decoding is greedy unless asked otherwise, and pi sends no temperature, so agents get greedy decoding,
which Qwen warns repeats in thinking mode. The fused sampler has no top-k or top-p, which Qwen's
recommended settings (0.6, top-p 0.95, top-k 20) need.

- [ ] Top-k/top-p in `kernels/sampling.py`; Qwen's settings as the default with thinking.
- [ ] A real tool-use reliability check with pi.

### 5. Agent latency: fixed cost per request

Measured before 64-row prompt chunks: each tool step read its new tokens at only 250–380 tok/s against ~600
for long prompts, about 0.3–0.4 s of fixed cost per request. Saving and restoring prompt snapshots accounts
for ~0.1 s (50 MB each way to host memory). Short prompts now read at ~400–500 tok/s against ~1,600; re-measure
with pi before profiling.

- [ ] Profile one short follow-up request; cut the fixed cost.

### 6. Raw speed

A DFlash round was 23 ms: the 16-row target verify 15.8 ms (its matmuls 11 ms, now 8.6 ms), the draft
4.1 ms, the rest ~1.6 ms. Plain decode is 10.3 ms per token (its matmuls 7.6 ms).

- [x] **Decode at long context** (done; see Done): plain decode at 16k 53 → 89 tok/s.
- [x] **DFlash at long context** (done; see Done): 102 → 158 tok/s at 16k.
- [ ] **Prompt chunk, after multi-chunk matmuls** (64 rows, ~32 ms of GPU work, RTX 5060): matmuls 14.7 ms, now
  bound by FP16 MMAs with FP32 accumulation (~32 TFLOPS) rather than weight reads; four 16-row GDN scans a layer
  ~4.6 ms (48 µs a call, was 83); attention 1.0 ms; the GDN β and α projections 0.3 ms (`gdn_gates`, was 4.4 ms);
  the rest ~4 ms, plus ~5 ms of replay. A single 64-row scan would not save much: the state projection and update
  (the bulk of a scan) happen once per 16 rows either way.
- [ ] **16-row attention at head dim 256** (the target verify and prompt reading): 393 µs a layer at 16k keys,
  ~2.5× the time to read its cache. The output fragments take 128 registers a lane (254 in all: 8 warps an
  SM), so loads aren't overlapped. Tried: 64–512-key segments (256 is best); values staged in shared memory
  (461 µs: barriers at this occupancy); Q fragments in shared memory (537 µs: the compiler spends the freed
  registers on more prefetch and spills). Left: splitting the head dimension's output across warps, or
  pipelining loads with cp.async. Worth ~2 ms a DFlash round at 16k, little at short contexts.
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

- 2026-10-17: the user side, after Ollama's workflow. `infurnace serve` keeps the model warm (listens at once,
  warms up on a thread, generation behind a GPU lock); `run` (interactive with /set, /clear, """, or one prompt
  and piped input) talks to it and starts one in the background when none answers; `ps`, `stop`, `pull`, `list`,
  `rm`, `launch pi` (pi's models.json and settings.json, entries marked `_launch`). infurnace's API at `/api`
  (chat, generate, tags, ps, show, status, version, shutdown) beside the OpenAI-compatible `/v1`. The context
  comes from free VRAM. A warm `run` answers in ~0.6 s end to end.
- 2026-10-17: the GDN chunk kernel (`gdn_chunk`, `gdn_prefill`) splits a head over 4 blocks of 32 state rows and,
  inside, 4 warps of 32 key columns (one 128-thread block a head left one warp per scheduler and thousands of
  dependent steps a thread); S0 staged coalesced, K/Q rows padded against bank conflicts, the state update as a
  loop of eight rows (unrolled it spilled). 87 → 48 µs a call. RMSNorm and the gate now follow the kernel as
  tensor ops (a head's values span four blocks). Prompt reading +11-14% (4k: 2,041 → 2,297 tok/s; 16k prompt
  7.4 s to first token), DFlash decode 1k 183 → 224 tok/s. Perplexity unchanged (16-row 16.9860, verifier 16.9861).
- 2026-10-17: `gdn_gates`: the GDN β and α projections (2,560 → 32 each) in one launch, one warp per output, for
  prompt chunks and DFlash's verifier, instead of tinygrad's generic matmul (~91 µs a call). Plain prompt reading
  1,734 → 1,958 tok/s (2k prompt); DFlash decode +9-15% (1k: 166 → 183 tok/s; 16k: 161 → 184), its prompt reading
  +16-18%. Perplexity unchanged (16-row 16.9860, verifier 16.9856).
- 2026-10-17: prompt-chunk matmuls take two 16-row chunks a warp (`mtiles`): each dequantized weight block feeds
  two MMAs. Prompt reading +8-10% (1k: 1,390 → 1,521 tok/s; 4k: 1,683 → 1,829). At 64 rows the kernels turned out
  bound by the MMAs, not weight reads: one, two and four chunks a warp are within ~5-25% of each other.
- 2026-10-17: startup. The one-row target graph is captured without an eager pass and unfused proposals share
  the fused round's draft graph (DFlash warmup 73 → 64 s); the server listens before capturing its prompt
  graph, then captures it in its first idle moment without disturbing a conversation (listening after ~50 s
  instead of ~84 s).
- 2026-10-16: prompt reading in 64-row chunks through one fixed-size captured graph that slices the prompt
  from the device (`Qwen35Model._prompt_chunk`); the row count is a value, not a shape (a symbolic count gave
  hundreds of kernels runtime launch sizes, patched on every replay: 65 vs 33 ms a chunk). DFlash reads its
  prompt through the same chunks, which also return the draft's feature taps. Plain prompt reading
  ~650 → ~1,600 tok/s, DFlash ~600 → ~1,300; 16k prompt 25 → 10 s to first token (12.4 s with DFlash).
  Warmup +11 s plain, +29 s with DFlash. Attention runs in 16-row slices on the FP32 kernel (its shared
  memory ran out at 64 rows).
- 2026-10-16: 16-row attention on FP16 tensor cores (`_mma_kernel`, FlashAttention-2's layout on mma.sync
  m16n8k16): the target verify and prompt reading, and the draft's sliding and full layers. At 16k keys: target
  1.51 → 0.39 ms a layer, draft full layer 1.60 → 0.33 ms, sliding layer 0.76 → 0.11 ms (values staged
  transposed in shared memory at head dim 128). DFlash 102 → 158 tok/s at 16k, 119 → 163 at 8k; 16k prompt
  30 → 25 s to first token. Perplexity unchanged.
- 2026-10-16: one-row decode attention after flash-decoding (`_decode_kernel`): each warp walks 128 keys of a
  KV head with one 16-byte load per lane per key, scores reduced by shuffles, softmax in registers. At 16k keys
  1.2 ms → 0.21 ms per layer (decode part 411 GB/s); plain decode 84 → 108 tok/s at 4k, 73 → 102 at 8k,
  53 → 89 at 16k. Perplexity within the one-row path's noise (17.006 vs 16.997; reordering the old kernel's
  sums alone gives 16.980).
- 2026-10-16: prompt reading through the attention kernel (16-row chunks padded like the verifier's): 16k-token
  prompt 255 → 30 s to first token, 8k 48 → 13 s; ~600 tok/s from 1k to 16k instead of 535 falling to 63.
  Perplexity unchanged (16.986, 6.511 at 4k). DFlash: drafting fewer than 15 tokens reuses the full-block graph,
  and the last positions of a full context step plainly, instead of capturing a draft graph per size (~3.5 s
  stalls each). `benchmarks/sweep.py`: TTFT, prompt and decode rates across prompt lengths.
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
