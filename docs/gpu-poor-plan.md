# Plan: easy and fast on modest GPUs

From a code review on 2026-10-12. Goal: infurnace should be easy to install and fast on the
cheap or older GPUs people actually have, not just the RTX 5060 it was tuned on. Items are
ordered by impact. Each one says where the problem is in the code and how to check the fix.

## Top three

1. [Multirow prefill for plain decoding](#2-plain-decoding-reads-prompts-one-token-at-a-time)
2. [Attention over the active length, then a larger default context](#4-small-default-context-and-dflash-cost-grows-with-it)
3. [Install and model download](#6-install-and-getting-models)

---

## 1. Only one setup gets the fast path

- [x] Run the kernels on tinygrad's `CUDA` backend too (the CUDA driver API; tinygrad loads it on Windows)
- [ ] Benchmark plain decoding on an AMD RDNA3/4 card through upstream tinygrad
- [ ] Try the packed kernels on AMD/Metal with the CUDA-renderer check relaxed
- [ ] Publish a "what works on what" table in the README

**Problem.** The packed kernels turn on only for tinygrad's `NV` backend with its CUDA renderer:
`supports_nv_quant` (`infurnace/device.py`) and
`nv_prefill_supported` (`kernels/quant_gemm.py`). `NV` drives the NVIDIA kernel driver directly:
Linux only, and probably Turing or newer (check tinygrad's support list). Everything else runs
upstream's generic Qwen3.5 path; benchmark each backend rather than assuming parity with NV.
Tensor-core GEMM also needs an `(8,16,16)` FP16 tensor core, so older cards that pass the
`NV` check cannot use that multirow fast path.

**Ideas.**
- tinygrad has its own RDNA3/4 kernels (`amd_custom_kernels_supported` in
  `tinygrad/llm/kernels/amd.py`). AMD cards such as the RX 7600 may already be fast through
  upstream, which could become a documented second tier.
- The kernels are written in tinygrad's UOp kernel language, not raw CUDA, so they may port to
  AMD/Metal with modest changes.

**Check.** tok/s for `infurnace --benchmark` on each backend tried, recorded in the table.

**Done: `CUDA`.** The kernels are CUDA C either way, so only device checks kept them to `NV`; they now
use `device.nvidia()` (NV or CUDA) and the loader still requires the CUDA C renderer (tinygrad's PTX and
NVCC renderers can't compile their intrinsics), so `CUDA` needs NVRTC from the CUDA toolkit. RTX 5060,
same tokens on both backends:

| | Generic path (before) | `CUDA` | `NV` |
|---|---|---|---|
| Plain decode, `--benchmark 128,64` | did not finish in 4 min 40 s | 112–114 tok/s | 112–114 tok/s |
| DFlash code / prose (256 tokens) | — | 290 / 164 tok/s | 290 / 163 tok/s |
| DFlash warmup | — | 92 s | 72 s |

Kernel outputs are bit-identical across the two (`tests/kernels/test_nv_cuda_backend.py`); the fused
DFlash rounds work on `CUDA` without the `NV` doorbell fix. `NV` stays first in device selection on
Linux (faster graph capture); `CUDA` is what Windows NVIDIA users would get. Untested on Windows.

## 2. Plain decoding reads prompts one token at a time

- [x] Read prompts in 16-row chunks through a fused NV kernel (`gdn_prefill`)
- [x] Cut the extra warmup it adds: skip TinyJit's eager pass (45 s → 31 s)

**Problem.** Upstream `Transformer.generate` (`tinygrad/llm/model.py`) does:
```python
if self.has_recurrent_block and not amd_custom_kernels_supported(...): chunk_size = 1
```
Its multirow recurrent scan is a row-by-row loop unrolled into the graph, so off AMD it reads
prompts one token per step: `--no-dflash`, `--prompt` without a draft, and all of `--serve`.

**Done.** The kernel route, like upstream's AMD branch:
- `gdn_prefill` (`kernels/gdn.py`) is DFlash's closed-form 16-row chunk kernel with one addition:
  it also advances the recurrent state in place, since prefill keeps every row.
- `_fused_gdn` (`models/qwen35.py`) runs upstream's input processing for 16-row chunks, then that
  kernel instead of upstream's loop.
- `Qwen35Model._prefill` feeds all but the last prompt token through upstream's own forward in
  16-row chunks and records them as the cached prefix; upstream's `generate` resumes from there.
  It respects existing prefix reuse. `INFURNACE_MULTIROW_PREFILL=0` turns it off.

Measured on the RTX 5060, plain decoding, 4,096 context:

| Prompt | First token before | After |
|---|---|---|
| 100 tokens | 0.91 s | 0.19 s |
| 2,532 tokens | 26.9 s | 6.5 s |

Decode speed is unchanged. Output matched on both prompts and on 4 of 5 test prompts. The fifth
flips its first token on a near-tie ("1" vs "M", 0.04 logits apart): one-row projections quantize
activations to 8 bits, the 16-row GEMM multiplies in FP16, so logits agree to ~1%, not exactly.
Tests: `tests/dflash/test_nv_dflash_kernels.py` (kernel vs one-row scan, padding, reset) and
`tests/core/test_nv_qwen35_prefill.py` (logits tolerance, generate, prefix reuse).

**Warmup cost.** Capturing the 16-row graph made plain-decoding warmup 12 s → 45 s: the first
chunk ran 19 s eagerly, then 14 s to capture. `_prefill` now realizes every state buffer first
(`Qwen35Model.realize_state`, shared with DFlash) and starts the JIT at its capture pass, as
DFlash's `captured()` does: warmup is **31 s**, output unchanged. The remaining ~19 s over the
one-row baseline is the capture itself, scheduling and compiling the 16-row graph.

**Who waits for it.** A cold one-shot `--prompt` doesn't warm up, so the capture landed on the
request: a short prompt took 21 s → 40 s. Outside warmup, `_prefill` now captures the 16-row graph
only for prompts of `PREFILL_CAPTURE_TOKENS` (2,048) or more, where it pays for itself; shorter
cold prompts read one row at a time as before (23.7 s). Chat now warms up on a background thread
while the user types the first message (`cli.BackgroundWarmup`), and waits for it only if the
message arrives first. `--serve` still warms up before it listens.

## 3. DFlash chat re-reads the whole conversation every turn

- [x] Save the recurrent state after each prompt prefill and reuse it on the next turn

**Problem.** Recurrent (GDN) state can't be rewound, so every chat turn read the whole conversation
again: DFlash reset to position 0 on every request, and plain decoding reused its cache only when a
new prompt extended the previous one exactly, which the chat template breaks whenever thinking is on
(earlier replies are re-rendered without their thinking).

**Done: prompt snapshots** (`Qwen35Model.save_snapshot` / `restore_snapshot`), the way llama.cpp's
server (`--ctx-checkpoints`) and Ollama's MLX runner keep them:
- One snapshot per conversation, taken just before the prompt's last turn header (`<|im_start|>`;
  4 tokens from the end for a raw prompt). The next turn re-renders the previous reply without its
  thinking, so its prompt diverges right after that point.
- It holds the prompt tokens up to the point, copies of every GDN layer's recurrent and conv state
  (~48 MB), and for DFlash the draft's position and 16-row staging buffer. Attention KV caches need no
  copy: they are indexed by position, and later tokens only overwrite what follows the point.
- A request whose prompt still matches restores it and reads only the rest; one that would overwrite
  positions the snapshot relies on drops it. DFlash only uses snapshots its own draft saved.

Measured on the RTX 5060, three-turn chat, thinking on, ~3.1k-token first message:

| Time to first token | Turn 1 | Turn 2 | Turn 3 |
|---|---|---|---|
| DFlash, from scratch (before) | 5.8 s | 6.0 s | 6.4 s |
| DFlash, with snapshot | 5.8 s | **0.62 s** | **0.56 s** |
| Plain, from scratch (before) | 9.0 s | 9.8 s | 10.7 s |
| Plain, with snapshot | 8.9 s | **0.95 s** | **1.0 s** |

Replies identical either way. Tests: `tests/dflash/test_prompt_snapshot.py` (CPU: resume point, output
equal to reading from scratch, divergence drops the snapshot, DFlash with its draft).

**Then: several snapshots, in host memory.** One snapshot failed agents: during a tool loop it moves
with every call, so the next user message (whose prompt only matches up to the previous user message,
since Qwen's template strips the loop's thinking) read the whole conversation again. pi showed it as
`in: 0 + 1816` and `in: 0 + 3911`. Like llama.cpp (up to 32 checkpoints, always one at the last user
message), Ollama (a trie of snapshots, 8 GiB, LRU), ExLlamaV3 (4 GiB host LRU) and SGLang (radix nodes,
int8 option): now up to 8 snapshots (`SNAPSHOTS`), copied to host memory (~50 MB each, no VRAM), the
newest match restored, the ones a prompt diverges from dropped, and the oldest evicted except the
last user message's. RTX 5060, 32k context, a 10-step tool loop then a new user message: resumed at
1,043 instead of 0 (reading only the 1,485 re-rendered tokens); the next message read 62 tokens.

## 4. Small default context, and DFlash cost grows with it

- [x] Attend over the active length in the DFlash verifier and draft
- [x] Then raise the default `--max-context`: now 8,192 (about 5.3 GB with DFlash on the 5060)

**Problem.** The default is 2,048 tokens. A thinking model can use most of that on one answer,
and then chat starts dropping turns. Qwen3.5's attention cache is cheap: 8 attention layers ×
K/V × 4 heads × 256 × 2 bytes = **32 KiB per token**, so 16k tokens ≈ 0.5 GB. But DFlash attended
over the whole allocated window every round, so a larger `--max-context` slowed every round.

**Done: `kernels/attention.py`.** A port of vLLM's unified-attention design to tinygrad UOps: all
query rows of the heads sharing a KV head are packed into one block (each key and value read once
per KV head); each row has its own causal limit (plus sliding windows for the draft); 256-key
segments are processed in parallel, and only those holding active keys launch; a second kernel
merges them, after llama.cpp's `flash_attn_combine_results`. One captured graph serves every
length. Other devices fall back to the old masked tensor ops.

Why a custom kernel: tinygrad's own attention over a runtime length scales badly (16-row call at
16k keys: 81.6 ms with `start_pos+T`, 8.9 ms as 256-key blocks), while its fixed-length kernel
costs the same at any active length (1.5 ms at a 16k window).

Measured on the RTX 5060 (16-row call, Qwen3.5 shapes): ~0.22 ms up to 1k active keys, 0.40 ms at
2k, 0.61 ms at 4.3k, 0.92 ms at 8k, 1.68 ms at 16k. End to end, same output and acceptance:

| DFlash, code prompt | Before | After |
|---|---|---|
| `--max-context 2048` | 266 tok/s | 346 tok/s |
| `--max-context 16384` | 91 tok/s | 332 tok/s |
| 16k context, 4,319-token prompt | 94 tok/s, first token 15.7 s | 276 tok/s, first token 7.7 s |

Memory at 16k with DFlash: 5.1 GB allocated by tinygrad, 6.9 GB peak for the whole card with the
desktop running. Tests: `tests/kernels/test_nv_attention.py` (against plain softmax, across segment
boundaries, causal/non-causal/sliding, one graph at every length).

**Plain decoding too.** One-row decode attention now goes through the same kernel (`_fused_attention`
in `models/qwen35.py`, after upstream's AMD branch), with 32-key segments and queries per warp sized
to the call. Same output tokens; 100-token prompt 107.5 → 108.9 tok/s, 2,532-token prompt
73.7 → 95.5 tok/s. Accumulating in FP32 instead of FP16 flips some measured near-ties (0.006–0.08
logits) between paths; the real-model tests now compare prefixes before them.

**Default context: 8,192.** At the default, infurnace used about 5.3 GB with DFlash (whole card
7.46 GB with 2.1 GB of desktop apps). 16k adds roughly half a GB of cache: it fits an 8 GB card only
if other apps leave room, so it is not the default.

**Measured limit, then a smaller draft cache.** Filling the context to the end on the RTX 5060 with
a desktop running (1.9–2.2 GB of other apps): 24k, 28k and 32k all fit with DFlash, 32k peaking at
7,721 of 8,151 MiB. Per token, the target's cache costs 32 KiB and the draft's 24 KiB, but five of
the draft's six layers only look back 4,096 tokens. Those now keep a ring of window + 2 blocks
(4,128) slots, position x at slot x % 4,128 (`store_rows` writes it, `attention(ring=True)` reads
it), and their attention skips segments before the window. Restoring an older prompt snapshot after
the ring lapped it masks the overwritten slots with a floor buffer (keys below it are hidden) instead
of copying the draft cache into every snapshot; output is exact either way, only proposals for the
next ~4k tokens see less history. At 32k: about 520 MB less once ready, peak 7,721 → 7,346 MiB.
Same output and speed on the pi tasks (4.3 s and 9.4 s). Tests: ring vs full cache in
`tests/kernels/test_nv_attention.py` and `tests/dflash/test_dflash.py` (60 positions through a
16-slot ring), the floor after a lapped restore.

## 5. Greedy decoding only

- [x] Add `--temperature` for plain decoding
- [x] Speculative sampling so DFlash can sample too

**Problem.** Everything decoded greedily: no temperature option, and `DFlashGenerator.stream` rejected
any temperature but 0. Qwen advises against greedy decoding in thinking mode because it tends to
repeat itself.

**Done.** `--temperature T` (default 0, greedy), per request. Plain decoding passes it to upstream's
sampler. DFlash needed no full rejection sampling: the draft proposes greedily, so its distribution is
one-hot, and for a one-hot draft speculative sampling reduces to sampling the target's token at each
verified row and accepting the proposal when the sample equals it. That accepts with probability p(x)
(the optimal rate), and a rejecting row's sample is already distributed as the target excluding x,
the residual. Host acceptance logic is unchanged.

**Fused sampler** (`kernels/sampling.py`). The verifier samples every row with one kernel pass:
score = logit + T·g, g = -ln(-ln u), u from tinygrad's threefry hash of (key, step, element), so the
noise never leaves registers; a second launch combines per-chunk winners and advances `step`. At
T = 0 the score is the logit, so greedy is exactly argmax (first maximum on ties) and one captured
graph serves every temperature; each request writes its temperature into the buffer the graph reads.
16 rows × 248,320 logits: 127 µs, against 162 µs for argmax alone and 1,094 µs for tensor-op Gumbel
sampling. (Lesson: gate loads with clamped indices, not lane-dependent `valid`, or tinygrad can shrink
a warp the shuffles need; that produced token 0 for rows spanning 2–31 chunks.) Surveyed for code to
port: exllamav3's `gumbel_sample_kernel` (one block per row, cuRAND Philox) is the closest; llama.cpp,
vLLM and sglang sample with separate noise passes. Kept the split-row design for parallelism.

Checks: `tests/kernels/test_nv_sampling.py` (T = 0 equals argmax at uneven chunk counts, chi-square
against softmax inside a captured graph, fresh noise per replay); `tests/dflash/test_sampling.py`
(tiny CPU model, seeded: first-token chi-square, sequence distribution at 98% acceptance, one generator
serving every temperature with greedy staying exact).

Measured on the RTX 5060, thinking on, 256 tokens, T = 0.6: DFlash 186 / 176 / 219 tok/s against plain
110 (1.70× history prose, 1.60× explanation, 2.00× code; 1.26 / 1.51 / 1.84× with tensor-op sampling).
Greedy DFlash unchanged within noise (343 tok/s on the code prompt; 332–357 across runs).

**Open: default.** Chat still defaults to greedy; Qwen's thinking-mode setting would be 0.6 (plus
top-p 0.95 / top-k 20, which upstream's sampler doesn't have).

**Serving uses it.** `--serve` now generates with DFlash (one generator, every request temperature)
and honors a per-request thinking switch (`chat_template_kwargs.enable_thinking`). Pi on the RTX 5060:
read task 5.1 → 4.1 s, read-and-fix 11.5 → 8.9 s; see [serve.md](serve.md).

## 6. Install and getting models

- [x] Make `tokenizers` a base dependency (the default model needs it): `pip install -e .` is all
- [x] Add a model download path: `--model NAME` downloads on first use via `tinygrad.helpers.fetch`, tinygrad-style
- [x] Say in the README where the DFlash draft GGUF comes from

**Problem.** A plain `pip install -e .` lacks the `qwen35` extra, so bare `infurnace` stops with
"install infurnace[qwen35]". Named models now download on first use; tokenizer installation
is the remaining extra step.

**Check.** On a clean machine, `pip install -e .` then `infurnace` reaches a chat prompt with no
other steps.

## 7. About a minute per launch

- [ ] Use one denoise graph padded to the full block instead of one per proposal count
- [ ] Time warmup before and after

**Problem.** Every launch spends ~18 s loading and 37–45 s on DFlash warmup. `_denoise_graph(count)`
(`dflash/draft.py`) captures a separate graph for each proposal count from 1 to 15, and warmup is
mostly capture time. Always padding to 15 proposals and slicing the result leaves one graph.

**Check.** The "Ready in N s" line, before and after. Output must not change.

## 8. Minimum GPU memory is unknown

- [ ] Measure peak memory for chat with and without DFlash at a few context sizes
- [ ] Document the minimum card size
- [ ] Turn DFlash off automatically when memory is tight

**Problem.** The weights are 2.74 GB (target) + 0.38 GB (draft), so 4–6 GB cards should fit, and
that's exactly the GPU-poor range. Peak memory has never been measured.

## 9. Check whether DFlash-by-default slows ordinary chat

- [x] Same prose prompts with and without DFlash, matched length
- [x] Time a fallback step against a plain step
- [ ] Make probing cheaper when drafting keeps losing (thinking-off prose)

**Measured** (RTX 5060, 2026-10-12): warm runs, 256 tokens each with EOS ignored, DFlash block 16.

| Prompt | Thinking on (default chat) | Thinking off (`--no-thinking`) |
|---|---|---|
| Prose: history of the printing press | 1.29× (135 vs 105 tok/s) | **0.79×** (83 vs 105) |
| Prose: making friends in a new city | 1.61× | **0.84×** |
| Prose: why the sky is blue | 1.39× | **0.85×** |
| Code: merge two sorted lists | 1.73× | 2.78× |

**Default chat is fine:** with thinking on, DFlash beats plain decoding on prose as well as code;
reasoning text is predictable. The earlier ~88 tok/s prose figure came from a thinking-off run.

**Thinking-off prose loses 15–21%.** Not the fallback step: inside DFlash it costs the same as a
plain step (9.1–9.3 ms vs 9.3–9.4 ms). The loss is the probing. In the history prompt, about 63 of
190 rounds were drafting rounds, averaging ~1 accepted token each at roughly three steps' cost,
and they produced about half the output. `_DraftPolicy` spends at least `min_timed_rounds` (4)
timed rounds per probe before it can back off, and its cooldown tops out at 8 × 16 = 128 steps.

**Ideas.** Decide sooner when a probe accepts almost nothing (e.g. back off after the first timed
round with ~0 accepted), or allow longer backoff. Check that a change doesn't cost the thinking-on
and code cases above.

## Code simplification

- [x] Limit the runner to Qwen3.5-4B and DFlash; move shared `_NVLinear` into
  `models/linear.py`, remove unused model-family attention adapters and loader options.
- [x] Rewrite `docs/roadmap.md` around the current target and validation gates.
- [ ] Add direct `_DraftPolicy` tests with fake timings (`dflash/generate.py`).
