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

## 2. Plain decoding reads prompts one token at a time

- [x] Read prompts in 16-row chunks through a fused NV kernel (`gdn_prefill`)
- [ ] Cut the extra warmup it adds (see below)

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

**Open: warmup cost.** Plain-decoding warmup went from 12 s to 45 s. The first 16-row chunk
costs 19 s eagerly plus 14 s to capture; replays are free. Realizing the KV caches first and
skipping the eager pass (as DFlash's `captured()` does) should save part of that, unmeasured.

## 3. DFlash chat re-reads the whole conversation every turn

- [ ] Save the recurrent state after each prompt prefill and reuse it on the next turn

**Problem.** `TargetObserver.__enter__` (`dflash/target.py`) clears `_cached_tokens` and resets
the position to 0, so every chat turn re-prefills the entire history. Replies get slower to start
as the conversation grows.

**Fix.** The hard part is the GDN recurrent state, which can't be rewound to an earlier position.
It's small: 24 GDN layers × 32 heads × 128×128 × 4 bytes ≈ **48 MB**. Save it, plus the conv
state and draft state, at the end of each prompt prefill. On the next turn, if the new prompt
extends the saved one, restore it and prefill only the new tokens. The chat template drops
earlier assistant reasoning, so the reusable prefix ends where the last reply starts.

**Check.** Time to first token on turn 5 of a chat, before and after. Output must equal a fresh
run on the same full prompt.

## 4. Small default context, and DFlash cost grows with it

- [ ] Attend over the active length (rounded up to a few fixed sizes) in the DFlash verifier and draft
- [ ] Then raise the default `--max-context` to 8–16k

**Problem.** The default is 2,048 tokens. A thinking model can use most of that on one answer,
and then chat starts dropping turns. Qwen3.5's attention cache is cheap: 8 attention layers ×
K/V × 4 heads × 256 × 2 bytes = **32 KiB per token**, so 16k tokens ≈ 0.5 GB.

The catch: DFlash attends over the **whole window every round, with a mask**, not only the tokens
in use:
- `ParallelVerifier._attention` (`dflash/parallel.py`) uses `length = c.max_context`;
- `DFlashDraft._denoise` (`dflash/draft.py`) attends over `cache_length`.

So a larger `--max-context` slows every DFlash round, even early in a conversation.

**Check.** DFlash tok/s at `--max-context 2048` and `16384` on the same short chat should be
about equal after the fix.

## 5. Greedy decoding only

- [ ] Add `--temperature` for plain decoding
- [ ] Later: speculative rejection sampling so DFlash can sample too

**Problem.** `cli.generator` calls `model.generate(ids, budget)` at temperature 0, the CLI has no
temperature option, and `DFlashGenerator.stream` rejects any temperature but 0. Qwen advises
against greedy decoding in thinking mode because it tends to repeat itself.

**Check.** Long thinking answers stop looping. Upstream `generate` already takes a temperature.

## 6. Install and getting models

- [ ] Make `tokenizers` a base dependency (the default model needs it)
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

- [ ] Same prose prompt with and without `--no-dflash`, matched length
- [ ] Time a fallback step against a plain step

**Problem.** A 192-token prose run measured ~88 tok/s with DFlash, against the ~110 the README
quotes for plain decoding. Not a matched comparison, but if it holds, the default makes ordinary
English chat slower. A likely cause: fallback steps still feed every target row's features to the
draft (`draft.observe`).

## Code simplification

- [x] Limit the runner to Qwen3.5-4B and DFlash; move shared `_NVLinear` into
  `models/linear.py`, remove unused model-family attention adapters and loader options.
- [x] Rewrite `docs/roadmap.md` around the current target and validation gates.
- [ ] Add direct `_DraftPolicy` tests with fake timings (`dflash/generate.py`).
