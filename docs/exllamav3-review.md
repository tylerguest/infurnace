# What infurnace can take from ExLlamaV3

A review of [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (local checkout at commit
`3a513ac`) for ideas that fit infurnace, 2026-10-12. Written for infurnace's scope:
Qwen3.5-4B Q4_K_M plus its DFlash draft, one user, tinygrad, tuned for **specific GPUs**
(the RTX 5060 8 GB first), not broad GPU families.

The rule throughout: borrow execution strategies, not ExLlamaV3's architecture (model registry,
scheduler, paged cache, CUDA/Triton extension). ExLlamaV3 is MIT-licensed; anything copied
needs attribution, and vendored parts (e.g. its `vendor/fla`) need their own license checks.

## Summary

| # | Idea | ExLlamaV3 source | infurnace today | Verdict |
|---|---|---|---|---|
| 1 | Measured, cached per-GPU kernel tuning | `exllamav3_ext/quant/coop_autotune.cu` | Hand-written constants; custom kernels skip tinygrad's BEAM | **Adopt, as checked-in per-GPU tables** |
| 2 | Launch geometry derived from SM count | `modules/attention_fn/bc_attn.py`, `exllamav3_ext/quant/exl3_kernel_map.cu` | "~64 blocks" hard-coded for the 5060 | **Adopt with #1** |
| 3 | Shared activation quantization / bundled projections | `modules/gated_delta_net.py` (sliced qkv+z bundle) | Kernel supports a shared `q8`; Qwen3.5 never passes it | **Adopt first: cheap** |
| 4 | Attention over the active length, split by SM count | `modules/attention_fn/bc_attn.py` | **Done** (`kernels/attention.py`, after vLLM's unified attention) | Split sizes left, under #1 |
| 5 | Bounded multirow recurrent prefill | `modules/gated_delta_net_fn/gated_delta_rule.py` | **Done** (`gdn_prefill`, 16-row chunks) | Only chunk size left, under #1 |
| 6 | Paired target/draft state for prompt reuse | `cache/recurrent.py` (`PairedState`) | DFlash re-prefills every turn | Adopt later, minimal version |
| 7 | Confidence-calibrated draft windows | `generator/draft_confidence.py` | `_DraftPolicy` decides by wall time | Defer: fixed-width verify makes it pointless |

## 1. Measured, cached per-GPU kernel tuning

**ExLlamaV3.** `coop_autotune.cu` keeps a list of launch candidates per kernel (block size, number
of SMs used, concurrency). The first time a shape appears it times every candidate on the actual
GPU, keeps the fastest, and stores the winner in `~/.cache/exllamav3/autotune/coop_autotune_v1.bin`
(`EXLLAMAV3_TUNE_CACHE` overrides), keyed by a hash of kernel and shape.

**infurnace.** No tuning mechanism. The custom UOp kernels set `opts_to_apply=()`, so tinygrad's
BEAM search never touches them, and their parameters are constants chosen by hand:

- `quant_gemm._schedule`: "tuned on an RTX 5060 … split K only until there are ~64 blocks".
- `quant_gemv._decode_warps`: warps per output block.
- `PREFILL_ROWS = 16` (`models/qwen35.py`) and the DFlash verify width.

**Some of that tuning is already stale.** `_decode_warps` contains:
```python
if ggml_type == Q6_K and (out_features, in_features) == (151936, 2560): return 8
```
151936 is **Qwen3's** vocabulary. Qwen3.5's output head is wider (a ~248k vocabulary), so this
5060 tuning no longer applies to the only model infurnace runs, and the output head is one of the
largest weights read every token.

**Plan, adapted to the "specific GPUs" goal.** Rather than tuning at runtime, tune offline once
per GPU you own and check the result in:
- `benchmarks/tune.py` times candidate parameters for each kernel and shape the model uses,
  measured inside whole-layer runs, not isolated microbenchmarks.
- It writes `infurnace/tuning/<gpu>.json` (e.g. keyed by `Device['NV'].arch`, `sm_120` on the 5060),
  mapping kernel + shape + quant type + row count → parameters.
- Kernels look parameters up there; an unknown GPU falls back to today's constants (or #2's
  SM-derived defaults).

Profiles stay explicit and reviewable in git, startup does no tuning, and adding a GPU means
running the script on it and committing a file.

**Measure.** Decode tok/s and prefill tok/s per profile; each tuned value must win end to end,
not only in its own kernel.

## 2. Launch geometry derived from SM count

**ExLlamaV3.** Split counts come from the GPU's streaming-multiprocessor count: split-decode
attention targets `2 * sm_count` programs (`bc_attn.py`, `target = 2 * _get_sm_count(dev)`), and
the GEMM caps slices per SM (`exl3_kernel_map.cu`).

**infurnace.** `quant_gemm._schedule`'s "~64 blocks" is the same rule written as a constant for
one card. Expressing it as "about two blocks per SM", read from tinygrad's device properties,
records why the number is what it is and gives a reasonable default on a card without a #1
profile. Small change; do it together with #1.

## 3. Shared activation quantization and bundled projections

**ExLlamaV3.** For Qwen3.5-style GDN at decode it bundles the qkv and z projections into one
multi-matrix GEMV over column slices (`gated_delta_net.py`, `SlicedMultiLinear`; `EXL3_QKV_SLICE=0`
disables it): projections that share an input are processed together.

**infurnace.** The GEMV already accepts a shared quantized activation
(`_NVLinear.decode(x, q8)`, `quant_gemv(..., q8=)`), but **nothing in the Qwen3.5 path passes
it**. The code that did lived in the Qwen3 attention block and was removed with Qwen3 support. So
every projection quantizes its input on its own, including the ones that read the same `x`:
- GDN: `attn_qkv` and `attn_gate` (plus the small `ssm_beta`/`ssm_alpha`);
- MLP: `ffn_gate` and `ffn_up`.

**Plan.**
1. Quantize `x` once per group and pass it to each projection (recovers what Qwen3 had).
2. Then try true bundling (one launch for gate+up, one for qkv+gate), keeping each projection's
   quant type and output semantics.

**Caution.** Fewer launches can lose to register pressure; require a whole-layer and
whole-model win.

## 4. Attention over the active length (done)

**ExLlamaV3.** `bc_attn.py` separates fixed allocations from runtime lengths: the live split count
and split length are kernel arguments derived from the active sequence, with the number of splits
tied to the SM count.

**infurnace before.** DFlash attended over the whole allocated window with a mask
(`ParallelVerifier._attention` used `length = c.max_context`; `DFlashDraft._denoise` attended over
`cache_length`), so a larger `--max-context` slowed every round. tinygrad's own attention over a
runtime length was no fix: it scales badly (81.6 ms for a 16-row call at 16k keys).

**Done (2026-10-12).** `kernels/attention.py`, a port of vLLM's unified-attention design
(`triton_unified_attention.py`) rather than ExLlamaV3's paged kernels: rows × grouped heads packed
per KV head, per-row causal limits and sliding windows, fixed-size key segments where only those
holding active keys launch, then a merge after llama.cpp's `flash_attn_combine_results`. One
captured graph serves every length. It serves the DFlash verifier and draft (256-key segments) and
plain one-row decoding (32-key segments, queries per warp sized to the call). Measured in
`docs/gpu-poor-plan.md` item 4: DFlash at a 16k window 91 → 332 tok/s; plain decoding at 2.5k
tokens 74 → 96 tok/s; default context raised to 8,192.

**Left.** Segment sizes are hand-picked for the 5060; they belong in #1's tuning tables, with #2's
SM-count rule as the default for other cards.

## 5. Bounded multirow recurrent prefill (done)

**ExLlamaV3.** `gated_delta_rule.py` runs long prefill through its chunk kernels in sub-ranges
(`EXL3_GDN_SUB_CHUNK`, default 2048), carrying FP32 state between them, so intermediate storage
scales with the sub-range rather than the prompt. The 2048 default was measured on a 5090.

**infurnace.** Implemented 2026-10-12, see `docs/gpu-poor-plan.md` item 2: `gdn_prefill`
(`kernels/gdn.py`) processes 16-row chunks with FP32 state carried in place, so memory is bounded
by construction; ExLlamaV3's sub-range limit addresses a problem infurnace doesn't have.
2,532-token prompt: first token 26.9 s → 6.4 s.

**Left.** The chunk size (16) is a tunable for #1. Larger chunks would need the kernel extended
beyond its 16-row specialization.

## 6. Paired target/draft state for prompt reuse

**ExLlamaV3.** `cache/recurrent.py` defines `PairedState`: one sequence's recurrent state across
the target and the draft, restored or evicted together. Verification advances only the target
side; a DFlash-family drafter's side is advanced by the rows it writes.

**infurnace.** `TargetObserver.__enter__` (`dflash/target.py`) resets the position and cached
tokens every request, so DFlash chat re-prefills the whole conversation each turn (plan item 3).

**Plan.** One reusable committed endpoint, not an LRU cache:
- associate an exact token prefix with the target's recurrent/conv state, KV validity and the
  draft's absorbed features/cache;
- resume only when every component agrees;
- invalidate on divergence, context rollover, reset or changed settings.

**Measure.** Second-turn time to first token, and correctness after interrupted generation or
partial acceptance.

## 7. Confidence-calibrated draft windows (defer)

**ExLlamaV3.** `draft_confidence.py` learns a mapping from draft confidence to observed acceptance
(decayed bins: `bin_width=1.0`, `decay=0.995`, `min_count=8`, `burn_in=64`) and truncates draft
blocks where estimated acceptance falls below a target.

**infurnace.** `_DraftPolicy` already decides by measured wall time whether drafting pays at all;
keep that. Shorter proposals save nothing while the captured verifier always runs a full 16-row
block, so this only becomes useful with several verify widths (which #1 could tune).

## What infurnace already does well

- **Speculative GDN state.** Like ExLlamaV3, infurnace's verifier leaves recurrent state untouched
  and `commit` reconstructs the accepted prefix from stored intermediates, instead of snapshotting
  state per token. Validation of the current design, not something to replace.
- **Graph capture.** ExLlamaV3 has its own CUDA graph layer (`graph.cu`); tinygrad's JIT already
  does this. Don't duplicate it.

## Not applicable

- **EXL3/QTIP weights:** a new format, decoder, layout and conversion pipeline. Stay on GGUF Q4_K_M.
- **`dflash2.cu`:** kernels for a different draft architecture ("DFlash2", dynamic convolution),
  not infurnace's DFlash draft.
- **Quantized KV cache:** only if KV memory or bandwidth is measured to be the limit.
- **Dynamic batching, paged scheduling, tensor parallelism, MoE offload, broad architecture
  dispatch:** outside the single-user scope.
- **Direct CUDA/Triton ports:** borrow algorithms and layouts, expressed as tinygrad UOps.

## Order of work

1. **Share Q8 activation quantization** in Qwen3.5's GDN inputs and MLP gate/up (#3). Small,
   immediately measurable.
2. **Re-tune the Q6 output head for Qwen3.5's shape** (#1's stale rule). One manual measurement.
3. **Tuning tables and script, SM-derived defaults** (#1 + #2): replace hand constants (GEMV warps,
   GEMM tiles/splits, prefill chunk) with measured per-GPU values.
4. ~~**Active-length DFlash attention** (#4), then raise the default context.~~ Done 2026-10-12.
5. **Paired prompt-state reuse for DFlash chat** (#6).
6. **Confidence windows** (#7), only once verification has cheaper narrow widths.

Each change needs a correctness check (logits or tokens), an end-to-end speed gain, and a memory
check on the 8 GB card. Add another GPU only as its own measured profile, never as a broad
compatibility layer.
