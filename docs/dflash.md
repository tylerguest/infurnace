# Tinygrad DFlash reference

`infurnace/dflash/` implements our own block-diffusion draft and greedy sequential
verification in tinygrad. The target is the **unmodified upstream tinygrad**
Qwen3.5 GGUF model. No PyTorch or llama.cpp model runtime is used.

`infurnace` alone uses DFlash when the target is Qwen3.5 on NV and the draft
(`qwen35-4b-dflash-Q4_K_M.gguf`) is installed; `--no-dflash` turns it off. Explicitly:

```sh
infurnace --model models/qwen35-validation/Qwen3.5-4B-Q4_K_M.gguf \
  --dflash models/dflash-qwen35/qwen35-4b-dflash-Q4_K_M.gguf \
  --prompt 'Write a Python function that returns the n-th Fibonacci number.' --max-tokens 64
```

## Parallel verification (default on NV)

Parallel verification (`infurnace/dflash/parallel.py`) is used on NV,
with the checkpoint's trained block size (16). Each round runs **one** multirow target
forward over the anchor plus all proposals, so projection weights are read once per
block via Infurnace's packed tensor-core GEMMs (Q4_K/Q5_K/Q6_K: each warp reuses
one activation fragment across four 8-row output tiles, with split-K only for narrow
shapes). Prompt prefill uses the same forward in 16-row chunks.

- Gated delta-net layers use the chunked (WY/UT) closed form of the gated delta rule
  instead of a per-row scan; `commit` forms the accepted-prefix state directly from
  stored U, K and log-decays. No per-row state snapshots are kept. On NV the chunk,
  RMSNorm and output gate run as one kernel per layer (`kernels/gdn.py`: one block
  per head, one thread per value row, forward substitution in registers).
- Packed GEMMs permute the k order so each lane's tensor-core slots map to
  contiguous packed bytes (`_prefill_q4_paired` for Q4_K/Q5_K, `_prefill_q6_paired`
  for prepared Q6_K): one 8/16-byte load fills four k-steps and every nibble is
  read once. The reduction order changes, not the products.
- Full-attention layers attend over the fixed-length cache with a causal mask
  (symbolic key lengths compiled to slow kernels). Rejected suffix K/V stays outside
  the authorized prefix and is overwritten later.
- Greedy decisions are argmax over multirow logits. They can differ from one-row
  decoding only through floating-point reduction order; no RNG is consumed.

The draft runs as two fixed-shape captures: `absorb` writes context K/V in 16-row
chunks, and `denoise` predicts the block, so neither recompiles per prompt length.

Measured on an RTX 5060 (8 GB), Qwen3.5-4B Q4_K_M target, Q4_K_M DFlash draft, block 16,
fused rounds (the default; `INFURNACE_DFLASH_FUSED=0` disables them), chat-formatted prompts with thinking off,
512 output tokens, **decode only** (prompt prefill excluded: plain decoding prefills
one row at a time, DFlash 16), warm, median of the last two of five requests:

| Prompt | Plain | Always draft | Adaptive (default) | Tokens/round (always draft) |
|---|---|---|---|---|
| Red-black tree with tests (code) | 84.5 tok/s | 124.5 tok/s | **123.3 tok/s** (1.46×) | 4.65 |
| Printing press essay (English) | 88.8 tok/s | 68.9 tok/s | **84.0 tok/s** (0.95×) | 2.43 |

Short, highly predictable prompts do much better (a 128-token Fibonacci answer
drafts ~9.9 tokens/round), and open-ended prose does worse; the adaptive policy keeps
the loss on unpredictable text to the cost of its periodic probes (~5%). The desktop
GPU's clocks vary by a few percent between runs.

Each fused round is a chain of small captured graphs launched back to back with one
readback: commit the previously verified rows, absorb their features into the draft,
select the anchor on device, draft, gather the block's embeddings from the packed
head, run the 16-row target forward, and pack decisions and proposals. Back-to-back
submissions need the pinned tinygrad fork, which reads GPPut back before ringing the
GPFIFO doorbell; without it the GPU can miss the last submission and the readback
never completes. A round takes about 35 ms.

\* Multirow and one-row kernels reduce in different orders; near-ties can flip a
greedy choice. Treat outputs as equivalent up to floating-point ties, not bitwise
identical to one-row decoding.

**Adaptive fallback.** The generator compares wall time per emitted token over the last
~20 parallel rounds (a ratio of running sums, so a short dip cannot decide; at least 4
timed rounds after each probe) with the measured time of a one-row step. When drafting loses it decodes ordinary rows for 8
steps, doubling up to 128 while probes keep losing. Samples taken while a graph is
compiling or capturing are not timed; until a one-row step has been timed, a
tokens-per-round threshold (3.5) decides. During fallback the target's feature rows
are copied into the draft's fixed-shape 16-row staging buffer as they are produced
and absorbed in full chunks, so returning to drafting never recompiles. Greedy only.

**Startup and graph cache.** Chat warms DFlash's captured graphs before the first
prompt (~45 s). With `INFURNACE_GRAPH_CACHE=1` (opt-in: an entry has been seen to hang
its first replay in a later process; under investigation) they are saved to
`~/.cache/infurnace/dflash-graphs/` (`XDG_CACHE_HOME` respected, ~130 MB, newest four
kept), and later launches load them in ~7 s. Weights and caches are not written:
buffers owned by the model, draft and verifier are stored by name and re-bound on load.
Any change to tinygrad or infurnace sources, either GGUF file, the context length,
block size, fused rounds, device or compile flags selects a new entry.
The target's small prepared
parameters (norms, gates; ~10 MB that otherwise take ~2 s of kernel scheduling) are
cached alongside in `~/.cache/infurnace/params/`; `INFURNACE_PARAM_CACHE=0` disables
that. With both caches warm, chat is ready ~16 s after launch; without the graph cache, ~55 s.

## Draft identity and metadata

- Repository: [Anbeeld/Qwen3.5-4B-DFlash-GGUF](https://huggingface.co/Anbeeld/Qwen3.5-4B-DFlash-GGUF)
- Revision: `a79bafa7f383f8627f925250f547fe1ebbc43ff6`
- File: `qwen35-4b-dflash-Q4_K_M.gguf`, 381,456,064 bytes
- Verified SHA-256: `83586a63eab58b4ef9dd1d72de667738fcedb289e97d98d01d8df66f23df610f`

The GGUF, **not its stale repository config.json**, defines this draft: six
layers, hidden2560/FFN9216, 32 query/8 KV heads, head128, mask248077, block16,
five sliding layers followed by a full-attention layer. Feature taps
`[2,6,10,14,18,22,26,30]` mean HF hidden_states[N] / before target block N.
Vocabulary identity and tensor names/shapes are checked before draft uploads.

Attention defaults now match the audited current official repository: sliding
layers are causal; the final full-attention layer is noncausal. An explicit GGUF
`dflash.attention.causal` overrides all layers. The selected GGUF has no such
flag and does not identify its inference-code version. Older official/converter
code defaulted every layer to noncausal; `DFlashDraft.from_gguf(..., legacy_noncausal=True)`
reproduces that policy. No weights or target math are substituted.

See the [pinned upstream audit](../benchmarks/experiments/dflash/upstream-audit.md)
for the version difference and operation-by-operation comparison. Transient
noise K/V is overwritten by actual target features and never authorizes prefix reuse.

Draft weights/projections, context fusion/norm, RoPE, attention, MLP, output
head and argmax are tinygrad operations. Packed NV draft projections reuse
Infurnace's tinygrad UOp kernels; they do not alter target projections.

## Validation and remaining work

Synthetic tests cover fully accepted/rejected blocks, feature alignment and
snapshot lifetime, output budgets/context limits, cancellation, EOS, reset/JIT
replay, and subsequent RNG draws. Uniform-attention scalar controls exercise
causal/noncausal and full/sliding masks, including stale future cache rows.
CPU/NV draft hidden-state comparisons also passed. The expanded focused suite
completed with 114 CPU tests and five NV tests after the upstream audit and
attention-policy correction. A new independent scalar oracle also checks a
nonuniform full draft forward, RoPE/GQA, incremental caches and proposal head. CPU block controls check bit-exact
recurrent/conv states, authorized KV prefixes, features and RNG for every prefix,
future ordinary decoding after rollback, repeated capture replay, injected
failure cleanup, cancellation mid-block and EOS inside an accepted block. NV
controls check exact output/RNG and repeated sessions.

A completed real upstream-target protocol (context64/output8/block4, two warmup
pairs/two repeats) matched output and RNG throughout. It was slower than baseline
and used the former all-noncausal draft policy. It is not a measurement of the
corrected layer defaults.
See [reports](../benchmarks/experiments/dflash/README.md). Old reports used a
superseded local target and must not be attributed to the upstream target.

Next: resolve full-model capture/reliability problems, broader correctness and
block16 checks, efficient drafting, and genuinely parallel target projections
without changing target decisions. Stochastic verification is not implemented;
full-block performance is not validated.
No general speedup, 200 tok/s, or NV reliability-fix claim is made.
