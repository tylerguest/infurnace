# DFlash speculative decoding

`infurnace/dflash/` implements a [DFlash](https://github.com/z-lab/dflash) block-diffusion draft and
its verification in tinygrad. The draft proposes 15 tokens at once from the target's hidden features;
the target checks them in one 16-row forward and keeps the longest prefix it agrees with. Output is
the target's own: greedy output is the target's greedy choice, and sampled output follows the target's
distribution.

It turns on automatically for the named model on an NVIDIA GPU (`NV` or `CUDA`) when the draft is
available, and downloads it on first use. `--no-dflash` turns it off; `--dflash PATH` or
`INFURNACE_DFLASH_DRAFT` picks another draft file.

## A round

The parallel verifier (`dflash/parallel.py`) runs each round as a chain of captured graphs with one
readback at the end:

1. Commit the previous round's accepted rows: the recurrent layers' state is formed for exactly the
   accepted prefix from stored intermediates, with no per-row state snapshots.
2. Absorb those rows' target features into the draft's cache.
3. Draft 15 proposals after the anchor token.
4. Run the 16-row target forward: `quant_gemm` projections, `gdn_chunk` for the recurrent layers, the
   active-length `attention` kernel, and the fused sampler for every row's decision.
5. Pack the decisions and proposals into one readback.

The rejected suffix's attention rows are simply overwritten later.

On `NV`, back-to-back submissions need the pinned tinygrad fork, which reads GPPut back before ringing
the doorbell; without it the GPU can miss the last submission and the readback never completes. `CUDA`
doesn't need the fix.

On the RTX 5060 a round takes ~23 ms: target verify 15.8 ms (its matmuls 11 ms), draft 4.1 ms,
the rest ~1.6 ms. CPU and other devices use the sequential verifier, a one-row reference that checks
4 rows per round.

## Sampling

At temperature T the target scores each row by logit + T·Gumbel noise and takes the maximum (at T = 0,
plain argmax). The draft proposes greedily, so it is a one-hot distribution, and accepting a proposal
exactly when the target's own sample equals it is standard speculative sampling. One captured graph
serves every temperature: each request writes its temperature into a buffer the graph reads.

## Adaptive drafting

Drafting loses on unpredictable text. `_DraftPolicy` (`dflash/generate.py`) compares the wall time per
emitted token over the last ~20 rounds (running averages, at least 4 timed rounds after each probe)
with a measured one-row step. When drafting loses it decodes plain rows for 8 steps, doubling up to 128
while probes keep losing; until a one-row step has been timed, it falls back below 3.5 tokens per round.
During fallback the target's features keep flowing into the draft, so drafting can resume at once.

## Draft caches

The draft's last layer attends over the whole context. Its five sliding-window layers see only the
last 4,096 positions, so they keep a ring of 4,128 slots (position x at slot x % 4,128) instead of one
slot per context position: ~520 MB less at a 32k context. Restoring an older prompt snapshot after the
ring has lapped it hides the overwritten slots with a floor value instead of copying the draft cache
into every snapshot; proposals then see less history for a while, and output is unaffected.

## Measurements

RTX 5060, Qwen3.5-4B Q4_K_M with the Q4_K_M draft, block 16, October 2026:

| Workload | Plain | DFlash |
|---|---|---|
| Code, greedy, thinking off | ~110 tok/s | ~340 tok/s steady (7.7–8.3 tokens per round) |
| Prose, thinking on, greedy | ~110 tok/s | ~160 tok/s (4 tokens per round) |
| Temperature 0.6, thinking on | 110 tok/s | 1.6–2.0× plain |
| Prose, thinking off | 105 tok/s | 0.79–0.85× plain (probing cost; [roadmap](roadmap.md)) |
| Code at a 16k-token window | — | 332 tok/s |

Acceptance depends heavily on content: predictable code drafts 8–10 tokens per round, chatty thinking 3–5.

## Draft identity

- Repository: [Anbeeld/Qwen3.5-4B-DFlash-GGUF](https://huggingface.co/Anbeeld/Qwen3.5-4B-DFlash-GGUF)
- Revision: `a79bafa7f383f8627f925250f547fe1ebbc43ff6`
- File: `qwen35-4b-dflash-Q4_K_M.gguf`, 381,456,064 bytes
- SHA-256: `83586a63eab58b4ef9dd1d72de667738fcedb289e97d98d01d8df66f23df610f`

The GGUF, not the repository's stale `config.json`, defines the draft: six layers, hidden 2560,
FFN 9216, 32 query and 8 KV heads of 128, mask token 248077, block 16, five sliding-window layers then
a full-attention layer, feature taps `[2, 6, 10, 14, 18, 22, 26, 30]` (HF `hidden_states[N]`, before
target block N). Vocabulary identity and tensor shapes are checked before upload.

Attention follows the current official repository: sliding layers causal, the final full layer
non-causal. A GGUF `dflash.attention.causal` flag overrides all layers;
`DFlashDraft.from_gguf(..., legacy_noncausal=True)` reproduces older converters' all-non-causal policy.
See the [upstream audit](../benchmarks/experiments/dflash/upstream-audit.md) and the historical
[experiment reports](../benchmarks/experiments/dflash/README.md).

## Tests

CPU tests (`tests/dflash/`) cover accepted and rejected blocks, feature alignment, snapshot lifetime,
budgets and context limits, cancellation, EOS inside an accepted block, sampling distributions
(chi-square), ring caches against a full cache, and resuming from snapshots. GPU tests
(`INFURNACE_TEST_NV=1`) check the real model's output against plain decoding and the kernels against
references.
