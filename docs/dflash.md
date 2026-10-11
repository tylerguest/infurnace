# Tinygrad DFlash reference

`infurnace/dflash/` implements our own block-diffusion draft and greedy sequential
verification in tinygrad. The target is the **unmodified upstream tinygrad**
Qwen3.5 GGUF model. No PyTorch or llama.cpp model runtime is used.

```sh
infurnace --device NV --model models/qwen35-validation/Qwen3.5-4B-Q4_K_M.gguf \
  --dflash models/dflash-qwen35/qwen35-4b-dflash-Q4_K_M.gguf \
  --dflash-block-size 4 --max-context 128 --no-warmup \
  --prompt 'The capital of France is' --max-tokens 16
```

This is **not yet a validated DFlash acceleration**. The default verifier is
sequential: rejected tokens never execute in the target. Both schedules leave the
final emitted token unprocessed in committed state, preserving ordinary target
RNG advancement. Generation is greedy-only, bounded, exclusive per target, and
resets conversation authorization on close/error. CPU and NV one-row upstream
prefill are currently supported. Block=1 is a no-drafting control; the CLI
defaults to block4, including the known anchor.

## Experimental block verification

`--dflash-verifier block` opts into `infurnace/dflash/verify.py`. It executes
candidate rows **layer-major in one captured graph**, still calling each
unchanged upstream block with one row. It does not introduce multirow GEMM or
replace target recurrence, attention, projections or their positional bounds.

Recurrent and convolution states have checkpoints at every row, including zero.
RNG advancement has matching checkpoints. Verification restores zero before
returning to Python; commitment then advances one row at a time across output
yields. Rejected inputs cannot be committed. This preserves cancellation/EOS
semantics even inside an accepted block. Speculative suffix K/V may remain in
storage but is outside the authorized causal prefix and is overwritten when
needed. Output features are independently snapshotted before another capture.

All shapes share one history allocation, resized between requests. The selected
4B target needs approximately 50 MB per checkpoint (about 250 MB for block4,
850 MB for block16), in addition to normal model/draft memory. Owned, allocated
hidden/feature scratch prevents a reproduced pinned-planner aliasing failure
across opaque upstream calls; no global planner disable is installed. Executed
RNG graphs are detached in place without changing counter storage identity.

This schedule is **opt-in**. Its first real-model warmup matched output/RNG,
but the protocol later aborted with the existing NV signal timeout and
`NV_ERR_INVALID_ARGUMENT`. It produced no measured comparison. That attempt
predates the RNG graph cleanup; only small NV controls have completed afterward.
No performance benefit or full-model reliability is established.

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
code defaulted every layer to noncausal; use `--dflash-legacy-noncausal` to
reproduce that policy. No weights or target math are substituted.

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
