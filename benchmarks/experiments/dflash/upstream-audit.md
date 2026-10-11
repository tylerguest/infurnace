# DFlash upstream contract audit

Inspected official [z-lab/dflash](https://github.com/z-lab/dflash) at
`07ebd93db9f472af339b644bb70221ad8428328a`: README, `dflash/model.py`
(model utilities, generation, attention, decoder, DFlash/DFlash2 model classes)
and the Qwen3.5-capable `dflash/model_mlx.py` (configuration, attention, generation,
GDN capture/rollback). These sources were read, not executed. No torch/MLX runtime
was installed or added to inference.

Also inspected official Qwen3.5-4B-DFlash config at HF revision
`9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf` and selected Anbeeld GGUF metadata
at `a79bafa7f383f8627f925250f547fe1ebbc43ff6`. The latter identifies
`z-lab/Qwen3.5-4B-DFlash` as its base. Architecture agreement is not an assertion
of bitwise checkpoint equivalence; original BF16 versus quantized GGUF draft
logit parity has not been measured.

## Substantive finding: version-dependent attention defaults

Current official Torch and MLX attention default to:

- Sliding layers: causal, past window `query_position-key_position < window`.
- Full-attention layers: noncausal.
- Explicit `is_causal`: global override.

Our code previously defaulted every layer to noncausal. That agrees with the
older official implementation at `c2489aa8e9446ffb4f6591060b3d696305754f8e`
and the inspected BeeLlama converter runtime, but **not current official code**.
The behavior changed in `c2394722e1df0b0b2f2d9d7db5923ec152021e1f`.
The selected six-layer GGUF and official HF config both lack an explicit causal
flag; the GGUF does not identify the inference implementation version.
Therefore metadata alone cannot establish which policy its conversion expected.

Corrected `DFlashConfig.causal` to an optional override, with per-layer defaults
matching current upstream: `[True,True,True,True,True,False]` for this draft.
Explicit GGUF `dflash.attention.causal` remains authoritative. For reproducing
older converted behavior, `--dflash-legacy-noncausal` (benchmark
`--legacy-noncausal`) forces the former all-noncausal policy. Weights and target
model are unchanged. New reports include the effective per-layer causal flags.

Historical reports predate this correction and use the legacy policy. Their
exact-target-output/RNG evidence remains valid: rejected guesses cannot change
the target. Their proposals, acceptance and timing do not describe the new
policy. No corrected-policy full-model performance result is asserted here.

## Contract comparison

| Operation | Official reference | Our implementation / finding |
|---|---|---|
| Feature extraction | Post-block IDs, HF hidden_states[id+1] | GGUF taps `[2,6,...,30]` mean before block N; matches official IDs `[1,5,...,29]` plus1 |
| Input block | Known anchor + B−1 mask tokens | Matches; mask248077 and trained B16 from actual GGUF |
| Context fusion | hidden_norm(fc(concatenated features)) | Matches; encoded context reused unchanged across draft layers |
| Layer input norm | Noise only; not encoded context | Matches |
| Q versus K/V | Q from normalized noise; K/V from context + normalized noise | Matches, including per-head Q/K RMS norms and GQA |
| Position IDs | Newly observed context at P−C..P−1; noise at P..P+B−1 | Matches; queries use tail/noise positions, keys use both segments |
| RoPE | Non-interleaved rotate-half, configured theta | Tinygrad split-half RoPE matches; full head128, theta1e7 |
| Attention policy | Layer-aware defaults; explicit override | Previous version mismatch corrected as described above |
| Residuals/MLP | Pre-norm attention residual; pre-norm gated SiLU MLP residual | Matches |
| Draft output | Final norm, target head, discard anchor position | Matches; normalizing only tail rows is equivalent because RMSNorm is per-row |
| Draft cache | Crop transient block to observed target prefix | Matches by authorization length: noise overwritten before prefix extension and future rows masked |
| Greedy acceptance | Longest prefix where proposal[i] == target_decision[i]; bonus at first mismatch or block end | Matches |
| Target cache | Commit known anchor + matching descendants, bounded by final output/EOS | Matches; sequential does not speculate, block mode restores prefix checkpoints |
| Next context features | Only processed/committed target rows | Matches; pending feature snapshots exclude the final unprocessed output |
| Sampling | Official greedy argmax; stochastic uses p/q rejection and residual sampling | We support greedy only, preserving tinygrad's ordinary greedy Gumbel/RNG behavior, not Torch RNG semantics |

The selected config does not require embedding scaling, output multipliers,
softcapping, DFlash2 dynamic convolutions or candidate selection. This audit does
not extend support to those architectures/features.

## Missing acceleration / rollback design

Official generation verifies all proposed tokens in one target forward. Our
experimental block mode still calls one-row upstream blocks in a layer-major
captured graph. It is not the official multirow execution strategy and has no
established speed benefit.

The official MLX Qwen3.5 path captures projected GDN operands and initial state.
On rejection it recomputes only the accepted-prefix delta update and slices the
convolution input window. It does **not** retain a full recurrent-state copy at
every speculative row. Our checkpoints are correct in focused controls but use
roughly50 MB per row on the selected target. Capturing/replaying projected
operands is a concrete memory/performance direction, subject to keeping the
upstream tinygrad target math unchanged and proving exact prefix-state parity.

Official MLX also uses rotating sliding-window draft caches. Our capacity-sized
masked cache has the right visibility in tested controls, but performs needless
work and consumes more memory for long contexts. Torch and MLX differ on explicit
noncausal sliding masks when the noise block exceeds the window; for this
selected model B<=16 and window4096, that edge does not arise.

## Validation after the correction

- **114 focused CPU tests passed**, including layer-default/override mask tests,
  target output/RNG parity, state commitment, EOS/cancellation/replay and CLI flags.
- **Five NV tests passed**, including both verifier schedules and hidden controls.
- New `tests/test_dflash_reference.py`: independent scalar full-draft oracle with
  actual rounded weights and half-cache storage, nonuniform Q/K, split-half RoPE,
  GQA, both residuals, final norm and target head. Two incremental calls check
  replacement of transient noise with actual context. Hidden tolerance4e-5,
  matching proposal argmax. No torch/MLX/NumPy runtime is used by the oracle.

This is source-contract and synthetic arithmetic validation, not complete
original-BF16/quantized-GGUF draft parity or a full-model performance result.
The earlier full-model block protocol remains aborted with the unresolved NV
signal timeout/`NV_ERR_INVALID_ARGUMENT`; no GPU reset or unrelated termination
was used during this audit.
