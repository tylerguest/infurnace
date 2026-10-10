# EAGLE-3 on tinygrad: implementation review

Status: checkpoint-loaded drafter (`infurnace/eagle/draft.py`), sequential reference,
experimental NV chain verifier (`infurnace/eagle/verify.py`), and opt-in Python
generation (`infurnace/eagle/generate.py`) are implemented. CLI chat/raw generation
can explicitly opt in with `--eagle3 CHECKPOINT`; ordinary generation remains the
default. No tree mode or measured end-to-end speedup yet.

## Exact projection-reduction follow-up

- Added opt-in `fused_projections` / benchmark `--fused-projections`. The CTA
  computes completed warp sums and residual-first chunk accumulation. Ordinary
  ADDs were flattened with the final shuffle ADD; `__fadd_rn` preserves the
  rounding boundary of stored partials. Summing chunks before adding the residual
  passed standalone checks but failed real-model parity (logit error 0.17657).
  The corrected implementation passed exact real-model logits/features and
  replay at 2/3/4 rows, plus **30 focused NV tests**.
- Realized reference snapshots now use buffer-only views before copying, avoiding
  unnecessary target-graph dependencies while retaining independent live rows.
  Full CPU validation after these changes: **377 passed, 238 skipped**.
- Three complete, exact-output, two-repeat protocols still showed no application
  gain: fused projections **93.33 / 93.19 tok/s** versus **106–109 baseline**;
  detached-snapshot control **94.19 / 94.51** versus **109–110**; isolated packed
  12-bit Q4 metadata **92.33 / 91.08** versus **104–106**. All accepted 94
  descendants in 162 rounds. Reports and caveats are retained separately in
  [`benchmarks/experiments/eagle3`](../benchmarks/experiments/eagle3/README.md).
  The 12-bit metadata prototype is not integrated: projection savings did not
  survive preparation/restoration and whole-request measurement.
- Compact device-side commitment, zero-copy features, and scored draft margin
  filtering remain optional experiments. Scored drafting preserves proposal IDs;
  filtering can increase acceptance percentage while losing useful accepted
  tokens. No speedup or acceptance fix is inferred from that percentage alone.
  Ordinary decode and all new experimental defaults remain unchanged.

## Quantization-aware head calibration follow-up

- Collected 1,280 teacher-forced positions from the original target across eight
  training and two separate validation prompts. Trained a rank-16 draft-head-only
  correction; no target or original checkpoint file was changed. The merged BF16
  head improved held-out one-step accuracy **56.64% → 72.27%**. This is small-data,
  prompt-limited evidence, not general EAGLE quality or application performance.
- The original 17 × 23 benchmark was excluded from both training and validation.
  A completed one-proposal protocol accepted **101 descendants / 155 rounds**,
  improving useful commitment from 1.58 to 1.65 tokens/round, but measured only
  **98.12 / 97.65 tok/s versus 109.52 / 109.39 baseline** (ratios 0.896 / 0.893).
  Two proposals reached **129 / 127**, or 2.02 tokens/round, yet measured
  **91.93 / 92.19 versus 109.54 / 109.78** (ratios 0.839 / 0.840). All outputs
  matched exactly; each protocol completed two warmup pairs and two repeats.
  There is no matched uncalibrated-two-proposal control in this follow-up set.
- Reproducible isolated tools and reports are under
  [`benchmarks/experiments/eagle3`](../benchmarks/experiments/eagle3/README.md).
  Calibration collection and fitting run in separate processes; optional NumPy
  was installed only into a temporary tooling directory. Hash manifests identify
  the target/original checkpoint before applying an explicit fresh-draft override.
  Six additional CPU guards passed. No inference defaults or ordinary CLI path
  load the calibrated head automatically.
- Two collection attempts failed during draft loading before collecting samples;
  a subsequent isolated retry completed. This does not establish a timeout cause
  or reliability fix. **Useful acceptance improved modestly, but the requested
  substantial end-to-end speedup remains unachieved.**

## Fused draft follow-up

- Added opt-in `observe_propose()` and benchmark `--fused-draft`: observation
  and proposals share one capture, preserving the true observed hidden and
  committed KV prefix. This is **not** fused target/root sampling. CPU/NV
  exactness, replay and continuation checks cover draft counts 1/2/4/7; all eight
  focused checks passed with `HCQ_CACHE_THRESH=0`. Full pinned CPU regression:
  **361 passed, 184 skipped**, including loading-report, maximum-draft and
  final-root short-circuit regressions. Ordinary decode and the unfused draft
  remain defaults.
- Completed an original-target 256-output, one-proposal protocol with two warmup
  pairs and two measured repeats, `--loop-projections --fused-draft` and
  `HCQ_CACHE_THRESH=0`. Exact output; 94 accepted descendants in 162 rounds.
  EAGLE: **94.75 / 94.29 tok/s**; bracketed baseline: **109.39–109.70 tok/s**;
  ratios **0.864 / 0.861** (`/tmp/eagle-fused-ring-normal.json`). A separate
  otherwise-matched unfused control completed at **93.50 / 93.34 tok/s**, versus
  **109.73–109.79 baseline** (`/tmp/eagle-unfused-control.json`), with identical
  acceptance. The small incremental difference is not a substantial end-to-end
  gain or statistical evidence of a reliable improvement.
- A normal-cache phase-profile protocol aborted during its second warmup
  **baseline** request (`/tmp/eagle-fused-profile-default.json`). Its completed
  first EAGLE warmup supplies diagnostic phase timings only: the final 50 rounds
  averaged **11.57 ms verification + 2.31 ms fused drafting**. These are not
  completed-protocol throughput evidence. Another no-cache attempt failed during
  draft loading, so disabling caching is not a universal reliability fix.
- Isolated newer tinygrad `2dc4ca083dad33473a05ba40a08b6b51290bf32e` passed CPU
  regression and 26 toy NV checks after compatibility adaptations. A real-model
  suite timed out after 16 passes, while starting a later reference prefill;
  the installed pin remains unchanged. Increasing runtime pools to 256 MiB
  exhausted GPU memory during target loading, so it is not adopted.
- BF16 draft-only tensor-core prototypes (both weight orientations) were slower
  on full shapes: vocabulary projection roughly **0.51 ms GEMV versus
  0.90–1.12 ms MMA**, with changed draft rounding. Neither is integrated.
- Benchmark JSON now checkpoints loading failures as `aborted`, and records
  tinygrad module path, HCQ cache threshold and runtime pool size. No speedup
  ratios are published for incomplete protocols.

## Replay and verification follow-up

- Updated the dependency and local environment to tinygrad
  `bc5a2d393bb37f819c6a87cc65b2c750c060c1cb`. A deterministic regression showed
  the previous pin skipped initialization when submission-signal storage was
  reused. The new pin fixes that invariant; this is not a claim that every NV
  replay failure is resolved.
- Draft observation now uses concrete buffer-view JIT inputs, and computes
  queries/attention/FFNs only for the last observed row. All accepted rows still
  generate K/V. CPU and NV capture/replay checks compare hidden outputs and caches
  against full-row observation.
- Full-row residual reductions passed real-target parity. Batched Q/K norms need
  **materialized projection sums first**: omitting this boundary changed real
  logits by up to 0.19. With the boundary, real-model 2/3/4-row logits/features
  match single-token decode exactly, including capture/replay. The focused NV
  suite passed **27 tests**; CPU regression passed **350, with 180 skipped**.
- Added experimental whole-row projection CTA loops (`--loop-projections`). Cold
  weight-cache screens retain exact partial sums. Q6 FFN-down for two rows
  improved approximately **95 to 75 us**; the output head improved **902 to
  816 us**. These are projection measurements, not whole-request speedups.
  A cooperative integer-MMA prototype preserved partial sums after fixing its
  opaque register-write dependency, but remained slower for 2/3-row Q4 work;
  it is not integrated.
- Fused root sampling and zero-copy feature handoff passed isolated replay tests,
  but their combined long-request trial aborted on its second measured EAGLE
  request (`/tmp/eagle-copy-root-residual.json`). Both remain **disabled by
  default**. A later one-proposal direct-base run also timed out with both
  disabled (`/tmp/eagle-base-one.json`), so they are **not necessary causes**;
  replay reliability is still open. A conservative retry completed all
  brackets with identical outputs: **100.45 / 101.21 tok/s EAGLE**, versus
  **113.47–115.83 baseline**, one proposal and 256 outputs. This retry preceded
  the materialized batched Q/K norm change (`/tmp/eagle-residual-safe.json`).
- Downloaded a separate direct-base quantization to
  `models/base-qwen3/Qwen3-4B-Q4_K_M.gguf` from
  `bartowski/Qwen_Qwen3-4B-GGUF` (declared base `Qwen/Qwen3-4B`). The original
  Instruct/AWQ GGUF is untouched; no model substitution occurs in CLI. Base
  provenance alone does not prove checkpoint revision identity. With two
  proposals, the new target's completed 256-output comparison measured
  **90.85 / 91.20 tok/s EAGLE** versus **116.56–117.61 baseline**, identical
  output, 113 accepted descendants and 143 rounds
  (`/tmp/eagle-base-two.json`). Thus this experiment **does not establish that
  model provenance was the speedup blocker**.
- Disabling small-schedule HCQ runtime caching (`HCQ_CACHE_THRESH=0`) on the
  updated pin completed the direct-base one-proposal protocol, including two
  warmup pairs and two measured repeats, with identical outputs. It was slower:
  **87.31 / 87.49 tok/s EAGLE**, **104.67–105.68 baseline**
  (`/tmp/eagle-base-one-no-cache.json`). One successful retry is not a universal
  reliability proof and the extra submission overhead is not a speedup fix.
  No GPU reset or unrelated process termination was performed.

No substantial end-to-end speedup is established. At 1.79 committed tokens per
round and a 117 tok/s baseline, whole rounds must average below about **15.3 ms**
to break even; the direct-base trial averaged about **19.6 ms**. Further gains
require materially cheaper verification or higher useful acceptance, not a
checkpoint download alone.

## October 8 review

- Reproduced exact-output, synchronized baseline/EAGLE/baseline on the RTX 5060:
  64 arithmetic tokens, context 512, two proposals, two warmup pairs. EAGLE
  **68.65 tok/s**, bracketing baseline **104.52 / 100.57 tok/s**. Accepted 28
  descendants across 36 rounds. Captures are excluded from those warmed rates.
  Full records: `/tmp/infurnace-eagle-before.json`.
- GPU-timestamp projection screening, not CPU submission timing, rejected FP16
  prefill as a verifier optimization (also different arithmetic). It rejected
  broad unrolled weight reuse and broad CTA-local reuse: neither helped Q4 FFN
  or the Q6 vocabulary head.
- A bounded CTA-local token axis preserves every integer dot and FP32 reduction
  while sharing the SM weight cache. For the sampled prepared Q6 FFN-down,
  three rows at 2560×9728 improved from about **104 to 74 microseconds**. This
  candidate is restricted to Q6 FFN-down and **disabled by default** pending
  repeatability; benchmark it explicitly with `--local-ffn-down`. Ordinary decode
  is untouched.
  Projection/JIT checks cover 2/3/4/8 rows, Q4/Q6, prepared layouts and both
  2560/9728 input widths. Full real-model 3-/4-row logits/features remain exact.
  **38 focused NV tests passed**, including the real-model check.
- Added token-level EAGLE streaming with close propagation, CLI eligibility
  checks, and reusable session captures. Removed device-context leakage across
  generator yields. A final already-target-selected root (budget/context/EOS)
  needs neither another target pass nor draft observation; avoid that redundant
  work and its size-one JIT capture. This does not bypass any target decision.
- The prior CLI tests had stale expectations for automatic prepared weights and
  fused attention; updated them. CLI/flag/streaming tests: **40 passed**.

A projection microbenchmark is not an application speedup. A 256-output,
one-proposal, two-warmup/two-repeat trial produced identical completed outputs
and observed roughly 93–94 tok/s against 112–113 tok/s baseline, but its final
bracketing baseline aborted with `NV signal wait timed out` / `PYTHON hang detected`.
**This protocol is aborted and establishes no accepted performance result.**
Records: `/tmp/infurnace-eagle-one.json`; trace: `/tmp/infurnace-eagle-one.log`.
Both new application trials used the CTA-local candidate. The 256-token
worker remained alive in tinygrad shutdown, holding 3388 MiB; an attempted
fresh model load then ran out of memory. Only that agent-started worker was
terminated. No GPU reset or desktop-process termination was performed. A fresh
minimal NV arithmetic smoke passed. After memory reclamation, a fresh 64-output
one-proposal retry also aborted on its measured EAGLE request, after completing
both warmup pairs and a baseline at 104.06 tok/s. It establishes no throughput
result (`/tmp/infurnace-eagle-short-final.json`). That agent-started worker was
also terminated after shutdown stalled. The older implementation already had
NV timeouts, so these observations neither establish nor exclude a new kernel
cause. The candidate remains explicitly opt-in. Backend reliability is open.

CPU regression after fixes: **336 passed, 154 skipped**. Final focused CLI,
generation, candidate-flag and benchmark-report suite: **60 passed**. CLI reports known device-sync
failures without swallowing unrelated runtime errors. Benchmark summaries suppress
speedup comparisons unless the entire protocol reaches `status=complete`.

## Current implementation evidence

The pinned checkpoint payload is downloaded locally. Whole-chain draft JIT
capture owns recurrent intermediates, replacing the broken cross-JIT persistent
store experiment. Four proposals matched the reference path and took about 4 ms
versus 111 ms in one local comparison (draft-only, not application throughput).
BF16 GEMV reinterprets native storage inside its UOp kernel, avoiding repeated
weight-sized Tensor bitcast packing kernels.

The initial batched verifier required per-row contiguous residual boundaries
for parity. On the updated pin, full-row residual reductions validate exactly;
Q/K projection sums still need an explicit materialization boundary before
batched normalization. Twelve real Qwen3-4B rows match sequential logits and
features exactly, including capture/replay. Sampling selects a symbolic row
inside its JIT; selecting a contiguous caller-side view produced stale row
sampling. RNG draws remain restricted to committed rows.

The thinking-enabled arithmetic prompt produced exactly the same 64 tokens as
ordinary greedy generation, accepting 28 descendants across 36 rounds. This is
one correctness smoke, not broad quality validation. Steady rounds still take
roughly 63–75 ms, with large first/capture costs. Dropping already-executed JIT
output graphs before host use retained exact output but did not materially
improve these timings. A subsequent candidate shares hidden-axis normalization
(not Q/K normalization, which failed exact continuation parity), and captures
accepted-feature copying in a small JIT for generation. Public verifier snapshots
remain independent; explicitly transient snapshots are consumed by the drafter
before the next verifier round. The same 64-token smoke remained exact, with
steady rounds roughly 49–57 ms. These sequential-process observations are not a
matched application throughput benchmark. All 11 focused verifier tests passed,
including the gated real-model comparison and transient-feature replay.
Subsequent work adds dedicated integer input buffers uploaded with tinygrad's
`Buffer.copy_from`, eliminating scheduler walks for each tiny token tensor;
shared FFN normalization and whole-row projection reductions also passed real
parity checks. Native BF16 GEMV now handles up to eight observation rows, avoiding
large generic BF16 observation matmuls. Steady arithmetic rounds reached roughly
22–31 ms in local checks, but this still does not establish an application win.
SiLU now computes internally in FP32 before BF16 rounding, and attention scaling
uses the FP32 scale before rounding, matching the reference's operation contract.

`EagleGenerator` retains kernel captures across requests. Its verifier cache is
keyed by model, feature taps, chunk/context configuration, weight/norm/frequency
and KV storage identities; changed storage/layout invalidates it. Token/session
state is never reused. Verification failures clear the cache and release target
ownership. `benchmarks/eagle.py` performs exact-output, synchronized whole-request
baseline/EAGLE/baseline comparisons, with warmups and incomplete runs separated.

Completed comparisons before the newest launch-consolidation candidates:
- Arithmetic, 256 output tokens: warmed EAGLE about 76 tok/s versus bracketing
  baseline about 105–113 tok/s; 119 accepted descendants across 137 rounds.
- JSON task, 512 output tokens, three proposals: EAGLE about 62 tok/s versus
  baseline about 113–114 tok/s; 242 accepted descendants across 270 rounds.
These are different tasks/protocols, not updates to the published chart.

Chain attention now uses separate query/head grid axes and one partial/merge pair
for all nodes. Exact standalone checks cover 2/3/4/8 queries, tile boundaries,
JIT replay and poisoned future slots; real-model 3-/4-node logits/features passed.
The first flattened-axis candidate failed parity and was discarded. Unrolled
DP4A weight reuse remains disabled (flat/slower projection results); the slower
INT8 MMA prototypes were removed from the repository and retained only in /tmp.
Root sampling fusion remains disabled pending repeated-request validation.

The newest repeated-request run aborted with `NV signal wait timed out`; a fresh
retry failed before its first measured baseline with NV/CPU synchronization
errors. No throughput result is accepted from either run, and no GPU reset or
user-process termination was attempted. This blocks further device measurements
until repeatability is rechecked. A subsequent minimal NV smoke and a smaller
64-output repeated-request benchmark completed with root fusion disabled:
EAGLE 67.24 tok/s versus bracketing baseline 107.17/104.50 tok/s, with identical
output and 28 accepted descendants across 36 rounds. This does not prove root
fusion caused the earlier timeout. The current focused NV suite passed 40 tests;
CPU regression passed 20/3 skipped; real-model 3-/4-node parity passed separately.
Ruff and whitespace checks passed. **No end-to-end speculative speedup is
established.**

Tests cover an independent scalar BF16 architecture fixture, strict tensor and
vocabulary validation, shifted feature/token pairing, recurrent cache prefix
preservation, count/position replay, reset, budget/EOS/context/cancellation,
accepted-chain continuation and live-row sampling. Real-model tests are gated
by `INFURNACE_TEST_MODEL`. Learned drafting and batched verification remain
explicitly opt-in; CLI integration requires `--eagle3 CHECKPOINT`.

## Implemented reference API

`SequentialVerifier` exclusively owns the existing target while its context is
active. It mirrors symbolic prompt chunking, captures three residual-stream taps,
then executes accepted tokens through the existing one-token decode path. Its
zero-temperature decision expression is the pinned upstream Gumbel-max expression,
not a replacement argmax sampler. Exact ties preserve that sampling behavior;
independent runs with different RNG states need not emit identical tokens.

```python
from infurnace.eagle.reference import SequentialVerifier

with SequentialVerifier(model, eos_token_ids=(eos_id,)) as verifier:
    prompt_steps = verifier.prefill(prompt_ids)
    # Empty proposals are useful for reference-only execution. Once a drafter
    # exists, proposals contain descendants AFTER the known target-selected root.
    round_result = verifier.verify_chain([], max_tokens=1)
    accepted_ids = round_result.tokens
    pending_root = round_result.next_token  # not yet written into target KV
```

Every returned step holds independent, concrete-shaped logit/feature snapshots;
TinyJit replay cannot overwrite previously returned features. Concatenated feature
rows are explicitly made contiguous before copying live symbolic-prefill rows.
This debug/reference path incurs extra copies and is not a throughput engine.

The sequential scaffold never executes rejected candidates. EOS/context exhaustion
stops commitment; per-round budgets bound committed output, not the whole response.
Session exit or a device exception invalidates upstream prefix reuse and releases
exclusive ownership. Prefix import, tree rollback/compaction, mid-round streaming
cancellation and optimized feature retention are still later work. Ordinary
upstream generation can safely resume after exit with a fresh prefill.

Focused tests in `tests/test_speculative.py` cover feature/logit agreement, replay
snapshot stability, shifted root/proposal acceptance, mismatch-slot reuse, EOS,
context bounds, budgets, ownership, failure cleanup and zero-temperature exact-tie
sampling. The NV toy model uses current prepared Q4 projections, packed prefill,
shared Q8 and fused attention, and poisons inactive KV tails before decode.

A pinned tinygrad CUDA renderer failure (`expected a "}"`) was found for small
symbolic prefill chunks (3/4) in the NV toy shape; it also reproduces through
ordinary `Qwen3Model.generate`, without this verifier. Default chunk size 32 passes
that toy test. No dependency/global optimizer patch or automatic NOOPT workaround
was introduced. Reproduction log:
`/tmp/infurnace-eagle-small-chunk-upstream.txt`.

Validation for this foundation:

- CPU verifier + runner + model configuration: **63 passed, 4 skipped**.
- Prepared-Q4 NV toy verifier, default chunk 32: **1 passed**; exact target logits
  and feature taps through decode capture/replay, with poisoned inactive KV slots.
- Bundled Qwen3-4B-Q4_K_M on NV, capacity 1024: **1 passed**; 35-token prompt in
  two symbolic prefill chunks plus four decode steps, exact target logits/features
  with packed Q4/Q6, prepared storage, shared Q8 and fused attention enabled.
  One target model was loaded; the reference rewrites the same prefix/slots in
  sequence rather than allocating a second full model. Output:
  `/tmp/infurnace-eagle-target-smoke.txt`.
- Ruff on the new module/tests and `git diff --check` passed.

These foundation-stage checks are numerical/state smokes, not broad language-quality
validation or an acceptance/performance benchmark. At that stage no EAGLE
checkpoint payload had been downloaded; the current implementation loads it.

```sh
DEV=CPU .venv/bin/python -m pytest -q \
  tests/test_speculative.py tests/test_runner.py tests/test_model_config.py
DEV=NV INFURNACE_TEST_NV=1 .venv/bin/python -m pytest -q \
  tests/test_speculative.py -k nv_packed
DEV=NV INFURNACE_TEST_NV=1 \
INFURNACE_TEST_MODEL=models/Qwen3-4B-Q4_K_M.gguf \
.venv/bin/python -m pytest -q tests/test_speculative.py -k local_nv
```

## Sources inspected

- [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE), checkout
  `cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b`:
  `eagle/model/cnets.py`, `ea_model.py`, `utils.py`,
  `modeling_qwen3_kv.py`, and the EAGLE-3 training feature selection.
- [Tencent/AngelSlim](https://github.com/Tencent/AngelSlim), source snapshot
  `6c0e1957748023e69db701d421c49d1c2ee2f09c`:
  `angelslim/compressor/speculative/inference/models/eagle3/draft/llama3_eagle3.py`,
  `target/modeling_qwen3_kv.py`, `eagle3_model.py`, and training draft implementation.
- [AngelSlim/Qwen3-4B_eagle3](https://huggingface.co/AngelSlim/Qwen3-4B_eagle3),
  checkpoint revision `fd331e59626c8e95c392381a16ee59d518727fbb`:
  configuration, model card, safetensors header and downloaded tensor payload.
- Infurnace's pinned upstream `Transformer.forward`, `TransformerBlock`,
  `apply_rope`, packed GEMV/GEMM and NV attention integration.

EAGLE's README lists the AngelSlim checkpoint as **unofficial**, and still lists
support for official EAGLE-3/Qwen3 as a TODO. Neither headline EAGLE speedups nor
results on Qwen3-8B establish performance for our quantized Qwen3-4B.

## Actual checkpoint architecture

This is not a stock small language model. It consumes target-model features and
runs a learned recurrent draft decoder:

1. Concatenate three selected target residual-stream features (width 7680).
2. Project through `fc.weight` to width 2560.
3. Separately RMS-normalize that projected feature and the next-token embedding.
4. Concatenate embedding first, feature second, into width 5120.
5. Run attention and add its output to the **unnormalized projected feature**.
6. Apply post-attention RMSNorm and SwiGLU FFN with residual addition.
7. Apply output RMSNorm and the draft's **own** 32,000-token vocabulary head.
8. Feed draft hidden states back into subsequent draft steps (do not apply the
   7680-to-2560 fusion projection again to already-2560-wide draft features).

Observed BF16 shapes:

| Tensor | Shape |
| --- | --- |
| `fc.weight` | 2560 × 7680 |
| Q projection | 4096 × 5120 |
| K/V projections, each | 1024 × 5120 |
| attention output | 2560 × 4096 |
| FFN gate/up, each | 9728 × 2560 |
| FFN down | 2560 × 9728 |
| `lm_head.weight` | 32000 × 2560 |
| four RMSNorm weights | 2560 each |
| `d2t` | 32000, int64 |
| `t2d` | 151936, bool |

Configuration: one draft layer, 32 Q heads, 8 KV heads, **head dimension 128**,
RoPE theta 1,000,000, norm epsilon 1e-6. The draft has **no Q/K RMSNorm weights**;
do not substitute the upstream Qwen3 block wholesale.

Important incompatibility: stock EAGLE `cnets.py` computes head dimension as
`hidden_size // num_attention_heads` (80 here), whereas AngelSlim honors explicit
`head_dim=128`. Checkpoint shapes confirm 128. The stock code is not a literal
reference implementation for these weights.

The tensor payload is **436,898,176 bytes (~417 MiB)**, including vocabulary maps.
It contains **no embedding tensor**. A separate dense BF16 embedding would add
777,912,320 bytes (~742 MiB); instead investigate sharing the target's existing
packed-backed embedding lookup. Quantized embeddings differ from the original
BF16 teacher and may reduce draft acceptance. Do not silently realize a second
full dense vocabulary matrix. Validate actual allocation/peak and per-step draft
latency, not just serialized size.

## Features, token alignment, positions and vocabulary

The inspected target implementation captures the residual stream **before**
blocks `2`, `num_layers//2`, and `num_layers-3`. For our 36 layers these are
before blocks **2, 18, 33** (after blocks 1, 17, 32, zero-based), in ascending
order. These are not the final output norm or last-layer output. Confirm the
specific checkpoint's training contract before locking these indices into an API.

Feature at token position i is paired with the embedding of token i+1. The
reference drops the first input token in `topK_genrate`; the target-selected next
token is the root of the next verification tree. After acceptance, feed only
accepted target features back into the drafter. Avoid duplicating already-cached
feature/token pairs in its stable KV prefix.

The reduced vocabulary is not the first 32,000 target IDs. In the inspected code:

```
target_token = draft_index + d2t[draft_index]
```

`d2t` is an **offset**, not a direct ID lookup. `t2d` is a boolean membership map,
not a target-to-draft integer index. Validate bounds, unique mapped IDs and
membership consistency after loading actual payloads. The target still samples
from its full vocabulary; omitted draft tokens reduce acceptance, not correctness.

HF/EAGLE uses split-half RoPE. Our pinned tinygrad `apply_rope` also chunks the
head into two halves and concatenates the rotated halves, so **no interleaving
permutation is needed for the draft with this implementation**. Verify frequency
construction, depth-based positions and checkpoint Q/K layout against reference
fixtures. Do not import older tinygrad recipes that permute Q/K into interleaved
pairs; that would be wrong for this pinned split-half implementation.

The current GGUF describes an Instruct/AWQ source. Dimensions alone do not prove
that its weights match the BF16 model used to train this drafter. Pin both model
identities and measure acceptance with **our exact quantized target**, including
thinking; do not silently change model, chat template, or thinking policy.

## Target verifier: correctness before acceleration

Current `Transformer.forward` emits only the final position's sampled token.
Verification needs per-position target decisions plus the three feature taps.
Build an opt-in target adapter using the existing blocks, weights and function
boundaries; ordinary generation remains upstream. No global monkeypatches and no
second target weight copy.

The current fast prefill path is **not** an exact single-token decode verifier:
its FP16 packed GEMM differs from Q8-activation GEMV, and multi-token attention
uses a different reduction path. Near-tie predictions can differ. Rechecking only
low-margin logits is not a proof that the other logits are safe.

Start with a sequential reference verifier using the existing decode arithmetic.
It establishes token alignment/cache/feature correctness but is not expected to
speed up generation. Then develop a small batched-Q8 verifier. `quant_gemv`
already accepts multiple activation rows, so it is a correctness scaffold, not
yet a weight-amortizing GEMM kernel. Keep integer group dots, scales/min
corrections, floating operation order and reduction topology consistent with
single-token decode when claiming identical greedy output. Test per-layer outputs
and logits on held-out prefixes, including ties and adversarial rounding inputs.
A differently-rounded batched verifier must be described as a different numerical
target, not called universally token-identical to today's fast decode.

For useful throughput, develop a measured UOp batched projection kernel that
reuses each packed weight tile across a small number of verification rows. An
INT8 MMA/groupwise correction prototype is a candidate, not a guaranteed win.
Start with 4/8-token chains; larger trees require more target head/attention work
and can lose despite higher acceptance.

## Tree verification is not ordinary causal prefill

For a flattened proposal tree:

- KV slot = committed prefix length + physical tree-node index.
- RoPE position = committed prefix length + **node depth**.
- A query can see the committed prefix, itself and its ancestors, never siblings.
- Normal lower-triangular masking and contiguous RoPE slices are incorrect.
- Verify the target-selected root as well as draft descendants; emit an extra
  target-selected continuation from the last accepted node.

Keep attention computation in `infurnace/kernels/attention.py`, integration in
`model.py`. Extend bounded multi-query attention with explicit positions,
ancestry masks, valid-node count and committed-prefix length. Preserve FP32
query/softmax/accumulation and native FP16 KV. Begin with query-by-query kernel
arithmetic matching current decode; optimize common-prefix KV reuse only after
correctness tests. Avoid a context-square mask allocation for a small tree.

Commit only the selected root-to-leaf path. Gather selected K/V into separate
bounded scratch before writing the contiguous cache tail: an in-place parallel
gather/scatter can overwrite another thread's source. Compact selected features
and reconcile the drafter's stable prefix too. Rejected slots must never become
visible to a later query. Returned tokens, KV length and prefix-reuse bookkeeping
must agree, including cancellation mid-burst, EOS, reset and context rollover.
Yield only accepted tokens; preserve streaming and avoid an arbitrary output cap.

## Greedy first; stochastic decoding is a separate proof obligation

For a chain, accept the longest prefix of draft tokens matching the preceding
target decisions; stop at the first mismatch and use the target decision there.
EAGLE's temperature-zero reference uses strict argmax; our initial scaffold
retains tinygrad's upstream zero-temperature Gumbel-max expression instead. Treat
that distinction explicitly at exact/near ties in accelerated verification.
For a tree, follow a valid matching path and compact that path only.
Drafter confidence scores select proposals; they never authorize bypassing target
verification. Target quantization can lower acceptance without invalidating the
algorithm, provided the verifier really implements the intended target.

The reference's non-greedy tree acceptance is not just a confidence threshold or
ordinary top-k sampling: it tests candidate tokens against the target
probabilities and renormalizes after rejection. A separate probabilistic chain
proposal would require its actual proposal probabilities and the corresponding
speculative rejection/residual sampling rule. Do not mix these algorithms.
Initially restrict acceleration to temperature zero and keep the upstream
nonzero-temperature sampler unchanged. Correct sampling distribution does not
imply matching the same seeded random token stream.

## Proposed staged implementation

1. **Verifier contract and tests:** sequential target feature/logit reference,
   chain acceptance, bonus token, EOS, rollback, reset, prefix reuse and
   cancellation. These are reusable for prompt-lookup speculation too.
2. **Checkpoint-correct drafter:** tinygrad safetensors loading with strict
   shape/key checks, BF16 reference arithmetic, correct 128-wide heads, RoPE,
   shifted token pairing and reduced-vocabulary mapping. Missing embedding is
   explicit, not swallowed by `strict=False`. Validate with reference fixtures;
   PyTorch is not an Infurnace runtime dependency.
3. **Small batched-Q8 verification:** preserve target arithmetic; optimize packed
   weight reuse and bounded multi-query attention. A sequential verifier alone
   cannot deliver the speculative throughput gain.
4. **Greedy chain integration:** conservative 4/8-token proposal caps, static
   JIT shapes, no speculative token emitted before verification. Measure actual
   drafter and verifier timings and accepted tokens per round.
5. **Tree mode only if chains justify it:** device-side top-k/parent selection,
   ancestor masks, depth-based RoPE and safe KV/feature compaction. Do not copy
   Python `.tolist()`/`.item()` synchronization into every tree-building stage.
6. **Adaptive enablement:** turn speculation down/off on poor acceptance or bad
   measured round cost. Keep an explicit escape hatch and ordinary decode ready.
   A fallback is not a universal zero-overhead guarantee.

Performance criterion at comparable context:

```
mean committed tokens per round / (draft time + verify time + commit time)
    > ordinary decode tokens per second
```

Count the target-selected root/bonus consistently, and distinguish raw proposal
acceptance from committed tokens per round. Include synchronization, output-head
work, EOS, fallback and startup/JIT costs in the appropriate reported metric.
Bracket baseline/speculative/baseline with unchanged model/settings and monitor
clocks/power. Test code edits, fresh code, JSON, math, prose and thinking sessions.
Report token agreement, peak allocation, draft acceptance, sustained throughput,
TTFT and streaming latency. No predicted 2x/5x gain until measured on this GPU.

**Remaining performance work:** materially amortize target projection work over
verification rows, improve measured draft acceptance on this exact quantized
target, and establish wins across realistic prompts and contexts. Checkpoint
loading, chain drafting and exact batched verification are present; they do not
by themselves establish acceleration. Do not substitute the existing FP16
prefill arithmetic while claiming exact Q8-decode parity.
