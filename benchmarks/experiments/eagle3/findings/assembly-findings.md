# Verification assembly inspection

[Experiments](../README.md) · [Reports](../results/README.md)

Report filenames below refer to `../results/`; commands run from the repository root.

Diagnostic follow-up on RTX 5060, installed tinygrad `bc5a2d393bb37f819c6a87cc65b2c750c060c1cb`. This is not a decode-chart measurement or evidence of application speedup. Inference defaults and checkpoint files are unchanged.

## Reproduce the projection dumps

```sh
PYTHONPATH=. DEV=NV HCQ_CACHE_THRESH=0 .venv/bin/python \
  benchmarks/experiments/eagle3/inspect_verification.py \
  --output-dir /tmp/verification-assembly --tokens 1 2 3 --debug 6
```

Requires `cuobjdump` and `nvdisasm`. In this tinygrad version, `DEBUG=6` prints CUDA source; automatic disassembly starts at **7**. The inspection tool explicitly emits separately compiled PTX, saves the actual runtime cubin and its SASS, and reports resources. PTX is not a substitute for disassembling the runtime binary. Timing runs use `DEBUG=0`; JSON reports mark running/aborted/complete and `diagnostic_only`.

## Findings

1. **No register-spill explanation found.** Inspected Q4/Q6 projections report zero stack/local memory. A subsequent cache dump of 157 shaders from a real baseline/EAGLE protocol (including prefill, baseline, draft, and verification shaders) also showed no local-memory allocations. Representative projection SASS contains vector global loads, `IDP.4A.S8.S8`, and warp `SHFL.BFLY` reductions, not tensor-core MMA or block-wide barriers.

2. **The experimental static weight-reuse path regresses activation-load vectorization.** Three-row Q4 gate CUDA changes eight Q8 activation words per token from `uint4` loads to scalar loads. Its SASS has more scalar loads, and registers rise from 38 to 52. Measured gate time is 81.12 versus 64.16 microseconds for chunk looping. This path is disabled; it is an obstacle to that proposed optimization, not the cause of the active verifier's deficit.

3. **Active projections already amortize significant work.** Cold-cache isolated timings in microseconds:

   | Projection / mode | One row | Two rows | Three rows |
   |---|---:|---:|---:|
   | Q4 gate / loop | 34.90 | 44.77 | 64.16 |
   | Q6 FFN down / local | 50.90 | 55.50 | 74.05 |
   | Q6 vocabulary / local | 767.34 | 780.50 | 1080.98 |

   The first Q4 screen aborted on an NV synchronization failure after 23 records; those completed per-kernel diagnostics are not a completed application protocol. The separate 24-case Q6 screen completed with exact partial-output checks. The two-row head is already close to one-row cost; its roughly 328 MB prepared weights approach the card's memory-bandwidth limit. Three rows increase arithmetic/issue cost substantially.

4. **Host transfer/submission overhead is a major additional problem with `HCQ_CACHE_THRESH=0`.** An isolated realized target-forward capture measured medians of about 8.01 / 9.45 / 12.68 ms for 1/2/3 rows at a fixed 128-token prefix. A separate instrumented exact-output 128-output baseline/EAGLE/baseline protocol completed. Its last 80 verification rounds averaged:

   - target forward: 9.44 ms;
   - sampling JIT calls combined: 1.41 ms;
   - feature-copy JIT: 0.16 ms;
   - time outside those JIT calls: 4.37 ms.

   Nested `Buffer.copy_from` timings locate most of that remainder in candidate upload and scalar decision readback: approximately **1.4–1.5 ms per tiny transfer**. Combined copy time is 3.78 ms/round; decision `item()` time is 2.46 ms/round and **overlaps** those copy timings. Do not add both figures. `tinygrad/device.py:Buffer.copy_from` constructs a new LINEAR and runs lowering/linking even for these tiny transfers. The profile uses additional synchronization, so it is diagnostic rather than an uninstrumented speed comparison.

## Consequence

The earlier claim that verification simply fails to share weights was incomplete. Two-row target compute is much cheaper than two separate forwards. Draft work, rejected work, and expensive host-driven uploads/readbacks consume that saving. The next useful experiment is reusable/coherent small-buffer I/O and fewer decision round trips, while preserving target arithmetic and RNG advancement. A host-mapped scratch prototype completed an exact-output 256-output baseline/EAGLE/baseline protocol (two warmup pairs, two measured repeats): **112.37 / 111.64 EAGLE versus 105.99 / 106.43 bracketed baseline tok/s**, ratios **1.0602 / 1.0490**, with unchanged **94 accepted descendants / 162 rounds**. This is a modest 5–6% gain on one request, not the substantial gain sought or a general result. An earlier attempt aborted during the second measured EAGLE request with an NV synchronization error. Do not enable it by default. The scoped archived wrapper also completed a repeated exact-output 64-output smoke, which remained slower than baseline (90.9–91.1 versus approximately 99 tok/s).

The completed 256-output scratch report is `qwen3-4b-eagle3-mapped-io.json`. `mapped_io_benchmark.py` archives the same I/O experiment with patches scoped to draft/verifier token allocations and restored on exit:

```sh
PYTHONPATH=. DEV=NV HCQ_CACHE_THRESH=0 .venv/bin/python \
  benchmarks/experiments/eagle3/mapped_io_benchmark.py \
  models/Qwen3-4B-Q4_K_M.gguf models/eagle3-qwen3-4b \
  --context 512 --output 256 --draft-tokens 1 --loop-projections --fused-draft \
  --warmup-pairs 2 --repeats 2 --json /tmp/mapped-io-comparison.json
```

Raw local artifacts: `/tmp/eagle-assembly/`, `/tmp/eagle-q6-assembly/`, `/tmp/eagle-real-kernels/`, `/tmp/eagle-verifier-phases.json`, and `/tmp/eagle-real-assembly/report.json`.
