# EAGLE3 experiments

Opt-in research tooling, separate from ordinary inference and the published
1024-output decode chart. Run commands from the **repository root**.

## Start here

Latest completed best: **131.65 / 131.82 tok/s**, versus **107.25 / 107.15**
bracketed baseline, with identical output. This is one 256-output arithmetic
request, not a general performance or reliability guarantee. NV timeouts remain
unresolved; ordinary decode stays the default.

- **Reproduce the best configuration:** [parallel reduction findings](findings/parallel-reduction-findings.md).
- **Exact benchmark data:** [results index](results/README.md).
- **Implementation status:** [EAGLE3 plan](../../../docs/eagle3-plan.md).

## Layout

```text
eagle3/
  README.md              # navigation and current status
  *.py                   # stable runnable tools and experimental kernel
  findings/              # explanations, investigations, historical notes
  results/               # preserved benchmark JSON reports and index
```

Model/checkpoint artifacts remain in `models/`; transient assembly dumps,
visualization traces, and diagnostic logs remain under `/tmp/`. Neither is moved
or deleted by this organization. Report contents and script/import paths are
unchanged.

## Tools

| Purpose | File |
| --- | --- |
| Run mapped-I/O, parallel sampling and related experiments | [mapped_io_benchmark.py](mapped_io_benchmark.py) |
| Benchmark an explicitly calibrated draft head | [calibrated_benchmark.py](calibrated_benchmark.py) |
| Collect calibration data / fit a separate head | [calibrate_head.py](calibrate_head.py) |
| Inspect projection CUDA/PTX/runtime SASS | [inspect_verification.py](inspect_verification.py) |
| Inspect sampling, JIT shaders and VIZ traces | [inspect_sampling.py](inspect_sampling.py) |
| Screen sampler-only BEAM optimization | [sampling_beam_benchmark.py](sampling_beam_benchmark.py) |
| Experimental two-pass NV reduction (imported helper) | [parallel_argmax.py](parallel_argmax.py) |

The ordinary comparison harness remains [`benchmarks/eagle.py`](../../eagle.py).
Calibration's NumPy dependency is optional tooling, not an inference dependency.

## Findings

Read these in order for the investigation history; use the first link above for
the latest reproduction command.

1. [Historical protocols and calibration instructions](findings/history.md)
2. [Projection assembly and small-buffer I/O](findings/assembly-findings.md)
3. [Score materialization and VIZ](findings/sampling-findings.md)
4. [tinygrad / TensorRT-LLM implementation ideas](findings/implementation-ideas.md)
5. [Parallel reduction, mapped draft outputs and fused projections](findings/parallel-reduction-findings.md)

Historical negative results are retained. Kernel timings, acceptance rates and
aborted runs are not substitutes for completed exact-output application tests.
