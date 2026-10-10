# EAGLE3 experimental results

[Back to experiments](../README.md) · [Latest findings and reproduction](../findings/parallel-reduction-findings.md)

These preserved JSON reports are separate from the published decode chart.
Moving them here did not change their contents. Each report includes its own
configuration, measurements and completion status; cross-session differences
are not isolated component effects.

## Latest parallel-reduction series

| Report | Experiment |
| --- | --- |
| [parallel-fused-projections](qwen3-4b-eagle3-parallel-fused-projections.json) | Latest best: 131.65 / 131.82 tok/s, approximately 23% above bracketed baseline |
| [parallel-mapped-draft](qwen3-4b-eagle3-parallel-mapped-draft.json) | Parallel sampling and mapped draft proposal outputs |
| [parallel-reduction](qwen3-4b-eagle3-parallel-reduction.json) | Parallel sampling without mapped draft outputs |
| [parallel-compact](qwen3-4b-eagle3-parallel-compact.json) | Compact GPU commitment |
| [parallel-confidence](qwen3-4b-eagle3-parallel-confidence.json) | Confidence filtering |
| [parallel-fused-root](qwen3-4b-eagle3-parallel-fused-root.json) | Fused root sampling |

## Earlier I/O and sampling series

| Report | Experiment |
| --- | --- |
| [mapped-io](qwen3-4b-eagle3-mapped-io.json) | Initial completed mapped-I/O prototype |
| [mapped-compact](qwen3-4b-eagle3-mapped-compact.json) | Mapped compact commitment |
| [mapped-materialized](qwen3-4b-eagle3-mapped-materialized.json) | Materialized scores, original draft head |
| [mapped-calibrated-materialized](qwen3-4b-eagle3-mapped-calibrated-materialized.json) | Materialized scores, calibrated head |
| [mapped-calibrated-gears](qwen3-4b-eagle3-mapped-calibrated-gears.json) | Separate bicycle-gears prompt |
| [mapped-calibrated-two](qwen3-4b-eagle3-mapped-calibrated-two.json) | Two proposals |
| [sampling-beam](qwen3-4b-eagle3-sampling-beam.json) | Sampler-only BEAM search |

## Calibration and projection controls

| Report | Experiment |
| --- | --- |
| [calibrated-one](qwen3-4b-eagle3-calibrated-one.json) | Calibrated head, one proposal, before I/O improvements |
| [calibrated-two](qwen3-4b-eagle3-calibrated-two.json) | Calibrated head, two proposals, before I/O improvements |
| [fused-projections](qwen3-4b-eagle3-fused-projections.json) | Earlier fused-projection control |
| [detached-prefill](qwen3-4b-eagle3-detached-prefill.json) | Detached prefill snapshots |
| [meta12-experiment](qwen3-4b-eagle3-meta12-experiment.json) | Repacked Q4 metadata |
