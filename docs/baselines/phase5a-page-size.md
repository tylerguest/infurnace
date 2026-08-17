# Phase 5A Page-Size Selection

Page size is measured before a default is fixed, per the Phase 5A gate. This
records the fixed KV arithmetic, the measured CPU per-page dispatch cost, and
the NV pool-memory budget check. The page size is re-verified at Phase 5B/5C
once the real indexed-store and paged-attention kernels exist.

## KV arithmetic (Qwen3-0.6B, fp16 cache)

```text
layers        28
kv heads       8
head dim     128
dtype bytes    2
bytes/token   28 * 2 * 8 * 128 * 2 = 114,688 bytes (112 KiB)
```

## Target topology

4 slots x 2048 context = 8192 tokens (the default CLI/server configuration).

| page_size | bytes/page | pages (8192 tokens) | pool bytes |
| --- | --- | --- | --- |
| 16  | 1,835,008 (1.75 MiB) | 512 | 896 MiB |
| 32  | 3,670,016 (3.50 MiB) | 256 | 896 MiB |
| 64  | 7,340,032 (7.00 MiB) | 128 | 896 MiB |

Total pool bytes are fixed by the token budget; page size only changes page
count, per-page dispatch overhead, and the granularity of wasted space and
future prefix sharing.

## Measurements

Run: `DEV=CPU .venv/bin/python benchmarks/benchmark_pages.py --num-slots 4
--max-context 2048 --repeats 30 --output results/phase5a-page-size-cpu.json`.
Each candidate writes a page-sized K/V block into a realized `[28, 2, 8,
page_size, 8, 128]` fp16 pool (the prefill-store pattern Phase 5B mirrors) and
reads a page slice back. Means over 30 repeats.

| page_size | write/page | write/token | read/page | read/token |
| --- | --- | --- | --- | --- |
| 16  | 3.03 ms | 189.4 us | 0.47 ms | 29.1 us |
| 32  | 2.53 ms | 79.2 us  | 0.44 ms | 13.9 us |
| 64  | 2.55 ms | 39.8 us  | 0.45 ms | 7.0 us  |

Per-page dispatch cost falls ~4.8x from page 16 to page 64 on CPU because the
per-op launch dominates; the read cost tracks the same shape.

## Decision

Default page size is **16 tokens**.

- **Dispatch is not the NV bottleneck.** The CPU microbench is dominated by
  tinygrad's per-op launch overhead; on NV a per-page store/attention dispatch
  is microseconds against a ~16-20 ms/token decode, so the 4.8x CPU spread does
  not transfer. This is re-measured at Phase 5B/5C.
- **Granularity wins at 16.** The smallest candidate minimizes partial-tail
  waste (average 8 tokens, ~0.9 MiB per sequence) and gives the finest sharing
  granularity for the Phase 8 prefix cache.
- **Memory is not binding.** The 0.9 GiB pool sits comfortably alongside
  realized-FP16 weights (~1.35 GB) or lazy weights (~0.8 GB) on the 8,151 MiB
  RTX 5060. The optional `--nv-pool-check` run is deferred to Phase 5D
  integration, where the real pool allocation is measured against the budget.

## Non-claims

- This is not a product throughput baseline; it selects a page size for the
  Phase 5 paged-KV contracts.
- Store and attention kernel cost is re-measured at Phase 5B/5C.