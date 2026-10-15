# Architecture

Infurnace runs **Qwen3.5-4B** on tinygrad's own Qwen3.5 implementation (`tinygrad.llm.model`), and
replaces its hot paths with infurnace's kernels on NVIDIA GPUs. It is a single-user application, not
a general inference engine.

## Modules

| Module | Responsibility |
|---|---|
| `cli.py` | Chat (warms up while you type), one raw prompt, the HTTP server, per-step benchmark |
| `serve.py` | tinygrad's OpenAI-compatible server with DFlash, per-request thinking and temperature |
| `loader.py` | Named models and first-use downloads, architecture checks, device dispatch, DFlash loading |
| `device.py` | Device probing; `nvidia()` (NV or CUDA) and the fast-path check |
| `gguf.py` | GGUF parsing (tinygrad's parser) and metadata |
| `tokenizer.py` | Qwen3.5 BPE, streaming decode, text-only chat formatting |
| `models/qwen35.py` | Upstream model loading, packed weights, fused layers, multirow prefill, prompt snapshots |
| `models/linear.py` | `_NVLinear`: packed projection used by both the target and the draft |
| `dflash/` | Draft model, target observation, parallel verification, adaptive drafting |
| `kernels/` | Custom tinygrad UOp kernels (below) |

## Devices

The fast path runs on NVIDIA GPUs through either of tinygrad's NVIDIA backends: `NV` (tinygrad's
own driver interface, Linux; probed first) or `CUDA` (the CUDA driver API). Both need tinygrad's
CUDA C renderer, which compiles with NVRTC from the CUDA toolkit. Without it tinygrad falls back to its
PTX renderer and infurnace runs tinygrad's generic Qwen3.5 code, which is far too slow to use.
Other devices (AMD, Metal, CPU) also run the generic code.

## Kernels

All are tinygrad UOp kernels that render as CUDA C.

| Kernel | Used for |
|---|---|
| `quant_gemv` | One-row projections: Q4_K/Q5_K/Q6_K weights against Q8 activations (`dp4a`). Projections that share an input share one quantization. |
| `quant_gemm` | 16-row projections on FP16 tensor cores (`mma.sync m16n8k16`, FP32 accumulation). The k order is permuted so each lane's tensor-core slots map to contiguous packed bytes. |
| `gdn` | Recurrent (gated delta net) layers: `gdn_decode` for one row, `gdn_prefill` for 16-row prompt chunks, `gdn_chunk` for DFlash verification |
| `attention` | Attention over the active length only: split-KV partials and a merge (after vLLM's unified attention and llama.cpp's combine step). Grouped query heads, causal/non-causal, sliding windows, ring caches. |
| `sampling` | Fused Gumbel-max sampling over all rows (greedy at temperature 0) |
| `embedding` | Token embedding gathered straight from packed weights |

Weights stay packed. Q4_K and Q6_K are rearranged once at load into layouts the kernels read with
wide loads. Qwen3.5's output head is tied to the token embedding and shares its packed storage.
Small parameters (norms, gates) are cached under `~/.cache/infurnace/params/`.

## Plain decoding

- **Prompt:** read 16 rows at a time with `quant_gemm` projections and `gdn_prefill`. Attention layers
  use tinygrad's own code for these rows. The 16-row graph is captured at warmup or for long prompts.
- **Each token:** `quant_gemv` projections, `gdn_decode` and the `attention` kernel (32-key segments).
- Upstream tinygrad still drives generation, its KV cache and its JIT.

## Prompt snapshots

Recurrent state can't be rewound, so a prompt that only partly matches the previous one would be read
from scratch. After each prompt, the recurrent and conv states (~50 MB, and the DFlash draft's position)
are copied to host memory at the prompt's last turn header. Up to 8 are kept. A new prompt resumes from
the newest snapshot it still matches, and snapshots it diverges from are dropped. When the list is full
the oldest goes, except the snapshot at the end of the last user message: Qwen's template drops earlier
replies' thinking when a new user message arrives, so that is where an agent's next message diverges.
Attention caches need no copy: they are indexed by position, and every kept snapshot is a prefix of
what they hold.

## DFlash

A block-diffusion draft proposes 15 tokens and the target verifies them in one 16-row forward. Details
in [DFlash](dflash.md).

## Numerics

- One-row projections round activations to Q8; 16-row projections round them to FP16 and accumulate
  in FP32. Neither equals tinygrad's generic math, and the two sum in different orders, so near-ties
  can flip a greedy token between paths.
- Output quality against the reference has not been measured yet ([roadmap](roadmap.md), priority 2).
- `loader.load_model(..., nv_quant=False)` (or `--no-nv-quant` in the benchmarks) runs the generic path.

## Switches

| Variable | Effect |
|---|---|
| `INFURNACE_DFLASH_DRAFT` | DFlash draft GGUF to use |
| `INFURNACE_DFLASH_FUSED=0` | Unfused DFlash rounds |
| `INFURNACE_MULTIROW_PREFILL=0` | One-row prompt reading for plain decoding |
| `INFURNACE_FUSED_GDN=0`, `INFURNACE_FUSED_ATTENTION=0` | tinygrad's own recurrent and attention layers |
| `INFURNACE_SHARED_Q8=0` | Quantize activations per projection |
| `INFURNACE_PARAM_CACHE=0` | Don't cache small parameters on disk |
| `INFURNACE_TEST_NV=1` | Run GPU tests |
