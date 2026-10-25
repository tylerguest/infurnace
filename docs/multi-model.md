# Multi-model plan

infurnace runs one model today: Qwen3.5-4B Q4_K_M with its DFlash draft. This plan makes it run any small
model tinygrad can load, one family at a time, without slowing the Qwen3.5 path. Work through the phases in
order; check items off as they land.

## What is Qwen3.5-only today

Less than it looks. tinygrad's `Transformer.from_gguf` already loads `llama`, `qwen2`, `qwen3`, `qwen3moe`,
`glm4`, `gpt-oss` and others; infurnace replaces its hot paths with passes applied on top.

| Piece | Where | Model-neutral? |
|---|---|---|
| Packed Q4_K/Q5_K/Q6_K matmuls (`_enable_packed`, `_NVLinear`) | `models/qwen35.py`, `models/linear.py` (now `linear.py`) | yes: per GGUF weight |
| Fused attention, one-row and 64-row chunks (`_fused_attention`) | `models/qwen35.py`, `kernels/attention.py` | yes: QK norm, optional output gate, any `head_dim % 32 == 0` |
| Fused feed-forward, embedding gather, sampling, parameter cache | `models/qwen35.py`, `kernels/` | yes |
| 64-row prompt graph (`_prompt_chunk`) | `models/qwen35.py` | mostly: its GDN masks apply only to GDN layers |
| Fused GDN, `gdn_gates` | `models/qwen35.py`, `kernels/gdn.py` | Qwen3.5 only |
| Prompt snapshots | `models/qwen35.py` | only recurrent models need them (GDN state can't be rewound); attention-only models reuse the KV cache |
| Tokenizer | `tokenizer.py` | hard-coded to the `qwen35` preset |
| Architecture check in `from_gguf`, `loader.py`, `MODELS` | | hard-coded |
| DFlash target checks, verifier GDN operands | `dflash/` | assume Qwen3.5; the rest is generic |

## Target layout

```
infurnace/
  engine.py         Model: upstream Transformer + infurnace's passes, generate, prompt reading, warmup,
                    parameter cache (most of today's qwen35.py; Qwen35Model becomes Model)
  linear.py         _NVLinear: packed projections (models/ holds model families only)
  tokenizer.py      BPETokenizer: HF tokenizers built from the GGUF, with the pretokenizer regex chosen
                    by tokenizer.ggml.pre (qwen35, qwen2, llama-bpe, ...)
  models/
    __init__.py     FAMILIES: general.architecture -> family
    qwen35.py       Qwen3.5: the fused GDN pass, snapshot markers
    qwen3.py        Qwen3: attention-only (no extra passes)
  catalog.py        name -> model URL, optional DFlash draft URL, family (replaces loader.MODELS)
  loader.py         reads general.architecture, picks the family, builds the Model and tokenizer
```

A family is a small object, not a subclass tree:

- `architectures`: the `general.architecture` values it covers
- `accelerate(model)`: its extra passes (Qwen3.5: fused GDN)
- `recurrent`: whether prompts need snapshots
- `turn_markers`: where snapshots go (just before the last turn header)

Everything else lives in the engine. Chat formatting always comes from the GGUF's Jinja template, as the
server already does; the hand-written `Qwen35Tokenizer.render_chat`, used only by the legacy `--prompt`
path, goes away.

## Phases

### P0: split the engine from Qwen3.5 (no behavior change)

- [x] `engine.py`: move the model-neutral parts of `models/qwen35.py` there; rename `Qwen35Model` to `Model`
      (tests, benchmarks and `dflash/` follow)
- [x] Family registry in `models/__init__.py`; `models/qwen35.py` keeps only the GDN pass and snapshot markers
- [x] `loader.py` dispatches on `general.architecture`; an unsupported one names the supported families
- [x] The in-process chat (legacy flags) renders through the GGUF template (`chat.py`); `render_chat` removed
- [x] Gate: CPU and NV tests pass; perplexity unchanged (16-row 16.986, verifier 16.986); `benchmarks/sweep.py`
      within noise of the current numbers
      (2026-10-18: one-row 17.006, 16-row 16.9860, verifier 16.9861; plain 1k/4k prompts 1,828/2,305 tok/s read,
      105/104 tok/s decode; DFlash 1,368/1,677 read, 223/197 decode. `test_multirow_logits_match_one_row` reads
      its prompt eagerly and no longer finishes in 5 minutes, at the previous commit too)

### P1: catalog

- [x] `catalog.py`: entries carry the architecture (the family that runs them) and an optional draft; downloads
      (`pull`, `rm`, what is installed) moved there from `loader.py`
- [x] `infurnace list` shows available models as well as downloaded ones; `pull` takes any entry
- [x] DFlash stays automatic for entries that have a draft and off for the rest; a local GGUF gets a downloaded
      draft of a catalog model with the same architecture

### P2: second model, attention-only (Qwen3-4B Q4_K_M)

Qwen3-4B adds the least new surface: head_dim 128, Q4_K/Q6_K weights, tied embeddings, ChatML, the `qwen2`
tokenizer preset. A failure points at the split, not at the kernels.

- [x] Generic `BPETokenizer`; encodings match llama.cpp's on the quality texts (eval.txt 10,098 tokens and
      multilingual.txt 3,936 tokens identical; llama.cpp skips the NFC step Qwen's tokenizer.json specifies)
- [x] Multirow prompt reading without GDN: the Qwen3 family turns prompt chunks on; `tests/core/test_nv_prompt.py`
      checks chunked against one-row prompts, prefix reuse and a later graph capture on both models
- [x] Prefix reuse through the KV cache; no snapshots for non-recurrent families
- [x] Perplexity against llama.cpp and the speed sweep (below)
- [x] Catalog entry `qwen3:4b` (Qwen's official GGUF); README models table

Measured 2026-10-18 (RTX 5060): perplexity on eval.txt (512 × 8) llama.cpp 27.8028, one-row 27.8171 (+0.05%),
16-row 27.8032; multilingual.txt (512 × 5) llama.cpp 8.0761, one-row 8.0090, 16-row 8.0015 (−0.8 to −0.9%, as
Qwen3.5's −0.4 to −0.5%). Plain decoding:

| Prompt | TTFT | Prompt reading | Decode |
|---|---|---|---|
| 124 | 68 ms | 1,822 tok/s | 108.2 tok/s |
| 1,024 | 360 ms | 2,846 tok/s | 106.7 tok/s |
| 4,024 | 1.56 s | 2,577 tok/s | 97.1 tok/s |
| 8,024 | 3.79 s | 2,116 tok/s | 81.6 tok/s |

Decode falls faster with length than Qwen3.5's (its 36 layers are all attention, against 8 of 32): the decode
attention kernel at head_dim 128 is the next thing to tune. Freeing a model inside a process works (GPU memory
3.67 → 0.80 GB after dropping every reference and `gc.collect()`), which P4 builds on.

### P3: DFlash for attention-only targets (not planned)

Set aside on 2026-10-18: z-lab publishes its Qwen3 drafts as safetensors only, so it would start with a GGUF
conversion, and the user chose to spend the effort elsewhere. Qwen3-4B runs plain decoding.

### P4: switching models in the server

- [x] A request for another model unloads the current one, loads the new one and warms it up; one model on the
      GPU at a time; a failed load puts the previous one back
- [x] The client stops asking to `infurnace stop` first: `run MODEL` downloads the model if needed and has the
      server load it (an empty `/api/generate`, as in Ollama)
- [x] Unload after a configurable idle time (`serve --keep-alive 30m`)

Measured 2026-10-18 (RTX 5060): switching Qwen3.5 → Qwen3 51 s to the first reply (load 11 s, warmup 39 s);
back to Qwen3.5 with DFlash 25 s (kernels already compiled); an unloaded server holds no GPU memory beyond the
desktop's. Along the way: the prompt graph now waits for an idle moment instead of delaying the first reply
(cold start to first reply 69 → 42 s), `stop` waits for the server process to exit, and with an NVIDIA GPU
present a failed open is retried and reported instead of falling back to the CPU (a server restarted right after
`stop` had silently run on the CPU).

### P5: more families

- [ ] Llama 3.2 3B: a different tokenizer and chat template, a test of the generic pieces
- [x] Bonsai-4B (`bonsai:4b`, Ternary-Bonsai-2-4B Q2_0): Qwen3.5-4B's architecture with ternary weights whose
      inputs are Hadamard-rotated (`prism.hadamard` metadata, `hadamard.py`)
  - [x] GGUF parsing in infurnace for types tinygrad lacks; Q2_0 decoding; the rotations before each folded
        projection and the inverse after embedding lookups, checked against llama.cpp's graph
  - [x] Decode: a Q2_0 one-row GEMV (dp4a over 2-bit codes, codes then scales as packed words, 4 warps) and a
        blockwise Walsh-Hadamard kernel (`kernels/hadamard.py`; tinygrad's matmul fused into each rotation's
        producer cost 21-125 us a rotation). 6.6 -> ~140 tok/s
  - [x] Prompt chunks: a Q2_0 tensor-core GEMM over a k order matching the packed codes. ~350 -> 1,340-1,610 tok/s;
        at 64 rows it runs at ~25 TFLOPS, near the FP16 tensor cores' FP32-accumulation ceiling on this card
  - [x] Int8 tensor cores for prompt chunks: ternary weights are exact in int8, and m16n8k32 s8 MMA (inline PTX:
        tinygrad's tensor cores are FP16/BF16/TF32/FP8) runs at 140 T ops/s on this card against 35 TFLOPS for FP16
        with FP32 accumulation. Activations take one int8 scale per 64 values (a Q2_0 block), so a block's two MMAs
        chain in int32 and are scaled once; four 16-token tiles a warp share each decoded weight register. The
        matmuls of a 64-row chunk: 20.0 ms (FP16) -> 10.5 ms; prompt reading 1,337-1,605 -> 1,609-2,052 tok/s;
        perplexity 25.4245 (llama.cpp 25.4209)
  - [x] One embedding lookup for every caller (`Model.lookup`): the Q2_0 token embedding, `Model.embed` and DFlash's
        verifier once used three implementations, and DFlash on Bonsai embedded through the K-quant gather with no
        inverse rotation (wrong replies). `packed_gather` decodes Q2_0 and rejects types it can't decode; a local GGUF
        gets a catalog draft only if it is that model's file, and is served under its catalog name. Decode ~138 ->
        ~148 tok/s, prompt reading ~2,050 -> ~2,240 (no whole-table scan per lookup)
  - [ ] Faster decode: the transform (warp shuffles for the first five stages, Q8 quantization fused in) and the
        GEMV (316 GB/s of ~450)
  - [x] DFlash2 draft (`Ternary-Bonsai-2-4B-DFlash2-BF16.gguf`, PrismML's, in the catalog entry): DFlash v1 plus a
        block convolution around attention and the feed-forward (kernel 2, coefficients projected per position) and
        a selector replacing argmax (top-16 candidates, chosen greedily by a rank-256 score given the previous choice;
        the walk runs on the GPU inside the draft graph, from an anchor buffer the fused round writes). Its BF16
        projections are quantized to Q4_K at load (`quantize.py`: 1.56 GB -> 0.44 GB, acceptance unchanged), and a
        top-k kernel (`kernels/topk.py`) replaces 16 argmax passes (3.9 -> ~0.3 ms). Output identical to plain
        decoding; accepted tokens per round ~2.9 on prose, ~9.7 on code; decode ~181-209 tok/s on the sweep (148
        plain), ~290 on code
  - [ ] Faster DFlash2 rounds (~30 ms each): the 16-row verification runs the int8 Q2_0 GEMM at one 16-token tile,
        where each decoded weight register feeds one MMA (~140 GB/s, ~8.4 ms a forward); the Hadamard transforms and
        ~1,000 small kernels add the rest
- [ ] Bonsai-2B and its third-party Qwen3.5-2B drafts

Measured 2026-10-18 (RTX 5060), Bonsai-4B against llama.cpp on the same GGUF: eval.txt (512 × 8) llama.cpp
25.4209, one-row 25.4294 (+0.03%), 16-row 25.4245 (+0.01%, int8 tensor cores); multilingual.txt (512 × 5) llama.cpp 15.2132, one-row
15.2253, 16-row 15.2242. Plain decoding:

| Prompt | TTFT | Prompt reading | Decode |
|---|---|---|---|
| 124 | 245 ms | 505 tok/s | 137.8 tok/s |
| 1,024 | 636 ms | 1,609 tok/s | 140.1 tok/s |
| 4,024 | 1.97 s | 2,047 tok/s | 132.8 tok/s |
| 8,024 | 3.91 s | 2,052 tok/s | 123.4 tok/s |

The rest of a 64-row chunk (~14 of ~25 ms): the Hadamard transforms (4.0 ms) and int8 quantization (1.2 ms), which
could be one kernel, GDN (4.1 ms, as Qwen3.5's) and ~1,000 small generated kernels.

## Risks

- **Decode attention shape.** The decode kernel was tuned on Qwen3.5's head_dim 256 full-attention layers.
  Qwen3 has 36 attention layers at head_dim 128: expected bandwidth-bound at roughly 90–110 tokens/s plain,
  but unmeasured.
- **Context fits.** Every Qwen3 layer is attention (Qwen3.5: one in four), so its KV cache per token is
  several times larger. `fit_context` already counts it; the automatic context is just smaller.
- **Quant coverage.** Only Q4_K, Q5_K and Q6_K are packed. A GGUF with Q8_0 or Q4_0 projections falls back to
  tinygrad's generic path, far slower. The catalog should list only covered quantizations.

## Open decisions

- Second model: Qwen3-4B (decided 2026-10-18)
- P4 timing: after P2, before P3 (decided 2026-10-18); P3 set aside
