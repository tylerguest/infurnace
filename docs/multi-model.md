# Multi-model plan

infurnace runs one model today: Qwen3.5-4B Q4_K_M with its DFlash draft. This plan makes it run any small
model tinygrad can load, one family at a time, without slowing the Qwen3.5 path. Work through the phases in
order; check items off as they land.

## What is Qwen3.5-only today

Less than it looks. tinygrad's `Transformer.from_gguf` already loads `llama`, `qwen2`, `qwen3`, `qwen3moe`,
`glm4`, `gpt-oss` and others; infurnace replaces its hot paths with passes applied on top.

| Piece | Where | Model-neutral? |
|---|---|---|
| Packed Q4_K/Q5_K/Q6_K matmuls (`_enable_packed`, `_NVLinear`) | `models/qwen35.py`, `models/linear.py` | yes: per GGUF weight |
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
  tokenizer.py      BPETokenizer: HF tokenizers built from the GGUF, with the pretokenizer regex chosen
                    by tokenizer.ggml.pre (qwen35, qwen2, llama-bpe, ...)
  models/
    __init__.py     FAMILIES: general.architecture -> family
    qwen35.py       Qwen3.5: the fused GDN pass, snapshot markers
    attention.py    attention-only models: qwen3, llama, ... (no extra passes)
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

- [ ] `engine.py`: move the model-neutral parts of `models/qwen35.py` there; rename `Qwen35Model` to `Model`
      (tests, benchmarks and `dflash/` follow)
- [ ] Family registry in `models/__init__.py`; `models/qwen35.py` keeps only the GDN pass and snapshot markers
- [ ] `loader.py` dispatches on `general.architecture`; an unsupported one names the supported families
- [ ] Legacy `--prompt` path renders through the GGUF template; `render_chat` removed
- [ ] Gate: CPU and NV tests pass; perplexity unchanged (16-row 16.986, verifier 16.986); `benchmarks/sweep.py`
      within noise of the current numbers

### P1: catalog

- [ ] `catalog.py`: entries carry the family and an optional draft
- [ ] `infurnace list` shows available models as well as downloaded ones; `pull` takes any entry
- [ ] DFlash stays automatic for entries that have a draft and off for the rest

### P2: second model, attention-only (Qwen3-4B Q4_K_M)

Qwen3-4B adds the least new surface: head_dim 128, Q4_K/Q6_K weights, tied embeddings, ChatML, the `qwen2`
tokenizer preset. A failure points at the split, not at the kernels.

- [ ] Generic `BPETokenizer`; encodings match llama.cpp's on the quality texts
- [ ] Multirow prompt reading without GDN: `_multirow_prefill` is only set inside the fused-GDN branch today,
      so attention-only models need their own switch, plus a test that chunked and one-row prompts give the
      same caches
- [ ] Prefix reuse through the KV cache; no snapshots for non-recurrent families
- [ ] Perplexity against llama.cpp (`benchmarks/quality.py`); speed sweep (decode, prompt reading, TTFT)
- [ ] Catalog entry `qwen3:4b`; README Models line

### P3: DFlash for attention-only targets

- [ ] Verifier without GDN operands: rolling back is only moving the position
- [ ] A GGUF draft: z-lab publishes Qwen3 DFlash drafts as safetensors; find or convert a GGUF
- [ ] Acceptance and speed measured against plain decoding; the draft policy falls back when it loses

### P4: switching models in the server

- [ ] A request for another model unloads the current one, loads the new one and warms it up (about a
      minute); one model on the GPU at a time
- [ ] The client stops asking to `infurnace stop` first
- [ ] Unload after a configurable idle time

### P5: more families

- [ ] Llama 3.2 3B: a different tokenizer and chat template, a test of the generic pieces
- [ ] Bonsai (Ternary-Bonsai-2-2B): Q2_0 ternary weights need new GEMV/GEMM kernels; its own project

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
- P4 timing: swap models on request now, or keep "stop it first" until P2 lands
