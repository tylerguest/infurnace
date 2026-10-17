# Qwen3.5-4B

The only supported target. Qwen3.5 is a hybrid: of its 32 layers, 24 are recurrent (gated delta net)
and 8 are full attention with 4 KV heads of 256, so its attention cache costs only 32 KiB per token.

## Files

Target: [Unsloth Qwen3.5-4B Q4_K_M](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF)

- Revision: `e87f176479d0855a907a41277aca2f8ee7a09523`
- File: `Qwen3.5-4B-Q4_K_M.gguf`, 2,740,937,888 bytes
- SHA-256: `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`

Draft: see [DFlash](dflash.md).

`--model qwen3.5:4b` (the default) downloads both on first use into `~/.local/share/infurnace/models/`
(`XDG_DATA_HOME` respected); copies already there or under a checkout's `models/` are reused.
`--model PATH` takes a local GGUF, which must be a dense Qwen3.5. Other sizes and quantizations load
but are not validated, and the draft only matches the 4B.

```sh
infurnace                                   # chat; downloads on first use
infurnace -m /path/to/Qwen3.5-4B-Q4_K_M.gguf --no-dflash
infurnace --no-warmup --prompt 'The capital of France is' --max-tokens 16
```

## What infurnace changes

tinygrad's `Transformer.from_gguf` builds the model and owns layer configuration, caches, RoPE and
generation. On an NVIDIA GPU infurnace then:

- keeps Q4_K/Q5_K/Q6_K projection weights packed and swaps in its projection kernels;
- replaces the one-row recurrent and attention layers with fused kernels;
- reads prompts 64 rows at a time;
- saves prompt snapshots so later turns resume where the conversation diverged.

Details: [architecture](architecture.md).

The tokenizer is infurnace's own, for Qwen3.5: GGUF vocabulary and merges, NFC normalization, Qwen3.5
pretokenization, byte-level streaming decode and text-only chat formatting (`tokenizers` is used as
the codec). The HTTP server uses the GGUF's embedded Jinja chat template instead; see
[serving](serve.md).
