# Qwen3.5-4B: upstream tinygrad target

Infurnace uses **tinygrad's existing model implementation directly**:
`tinygrad.llm.model.Transformer.from_gguf()`. The small adapter in
`infurnace/models/qwen35.py` provides only Infurnace's bounded generation/lifecycle API.
There is no local target recurrence, attention, FFN, weight mapping or projection
replacement. The tinygrad pin and separate local tinygrad checkout are unchanged.

The selected file is [Unsloth Qwen3.5-4B Q4_K_M](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF):

- Revision: `e87f176479d0855a907a41277aca2f8ee7a09523`
- File: `Qwen3.5-4B-Q4_K_M.gguf`, 2,740,937,888 bytes
- Verified SHA-256: `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`

```sh
pip install -e '.[qwen35]'
infurnace --device NV --model models/qwen35-validation/Qwen3.5-4B-Q4_K_M.gguf \
  --max-context 128 --no-warmup --prompt 'The capital of France is' --max-tokens 16
```

## Responsibility boundary

Tinygrad owns GGUF tensor loading/dequantization, hybrid layer configuration,
gated delta-net recurrence, full attention, projections, cache reset/prefix reuse,
sampling and JIT. Its GGUF implementation already supports this model and its
Q4_K/Q5_K/Q6_K/Q8_0 tensors. Tiled recurrent head repetition is intentional for
converted GGUF weights.

On NV (default `--nv-quant`), quantized projection weights are swapped for
Infurnace's packed tinygrad UOp kernels (`_NVLinear`: Q4_K/Q5_K/Q6_K GEMV for one row,
tensor-core GEMM for multiple rows), including the output head tied to the token
embedding. Block math, recurrence, attention and caches stay upstream. Upstream's
lazy GGUF path otherwise re-unpacks every quantized weight inside each step's
kernels (about 3 tok/s on the 5060); with packed projections ordinary decode is
about 69 tok/s. Multirow embedding lookups use a host FP16 copy of the table, since
a one-hot gather over the packed 248k-row table scans the whole vocabulary on GPU.

The Qwen3.5 text codec remains separate: GGUF vocab/merges, NFC normalization,
Qwen3.5 pretokenization, byte streaming and basic text chat. The optional
`tokenizers` dependency is a text codec, not a model/inference runtime. Tools and
multimodal messages are not supported. There is no local safetensors model path.

Our own [DFlash reference](dflash.md) observes upstream block outputs. Default
sequential verification never executes rejected inputs. Opt-in layer-major block
verification checkpoints/rolls back speculative state. Neither replaces target
blocks or their arithmetic.

## Validation

- 114 focused CPU tests passed after the upstream DFlash audit, including direct-upstream versus
  wrapper output/RNG parity and exact upstream block/linear types.
- Five NV DFlash tests passed with the upstream target: sequential/block
  output/RNG parity, repeated reset/capture replay, and CPU/NV draft-hidden comparisons.
- The selected real 4B GGUF completed ordinary NV decoding.
- A completed context64/output8, block4, two-warmup-pair/two-repeat DFlash
  protocol matched upstream-baseline output and RNG advancement throughout.
  Its short-request baseline was about 2.95 tok/s and sequential DFlash 2.38;
  this is correctness evidence, not a performance success or general benchmark.
  The draft used the former all-noncausal attention policy; it does not measure
  the newly audited layer-default policy.

The experimental layer-major block protocol aborted after its first exact
warmup with the existing NV signal timeout/`NV_ERR_INVALID_ARGUMENT`. There is
no completed block-verifier performance comparison or reliability claim.
Full-suite validation and broader real-model prompts/contexts are pending. Prior
results from the local target implementation do **not** validate this new target.
They are retained as explicitly superseded history. Successful runs do not prove
that the earlier NV timeout issue is fixed.
