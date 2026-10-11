"""Single-model local CLI over tinygrad; no HTTP or scheduler."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time
from tinygrad.helpers import GlobalCounters, Timing, profile_marker
from infurnace.device import select_device, supports_nv_quant
from infurnace.kernels.quant_gemm import nv_prefill_supported
from infurnace.benchmark import DEFAULT_PROMPT_TOKENS, benchmark_prompt_ids, validate_benchmark_context
from infurnace.model import Qwen3Model
from infurnace.tokenizer import Tokenizer

_MODEL_FILENAME = "Qwen3-4B-Q4_K_M.gguf"

def _default_model() -> str:
  """Prefer explicit configuration, then user storage, then an editable checkout."""
  if path := os.environ.get("INFURNACE_MODEL"): return str(Path(path).expanduser())
  data_home = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))).expanduser()
  installed_model = data_home / "infurnace/models" / _MODEL_FILENAME
  checkout_model = Path(__file__).resolve().parent.parent / "models" / _MODEL_FILENAME
  if installed_model.is_file(): return str(installed_model)
  if checkout_model.is_file(): return str(checkout_model)
  return str(installed_model)
_SINGLE_SHOT_MAX_TOKENS = 64


def _build(path: str, max_context: int, device: str | None, nv_quant: bool | None = None, *,
           nv_prefill: bool | None = None, nv_attention: bool = False,
           nv_fused_attention: bool | None = None, nv_prepared_weights: bool | None = None) -> tuple[Qwen3Model, Tokenizer]:
  device = select_device(device)
  if Path(path).is_dir(): raise ValueError("--model must name a local GGUF file; Qwen3.5 uses tinygrad's GGUF loader")
  if Path(path).is_file():
    from infurnace.gguf import read_gguf_metadata
    metadata = read_gguf_metadata(path)
    if metadata.get("general.architecture") == "qwen35":
      from infurnace.qwen35 import Qwen35Model
      from infurnace.qwen35_tokenizer import Qwen35Tokenizer
      if any((nv_attention, nv_fused_attention, nv_prepared_weights)):
        raise ValueError("Qwen3.5 supports only --nv-quant/--nv-prefill; upstream tinygrad owns attention and recurrence")
      packed = supports_nv_quant(device) if nv_quant is None else nv_quant
      if packed and not supports_nv_quant(device): raise ValueError("--nv-quant requires NV with the CUDA C renderer")
      if nv_prefill and not packed: raise ValueError("NV prefill requires NV quantized integration")
      tokenizer = Qwen35Tokenizer.from_gguf(metadata)
      model = Qwen35Model.from_gguf(path, max_context=max_context, device=device, nv_quant=packed, nv_prefill=nv_prefill)
      return model, tokenizer
  enabled = supports_nv_quant(device) if nv_quant is None else nv_quant
  if enabled and not supports_nv_quant(device): raise ValueError("--nv-quant requires NV with the CUDA C renderer")
  kwargs = {"max_context": max_context, "device": device}
  if enabled: kwargs["nv_quant"] = True
  if nv_prepared_weights is None: nv_prepared_weights = enabled
  if nv_prepared_weights and not enabled: raise ValueError("--nv-prepared-weights requires NV quantized acceleration")
  if enabled: kwargs["nv_prepared_weights"] = nv_prepared_weights
  prefill_enabled = enabled and nv_prefill_supported(device) if nv_prefill is None else nv_prefill
  if prefill_enabled:
    if not enabled: raise ValueError("--nv-prefill requires NV quantized acceleration")
    if not nv_prefill_supported(device): raise ValueError("--nv-prefill requires CUDA-C FP16 tensor cores")
    kwargs["nv_prefill"] = True
  if nv_fused_attention is None: nv_fused_attention = enabled and not nv_attention
  if nv_attention or nv_fused_attention:
    if not enabled: raise ValueError("NV attention requires NV quantized acceleration")
    if nv_attention: kwargs["nv_attention"] = True
    if nv_fused_attention: kwargs["nv_fused_attention"] = True
  model = Qwen3Model.from_gguf(path, **kwargs)
  return model, Tokenizer.from_gguf(model.metadata)


def _prompt_ids(tokenizer: Tokenizer, prompt: str) -> list[int]:
  ids = tokenizer.encode(prompt)
  return ([tokenizer.bos_id] if tokenizer.bos_id is not None else []) + ids


def _stream(model: Qwen3Model, tokenizer: Tokenizer, ids: list[int], max_tokens: int | None, *,
            eagle=None, draft_tokens: int = 2, dflash=None, dflash_block_size: int = 4) -> str:
  decoder, reply = tokenizer.stream_decoder(), []
  budget = model.config.context_length - len(ids) if max_tokens is None else max_tokens
  if dflash is not None:
    eos = tuple(i for i in (tokenizer.eos_id, tokenizer.eot_id) if i is not None)
    stream = dflash.stream(ids, max_new_tokens=budget, block_size=dflash_block_size, eos_token_ids=eos)
  elif eagle is None:
    stream = model.generate(ids, budget)
  else:
    eos = tuple(i for i in (tokenizer.eos_id, tokenizer.eot_id) if i is not None)
    stream = eagle.stream(ids, max_new_tokens=budget, draft_tokens=draft_tokens, eos_token_ids=eos, batched=True)
  generated, ended, completed = 0, False, False
  try:
    for token in stream:
      generated += 1
      if tokenizer.is_end(token):
        ended = True
        break
      piece = decoder(token)
      reply.append(piece)
      print(piece, end="", flush=True)
    completed = True
  finally:
    stream.close()
    piece = decoder(None)
    reply.append(piece)
    print(piece, end="", flush=True)
    print()
  if completed and not ended:
    remaining = model.config.context_length - len(ids)
    if generated >= remaining:
      print("[Response stopped: context limit reached. Increase --max-context for longer answers.]", file=sys.stderr)
    else:
      print(f"[Response stopped: {budget}-token limit reached (includes thinking). Increase --max-tokens or use --no-thinking.]",
            file=sys.stderr)
  return "".join(reply)


def _run_single_shot(model: Qwen3Model, tokenizer: Tokenizer, prompt: str, max_tokens: int, **stream_options) -> int:
  _stream(model, tokenizer, _prompt_ids(tokenizer, prompt), max_tokens, **stream_options)
  return 0


def _warmup(model: Qwen3Model) -> None:
  print(f"Warming up {model.device} (first startup may take a while)...", file=sys.stderr, flush=True)
  start = time.perf_counter()
  model.warmup()
  print(f"Ready in {time.perf_counter()-start:.1f}s.", file=sys.stderr, flush=True)


def _run_chat(model: Qwen3Model, tokenizer: Tokenizer, max_tokens: int | None, max_context: int,
              enable_thinking: bool = True, **stream_options) -> int:
  history = []
  while True:
    try: text = input(">>> ")
    except EOFError: return 0
    context = min(max_context, model.config.context_length)
    reserve = min(max_tokens if max_tokens is not None else 1024, max(1, context // 2))
    retained, dropped = list(history), 0
    while True:
      messages = retained + [{"role": "user", "content": text}]
      ids = _prompt_ids(tokenizer, tokenizer.render_chat(messages, enable_thinking=enable_thinking))
      if len(ids) < context - reserve or not retained: break
      # CLI history consists of complete user/assistant pairs. Keep newest turns;
      # upstream token-prefix comparison safely invalidates the changed cache.
      retained = retained[2:]
      dropped += 1
    if len(ids) >= context:
      print("Message exceeds context even without history; shorten it or increase --max-context. History was not truncated.", file=sys.stderr)
      continue
    if dropped:
      print(f"[Context window: discarded {dropped} oldest turn(s); earlier details are no longer remembered.]", file=sys.stderr)
    reply = _stream(model, tokenizer, ids, max_tokens, **stream_options)
    history = messages + [{"role": "assistant", "content": reply}]


def _measure(model: Qwen3Model, ids: list[int], count: int) -> dict:
  model.reset()  # Include full prefill, not a cached prompt.
  stream = model.generate(ids, count)
  start = time.perf_counter()
  first = None
  actual = 0
  try:
    for _ in stream:
      now = time.perf_counter()  # Upstream .item() synchronizes each generated token.
      if first is None: first = now
      actual += 1
  finally: stream.close()
  end = time.perf_counter()
  decode_seconds = end - first if first is not None else 0
  ttft = first - start if first is not None else None
  return {
    "prompt_tokens": len(ids), "generated_tokens": actual,
    "ttft_seconds": ttft, "total_seconds": end-start,
    "prefill_tokens_per_second": len(ids)/ttft if ttft else None,
    "decode_tokens_per_second": (actual-1)/decode_seconds if actual > 1 and decode_seconds > 0 else None,
  }


def _benchmark_spec(value: str) -> tuple[int | None, int]:
  try:
    parts = [int(part) for part in value.split(",")]
    if len(parts) not in (1, 2) or any(part <= 0 for part in parts): raise ValueError
  except ValueError:
    raise argparse.ArgumentTypeError("benchmark must be COUNT or PROMPT,COUNT with positive integers") from None
  return (None, parts[0]) if len(parts) == 1 else (parts[0], parts[1])


def _run_step_benchmark(model: Qwen3Model, tokenizer: Tokenizer, count: int, *,
                        prompt_tokens: int, prompt_text: str | None = None, warmup: bool = False) -> int:
  ids = benchmark_prompt_ids(tokenizer, prompt_tokens, prompt_text)
  validate_benchmark_context(len(ids), count, model.config.context_length)
  if warmup: _warmup(model)
  model.reset()
  print(f"device={model.device}, context={model.config.context_length}, prompt={len(ids)}, outputs={count}, "
        f"fused_attention={getattr(model, 'nv_fused_attention', False)}, "
        f"prepared_weights={getattr(model, 'nv_prepared_weights', False)}, "
        f"weight_layout={getattr(model, 'nv_prepared_weight_layout', 'gguf')}, "
        f"shared_q8_blocks={getattr(model, 'nv_shared_q8_blocks', 0)}, warmup={warmup}")
  print("GB/s and traffic MB are tinygrad estimates; allocated MB is current allocation, not GPU peak.")
  stream = model.generate(ids, count)
  decoder = tokenizer.stream_decoder()
  text, decode_seconds, decoded = "", 0.0, 0
  try:
    for step in range(count):
      prefill = step == 0
      profile_marker(f"{'prefill' if prefill else 'decode'} @ {len(ids) if prefill else len(ids)+step-1}")
      GlobalCounters.reset()
      timer = Timing(enabled=False)
      with timer: token = next(stream)  # upstream .item() synchronizes
      seconds = timer.et*1e-9
      traffic, allocated = GlobalCounters.global_mem, GlobalCounters.mem_used
      text += decoder(token)
      if not prefill:
        decode_seconds += seconds
        decoded += 1
      # Formatting/token decoding/terminal output are outside the measured step.
      label = f"prefill {len(ids):5d}" if prefill else f"decode  {step-1:5d}"
      rate = len(ids)/seconds if prefill else 1/seconds
      print(f"{label}: {seconds*1000:8.2f} ms, {rate:8.2f} tok/s, {traffic/seconds/1e9:7.2f} est GB/s, "
            f"{traffic/1e6:.2f}/{allocated/1e6:.2f} MB, active={len(ids)+step}  --  "
            +text.replace("\n", "\\n"), flush=True)
  finally:
    stream.close()
  if decoded:
    print(f"decode aggregate: {decoded/decode_seconds:.2f} tok/s ({decoded} steps; excludes prefill, includes any decode capture)")
  print("Prefill includes first-token work; without --warmup, compilation/capture can appear in timed steps.")
  return 0


def _run_benchmark(model: Qwen3Model, tokenizer: Tokenizer, count: int, load_seconds: float = 0.0, *,
                   prompt_tokens: int = DEFAULT_PROMPT_TOKENS, prompt_text: str | None = None) -> int:
  ids = benchmark_prompt_ids(tokenizer, prompt_tokens, prompt_text)
  validate_benchmark_context(len(ids), count, model.config.context_length)
  cold = _measure(model, ids, count)
  # Two complete passes capture upstream prefill/decode JIT paths before warm timing.
  _measure(model, ids, count)
  warm = [_measure(model, ids, count) for _ in range(3)]
  rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  print(json.dumps({
    "model": model.metadata.get("general.name", "Qwen3"), "device": model.device,
    "context": model.config.context_length, "load_seconds": load_seconds,
    "prompt_tokens": len(ids), "prompt_ids": ids,
    "prompt_source": "raw text" if prompt_text is not None else "synthetic repeated prose",
    "decode_input_positions": [len(ids), len(ids)+count-2] if count > 1 else None,
    "nv_quant_linears": getattr(model, "nv_quant_linears", 0),
    "nv_shared_q8_blocks": getattr(model, "nv_shared_q8_blocks", 0),
    "nv_prefill": getattr(model, "nv_prefill", False),
    "nv_prepared_weights": getattr(model, "nv_prepared_weights", False),
    "nv_prepared_weight_layout": getattr(model, "nv_prepared_weight_layout", "gguf"),
    "nv_attention": getattr(model, "nv_attention", False),
    "nv_fused_attention": getattr(model, "nv_fused_attention", False),
    "cold": cold, "warm_runs": warm,
    "host_peak_rss_bytes": rss if sys.platform == "darwin" else rss*1024,
    "tinygrad_current_allocated_bytes": GlobalCounters.mem_used,
    "notes": "Prefill rate includes first-token work; cold includes compilation. GPU peak is not measured. EOS is ignored for benchmarking.",
  }, indent=2))
  return 0


def main() -> int:
  ap = argparse.ArgumentParser(description="infurnace local Qwen3 runner")
  ap.add_argument("--model", default=_default_model(), help="local Qwen3/Qwen3.5 GGUF (or set INFURNACE_MODEL)")
  mode = ap.add_mutually_exclusive_group()
  mode.add_argument("--prompt", help="raw single-shot prompt (not chat formatted)")
  mode.add_argument("--chat", action="store_true", help="interactive chat (default)")
  mode.add_argument("--benchmark", nargs="?", type=_benchmark_spec, const=(None, 20), metavar="[PROMPT,]COUNT",
                    help="per-step benchmark; default 20 outputs and 512 prompt tokens")
  ap.add_argument("--benchmark-format", choices=("text", "json"), default="text", help="benchmark output (default: text)")
  ap.add_argument("--warmup", action="store_true", help="warm up JIT before benchmarking (chat already warms up)")
  benchmark_prompt = ap.add_mutually_exclusive_group()
  benchmark_prompt.add_argument("--benchmark-prompt-tokens", type=int, help="synthetic prompt length including BOS (default: 512)")
  benchmark_prompt.add_argument("--benchmark-prompt-file", type=Path, help="UTF-8 raw prompt file; not truncated")
  ap.add_argument("--max-tokens", type=int, help="optional generation cap including thinking (chat: remaining context; raw prompt: 64)")
  ap.add_argument("--max-context", type=int, default=2048)
  ap.add_argument("--device", help="override automatic GPU/CPU selection, e.g. CPU or NV")
  acceleration = ap.add_mutually_exclusive_group()
  acceleration.add_argument("--nv-quant", dest="nv_quant", action="store_true", help="require NV quantized acceleration")
  acceleration.add_argument("--no-nv-quant", dest="nv_quant", action="store_false", help="use generic tinygrad linears")
  ap.set_defaults(nv_quant=None)
  prepared = ap.add_mutually_exclusive_group()
  prepared.add_argument("--nv-prepared-weights", dest="nv_prepared_weights", action="store_true",
                        help="require tiled/swizzled Q4/Q6 packing (automatic with NV quantized acceleration)")
  prepared.add_argument("--no-nv-prepared-weights", dest="nv_prepared_weights", action="store_false",
                        help="use native GGUF Q4/Q6 layouts instead of prepared weights")
  ap.set_defaults(nv_prepared_weights=None)
  prefill = ap.add_mutually_exclusive_group()
  prefill.add_argument("--nv-prefill", dest="nv_prefill", action="store_true", help="require packed tensor-core prefill")
  prefill.add_argument("--no-nv-prefill", dest="nv_prefill", action="store_false", help="use generic prefill, retaining NV decode")
  ap.set_defaults(nv_prefill=None)
  ap.add_argument("--nv-attention", action="store_true", help="experimental parallel-softmax single-token NV attention")
  fused = ap.add_mutually_exclusive_group()
  fused.add_argument("--nv-fused-attention", dest="nv_fused_attention", action="store_true",
                     help="require fused GQA decode attention (automatic on NV; supersedes --nv-attention)")
  fused.add_argument("--no-nv-fused-attention", dest="nv_fused_attention", action="store_false",
                     help="disable automatic fused attention")
  ap.set_defaults(nv_fused_attention=None)
  ap.add_argument("--eagle3", type=Path, metavar="CHECKPOINT", help="experimental greedy EAGLE-3 checkpoint directory (NV only)")
  ap.add_argument("--eagle-draft-tokens", type=int, default=None, metavar="N", help="EAGLE descendants per round, 0–7 (default: 1)")
  ap.add_argument("--no-eagle-fast", action="store_true", help="disable fast EAGLE sampling, mapped I/O and projection optimizations")
  ap.add_argument("--dflash", type=Path, metavar="GGUF", help="experimental tinygrad DFlash draft GGUF (Qwen3.5 target, greedy only)")
  ap.add_argument("--dflash-block-size", type=int, default=None, metavar="N",
                  help="DFlash block including anchor (default: trained size with parallel verification, else 4)")
  ap.add_argument("--dflash-legacy-noncausal", action='store_true', help="use older all-noncausal draft attention instead of upstream layer defaults")
  ap.add_argument("--dflash-verifier", choices=('parallel', 'sequential', 'block'), default=None,
                  help="DFlash verification schedule (default: parallel multirow on NV, else sequential; block is experimental)")
  ap.add_argument("--no-warmup", action="store_true", help="skip automatic startup warmup for interactive chat")
  ap.add_argument("--no-thinking", action="store_true", help="disable thinking in chat formatting")
  args = ap.parse_args()
  if args.benchmark is not None:
    inline_prompt, args.benchmark = args.benchmark
    if inline_prompt is not None:
      if args.benchmark_prompt_tokens is not None or args.benchmark_prompt_file is not None:
        ap.error("PROMPT,COUNT conflicts with --benchmark-prompt-* options")
      args.benchmark_prompt_tokens = inline_prompt
  for name in ("max_context", "max_tokens", "benchmark", "benchmark_prompt_tokens"):
    value = getattr(args, name)
    if value is not None and value <= 0: ap.error(f"--{name.replace('_', '-')} must be positive")
  if args.nv_prepared_weights and args.nv_quant is False: ap.error("--nv-prepared-weights conflicts with --no-nv-quant")
  if args.nv_prefill and args.nv_quant is False: ap.error("--nv-prefill conflicts with --no-nv-quant")
  if (args.nv_attention or args.nv_fused_attention) and args.nv_quant is False: ap.error("NV attention conflicts with --no-nv-quant")
  if args.benchmark is None and (args.benchmark_prompt_tokens is not None or args.benchmark_prompt_file is not None):
    ap.error("--benchmark-prompt-* options require --benchmark")
  if args.no_thinking and (args.prompt is not None or args.benchmark is not None): ap.error("--no-thinking applies only to chat")
  if args.dflash is not None:
    if args.eagle3 is not None: ap.error("--dflash and --eagle3 are mutually exclusive")
    if args.benchmark is not None: ap.error("use benchmarks/dflash.py for matched DFlash measurements")
    if not args.dflash.is_file(): ap.error("--dflash must name a local draft GGUF file")
  if args.dflash_verifier is not None and args.dflash is None: ap.error("--dflash-verifier requires --dflash")
  if args.dflash_legacy_noncausal and args.dflash is None: ap.error("--dflash-legacy-noncausal requires --dflash")
  if args.dflash_block_size is not None:
    if args.dflash is None: ap.error("--dflash-block-size requires --dflash")
    if args.dflash_block_size <= 0: ap.error("--dflash-block-size must be positive")
  if args.no_eagle_fast and args.eagle3 is None: ap.error("--no-eagle-fast requires --eagle3")
  if args.eagle_draft_tokens is not None:
    if args.eagle3 is None: ap.error("--eagle-draft-tokens requires --eagle3")
    if not 0 <= args.eagle_draft_tokens <= 7: ap.error("--eagle-draft-tokens must be between zero and seven")
  if args.eagle3 is not None:
    if Path(args.model).is_dir(): ap.error("--eagle3 does not support the Qwen3.5 target")
    if args.benchmark is not None: ap.error("use benchmarks/eagle.py for matched EAGLE measurements")
    if args.nv_quant is False or args.nv_fused_attention is False or args.nv_attention:
      ap.error("--eagle3 requires NV quantized acceleration and fused attention")
    if args.device is not None and args.device.split(':')[0] != 'NV': ap.error("--eagle3 requires --device NV")
    if not (args.eagle3/'config.json').is_file() or not (args.eagle3/'model.safetensors').is_file():
      ap.error("--eagle3 directory must contain config.json and model.safetensors")
  try:
    start = time.perf_counter()
    kwargs = {"nv_prefill": args.nv_prefill} if args.nv_prefill is not None else {}
    if args.nv_prepared_weights is not None: kwargs["nv_prepared_weights"] = args.nv_prepared_weights
    if args.nv_attention: kwargs["nv_attention"] = True
    if args.nv_fused_attention is not None: kwargs["nv_fused_attention"] = args.nv_fused_attention
    model, tokenizer = _build(args.model, args.max_context, args.device, args.nv_quant, **kwargs)
    load_seconds = time.perf_counter()-start
    if args.benchmark is not None:
      text = args.benchmark_prompt_file.read_text(encoding="utf-8") if args.benchmark_prompt_file is not None else None
      if args.benchmark_format == "text":
        return _run_step_benchmark(model, tokenizer, args.benchmark,
                                   prompt_tokens=args.benchmark_prompt_tokens or DEFAULT_PROMPT_TOKENS,
                                   prompt_text=text, warmup=args.warmup)
      if args.warmup: _warmup(model)
      return _run_benchmark(model, tokenizer, args.benchmark, load_seconds,
                            prompt_tokens=args.benchmark_prompt_tokens or DEFAULT_PROMPT_TOKENS, prompt_text=text)
    stream_options = {}
    if args.eagle3 is not None:
      from infurnace.eagle.verify import BatchedVerifier
      from infurnace.eagle import EagleDraft
      from infurnace.eagle.generate import EagleGenerator
      BatchedVerifier(model)  # Fail eligibility checks before loading draft weights.
      eagle = EagleGenerator(model, EagleDraft.from_checkpoint(args.eagle3, model), fast=not args.no_eagle_fast)
      stream_options = {'eagle': eagle, 'draft_tokens': 1 if args.eagle_draft_tokens is None else args.eagle_draft_tokens}
      mode = 'basic batched verifier' if args.no_eagle_fast else 'fast sampling and mapped I/O'
      print(f"EAGLE-3 experimental ({mode}): first-request JIT capture can be slow; NV timeouts remain possible.",
            file=sys.stderr, flush=True)
    elif args.dflash is not None:
      from infurnace.dflash import DFlashDraft, DFlashGenerator
      draft = DFlashDraft.from_gguf(args.dflash, model, **({'legacy_noncausal': True} if args.dflash_legacy_noncausal else {}))
      verifier = args.dflash_verifier or ('parallel' if model.device.split(':')[0] == 'NV' else 'sequential')
      default_block = draft.config.block_size if verifier == 'parallel' else 4
      block = default_block if args.dflash_block_size is None else args.dflash_block_size
      if block > draft.config.block_size: raise ValueError("--dflash-block-size exceeds the checkpoint's trained block size")
      if verifier == 'parallel' and block < 2: raise ValueError("parallel DFlash verification needs --dflash-block-size of at least 2")
      dflash = DFlashGenerator(model, draft, verifier=verifier)
      stream_options = {'dflash': dflash, 'dflash_block_size': block}
      print(f"DFlash experimental: tinygrad {verifier} verification, block {block}; first capture can take minutes. "
            "Draft attention: "
            f"{'legacy all-noncausal' if args.dflash_legacy_noncausal else 'metadata/upstream layer defaults'}.",
            file=sys.stderr, flush=True)
    if args.prompt is not None:
      return _run_single_shot(model, tokenizer, args.prompt, args.max_tokens or _SINGLE_SHOT_MAX_TOKENS, **stream_options)
    if not args.no_warmup: _warmup(model)
    return _run_chat(model, tokenizer, args.max_tokens, args.max_context, not args.no_thinking, **stream_options)
  except KeyboardInterrupt:
    print("\nInterrupted.", file=sys.stderr)
    return 130
  except RuntimeError as exc:
    if 'signal wait timed out' not in str(exc) and 'hang detected' not in str(exc): raise
    print(f"infurnace: device synchronization failed ({exc}). Restart Infurnace; "
          "Speculative decoding is experimental; retry ordinary decode without --eagle3/--dflash. No GPU reset was attempted.", file=sys.stderr)
    return 1
  except MemoryError:
    print("infurnace: out of memory. Stop other model processes or reduce --max-context.", file=sys.stderr)
    return 1
  except FileNotFoundError as exc:
    print(f"infurnace: file not found: {exc.filename}\nPlace the Qwen3 GGUF in {_default_model()} "
          "or pass --model /path/to/model.gguf.",
          file=sys.stderr)
    return 1
  except (ValueError, OSError) as exc:
    print(f"infurnace: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
