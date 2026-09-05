#!/usr/bin/env python3
"""Per-shape decode GEMV bandwidth: custom Q4_K/Q6_K kernel vs generic matmul.

Captures each of the model's decode GEMV shapes (attn_q/k/v, attn_output,
ffn_gate/up/down, output) as a TinyJit replay and measures sync-bracketed
bandwidth at the *compressed* weight bytes read per call. Compressed bytes are
the reliable aggregate (the eager per-op microbench in the A3 profile was
inflated by planning overhead); a JIT capture amortizes planning so the
measured time is kernel time.

Run on the pinned checkpoint:

    DEV=NV .venv/bin/python tools/gemv_bench.py \
      --manifest models/qwen3-4b-q4_k_m.json \
      --artifact artifacts/models/Qwen3-4B-Q4_K_M.gguf \
      --output results/infurnace-4b-c1-gemv.json
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

try:
  from tools.gguf_inspection import stable_artifact_path
except ModuleNotFoundError:
  sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
  from gguf_inspection import stable_artifact_path
from infurnace.models.manifest import ArtifactError, ManifestError, load_manifest, verify_artifact

REPO_ROOT = Path(__file__).parents[1]

# compressed bytes per element by GGML quant type (bytes per 256-weight block / 256)
Q4_K, Q6_K = 12, 14
BYTES_PER_ELEM = {Q4_K: 144 / 256, Q6_K: 210 / 256}

def median_ms(samples_s: list[float]) -> float:
  return statistics.median(samples_s) * 1e3

def bracketed_replay(device, jit, *inputs, reps: int = 7) -> list[float]:
  samples = []
  for _ in range(reps):
    device.synchronize()
    started = time.perf_counter()
    jit(*inputs)
    device.synchronize()
    samples.append(time.perf_counter() - started)
  return samples

def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", default=str(REPO_ROOT / "models" / "qwen3-4b-q4_k_m.json"), type=Path)
  parser.add_argument("--artifact", default=str(REPO_ROOT / "artifacts" / "models" / "Qwen3-4B-Q4_K_M.gguf"), type=Path)
  parser.add_argument("--reps", type=int, default=7)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()

  if os.environ.get("DEV") != "NV":
    print("error: gemv_bench requires DEV=NV", file=sys.stderr)
    return 1
  if not (args.manifest.exists() and args.artifact.exists()):
    print("error: manifest or artifact not found", file=sys.stderr)
    return 1

  try:
    from tinygrad import Device, Tensor, TinyJit, dtypes
    from infurnace.executor.tinygrad.weights import WeightPolicy, load_qwen3_checkpoint
    from infurnace.executor.tinygrad.model import Qwen3Model
    from infurnace.kernels.quant_gemv import packed_weight, quant_gemv
    from infurnace.models.config import Qwen3Config

    manifest = load_manifest(args.manifest)
    with stable_artifact_path(args.artifact) as stable_path:
      verify_artifact(stable_path, manifest)
      checkpoint = load_qwen3_checkpoint(stable_path, manifest, WeightPolicy.LAZY_FP16)
    config: Qwen3Config = checkpoint.weights.config
    model = Qwen3Model(checkpoint.weights)
    device = Device["NV"]

    inspection = json.loads((REPO_ROOT / "models" / f"{manifest.id}.inspection.json").read_text(encoding="utf-8"))
    weights_bytes = sum(tensor["serialized_nbytes"] for tensor in inspection["gguf"]["tensors"])

    roles = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
    rows = []

    def measure(name: str, out_features: int, in_features: int, ggml_type: int | None,
                custom_fxn, generic_fxn) -> dict:
      x = Tensor.randn(1, in_features, dtype=dtypes.float32).realize()
      compressed = out_features * in_features * (BYTES_PER_ELEM[ggml_type] if ggml_type is not None else 2)

      if custom_fxn is not None:
        jit = TinyJit(custom_fxn)
        jit(x)
        custom_ms = median_ms(bracketed_replay(device, jit, x, reps=args.reps))

      jit_g = TinyJit(generic_fxn)
      jit_g(x)
      generic_ms = median_ms(bracketed_replay(device, jit_g, x, reps=args.reps))

      row = {
        "name": name, "shape": [1, in_features, out_features],
        "ggml_type": {Q4_K: "Q4_K", Q6_K: "Q6_K"}.get(ggml_type, "none"),
        "compressed_bytes_per_call": compressed,
        "custom_ms": custom_ms if custom_fxn is not None else None,
        "custom_gb_per_s": compressed / (custom_ms / 1e3) / 1e9 if custom_fxn is not None else None,
        "generic_ms": generic_ms,
        "generic_gb_per_s": compressed / (generic_ms / 1e3) / 1e9,
      }
      rows.append(row)
      print(f"  {row['name']:12s} {str(row['shape']):>22s} {row['ggml_type']:5s} "
            f"custom {row['custom_gb_per_s'] if row['custom_gb_per_s'] is not None else 0:6.1f} GB/s "
            f"generic {row['generic_gb_per_s']:6.1f} GB/s")
      return row

    print("per-shape decode GEMV (compressed bytes, JIT-captured replay):")
    for role in roles:
      weight = model._w[f"blk.0.{role}.weight"]
      out_features, in_features = tuple(weight.shape)
      entry = packed_weight(weight)
      ggml_type = entry[1] if entry is not None else None
      packed = entry[0] if entry is not None else None
      custom = (lambda x, p=packed, o=out_features, i=in_features, g=ggml_type:
                quant_gemv(x, p, out_features=o, in_features=i, ggml_type=g)) if packed is not None else None
      generic = (lambda x, w=weight: x.linear(w.transpose()))
      measure(role, out_features, in_features, ggml_type, custom, generic)

    weight = model._w["token_embd.weight"]
    out_features, in_features = tuple(weight.shape)
    entry = packed_weight(weight)
    measure("output", out_features, in_features, entry[1] if entry else None,
            (lambda x, p=entry[0], o=out_features, i=in_features, g=entry[1]:
             quant_gemv(x, p, out_features=o, in_features=i, ggml_type=g)) if entry else None,
            (lambda x, w=weight: x.linear(w.transpose())))

    # whole decode graph replay (custom path is the default now); reuse the
    # profile_decode measurement methodology for the aggregate.
    from infurnace.engine import Engine
    from infurnace.scheduler.scheduler import Scheduler
    from infurnace.sampler import GreedySampler
    from infurnace.tokenizer import GGUFTokenizer
    from infurnace.executor.tinygrad.runner import Qwen3Runner

    tokenizer = GGUFTokenizer.from_gguf_metadata(checkpoint.metadata)
    runner = Qwen3Runner.from_weights(checkpoint.weights, num_slots=1, max_context=512)
    engine = Engine(runner, Scheduler(num_slots=1, max_context=512), GreedySampler(), tokenizer)
    from infurnace.engine.request import Request, SamplingParams
    prompt = [257] + [1000 + i for i in range(15)]
    warmup = Request("warmup", prompt, SamplingParams(max_tokens=4), 512)
    engine.add_request(warmup)
    while not engine.is_done(): engine.step()
    runner.prefill(Tensor([prompt], dtype=dtypes.int32), slot=0, start_position=0)
    position = 16
    ids = Tensor([[1234]], dtype=dtypes.int32)
    engine_request = Request("eng", prompt, SamplingParams(max_tokens=128), 512)
    engine.add_request(engine_request)
    engine.step(); engine.step(); engine.step(); device.synchronize()
    chain = 16
    engine_ms = median_ms(bracketed_replay(device, lambda: [engine.step() for _ in range(chain)], reps=3)) / chain
    samples = []
    for _ in range(3):
      device.synchronize(); started = time.perf_counter()
      for _ in range(chain): runner.decode_batch(ids, [position], [0])
      device.synchronize()
      samples.append((time.perf_counter() - started) / chain)
    replay_ms = median_ms(samples)
    print(f"\nwhole decode graph: engine {engine_ms:.1f} ms/token | "
          f"jit replay {replay_ms:.1f} ms/token | {weights_bytes / (replay_ms/1e3) / 1e9:.1f} GB/s "
          f"(ceiling ~448 GB/s)")

    result = {
      "schema_version": 1,
      "profile": "infurnace_c1_gemv",
      "created_at_utc": datetime.now(timezone.utc).isoformat(),
      "checkpoint": {"id": manifest.id, "sha256": manifest.sha256, "size_bytes": manifest.size_bytes},
      "system": {"python": platform.python_version(), "platform": platform.platform(),
                 "device": os.environ.get("DEV", "NV")},
      "execution": {"reps": args.reps, "weight_policy": "lazy-fp16",
                    "note": "per-shape rows are TinyJit-captured replay at compressed "
                            "weight bytes; custom is the fused Q4_K/Q6_K kernel, "
                            "generic is x.linear(w.T)"},
      "weights_bytes": weights_bytes,
      "gemv": rows,
      "decode_path": {
        "engine_decode_ms_per_token": engine_ms,
        "jit_replay_amortized_ms": replay_ms,
        "jit_replay_gb_per_s": weights_bytes / (replay_ms / 1e3) / 1e9,
      },
    }
    if args.output is not None:
      args.output.parent.mkdir(parents=True, exist_ok=True)
      descriptor, temporary_name = tempfile.mkstemp(prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent)
      temporary = Path(temporary_name)
      try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
          destination.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
          destination.flush()
          os.fsync(destination.fileno())
        os.replace(temporary, args.output)
      except BaseException:
        temporary.unlink(missing_ok=True)
        raise
  except (ArtifactError, ManifestError, OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
    print(f"error: {error}", file=sys.stderr)
    return 1
  return 0

if __name__ == "__main__": raise SystemExit(main())