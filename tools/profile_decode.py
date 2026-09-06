#!/usr/bin/env python3
"""Profile one warm decode step of the Infurnace engine on DEV=NV.

Builds the pinned Qwen3-4B Q4_K_M engine, captures the shape-1 decode TinyJit
contract, then decomposes the serving decode path by reliable sync-bracketed
timing:

  engine_decode_ms_per_token     engine decode step (sampler, .tolist(), JIT)
  jit_replay_single_ms           one JIT replay bracketed by syncs
  jit_replay_amortized_ms        N JIT replays queued back-to-back / N
  host_overhead_ms_per_token     engine - jit_replay_amortized
  kernel_count                   kernels per decode graph

If jit_replay_amortized is close to engine_decode, decode is device-bound and
the GEMV kernels are the lever. If it is much smaller, host work (the .tolist()
roundtrip, sampler, scheduling) dominates and persistent device buffers matter
more. Per-shape GEMV device times (amortized over a chained graph) and
attention give the kernel tuning baseline.
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

def make_prompt(length: int) -> list[int]:
  return [257] + [1000 + index for index in range(length - 1)]

def build_engine(artifact: str, manifest_path: str, *, num_slots: int, max_context: int):
  from infurnace.engine import Engine
  from infurnace.scheduler.scheduler import Scheduler
  from infurnace.sampler import GreedySampler
  from infurnace.tokenizer import GGUFTokenizer
  from infurnace.executor.tinygrad.runner import Qwen3Runner
  from infurnace.executor.tinygrad.weights import WeightPolicy, load_qwen3_checkpoint
  manifest = load_manifest(manifest_path)
  checkpoint = load_qwen3_checkpoint(artifact, manifest, WeightPolicy.LAZY_FP16)
  tokenizer = GGUFTokenizer.from_gguf_metadata(checkpoint.metadata)
  runner = Qwen3Runner.from_weights(checkpoint.weights, num_slots=num_slots, max_context=max_context)
  return Engine(runner, Scheduler(num_slots=num_slots, max_context=max_context), GreedySampler(), tokenizer)

def median_ms(samples_s: list[float]) -> float:
  return statistics.median(samples_s) * 1e3

def bracketed(device, fxn, reps: int = 7) -> list[float]:
  samples = []
  for _ in range(reps):
    device.synchronize()
    started = time.perf_counter()
    fxn()
    device.synchronize()
    samples.append(time.perf_counter() - started)
  return samples

def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", default=str(REPO_ROOT / "models" / "qwen3-4b-q4_k_m.json"), type=Path)
  parser.add_argument("--artifact", default=str(REPO_ROOT / "artifacts" / "models" / "Qwen3-4B-Q4_K_M.gguf"), type=Path)
  parser.add_argument("--max-context", type=int, default=512)
  parser.add_argument("--prompt-tokens", type=int, default=16)
  parser.add_argument("--decode-position", type=int, default=200)
  parser.add_argument("--reps", type=int, default=5)
  parser.add_argument("--chain", type=int, default=16)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()

  if os.environ.get("DEV") != "NV":
    print("error: profile requires DEV=NV", file=sys.stderr)
    return 1
  if not (args.manifest.exists() and args.artifact.exists()):
    print("error: manifest or artifact not found", file=sys.stderr)
    return 1

  try:
    manifest = load_manifest(args.manifest)
    with stable_artifact_path(args.artifact) as stable_path:
      verify_artifact(stable_path, manifest)
      engine = build_engine(stable_path, args.manifest, num_slots=1, max_context=args.max_context)
      runner = engine.runner
      model = runner.model
      config = model.config

      from tinygrad import Device, Tensor, dtypes
      device = Device["NV"]

      prompt = make_prompt(args.prompt_tokens)
      from infurnace.engine.request import Request, SamplingParams
      warmup = Request("warmup", prompt, SamplingParams(max_tokens=4), args.max_context)
      engine.add_request(warmup)
      while not engine.is_done():
        engine.step()

      runner.prefill(Tensor([prompt], dtype=dtypes.int32), slot=0, start_position=0)
      position = args.prompt_tokens
      ids = Tensor([[1234]], dtype=dtypes.int32)

      # --- engine decode steps on one request (sampler, .tolist(), JIT) ---
      engine_request = Request("eng", prompt, SamplingParams(max_tokens=args.chain + 256), args.max_context)
      engine.add_request(engine_request)
      engine.step()  # prefill + first token
      engine.step()  # one warm decode (JIT captured)
      engine.step()
      device.synchronize()
      engine_decode_ms = median_ms(bracketed(device, lambda: [engine.step() for _ in range(args.chain)], reps=args.reps)) / args.chain

      # --- JIT replay: one call, bracketed ---
      runner.decode_batch(ids, [position], [0])
      device.synchronize()
      jit_single_ms = median_ms(bracketed(device, lambda: runner.decode_batch(ids, [position], [0]), reps=args.reps))

      # --- JIT replay amortized: N calls queued back-to-back, one trailing sync ---
      samples = []
      for _ in range(args.reps):
        device.synchronize()
        started = time.perf_counter()
        for _ in range(args.chain):
          runner.decode_batch(ids, [position], [0])
        device.synchronize()
        samples.append((time.perf_counter() - started) / args.chain)
      jit_amortized_ms = median_ms(samples)

      # Kernel count of the captured shape-1 decode graph. JIT replays run with
      # update_stats=False, so read the captured graph's kernel calls directly.
      jit = runner._decode_batch_jit[(0,)]
      kernel_count = len(jit.captured.linear.src)  # top-level CALL nodes (SSA executor)

      # --- per-shape GEMV device time amortized over a chained graph ---
      def chained_ms(weight) -> float:
        out_features, in_features = tuple(weight.shape)
        inputs = x if in_features == config.embedding_length else Tensor.randn(1, in_features, dtype=dtypes.float32).realize()
        weight_transposed = weight.transpose()
        samples = []
        for _ in range(args.reps):
          device.synchronize()
          started = time.perf_counter()
          outputs = [inputs.linear(weight_transposed) for _ in range(args.chain)]
          Tensor.realize(*outputs)
          device.synchronize()
          samples.append((time.perf_counter() - started) / args.chain)
        return median_ms(samples)

      x = Tensor.randn(1, config.embedding_length, dtype=dtypes.float32).realize()
      roles = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
      gemv_rows = []
      gemv_sum_ms = 0.0
      for role in roles:
        weight = model._w[f"blk.0.{role}.weight"]
        out_features, in_features = tuple(weight.shape)
        ms = chained_ms(weight)
        count = config.block_count
        per_call_bytes = out_features * in_features * 2
        gemv_rows.append({
          "role": role, "shape": [1, in_features, out_features], "storage": weight.dtype.name,
          "count": count, "ms_per_call": ms, "total_ms": ms * count,
          "achieved_gb_per_s": per_call_bytes / (ms / 1e3) / 1e9,
        })
        gemv_sum_ms += ms * count
      weight = model._w["token_embd.weight"]
      out_features, in_features = tuple(weight.shape)
      ms = chained_ms(weight)
      per_call_bytes = out_features * in_features * 2
      gemv_rows.append({
        "role": "output", "shape": [1, in_features, out_features], "storage": weight.dtype.name,
        "count": 1, "ms_per_call": ms, "total_ms": ms,
        "achieved_gb_per_s": per_call_bytes / (ms / 1e3) / 1e9,
      })
      gemv_sum_ms += ms

      # --- attention at a representative position ---
      def chained_attention():
        kv = runner.kv_cache.kv
        q = Tensor.randn(1, 1, config.attention_head_count, config.key_length, dtype=dtypes.float32).realize()
        k = kv[0, 0, 0, :position].permute(1, 0, 2).unsqueeze(0).float()
        v = kv[0, 1, 0, :position].permute(1, 0, 2).unsqueeze(0).float()
        samples = []
        for _ in range(args.reps):
          device.synchronize()
          started = time.perf_counter()
          for _ in range(args.chain):
            q.scaled_dot_product_attention(k, v, is_causal=False, enable_gqa=True).realize()
          device.synchronize()
          samples.append((time.perf_counter() - started) / args.chain)
        return median_ms(samples)
      attention_ms = chained_attention()

      inspection = json.loads((REPO_ROOT / "models" / f"{manifest.id}.inspection.json").read_text(encoding="utf-8"))
      weights_bytes = sum(tensor["serialized_nbytes"] for tensor in inspection["gguf"]["tensors"])

      result = {
        "schema_version": 1,
        "profile": "infurnace_decode_step",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {"id": manifest.id, "sha256": manifest.sha256, "size_bytes": manifest.size_bytes},
        "system": {"python": platform.python_version(), "platform": platform.platform(),
                   "device": os.environ.get("DEV", "NV")},
        "execution": {"max_context": args.max_context, "prompt_tokens": args.prompt_tokens,
                      "decode_position": position, "reps": args.reps, "chain": args.chain,
                      "weight_policy": "lazy-fp16"},
        "decode_path": {
          "engine_decode_ms_per_token": engine_decode_ms,
          "jit_replay_single_ms": jit_single_ms,
          "jit_replay_amortized_ms": jit_amortized_ms,
          "host_overhead_ms_per_token": max(0.0, engine_decode_ms - jit_amortized_ms),
          "kernel_count": kernel_count,
          "jit_amortized_gb_per_s": weights_bytes / (jit_amortized_ms / 1e3) / 1e9,
          "note": "engine_decode is the serving path (sampler, .tolist(), JIT); "
                  "jit_replay_amortized queues N replays with one trailing sync so it "
                  "approximates device+launch time; the difference is host work per token; "
                  "kernel_count is the captured graph top-level CALL nodes, not raw kernels",
        },
        "gemv": {
          "roles": gemv_rows,
          "note": "isolated per-shape microbench; absolute ms are inflated by per-op "
                  "eager planning overhead (the whole decode graph is ~jit_replay_amortized ms, "
                  "not the sum here). The relative picture is the signal: the large output "
                  "GEMV runs near the bandwidth ceiling while the small per-layer GEMVs do not.",
        },
        "attention": {"per_layer_ms": attention_ms, "layers": config.block_count,
                      "position": position,
                      "note": "isolated microbench; inflated like the GEMVs above"},
      }
      json_bytes = json.dumps(result, indent=2, sort_keys=True) + "\n"
      if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent)
        temporary = Path(temporary_name)
        try:
          with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            destination.write(json_bytes)
            destination.flush()
            os.fsync(destination.fileno())
          os.replace(temporary, args.output)
        except BaseException:
          temporary.unlink(missing_ok=True)
          raise

      print(f"decode path per token: engine {engine_decode_ms:.1f} ms | "
            f"JIT replay single {jit_single_ms:.1f} ms | "
            f"JIT replay amortized {jit_amortized_ms:.1f} ms | "
            f"host overhead {max(0.0, engine_decode_ms - jit_amortized_ms):.1f} ms | "
            f"{kernel_count} top-level CALL nodes in the captured graph")
      print(f"  jit amortized bandwidth: {result['decode_path']['jit_amortized_gb_per_s']:.1f} GB/s "
            f"(ceiling ~448 GB/s)")
      print("\nisolated GEMV microbench (relative; inflated absolute ms, see notes):")
      for row in gemv_rows:
        print(f"  {row['role']:12s} {str(row['shape']):>22s} "
              f"{row['ms_per_call']:6.3f} ms/call  {row['achieved_gb_per_s']:6.1f} GB/s")
      print(f"  attention: {attention_ms:.3f} ms/layer (isolated, inflated)")
  except (ArtifactError, ManifestError, OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
    print(f"error: {error}", file=sys.stderr)
    return 1
  return 0

if __name__ == "__main__": raise SystemExit(main())