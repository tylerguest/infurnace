#!/usr/bin/env python3
"""Decompose the C1 decode replay and engine step into measured pieces.

Reuses the profile_decode engine/warmup setup, then probes the captured
shape-1 decode TinyJit and the serving engine step with sync-bracketed and
host-dispatch timing:

  replay probes (per JIT contract, sync-bracketed and amortized):
    baseline           stock C1 decode graph
    attention-noop     per-layer SDPA replaced by the identity (KV store +
                       cached gather still run)
    generic            INFURNACE_DISABLE_QUANT_GEMV=1 (fp16 matmul path)
    position sweep     baseline at a few KV lengths

  per-replay kernel count from tinygrad GlobalCounters deltas, and a
  host-dispatch-only replay (jit() without a trailing sync).

  engine bisect (one serving decode step, per-piece median):
    scheduler.schedule(), input Tensor build, input_ids.tolist(), padded
    rebuild, JIT dispatch (no sync), sampler argmax+sync, detokenizer,
    commit. The pieces should sum to the bracketed engine step; the residue
    is the unidentified host cost.

Run on the pinned checkpoint:

    DEV=NV JIT=1 .venv/bin/python tools/profile_decode_probes.py \
      --manifest models/qwen3-4b-q4_k_m.json \
      --artifact artifacts/models/Qwen3-4B-Q4_K_M.gguf \
      --output results/infurnace-4b-decode-probes.json
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

def median_ms(samples_s: list[float]) -> float:
  return statistics.median(samples_s) * 1e3

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

def make_prompt(length: int) -> list[int]:
  return [257] + [1000 + index for index in range(length - 1)]

def amortized_ms(device, fxn, chain: int = 16, reps: int = 5) -> float:
  samples = []
  for _ in range(reps):
    device.synchronize()
    started = time.perf_counter()
    for _ in range(chain):
      fxn()
    device.synchronize()
    samples.append((time.perf_counter() - started) / chain)
  return median_ms(samples)

def bracketed_ms(device, fxn, reps: int = 5) -> float:
  samples = []
  for _ in range(reps):
    device.synchronize()
    started = time.perf_counter()
    fxn()
    device.synchronize()
    samples.append(time.perf_counter() - started)
  return median_ms(samples)

def kernel_delta(fxn, reps: int = 3) -> int:
  from tinygrad import GlobalCounters
  deltas = []
  for _ in range(reps):
    before = GlobalCounters.kernel_count
    fxn()
    deltas.append(GlobalCounters.kernel_count - before)
  return statistics.median(deltas)

def probe_replay(device, runner, tid: int, position: int, *, label: str,
                 chain: int = 16, reps: int = 5) -> dict:
  from tinygrad import Tensor, Variable, dtypes
  max_context = runner.kv_cache.max_context
  ids = Tensor([[tid]], dtype=dtypes.int32).contiguous().realize()
  bound = Variable("position_0", 0, max_context - 1).bind(position)
  jit = runner._decode_batch_jit[(0,)]
  fxn = lambda: jit(ids, bound)
  amortized = amortized_ms(device, fxn, chain=chain, reps=reps)
  bracketed = bracketed_ms(device, fxn, reps=reps)
  # host-dispatch-only: no trailing sync, so this is SSA executor host time.
  samples = []
  for _ in range(reps):
    started = time.perf_counter()
    fxn()
    samples.append(time.perf_counter() - started)
  dispatch_only = statistics.median(samples) * 1e3
  kernels = kernel_delta(fxn)
  return {
    "label": label, "position": position, "replay_amortized_ms": amortized,
    "replay_bracketed_ms": bracketed, "jit_dispatch_host_ms": dispatch_only,
    "kernels_per_replay": kernels,
  }

def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--manifest", default=str(REPO_ROOT / "models" / "qwen3-4b-q4_k_m.json"), type=Path)
  parser.add_argument("--artifact", default=str(REPO_ROOT / "artifacts" / "models" / "Qwen3-4B-Q4_K_M.gguf"), type=Path)
  parser.add_argument("--max-context", type=int, default=512)
  parser.add_argument("--prompt-tokens", type=int, default=16)
  parser.add_argument("--positions", type=str, default="16,64,219")
  parser.add_argument("--chain", type=int, default=16)
  parser.add_argument("--reps", type=int, default=5)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()

  if os.environ.get("DEV") != "NV":
    print("error: probes require DEV=NV", file=sys.stderr)
    return 1
  if not (args.manifest.exists() and args.artifact.exists()):
    print("error: manifest or artifact not found", file=sys.stderr)
    return 1
  positions = [int(p) for p in args.positions.split(",")]
  if not positions or any(p < 1 or p >= args.max_context for p in positions):
    print("error: positions must be in [1, max_context)", file=sys.stderr)
    return 1

  try:
    manifest = load_manifest(args.manifest)
    with stable_artifact_path(args.artifact) as stable_path:
      verify_artifact(stable_path, manifest)
      engine = build_engine(stable_path, args.manifest, num_slots=1, max_context=args.max_context)
      runner = engine.runner
      model = runner.model

      from tinygrad import Device, Tensor, Variable, dtypes
      device = Device["NV"]

      prompt = make_prompt(args.prompt_tokens)
      from infurnace.engine.request import Request, SamplingParams
      warmup = Request("warmup", prompt, SamplingParams(max_tokens=4), args.max_context)
      engine.add_request(warmup)
      while not engine.is_done():
        engine.step()

      # Prefill slot 0 directly so the probes drive a clean decode state.
      runner.prefill(Tensor([prompt], dtype=dtypes.int32), slot=0, start_position=0)
      position0 = args.prompt_tokens
      tid = 1234

      def run_probe(label: str, position: int, *, attention_noop: bool = False,
                    quant_gemv_disabled: bool = False) -> dict:
        real_sdpa = Tensor.scaled_dot_product_attention
        if attention_noop:
          def noop_sdpa(self, k, v, *kwn, **kw):
            return self
          Tensor.scaled_dot_product_attention = noop_sdpa
        try:
          runner._decode_batch_jit.pop((0,), None)
          if quant_gemv_disabled:
            os.environ["INFURNACE_DISABLE_QUANT_GEMV"] = "1"
          else:
            os.environ.pop("INFURNACE_DISABLE_QUANT_GEMV", None)
          if not quant_gemv_disabled:
            model._packed.clear()
          runner.decode_batch(Tensor([[tid]], dtype=dtypes.int32), [position], [0])
          device.synchronize()
          return probe_replay(device, runner, tid, position, label=label, chain=args.chain, reps=args.reps)
        finally:
          Tensor.scaled_dot_product_attention = real_sdpa
          os.environ.pop("INFURNACE_DISABLE_QUANT_GEMV", None)

      probes = []
      probes.append(run_probe("baseline", position0))
      probes.append(run_probe("attention-noop", position0, attention_noop=True))
      probes.append(run_probe("generic", position0, quant_gemv_disabled=True))
      # restore C1 baseline for the position sweep and engine bisect
      probes.append(run_probe("baseline", max(position0, 16)))
      for position in positions:
        if position == position0: continue
        probes.append(run_probe("baseline", position))

      # --- engine step bisect at a representative position ---
      bisect_position = max(position0, positions[-1])
      bisect_rows: list[dict] = []
      all_token_ids = list(prompt) + [tid]
      from infurnace.tokenizer import StreamingDetokenizer
      detok = StreamingDetokenizer(engine.tokenizer, skip_prompt_len=len(prompt))
      sampler = engine.sampler
      scheduler = engine.scheduler
      sampling_params = SamplingParams(max_tokens=8)
      raw = prompt[-1]
      step_samples: dict[str, list[float]] = {}
      def record(name: str, seconds: float):
        step_samples.setdefault(name, []).append(seconds)
      for _ in range(8):
        total_started = time.perf_counter()
        started = time.perf_counter(); plans = scheduler.schedule(); record("schedule", time.perf_counter() - started)
        started = time.perf_counter(); ids = Tensor([[raw]], dtype=dtypes.int32); record("tensor_build", time.perf_counter() - started)
        started = time.perf_counter(); raw_list = ids.tolist(); record("tolist", time.perf_counter() - started)
        started = time.perf_counter(); padded = Tensor(raw_list, dtype=dtypes.int32).contiguous().realize(); record("padded_build", time.perf_counter() - started)
        started = time.perf_counter()
        bound = [Variable(f"position_{i}", 0, runner.kv_cache.max_context - 1).bind(bisect_position) for i in range(1)]
        record("var_bind", time.perf_counter() - started)
        started = time.perf_counter(); logits = runner._decode_batch_jit[(0,)](padded, *bound); record("jit_dispatch", time.perf_counter() - started)
        started = time.perf_counter(); tokens = sampler.sample_batch(logits, sampling_params); record("sampler_sync", time.perf_counter() - started)
        started = time.perf_counter(); delta = detok.update(all_token_ids); record("detok", time.perf_counter() - started)
        step_samples.setdefault("engine_step", []).append(time.perf_counter() - total_started)
        all_token_ids.append(tokens[0])
      for name, samples in step_samples.items():
        bisect_rows.append({"piece": name, "median_ms": statistics.median(samples) * 1e3})

      result = {
        "schema_version": 1,
        "profile": "infurnace_decode_probes",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {"id": manifest.id, "sha256": manifest.sha256, "size_bytes": manifest.size_bytes},
        "system": {"python": platform.python_version(), "platform": platform.platform(), "device": "NV"},
        "execution": {"max_context": args.max_context, "prompt_tokens": args.prompt_tokens,
                      "positions": positions, "chain": args.chain, "reps": args.reps,
                      "weight_policy": "lazy-fp16"},
        "replay_probes": probes,
        "engine_bisect": {
          "at_position": bisect_position,
          "pieces": bisect_rows,
          "note": "engine_step is the bracketed median of a full serving step; "
                  "jit_dispatch has no trailing sync (SSA executor host time); "
                  "sampler_sync absorbs the device replay + argmax + copy-back",
        },
      }
      serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
      if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent)
        temporary = Path(temporary_name)
        try:
          with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            destination.write(serialized)
            destination.flush()
            os.fsync(destination.fileno())
          os.replace(temporary, args.output)
        except BaseException:
          temporary.unlink(missing_ok=True)
          raise

      print("replay probes (ms per token, shape-1 decode JIT):")
      print(f"  {'label':16s} {'pos':>5s} {'amortized':>10s} {'bracketed':>10s} "
            f"{'jit-dispatch':>12s} {'kernels':>8s}")
      for row in probes:
        print(f"  {row['label']:16s} {row['position']:5d} {row['replay_amortized_ms']:10.2f} "
              f"{row['replay_bracketed_ms']:10.2f} {row['jit_dispatch_host_ms']:12.2f} "
              f"{row['kernels_per_replay']:8d}")
      print(f"\nengine step bisect at position {bisect_position} (median ms):")
      for row in bisect_rows:
        print(f"  {row['piece']:14s} {row['median_ms']:9.2f}")
  except (ArtifactError, ManifestError, OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
    print(f"error: {error}", file=sys.stderr)
    return 1
  return 0

if __name__ == "__main__": raise SystemExit(main())