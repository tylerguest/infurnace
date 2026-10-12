"""Real GGUF greedy controls against a separately owned CPU llama-server.

Diagnostic correctness protocol, not a throughput benchmark. Reports aborted
runs explicitly. Never resets the GPU or terminates an existing server.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request

from tinygrad import Device
from infurnace.loader import load_model as _build


def post(url, body):
  request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
  with urllib.request.urlopen(request, timeout=180) as response: return json.load(response)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("model", type=Path)
  parser.add_argument("--llama-server", type=Path, required=True)
  parser.add_argument("--device", default="NV")
  parser.add_argument("--output", type=int, default=32)
  parser.add_argument("--context", type=int, default=256)
  parser.add_argument("--json", type=Path, required=True)
  args = parser.parse_args()
  if args.output <= 0 or args.context <= args.output: parser.error("invalid output/context limits")
  report = {"status": "running", "model": str(args.model.resolve()), "device": args.device,
    "target_implementation": "unmodified tinygrad Transformer.from_gguf",
    "output_limit": args.output, "context": args.context, "cases": [], "comparison": "same-GGUF CPU llama.cpp greedy"}
  def save():
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
  save()
  process = None
  log = None
  try:
    with args.model.open("rb") as handle: report["sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    model, tokenizer = _build(str(args.model), args.context, args.device)
    Device[model.device].synchronize()
    report["packed_linears"] = getattr(model, "nv_quant_linears", 0)
    with socket.socket() as sock:
      sock.bind(("127.0.0.1", 0))
      port = sock.getsockname()[1]
    log_path = args.json.with_suffix(".llama.log")
    log = log_path.open("w")
    process = subprocess.Popen([str(args.llama_server.resolve()), "-m", str(args.model.resolve()), "-ngl", "0", "-t", "8",
      "-c", str(args.context), "--parallel", "1", "--host", "127.0.0.1", "--port", str(port)], stdout=log, stderr=log)
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 120
    while True:
      if process.poll() is not None: raise RuntimeError(f"owned llama-server exited; see {log_path}")
      try:
        with urllib.request.urlopen(url + "/health", timeout=2) as response:
          if response.status == 200: break
      except (urllib.error.URLError, TimeoutError): pass
      if time.monotonic() > deadline: raise TimeoutError("llama-server startup timed out")
      time.sleep(0.1)
    prompts = ["The capital of France is", "One, two, three,", "A bicycle has two wheels because",
      tokenizer.render_chat([{"role": "user", "content": "What is 2 + 2? Answer briefly."}], enable_thinking=False)]
    for index, prompt in enumerate(prompts):
      ids = tokenizer.encode(prompt)
      reference = post(url + "/completion", {"prompt": ids, "n_predict": args.output, "temperature": 0.0,
        "top_k": 1, "cache_prompt": False, "return_tokens": True, "seed": 1})
      expected = reference["tokens"]
      if not isinstance(expected, list) or any(type(t) is not int for t in expected): raise ValueError("invalid reference tokens")
      runs = []
      for replay in range(3):
        stream = model.generate(ids, args.output, reuse_cache=False)
        actual = []
        try:
          for token in stream:
            # return_tokens includes EOS even though content omits it.
            actual.append(token)
            if tokenizer.is_end(token): break
        finally: stream.close()
        Device[model.device].synchronize()
        runs.append(actual)
        print(f"case {index}, replay {replay}: {len(actual)} tokens, reference_equal={actual == expected}", flush=True)
      case = {"prompt": prompt, "prompt_tokens": ids, "reference_tokens": expected, "reference_text": reference["content"],
        "target_runs": runs, "target_text": tokenizer.decode([t for t in runs[0] if not tokenizer.is_end(t)]),
        "reference_equal": all(r == expected for r in runs),
        "reset_replay_equal": all(r == runs[0] for r in runs)}
      report["cases"].append(case)
      save()
    report["status"] = "complete"
    report["all_reference_equal"] = all(c["reference_equal"] for c in report["cases"])
    report["all_reset_replay_equal"] = all(c["reset_replay_equal"] for c in report["cases"])
    save()
  except BaseException as exc:
    report["status"], report["error"] = "aborted", f"{type(exc).__name__}: {exc}"
    save()
    raise
  finally:
    if process is not None and process.poll() is None:
      process.terminate()
      try: process.wait(timeout=10)
      except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    if log is not None: log.close()


if __name__ == "__main__": main()
