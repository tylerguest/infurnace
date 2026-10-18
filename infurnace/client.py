"""Talking to an infurnace server: find it (or start one in the background), chat with it,
list and stop it, and point agents at it.

The server stays warm between commands, so only its first start pays the GPU warmup. INFURNACE_HOST picks the
server (default 127.0.0.1:8000).
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator

DEFAULT_HOST = "127.0.0.1:8000"

class ServerError(Exception): pass

def host() -> str:
  """The server's address as host:port."""
  value = os.environ.get("INFURNACE_HOST", DEFAULT_HOST).strip()
  value = value.split("://", 1)[-1].rstrip("/")
  return value if ":" in value else f"{value}:8000"

def base_url() -> str: return f"http://{host()}"

def _error(exc: urllib.error.HTTPError) -> ServerError:
  """The server's {"error": ...} message as a ServerError."""
  try: message = json.loads(exc.read()).get("error", str(exc))
  except (ValueError, AttributeError): message = str(exc)
  return ServerError(message if isinstance(message, str) else json.dumps(message))

def request(method: str, path: str, body: dict | None = None, timeout: float | None = 10) -> dict:
  data = json.dumps(body).encode() if body is not None else None
  req = urllib.request.Request(base_url() + path, data=data, method=method, headers={"Content-Type": "application/json"})
  try:
    with urllib.request.urlopen(req, timeout=timeout) as response: return json.loads(response.read() or b"{}")
  except urllib.error.HTTPError as exc: raise _error(exc) from exc

def stream(path: str, body: dict) -> Iterator[dict]:
  """POST and yield each record of a newline-delimited JSON reply. Closing the iterator closes the connection,
  which stops generation on the server."""
  req = urllib.request.Request(base_url() + path, data=json.dumps(body).encode(), method="POST", headers={"Content-Type": "application/json"})
  try: response = urllib.request.urlopen(req, timeout=None)
  except urllib.error.HTTPError as exc: raise _error(exc) from exc
  with response:
    for line in response:
      if line.strip(): yield json.loads(line)

def status() -> dict | None:
  """The server's /api/status, or None when nothing listens. A server that accepts the connection but is slow to
  answer is busy, not gone."""
  try: return request("GET", "/api/status", timeout=5)
  except TimeoutError: return {"status": "busy"}
  except urllib.error.URLError as exc:
    if isinstance(exc.reason, TimeoutError): return {"status": "busy"}
    return None
  except (ConnectionError, OSError): return None

def log_path() -> Path:
  return Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state"))).expanduser() / "infurnace/serve.log"

def start_server(serve_args: list[str]) -> None:
  """Start `infurnace serve` detached, logging to log_path(), and wait until it answers."""
  log = log_path()
  log.parent.mkdir(parents=True, exist_ok=True)
  hostname, port = host().rsplit(":", 1)
  command = [sys.executable, "-m", "infurnace.cli", "serve", "--host", hostname, "--port", port, *serve_args]
  print(f"Starting the infurnace server in the background (log: {log})...", file=sys.stderr, flush=True)
  with open(log, "ab") as out:
    process = subprocess.Popen(command, stdout=out, stderr=out, stdin=subprocess.DEVNULL, start_new_session=True)
  deadline = time.monotonic() + 300
  while time.monotonic() < deadline:
    if status() is not None: return
    if process.poll() is not None: raise ServerError(f"the server exited during startup; see {log}")
    time.sleep(0.5)
  raise ServerError(f"the server did not start within 5 minutes; see {log}")

def connect(model: str | None, serve_args: list[str]) -> dict:
  """The running server's status, starting one when none answers. A different model than the one it serves is an error."""
  current = status()
  if current is None:
    start_server((["--model", model] if model else []) + serve_args)
    current = status() or {}
  # A GGUF path is served under its file name's stem
  if model and current.get("model") not in (None, model, Path(model).stem if model.endswith(".gguf") else model):
    raise ServerError(f"the server at {host()} is serving {current.get('model')}; stop it first (infurnace stop) to run {model}")
  return current

def wait_ready() -> None:
  """Block until the server has finished warming up, saying so once if it has not."""
  told = False
  while (current := status()) is not None and current.get("status") == "warming up":
    if not told:
      print("Warming up the GPU (about a minute on a fresh start)...", file=sys.stderr, flush=True)
      told = True
    time.sleep(0.5)
  if current is None: raise ServerError(f"the server at {host()} went away; see {log_path()}")
  if current.get("status") == "failed": raise ServerError(f"the server failed to warm up: {current.get('error')}")

# --- run: one prompt, or an interactive chat --------------------------------------------------------------------

DIM, RESET = "\x1b[2m", "\x1b[0m"

def _reply(messages: list[dict], model: str, think: bool | None, options: dict, show_thinking: bool) -> str:
  """Stream one reply to stdout (thinking dimmed, when shown); returns its content."""
  body = {"model": model, "messages": messages, "stream": True, "options": options, **({"think": think} if think is not None else {})}
  content, thinking_open, records = [], False, stream("/api/chat", body)
  try:
    for record in records:
      message = record.get("message", {})
      if (thinking := message.get("thinking")) and show_thinking:
        if not thinking_open: print(DIM, end="")
        thinking_open = True
        print(thinking, end="", flush=True)
      if text := message.get("content"):
        if thinking_open:
          print(RESET + "\n", end="")
          thinking_open = False
        print(text, end="", flush=True)
        content.append(text)
      for call in message.get("tool_calls", []):
        print(f"\n[tool call] {call['function']['name']} {json.dumps(call['function']['arguments'])}", end="", flush=True)
      if record.get("done") and record.get("done_reason") == "length":
        print("\n[Response stopped: token limit reached.]", file=sys.stderr, end="")
  finally:
    records.close()
    if thinking_open: print(RESET, end="")
    print(flush=True)
  return "".join(content)

HELP = """Commands:
  /set think | /set nothink      turn thinking on or off
  /set temperature <float>       sampling temperature (0: greedy)
  /clear                         forget the conversation
  /bye                           exit
  \"\"\"                            begin and end a multi-line message"""

def _read_message() -> str | None:
  try: line = input(">>> ")
  except EOFError: return None
  if not line.strip().startswith('"""'): return line
  lines, first = [], line.strip()[3:]
  if first.endswith('"""'): return first[:-3]
  if first: lines.append(first)
  while True:
    try: line = input("... ")
    except EOFError: return "\n".join(lines)
    if line.rstrip().endswith('"""'):
      lines.append(line.rstrip()[:-3])
      return "\n".join(lines)
    lines.append(line)

def chat(model: str, think: bool | None, options: dict) -> int:
  """Interactive chat. When a conversation outgrows the context, its oldest turns are dropped."""
  history: list[dict] = []
  print("Send a message (/? for help)", file=sys.stderr)
  while True:
    text = _read_message()
    if text is None: return 0
    command = text.strip()
    if command in ("/bye", "/exit"): return 0
    if command in ("/?", "/help"):
      print(HELP)
      continue
    if command == "/clear":
      history = []
      print("Cleared the conversation.")
      continue
    if command.startswith("/set "):
      parts = command.split()
      if parts[1:] == ["think"]: think = True
      elif parts[1:] == ["nothink"]: think = False
      elif len(parts) == 3 and parts[1] == "temperature":
        try: options["temperature"] = float(parts[2])
        except ValueError: print("temperature must be a number")
      else: print(HELP)
      continue
    if command.startswith("/"):
      print(f"Unknown command {command.split()[0]!r}. Type /? for help.")
      continue
    if not command: continue
    message = {"role": "user", "content": text}
    wait_ready()
    while True:
      try:
        answer = _reply(history + [message], model, think, options, show_thinking=True)
        break
      except ServerError as exc:
        if "context" in str(exc) and history:
          history = history[2:]
          print("[Context window: discarded the oldest turn; earlier details are no longer remembered.]", file=sys.stderr)
          continue
        print(f"[Error: {exc}]", file=sys.stderr)  # the conversation goes on without this message
        answer = None
        break
      except KeyboardInterrupt:
        print("\n[Interrupted.]", file=sys.stderr)
        answer = None
        break
    if answer is not None: history += [message, {"role": "assistant", "content": answer}]

def run(model: str | None, prompt: str | None, think: bool | None, options: dict, serve_args: list[str]) -> int:
  """`infurnace run`: chat interactively, or answer one prompt (an argument or piped input) and exit."""
  piped = not sys.stdin.isatty()
  if piped:
    stdin = sys.stdin.read()
    prompt = "\n".join(p for p in (stdin, prompt) if p) or prompt
    if prompt is None: return 0  # nothing to answer: don't start a server for it
  current = connect(model, serve_args)
  model = model or current.get("model") or ""
  if prompt is None: return chat(model, think, options)
  wait_ready()
  _reply([{"role": "user", "content": prompt}], model, think, options, show_thinking=sys.stdout.isatty())
  return 0

# --- ps, stop ----------------------------------------------------------------------------------------------------

def ps() -> int:
  if (current := status()) is None:
    print(f"No infurnace server is running at {host()}.")
    return 0
  for loaded in request("GET", "/api/ps").get("models", []):
    details = loaded.get("details", {})
    flags = ["DFlash"] if loaded.get("infurnace", {}).get("dflash") else []
    print(f"{loaded['name']:<16} {details.get('quantization_level', ''):<8} {loaded.get('size', 0) / 1e9:5.1f} GB  "
          f"context {loaded.get('context_length')}  {current.get('status')}  {' '.join(flags)}  {base_url()}")
  return 0

def stop() -> int:
  if status() is None:
    print(f"No infurnace server is running at {host()}.")
    return 0
  # The server handles one thing at a time: a reply in progress, or the prompt graph it captures after its warmup
  # (~30 s), finishes first.
  print("Stopping the server (it finishes what it is doing first)...", file=sys.stderr, flush=True)
  request("POST", "/api/shutdown", {}, timeout=120)
  for _ in range(40):
    if status() is None:
      print("Stopped the infurnace server.")
      return 0
    time.sleep(0.25)
  raise ServerError("the server did not stop")

# --- launch: point an agent at the server -----------------------------------------------------------------------

def _write_with_backup(path: Path, value: dict) -> None:
  if path.is_file(): shutil.copy2(path, path.with_name(path.name + ".infurnace.bak"))
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value, indent=2) + "\n")

def configure_pi(model: str, context: int, home: Path | None = None) -> Path:
  """Add (or refresh) this server's model in pi's models.json under the `infurnace` provider: entries this writes
  are marked `_launch`; others (yours) are left as they are. Also makes it pi's default model. The previous files
  are kept as *.infurnace.bak."""
  agent = (home or Path.home()) / ".pi" / "agent"
  models_path, settings_path = agent / "models.json", agent / "settings.json"
  config = json.loads(models_path.read_text()) if models_path.is_file() else {}
  providers = config.setdefault("providers", {})
  provider = providers.setdefault("infurnace", {})
  if provider.get("baseUrl") not in (None, base_url() + "/v1"):
    print(f"Pointing pi's infurnace provider at {base_url()}/v1 (was {provider['baseUrl']}).", file=sys.stderr)
  provider["baseUrl"] = base_url() + "/v1"  # the server just configured, wherever an earlier one was
  provider.setdefault("api", "openai-completions")
  provider.setdefault("apiKey", "infurnace")
  entry = {"id": model, "_launch": True, "input": ["text"], "reasoning": True, "contextWindow": context,
           "maxTokens": min(8192, max(1024, context // 4)),
           "compat": {"thinkingFormat": "qwen-chat-template", "supportsDeveloperRole": False, "supportsReasoningEffort": False}}
  kept = [m for m in provider.get("models", []) if not (isinstance(m, dict) and m.get("_launch") and m.get("id") == model)]
  provider["models"] = kept + [entry]
  _write_with_backup(models_path, config)
  settings = json.loads(settings_path.read_text()) if settings_path.is_file() else {}
  settings.update(defaultProvider="infurnace", defaultModel=model)
  _write_with_backup(settings_path, settings)
  return models_path

def launch(integration: str, model: str | None, serve_args: list[str], extra: list[str], configure_only: bool = False) -> int:
  if integration != "pi": raise ServerError(f"unknown integration {integration!r}; supported: pi")
  current = connect(model, serve_args)
  model = model or current.get("model") or ""
  context = next((m.get("context_length") for m in request("GET", "/api/ps").get("models", [])), 8192)
  path = configure_pi(model, context)
  print(f"Configured pi for {model} at {base_url()}/v1 (context {context}) in {path}.", file=sys.stderr)
  if configure_only: return 0
  pi = shutil.which("pi") or str(Path.home() / ".pi/agent/bin/pi")
  if not Path(pi).exists(): raise ServerError("pi is not installed (npm install -g @earendil-works/pi-coding-agent)")
  return subprocess.call([pi, *extra])
