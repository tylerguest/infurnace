"""Single-user HTTP serving: tinygrad's OpenAI-compatible server (/v1), with DFlash and per-request thinking, plus
infurnace's own API (/api: chat, generate, tags, ps, show, version, status, shutdown), which the CLI uses.

The server listens before its warmup ends: warmup (and then the prompt graph) runs on a thread while the main
thread answers the endpoints that don't touch the GPU; generation takes the GPU lock (tinygrad is not
thread-safe), so it waits for whichever of them is running."""
from __future__ import annotations
import datetime
import io
import json
import sys
import threading
import time
from importlib import metadata as _metadata

from tinygrad.llm.serve import Handler, LLMServer, normalize_messages
from tinygrad.viz.serve import TCPServerWithReuse

try: VERSION = _metadata.version('infurnace')
except _metadata.PackageNotFoundError: VERSION = '0.0.0'

def _now() -> str:
  return datetime.datetime.now(datetime.timezone.utc).isoformat()

class ServerModel:
  """What tinygrad's server generates with: DFlash when a draft is loaded (one generator serves every
  temperature), else plain decoding, bounded by the context and closed when a client goes away."""
  def __init__(self, model, dflash=None, *, eos_token_ids: tuple[int, ...] = (), block_size: int = 16):
    self.model, self.dflash, self.eos, self.block = model, dflash, eos_token_ids, block_size
    self.max_context = model.config.context_length

  def get_start_pos(self, ids):  # only for the server's log line: the reused prefix, counting the prompt snapshot
    snapshots = getattr(self.model, 'snapshot_length', None)
    return max(self.model.transformer.get_start_pos(ids), snapshots(list(ids)) if snapshots else 0)

  def generate(self, ids, temperature=0.0):
    budget = self.max_context - len(ids)
    if self.dflash is not None:
      stream = self.dflash.stream(list(ids), max_new_tokens=budget, block_size=self.block, eos_token_ids=self.eos, temperature=temperature)
    else: stream = self.model.generate(ids, budget, temperature=temperature)
    try:
      yield from stream
    finally:
      stream.close()

class ChatTemplate:
  def __init__(self, metadata, tokenizer, enable_thinking=True):
    try:
      import jinja2
    except ImportError as exc:
      raise ValueError("HTTP serving requires Jinja2: python3 -m pip install jinja2 (or reinstall infurnace)") from exc
    source = metadata.get('tokenizer.chat_template')
    if not isinstance(source, str) or not source:
      raise ValueError("HTTP serving requires a GGUF chat template (no tool-unsafe fallback)")
    env = jinja2.Environment()
    env.filters['tojson'] = lambda obj, **kwargs: json.dumps(obj, **kwargs)
    env.globals['raise_exception'] = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
    env.globals['strftime_now'] = lambda fmt: time.strftime(fmt)
    env.globals['bos_token'] = tokenizer.decode([tokenizer.bos_id]) if tokenizer.bos_id is not None else ''
    env.globals['eos_token'] = tokenizer.decode([tokenizer.eos_id])
    self.template = env.from_string(source)
    self.enable_thinking = enable_thinking
    self.request_thinking: bool | None = None  # the current request's switch, if it sent one

  def render(self, messages, **kwargs):
    # OpenAI clients send the system prompt as `developer` to reasoning models; Qwen's template only knows `system`.
    messages = [{**m, 'role': 'system'} if isinstance(m, dict) and m.get('role') == 'developer' else m for m in messages]
    thinking = self.enable_thinking if self.request_thinking is None else self.request_thinking
    return self.template.render(messages=messages, **kwargs, enable_thinking=thinking)

def request_thinking(body) -> bool | None:
  """A request's thinking switch: `chat_template_kwargs.enable_thinking` (vLLM and sglang's convention,
  pi's qwen-chat-template format) or a top-level `enable_thinking` (pi's qwen format)."""
  if not isinstance(body, dict): return None
  kwargs = body.get('chat_template_kwargs')
  if isinstance(kwargs, dict) and isinstance(kwargs.get('enable_thinking'), bool): return kwargs['enable_thinking']
  return body['enable_thinking'] if isinstance(body.get('enable_thinking'), bool) else None

def _think(body) -> bool | None:
  """A request's `think`: a bool, or a level ("low", "medium", "high"), which this model treats as on."""
  value = body.get('think')
  if isinstance(value, bool): return value
  if isinstance(value, str): return value.lower() not in ('false', 'off', 'none', '')
  return None

class RequestHandler(Handler):
  """tinygrad's handler, with the request's thinking switch applied while its prompt is rendered, and the /api endpoints."""
  def send_json(self, value, status_code: int = 200):
    self.send_data(json.dumps(value).encode(), status_code=status_code)

  def do_GET(self):
    route, server = self.path.split('?')[0], self.server
    if route == '/api/version': self.send_json({'version': VERSION})
    elif route == '/api/status': self.send_json(server.status())
    elif route == '/api/tags': self.send_json({'models': server.installed()})
    elif route == '/api/ps': self.send_json({'models': [server.loaded()]})
    else: super().do_GET()

  def do_POST(self):
    raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
    try: body = json.loads(raw) if raw else {}
    except ValueError: body = None
    route = self.path.split('?')[0]
    if route.startswith('/api/'):
      if not isinstance(body, dict): return self.send_json({'error': 'invalid JSON body'}, 400)
      if route == '/api/shutdown':
        self.send_json({'status': 'stopping'})
        def close(server=self.server):
          server.shutdown()
          server.finish()
        threading.Thread(target=close, daemon=True).start()
      elif route == '/api/show': self.send_json(self.server.show())
      elif route in ('/api/chat', '/api/generate'):
        if not self.wait_usable(): return
        with self.server.gpu: self.api_generate(body, chat=route == '/api/chat')
      else: self.send_json({'error': f'unknown endpoint {route}'}, 404)
      return
    if not self.wait_usable(): return
    self.rfile = io.BytesIO(raw)  # tinygrad's handler reads the body again
    template = self.server.template
    template.request_thinking = request_thinking(body)
    try:
      with self.server.gpu: super().do_POST()
    finally: template.request_thinking = None

  def wait_usable(self) -> bool:
    """Wait for warmup; after a failed one, answer 503 with its error instead of running a half-set-up model."""
    self.server.ready.wait()
    if self.server.failure is None: return True
    self.send_json({'error': f'the server failed to warm up: {self.server.failure}'}, 503)
    return False

  def api_generate(self, body: dict, chat: bool):
    """/api/chat or /api/generate: tinygrad's streamed chunks as newline-delimited JSON records."""
    generate = not chat
    server, started = self.server, time.perf_counter_ns()
    name = server.model_name
    if generate:
      messages = ([{'role': 'system', 'content': body['system']}] if body.get('system') else []) + \
                 [{'role': 'user', 'content': body.get('prompt') or ''}]
    else: messages = body.get('messages')
    if not isinstance(messages, list): return self.send_json({'error': 'messages must be a list'}, 400)
    normalize_messages(messages)
    template = server.template
    template.request_thinking = _think(body)
    try:
      rendered = (body.get('prompt') or '') if generate and body.get('raw') else \
                 template.render(messages=messages, tools=body.get('tools'), add_generation_prompt=True, preserve_thinking=True)
    finally: template.request_thinking = None
    ids = server.tok.encode(rendered)
    if len(ids) >= server.model.max_context:
      return self.send_json({'error': f'prompt has {len(ids)} tokens, but the model context is {server.model.max_context}'}, 400)
    options = body.get('options') if isinstance(body.get('options'), dict) else {}
    limit = options.get('num_predict')
    max_tokens = limit if isinstance(limit, int) and limit > 0 else None
    temperature = options.get('temperature')
    try: temperature = 0.0 if temperature is None else float(temperature)
    except (TypeError, ValueError): temperature = -1.0
    if not 0 <= temperature < float('inf'): return self.send_json({'error': 'options.temperature must be a number >= 0'}, 400)
    chunks = self.run_model(ids, name, True, max_tokens=max_tokens, temperature=temperature, reasoning=rendered.rstrip().endswith('<think>'))
    text_key = 'response' if generate else 'content'
    def records():
      first, done_reason, usage = None, 'stop', {}
      for chunk in chunks:
        if not chunk['choices']:
          usage = chunk.get('usage', {})
          continue
        choice = chunk['choices'][0]
        if choice.get('finish_reason'): done_reason = 'length' if choice['finish_reason'] == 'length' else 'stop'
        delta = choice.get('delta') or {}
        content, thinking = delta.get('content') or '', delta.get('reasoning_content') or ''
        calls = [{'function': {'name': c['function']['name'], 'arguments': _arguments(c['function']['arguments'])}}
                 for c in delta.get('tool_calls') or []]
        if not (content or thinking or calls): continue
        if first is None: first = time.perf_counter_ns()
        part = {text_key: content, **({'thinking': thinking} if thinking else {}), **({'tool_calls': calls} if calls else {})}
        yield {'model': name, 'created_at': _now(), **(part if generate else {'message': {'role': 'assistant', **part}}), 'done': False}
      end = time.perf_counter_ns()
      first = first or end
      yield {'model': name, 'created_at': _now(), **({'response': ''} if generate else {'message': {'role': 'assistant', 'content': ''}}),
             'done': True, 'done_reason': done_reason, 'total_duration': end - started, 'load_duration': 0,
             'prompt_eval_count': usage.get('prompt_tokens', len(ids)), 'prompt_eval_duration': first - started,
             'eval_count': usage.get('completion_tokens', 0), 'eval_duration': end - first}
    if body.get('stream', True):
      try:
        self.send_response(200)
        self.send_header('Content-Type', 'application/x-ndjson')
        self.end_headers()
        for record in records():
          self.wfile.write((json.dumps(record) + '\n').encode())
          self.wfile.flush()
      except (BrokenPipeError, ConnectionResetError): chunks.close()
      return
    merged: dict = {}
    for record in records():
      part = record if generate else record['message']
      if record['done']:
        final = record
        continue
      for key in (text_key, 'thinking'):
        if part.get(key): merged[key] = merged.get(key, '') + part[key]
      if part.get('tool_calls'): merged['tool_calls'] = merged.get('tool_calls', []) + part['tool_calls']
    merged.setdefault(text_key, '')
    self.send_json({**final, **(merged if generate else {'message': {'role': 'assistant', **merged}})})

def _arguments(value):
  if isinstance(value, str):
    try: return json.loads(value)
    except ValueError: return value
  return value

class Server(LLMServer):
  """`warmup` runs on a thread from the start, then `idle` (work a request may wait for, e.g. the prompt graph);
  generating requests wait for `ready` and take the `gpu` lock. `info` describes the loaded model for /api/ps and
  /api/show; `installed()` lists downloaded models for /api/tags."""
  def __init__(self, server_address: tuple, model, model_name: str, tok, template, idle=None, *, warmup=None, info=None,
               installed=None):
    self.model, self.model_name, self.tok, self.template, self.idle = model, model_name, tok, template, idle
    self.info, self.installed = info or {}, installed or (lambda: [])
    self.ready, self.failure, self.warmup_seconds, self.gpu = threading.Event(), None, None, threading.Lock()
    self.stopping, self.closed = False, threading.Event()
    TCPServerWithReuse.__init__(self, server_address, RequestHandler)
    if warmup is None and idle is None: self.ready.set()
    else: threading.Thread(target=self._warm, args=(warmup, idle), name='infurnace-warmup', daemon=True).start()

  def _warm(self, warmup, idle):
    start = time.perf_counter()
    with self.gpu:
      try:
        if warmup is not None: warmup()
      except BaseException as exc:  # reported by /api/status and the log; requests then fail instead of hanging
        self.failure = f'{type(exc).__name__}: {exc}'
        print(f"Warmup failed: {self.failure}", file=sys.stderr, flush=True)
      self.warmup_seconds = time.perf_counter() - start
      self.ready.set()
    if idle is not None and not self.failure:
      with self.gpu:  # a request that arrives meanwhile waits for it
        if self.stopping: return
        try: idle()
        except BaseException as exc: print(f"Background work failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

  def finish(self):
    """Stop listening once the warmup thread is off the GPU (it skips the prompt graph if that has not started),
    so the process never exits in the middle of GPU work. Until then connections wait rather than being refused,
    so clients see a busy server, not a stopped one. The lock stays held: nothing starts on the GPU afterwards."""
    if self.stopping: return self.closed.wait()
    self.stopping = True
    if not self.gpu.acquire(blocking=False):
      print("Waiting for the GPU work in progress to finish...", file=sys.stderr, flush=True)
      self.gpu.acquire()
    self.server_close()
    self.closed.set()

  def status(self) -> dict:
    state = 'failed' if self.failure else 'ready' if self.ready.is_set() else 'warming up'
    return {'status': state, 'model': self.model_name, **({'error': self.failure} if self.failure else {}),
            **({'warmup_seconds': round(self.warmup_seconds, 1)} if self.warmup_seconds is not None else {})}

  def loaded(self) -> dict:
    return {'name': self.model_name, 'model': self.model_name, 'size': self.info.get('size', 0), 'size_vram': self.info.get('size', 0),
            'context_length': self.model.max_context, 'expires_at': None, 'details': self.info.get('details', {}),
            'infurnace': {'ready': self.ready.is_set() and not self.failure, 'dflash': self.info.get('dflash', False)}}

  def show(self) -> dict:
    return {'details': self.info.get('details', {}), 'model_info': self.info.get('model_info', {}),
            'capabilities': ['completion', 'tools', 'thinking'], 'context_length': self.model.max_context}

def serve(model, tokenizer, port=8000, host='127.0.0.1', *, enable_thinking=True, dflash=None, block_size=16, idle=None,
          warmup=None, name=None, info=None, installed=None):
  template = ChatTemplate(model.metadata, tokenizer, enable_thinking)
  name = name or model.metadata.get('general.name') or 'infurnace'
  eos = tuple(i for i in (tokenizer.eos_id, tokenizer.eot_id) if i is not None)
  with Server((host, port), ServerModel(model, dflash, eos_token_ids=eos, block_size=block_size), name, tokenizer, template,
              idle, warmup=warmup, info=info, installed=installed) as server:
    print(f"Serving {name} at http://{host}:{server.server_address[1]} (OpenAI-compatible API at /v1, infurnace API at /api)"
          f"{' with DFlash' if dflash is not None else ''}{'; warming up' if warmup is not None else ''}", flush=True)
    try: server.serve_forever()
    finally: server.finish()  # after /api/shutdown or Ctrl-C
