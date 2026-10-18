"""Single-user HTTP serving: tinygrad's OpenAI-compatible server (/v1), with DFlash and per-request thinking, plus
infurnace's own API (/api: chat, generate, tags, ps, show, version, status, shutdown), which the CLI uses.

The server listens before its warmup ends: warmup (and then the prompt graph) runs on a thread. Each request has
its own thread, so the endpoints that don't touch the GPU (status, ps, shutdown) answer at any time; generation
takes the GPU lock (tinygrad is not thread-safe), so it waits for the warmup or another reply.

One model is on the GPU at a time. A request naming another model the server can load (a downloaded catalog
model) switches to it: the current one is freed, the new one loaded and warmed up while the request waits. With
a keep-alive, a model left idle that long is unloaded, and the next request loads it again."""
from __future__ import annotations
import contextlib
import datetime
import gc
import io
import json
import os
import select
import socket
import socketserver
import sys
import threading
import time
from dataclasses import dataclass, field
from importlib import metadata as _metadata
from typing import Any, Callable

from tinygrad import Device
from tinygrad.llm.serve import Handler, LLMServer, normalize_messages
from tinygrad.viz.serve import TCPServerWithReuse
from infurnace.chat import ChatTemplate

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

@dataclass
class Loaded:
  """A model as the server serves it: what generates, its tokenizer and chat template, what /api/ps and /api/show
  report, and its warmup and background work (e.g. the prompt graph)."""
  name: str
  model: ServerModel
  tokenizer: Any
  template: ChatTemplate
  info: dict = field(default_factory=dict)
  warmup: Callable[[], None] | None = None
  idle: Callable[[], None] | None = None
  dflash: bool = False

def loaded(model, tokenizer, *, name=None, dflash=None, block_size=16, enable_thinking=True, info=None, warmup=None, idle=None) -> Loaded:
  name = name or model.metadata.get('general.name') or 'infurnace'
  eos = tuple(i for i in (tokenizer.eos_id, tokenizer.eot_id) if i is not None)
  return Loaded(name, ServerModel(model, dflash, eos_token_ids=eos, block_size=block_size), tokenizer,
                ChatTemplate(model.metadata, tokenizer, enable_thinking), info or {}, warmup, idle, dflash is not None)

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
    elif route == '/api/ps': self.send_json({'models': [server.loaded()] if server.model is not None else []})
    else: super().do_GET()

  def do_POST(self):
    try: length = int(self.headers.get("Content-Length", "0"))
    except ValueError: length = -1
    if length < 0: return self.send_json({'error': 'invalid Content-Length'}, 400)
    raw = self.rfile.read(length)
    try: body = json.loads(raw) if raw else {}
    except ValueError: body = None
    route = self.path.split('?')[0]
    if route.startswith('/api/'):
      if not isinstance(body, dict): return self.send_json({'error': 'invalid JSON body'}, 400)
      if route == '/api/shutdown':
        self.send_json({'status': 'stopping'})
        def close(server=self.server):
          server.stopping = True  # no new GPU work (the prompt graph) from now on
          server.shutdown()
          server.finish()
        threading.Thread(target=close, daemon=True).start()
      elif route == '/api/show': self.send_json(self.server.show())
      elif route in ('/api/chat', '/api/generate'):
        with self.server.request_gpu():
          if self.usable() and self.use_model(body.get('model'), strict=True): self.api_generate(body, chat=route == '/api/chat')
          self.server.last_used = time.monotonic()
      else: self.send_json({'error': f'unknown endpoint {route}'}, 404)
      return
    self.rfile = io.BytesIO(raw)  # tinygrad's handler reads the body again
    with self.server.request_gpu():  # the template's switch is shared: set it only while this request holds the GPU
      # OpenAI clients often send a fixed model name: an unknown one is served by the current model
      if not self.usable() or not self.use_model(body.get('model') if isinstance(body, dict) else None, strict=False): return
      template = self.server.template
      template.request_thinking = request_thinking(body)
      try: super().do_POST()
      finally:
        template.request_thinking = None
        self.server.last_used = time.monotonic()

  def usable(self) -> bool:
    """After a failed warmup, answer 503 with its error instead of running a half-set-up model."""
    if self.server.failure is None: return True
    self.send_json({'error': f'the server failed to warm up: {self.server.failure}'}, 503)
    return False

  def use_model(self, requested, strict: bool) -> bool:
    """With the GPU lock held: load the requested model if it isn't the loaded one (or none is). Answers the
    request with an error and returns False when it can't: an unknown model (404, if `strict`) or a failed load."""
    try: self.server.use(requested if isinstance(requested, str) and requested else None, strict)
    except LookupError as exc:
      self.send_json({'error': str(exc)}, 404)
      return False
    except Exception as exc:
      self.send_json({'error': str(exc)}, 500)
      return False
    return True

  def client_gone(self) -> bool:
    """The client closed its connection (a readable socket with nothing to read)."""
    try: return bool(select.select([self.connection], [], [], 0)[0]) and not self.connection.recv(1, socket.MSG_PEEK)
    except OSError: return True

  def api_generate(self, body: dict, chat: bool):
    """/api/chat or /api/generate: tinygrad's streamed chunks as newline-delimited JSON records."""
    generate = not chat
    server, started = self.server, time.perf_counter_ns()
    name = server.model_name
    if (body.get('prompt') if generate else body.get('messages')) in (None, '', []):
      # Nothing to answer: the request only loads the model, or unloads it with keep_alive 0
      unload = body.get('keep_alive') in (0, '0', '0s')
      if unload: server.unload()
      return self.send_json({'model': name, 'created_at': _now(), **({'response': ''} if generate else
                             {'message': {'role': 'assistant', 'content': ''}}), 'done': True, 'done_reason': 'unload' if unload else 'load'})
    if generate:
      messages = ([{'role': 'system', 'content': body['system']}] if body.get('system') else []) + \
                 [{'role': 'user', 'content': body.get('prompt') or ''}]
    else: messages = body.get('messages')
    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
      return self.send_json({'error': 'messages must be a list of objects'}, 400)
    template = server.template
    template.request_thinking = _think(body)
    try:
      normalize_messages(messages)
      rendered = (body.get('prompt') or '') if generate and body.get('raw') else \
                 template.render(messages=messages, tools=body.get('tools'), add_generation_prompt=True, preserve_thinking=True)
    except Exception as exc:  # the chat template's raise_exception, or messages it can't render
      return self.send_json({'error': f'could not render the chat template: {exc}'}, 400)
    finally: template.request_thinking = None
    ids = server.tok.encode(rendered)
    if len(ids) >= server.model.max_context:
      return self.send_json({'error': f'prompt has {len(ids)} tokens, but the model context is {server.model.max_context}',
                             'code': 'context_length_exceeded'}, 400)
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
      if self.client_gone():  # nobody is waiting for the reply: free the GPU
        chunks.close()
        return
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

class Server(socketserver.ThreadingMixIn, LLMServer):
  """`warmup` runs on a thread from the start, then `idle` (work a request may wait for, e.g. the prompt graph);
  generating requests wait for `ready` and take the `gpu` lock. `info` describes the loaded model for /api/ps and
  /api/show; `installed()` lists downloaded models for /api/tags.

  `loader` (optional) makes other models servable: `loader.names()` lists them and `loader(name)` returns a Loaded.
  `keep_alive` (seconds, optional) unloads a model left idle that long."""
  daemon_threads = True  # finish() waits for the GPU instead
  def __init__(self, server_address: tuple, model, model_name: str, tok, template, idle=None, *, warmup=None, info=None,
               installed=None, loader=None, keep_alive: float | None = None, dflash: bool = False):
    self.model, self.model_name, self.tok, self.template, self.idle = model, model_name, tok, template, idle
    self.info, self.installed, self.dflash = info or {}, installed or (lambda: []), dflash
    self.loader, self.keep_alive, self.switching, self.last_used = loader, keep_alive, None, time.monotonic()
    self.waiting, self._waiting = 0, threading.Lock()  # requests waiting for the GPU (background work lets them go first)
    self.ready, self.failure, self.warmup_seconds, self.gpu = threading.Event(), None, None, threading.Lock()
    self.stopping, self.closed, self._finishing = False, threading.Event(), threading.Lock()
    TCPServerWithReuse.__init__(self, server_address, RequestHandler)
    if warmup is None and idle is None: self.ready.set()
    else: threading.Thread(target=self._warm, args=(warmup,), name='infurnace-warmup', daemon=True).start()
    if keep_alive: threading.Thread(target=self._unload_when_idle, name='infurnace-keep-alive', daemon=True).start()

  def _install(self, model: Loaded):
    self.model, self.model_name, self.tok, self.template = model.model, model.name, model.tokenizer, model.template
    self.info, self.dflash, self.idle = model.info, model.dflash, model.idle

  def unload(self):
    """With the GPU lock held: free the loaded model's GPU memory (its name stays, to load it again)."""
    if self.model is None: return
    device = self.model.model.device
    self.model = self.tok = self.template = self.idle = None
    self.info, self.dflash = {}, False
    gc.collect()  # the model's buffers go once nothing refers to them
    allocator = getattr(Device[device], 'allocator', None)
    if hasattr(allocator, 'free_cache'): allocator.free_cache()  # tinygrad keeps freed buffers for reuse otherwise

  def servable(self) -> list[str]:
    return list(dict.fromkeys([self.model_name, *(self.loader.names() if self.loader is not None else [])]))

  def use(self, requested: str | None, strict: bool):
    """With the GPU lock held: make sure a model is loaded, the requested one when it is servable. An unknown name
    raises LookupError when `strict`, else is served by the current model."""
    if requested is not None and requested not in self.servable():
      if strict: raise LookupError(f"model {requested!r} is not available on this server (available: {', '.join(self.servable())}; "
                                   f"`infurnace pull {requested}` downloads a catalog model)")
      requested = None
    target = requested or self.model_name
    if self.model is not None and target == self.model_name: return
    if self.loader is None: raise RuntimeError(f'this server cannot load {target}')
    self._switch(target)

  def _switch(self, target: str):
    previous = self.model_name if self.model is not None else None
    start, self.switching = time.perf_counter(), target
    print(f"Loading {target}" + (f" in place of {previous}" if previous else "") + "...", file=sys.stderr, flush=True)
    try:
      self.unload()
      try:
        model = self.loader(target)
        if model.warmup is not None: model.warmup()
      except BaseException as exc:
        model, error = None, f'{type(exc).__name__}: {exc}'
        print(f"Loading {target} failed: {error}", file=sys.stderr, flush=True)
        if previous is not None: self._restore(previous)
        raise RuntimeError(f'could not load {target}: {error}') from exc
      self._install(model)
      print(f"Switched to {target} in {time.perf_counter() - start:.1f}s.", file=sys.stderr, flush=True)
    finally: self.switching = None
    if self.idle is not None: self._start_background()

  def _restore(self, name: str):
    """After a failed load, put `name` back so the server stays usable (what the failure left is freed first)."""
    gc.collect()
    try:
      back = self.loader(name)
      if back.warmup is not None: back.warmup()
      self._install(back)
    except BaseException as exc:  # nothing is loaded now; the next request tries again
      print(f"Loading {name} again failed too: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

  def _start_background(self):
    threading.Thread(target=self._background, name='infurnace-idle', daemon=True).start()

  def _background(self, quiet: float = 1.0):
    """Run the loaded model's background work (`self.idle`: the prompt graph) in the first idle moment: no request
    waiting and none for `quiet` seconds (a client that just loaded a model sends its reply's request right after).
    A request that arrives while it runs waits for it. The thread holds no reference to the work while it waits,
    so unloading the model (which drops `self.idle`) frees it; the work is then skipped."""
    def idle() -> bool: return not self.waiting and time.monotonic() - self.last_used >= quiet
    while True:
      if idle():
        with self.gpu:
          if self.stopping or self.idle is None: return
          if idle():  # again: a request may have held the lock meanwhile, and another be about to follow it
            work, self.idle = self.idle, None
            try: work()
            except BaseException as exc: print(f"Background work failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return
      time.sleep(0.05)

  def _unload_when_idle(self):
    while not self.stopping:
      time.sleep(min(5.0, self.keep_alive))
      if self.model is None or time.monotonic() - self.last_used < self.keep_alive or not self.gpu.acquire(blocking=False): continue
      try:
        if self.model is not None and not self.stopping and time.monotonic() - self.last_used >= self.keep_alive:
          name = self.model_name
          self.unload()
          print(f"Unloaded {name} after {self.keep_alive:g}s idle; the next request loads it again.", file=sys.stderr, flush=True)
      finally: self.gpu.release()

  def _warm(self, warmup):
    start = time.perf_counter()
    with self.gpu:
      try:
        if warmup is not None: warmup()
      except BaseException as exc:  # reported by /api/status and the log; requests then fail instead of hanging
        self.failure = f'{type(exc).__name__}: {exc}'
        print(f"Warmup failed: {self.failure}", file=sys.stderr, flush=True)
      self.warmup_seconds = time.perf_counter() - start
      self.last_used = time.monotonic()  # clients polling the status send their request right after
      self.ready.set()
    if self.idle is not None and not self.failure: self._start_background()

  @contextlib.contextmanager
  def request_gpu(self):
    """The GPU lock for a request, after the warmup; while it waits, background work doesn't start."""
    with self._waiting: self.waiting += 1
    try:
      self.ready.wait()
      self.gpu.acquire()
    finally:
      with self._waiting: self.waiting -= 1
    try: yield
    finally: self.gpu.release()

  def finish(self):
    """Stop listening once the warmup thread is off the GPU (it skips the prompt graph if that has not started),
    so the process never exits in the middle of GPU work. Until then connections wait rather than being refused,
    so clients see a busy server, not a stopped one. The lock stays held: nothing starts on the GPU afterwards."""
    self.stopping = True
    with self._finishing:  # once: /api/shutdown and the end of serve() can both get here
      if self.closed.is_set(): return
      if not self.gpu.acquire(blocking=False):
        print("Waiting for the GPU work in progress to finish...", file=sys.stderr, flush=True)
        self.gpu.acquire()
      self.server_close()
      self.closed.set()

  def status(self) -> dict:
    state = ('failed' if self.failure else 'warming up' if not self.ready.is_set() else 'loading' if self.switching
             else 'unloaded' if self.model is None else 'ready')
    return {'status': state, 'model': self.switching or self.model_name, 'pid': os.getpid(), **({'error': self.failure} if self.failure else {}),
            **({'warmup_seconds': round(self.warmup_seconds, 1)} if self.warmup_seconds is not None else {})}

  def loaded(self) -> dict:
    return {'name': self.model_name, 'model': self.model_name, 'size': self.info.get('size', 0), 'size_vram': self.info.get('size', 0),
            'context_length': self.model.max_context, 'expires_at': None, 'details': self.info.get('details', {}),
            'expires_in': self.keep_alive, 'infurnace': {'ready': self.ready.is_set() and not self.failure, 'dflash': self.info.get('dflash', False)}}

  def show(self) -> dict:
    return {'details': self.info.get('details', {}), 'model_info': self.info.get('model_info', {}),
            'capabilities': ['completion', 'tools', 'thinking'], 'context_length': self.model.max_context if self.model is not None else 0}

def serve(model, tokenizer, port=8000, host='127.0.0.1', *, enable_thinking=True, dflash=None, block_size=16, idle=None,
          warmup=None, name=None, info=None, installed=None):
  """Serve one model as given (the in-process flags); `infurnace serve` uses serve_models."""
  serve_models(loaded(model, tokenizer, name=name, dflash=dflash, block_size=block_size, enable_thinking=enable_thinking, info=info,
                      warmup=warmup, idle=idle), port, host, installed=installed)

def serve_models(first: Loaded, port=8000, host='127.0.0.1', *, installed=None, loader=None, keep_alive=None):
  """Serve `first`, and with a `loader` the other models it can load (see Server)."""
  with Server((host, port), first.model, first.name, first.tokenizer, first.template, first.idle, warmup=first.warmup,
              info=first.info, installed=installed, loader=loader, keep_alive=keep_alive, dflash=first.dflash) as server:
    name, has_draft, warming = first.name, first.dflash, first.warmup is not None
    del first  # the Server holds the model now: unloading it must leave no other reference
    print(f"Serving {name} at http://{host}:{server.server_address[1]} (OpenAI-compatible API at /v1, infurnace API at /api)"
          f"{' with DFlash' if has_draft else ''}{'; warming up' if warming else ''}", flush=True)
    try: server.serve_forever()
    finally: server.finish()  # after /api/shutdown or Ctrl-C
