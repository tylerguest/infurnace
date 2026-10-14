"""Single-user HTTP serving on tinygrad's OpenAI-compatible server, with DFlash and per-request thinking."""
import io
import json
import time

from tinygrad.llm.serve import Handler, LLMServer
from tinygrad.viz.serve import TCPServerWithReuse

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
      raise ValueError("HTTP serving requires Jinja2: install infurnace[server]") from exc
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

class RequestHandler(Handler):
  """tinygrad's handler, with the request's thinking switch applied while its prompt is rendered."""
  def do_POST(self):
    raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
    try: thinking = request_thinking(json.loads(raw))
    except ValueError: thinking = None
    self.rfile = io.BytesIO(raw)  # tinygrad's handler reads the body again
    template = self.server.template
    template.request_thinking = thinking
    try: super().do_POST()
    finally: template.request_thinking = None

class Server(LLMServer):
  def __init__(self, server_address: tuple, model, model_name: str, tok, template):
    self.model, self.model_name, self.tok, self.template = model, model_name, tok, template
    TCPServerWithReuse.__init__(self, server_address, RequestHandler)

def serve(model, tokenizer, port=8000, host='127.0.0.1', *, enable_thinking=True, dflash=None, block_size=16):
  template = ChatTemplate(model.metadata, tokenizer, enable_thinking)
  name = model.metadata.get('general.name') or 'infurnace'
  eos = tuple(i for i in (tokenizer.eos_id, tokenizer.eot_id) if i is not None)
  with Server((host, port), ServerModel(model, dflash, eos_token_ids=eos, block_size=block_size), name, tokenizer, template) as server:
    print(f"Serving {name} at http://{host}:{server.server_address[1]}/v1{' with DFlash' if dflash is not None else ''}", flush=True)
    server.serve_forever()
