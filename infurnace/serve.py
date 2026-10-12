"""Single-user HTTP serving using tinygrad's OpenAI-compatible server."""
import json
import time

from tinygrad.llm.serve import LLMServer


class ServerModel:
  """Preserve Infurnace's device contexts and bounded generation lifecycle."""
  def __init__(self, model):
    self.model = model
    self.max_context = model.config.context_length

  def get_start_pos(self, ids):
    return self.model.transformer.get_start_pos(ids)

  def generate(self, ids, temperature=0.0):
    stream = self.model.generate(ids, self.max_context - len(ids), temperature=temperature)
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

  def render(self, **kwargs):
    return self.template.render(**kwargs, enable_thinking=self.enable_thinking)


def serve(model, tokenizer, port=8000, host='127.0.0.1', *, enable_thinking=True):
  template = ChatTemplate(model.metadata, tokenizer, enable_thinking)
  name = model.metadata.get('general.name') or 'infurnace'
  with LLMServer((host, port), ServerModel(model), name, tokenizer, template) as server:
    print(f"Serving {name} at http://{host}:{server.server_address[1]}/v1", flush=True)
    server.serve_forever()
