"""Chat prompts from the GGUF's own Jinja chat template, for every model family (the server and the local chat)."""
from __future__ import annotations
import json
import time

class ChatTemplate:
  def __init__(self, metadata, tokenizer, enable_thinking=True):
    try:
      import jinja2
    except ImportError as exc:
      raise ValueError("chat requires Jinja2: python3 -m pip install jinja2 (or reinstall infurnace)") from exc
    source = metadata.get('tokenizer.chat_template')
    if not isinstance(source, str) or not source:
      raise ValueError("chat requires the GGUF's chat template (tokenizer.chat_template)")
    env = jinja2.Environment()
    env.filters['tojson'] = lambda obj, **kwargs: json.dumps(obj, **kwargs)
    env.globals['raise_exception'] = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
    env.globals['strftime_now'] = lambda fmt: time.strftime(fmt)
    env.globals['bos_token'] = tokenizer.decode([tokenizer.bos_id]) if tokenizer.bos_id is not None else ''
    env.globals['eos_token'] = tokenizer.decode([tokenizer.eos_id])
    self.template = env.from_string(source)
    self.enable_thinking = enable_thinking
    self.request_thinking: bool | None = None  # the current request's switch, if it sent one

  def render(self, messages, *, thinking: bool | None = None, **kwargs):
    """The prompt text for `messages`; `thinking` overrides the request's switch and the default."""
    # OpenAI clients send the system prompt as `developer` to reasoning models; Qwen's template only knows `system`.
    messages = [{**m, 'role': 'system'} if isinstance(m, dict) and m.get('role') == 'developer' else m for m in messages]
    if thinking is None: thinking = self.enable_thinking if self.request_thinking is None else self.request_thinking
    return self.template.render(messages=messages, **kwargs, enable_thinking=thinking)

def user_prompt(model, tokenizer, text: str, thinking: bool = False) -> str:
  """One user message as a prompt, ready for the reply (tests and benchmarks)."""
  return ChatTemplate(model.metadata, tokenizer).render([{'role': 'user', 'content': text}], thinking=thinking,
                                                        add_generation_prompt=True)
