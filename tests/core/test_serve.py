import http.client
import json
import threading
from types import SimpleNamespace

from infurnace.serve import ChatTemplate, ServerModel
from tinygrad.llm.serve import LLMServer


class Tokenizer:
  bos_id = None
  eos_id = 999

  def encode(self, text):
    self.rendered = text
    return [1, 2]

  def decode(self, ids):
    return '<eos>'

  def is_end(self, token):
    return token == 999

  def stream_decoder(self):
    return lambda token=None: chr(token) if token is not None else ''


class Model:
  config = SimpleNamespace(context_length=1024)
  transformer = SimpleNamespace(get_start_pos=lambda ids: 0)

  def __init__(self):
    self.closed = False
    self.output = '<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>'

  def generate(self, ids, limit, temperature=0):
    self.limit = limit
    try:
      yield from map(ord, self.output[:limit])
    finally:
      self.closed = True


def test_adapter_close_and_bound():
  model = Model()
  stream = ServerModel(model).generate([1, 2])
  next(stream)
  stream.close()
  assert model.closed
  assert model.limit == 1022


def test_http_tool_round_trip():
  model, tok = Model(), Tokenizer()
  template = ChatTemplate({'tokenizer.chat_template':
    '{{ tools | tojson }}{% for m in messages %}{{ m.content }}'
    '{% for tc in m.tool_calls or [] %}{{ tc.function.arguments.path }}{% endfor %}{% endfor %}'}, tok)
  with LLMServer(('127.0.0.1', 0), ServerModel(model), 'test', tok, template) as server:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def request(messages, stream=False):
      conn = http.client.HTTPConnection(*server.server_address)
      conn.request('POST', '/v1/chat/completions', json.dumps({
        'model': 'test', 'messages': messages, 'tools': [], 'stream': stream,
        'stream_options': {'include_usage': True}}), {'Content-Type': 'application/json'})
      response = conn.getresponse()
      assert response.status == 200
      data = response.read().decode()
      conn.close()
      return data
    try:
      first = json.loads(request([{'role': 'user', 'content': 'read'}]))
      message = first['choices'][0]['message']
      assert first['choices'][0]['finish_reason'] == 'tool_calls'
      assert json.loads(message['tool_calls'][0]['function']['arguments']) == {'path': 'README.md'}
      model.output = 'done'
      second = request([{'role': 'user', 'content': 'read'}, message,
                        {'role': 'tool', 'tool_call_id': message['tool_calls'][0]['id'], 'content': 'file contents'}], True)
      assert 'README.md' in tok.rendered
      assert 'file contents' in tok.rendered
      assert 'data: [DONE]' in second
      assert 'prompt_tokens' in second
      assert model.closed
    finally:
      server.shutdown()
      thread.join()
