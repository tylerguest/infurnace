import http.client
import json
import socket
import threading
import time
from types import SimpleNamespace

from infurnace.serve import ChatTemplate, Server, ServerModel

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

class DFlash:
  def stream(self, ids, **kwargs):
    self.ids, self.kwargs = ids, kwargs
    try:
      yield 7
    finally:
      self.closed = True

def test_adapter_uses_dflash_with_request_temperature():
  dflash = DFlash()
  stream = ServerModel(Model(), dflash, eos_token_ids=(999,), block_size=16).generate((1, 2), temperature=0.7)
  assert next(stream) == 7
  stream.close()
  assert dflash.closed and dflash.ids == [1, 2]
  assert dflash.kwargs == {'max_new_tokens': 1022, 'block_size': 16, 'eos_token_ids': (999,), 'temperature': 0.7}

def test_request_switches_thinking():
  model, tok = Model(), Tokenizer()
  model.output = 'done'
  template = ChatTemplate({'tokenizer.chat_template': '{{ "think" if enable_thinking else "plain" }}'}, tok)
  with Server(('127.0.0.1', 0), ServerModel(model), 'test', tok, template) as server:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def rendered(**extra):
      conn = http.client.HTTPConnection(*server.server_address)
      conn.request('POST', '/v1/chat/completions', json.dumps({'model': 'test', 'messages': [{'role': 'user', 'content': 'hi'}]} | extra))
      assert conn.getresponse().status == 200
      conn.close()
      return tok.rendered
    try:
      assert rendered() == 'think'
      assert rendered(chat_template_kwargs={'enable_thinking': False}) == 'plain'
      assert rendered() == 'think'  # the switch lasts one request
      assert rendered(enable_thinking=False) == 'plain'
      template.enable_thinking = False
      assert rendered(chat_template_kwargs={'enable_thinking': True}) == 'think'
      template.template = template.template.environment.from_string('{% for m in messages %}{{ m.role }}{% endfor %}')
      assert rendered(messages=[{'role': 'developer', 'content': 'be brief'}, {'role': 'user', 'content': 'hi'}]) == 'systemuser'
    finally:
      server.shutdown()
      thread.join()

def test_http_tool_round_trip():
  model, tok = Model(), Tokenizer()
  template = ChatTemplate({'tokenizer.chat_template':
    '{{ tools | tojson }}{% for m in messages %}{{ m.content }}'
    '{% for tc in m.tool_calls or [] %}{{ tc.function.arguments.path }}{% endfor %}{% endfor %}'}, tok)
  with Server(('127.0.0.1', 0), ServerModel(model), 'test', tok, template) as server:
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

def test_background_work_keeps_status_responsive():
  # Work after warmup (the prompt graph, ~30 s) runs on the warmup thread: status answers meanwhile, generation waits.
  model, tok = Model(), Tokenizer()
  model.output = 'ok'
  template = ChatTemplate({'tokenizer.chat_template': '{% for m in messages %}{{ m.content }}{% endfor %}'}, tok)
  started, release, calls = threading.Event(), threading.Event(), []
  def idle():
    calls.append(1)
    started.set()
    release.wait(5)
  server, call, stop = serving(model, tok, template, idle=idle)
  try:
    assert started.wait(5)
    assert json.loads(call('GET', '/api/status')[1])['status'] == 'ready'  # not blocked by the work
    threading.Timer(0.2, release.set).start()
    status, data = call('POST', '/api/chat', {'model': 'test', 'messages': [{'role': 'user', 'content': 'x'}], 'stream': False})
    assert status == 200 and release.is_set() and calls == [1]  # the reply came after the work
  finally: stop()

def test_shutdown_waits_for_warmup():
  # /api/shutdown during warmup: the server keeps listening until the warmup is off the GPU, and the prompt graph
  # (idle) never starts, so the process never exits in the middle of GPU work.
  model, tok = Model(), Tokenizer()
  template = ChatTemplate({'tokenizer.chat_template': '{% for m in messages %}{{ m.content }}{% endfor %}'}, tok)
  release, calls = threading.Event(), []
  server, call, stop = serving(model, tok, template, warmup=lambda: release.wait(5), idle=lambda: calls.append(1))
  try:
    assert json.loads(call('POST', '/api/shutdown', {})[1])['status'] == 'stopping'
    time.sleep(0.3)
    assert not server.closed.is_set()  # still listening: clients see a busy server, not a stopped one
    release.set()
    assert server.closed.wait(5) and calls == []
  finally: stop()

def test_bad_requests_get_400():
  # Unrenderable messages, an oversized prompt and a malformed Content-Length are answered, not dropped.
  model, tok = Model(), Tokenizer()
  template = ChatTemplate({'tokenizer.chat_template': "{% if messages[-1].role != 'user' %}{{ raise_exception('No user query') }}"
                                                      "{% endif %}{{ messages[-1].content }}"}, tok)
  server, call, stop = serving(model, tok, template)
  try:
    status, data = call('POST', '/api/chat', {'messages': [{'role': 'assistant', 'content': 'hi'}]})
    assert status == 400 and 'No user query' in json.loads(data)['error']
    status, data = call('POST', '/api/chat', {'messages': ['not an object']})
    assert status == 400
    model.config = SimpleNamespace(context_length=2)  # the fake tokenizer always encodes two tokens
    server.model.max_context = 2
    status, data = call('POST', '/api/chat', {'messages': [{'role': 'user', 'content': 'x'}]})
    assert status == 400 and json.loads(data)['code'] == 'context_length_exceeded'
    conn = http.client.HTTPConnection(*server.server_address)
    conn.putrequest('POST', '/api/chat')
    conn.putheader('Content-Length', 'abc')
    conn.endheaders()
    assert conn.getresponse().status == 400
    conn.close()
  finally: stop()

class SlowModel(Model):
  """Yields a token every 10 ms until `release` is set (or 500 tokens)."""
  def __init__(self):
    super().__init__()
    self.started, self.release, self.count = threading.Event(), threading.Event(), 0

  def generate(self, ids, limit, temperature=0):
    self.started.set()
    try:
      while not self.release.is_set() and self.count < 500:
        self.count += 1
        time.sleep(0.01)
        yield ord('a')
    finally:
      self.closed = True

def test_status_answers_during_a_reply():
  # Requests have their own threads: status and ps answer while another request generates.
  model, tok = SlowModel(), Tokenizer()
  template = ChatTemplate({'tokenizer.chat_template': '{% for m in messages %}{{ m.content }}{% endfor %}'}, tok)
  server, call, stop = serving(model, tok, template)
  reply = threading.Thread(target=call, args=('POST', '/api/chat', {'messages': [{'role': 'user', 'content': 'x'}], 'stream': False}))
  reply.start()
  try:
    assert model.started.wait(5)
    assert json.loads(call('GET', '/api/status')[1])['status'] == 'ready' and not model.closed
    assert json.loads(call('GET', '/api/ps')[1])['models'][0]['name'] == 'test'
  finally:
    model.release.set()
    reply.join(5)
    stop()

def test_non_streamed_reply_stops_when_the_client_leaves():
  model, tok = SlowModel(), Tokenizer()
  template = ChatTemplate({'tokenizer.chat_template': '{% for m in messages %}{{ m.content }}{% endfor %}'}, tok)
  server, call, stop = serving(model, tok, template)
  try:
    body = json.dumps({'messages': [{'role': 'user', 'content': 'x'}], 'stream': False}).encode()
    with socket.create_connection(server.server_address) as conn:
      conn.sendall(b'POST /api/chat HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n' % len(body) + body)
      assert model.started.wait(5)
    deadline = time.monotonic() + 3
    while not model.closed and time.monotonic() < deadline: time.sleep(0.02)
    assert model.closed and model.count < 500
  finally:
    model.release.set()
    stop()

def serving(model, tok, template, **kwargs):
  server = Server(('127.0.0.1', 0), ServerModel(model), 'test', tok, template, **kwargs)
  thread = threading.Thread(target=server.serve_forever, daemon=True)
  thread.start()
  def call(method, path, body=None):
    conn = http.client.HTTPConnection(*server.server_address)
    conn.request(method, path, json.dumps(body) if body is not None else None, {'Content-Type': 'application/json'})
    response = conn.getresponse()
    data = response.read().decode()
    conn.close()
    return response.status, data
  def stop():
    server.shutdown()
    thread.join()
    server.server_close()
  return server, call, stop

def test_api_chat_and_generate():
  model, tok = Model(), Tokenizer()
  template = ChatTemplate({'tokenizer.chat_template': '{% for m in messages %}{{ m.content }}{% endfor %}'}, tok)
  server, call, stop = serving(model, tok, template, info={'details': {'family': 'qwen35'}},
                               installed=lambda: [{'name': 'qwen3.5:4b', 'size': 1}])
  try:
    status, data = call('POST', '/api/chat', {'model': 'test', 'messages': [{'role': 'user', 'content': 'read'}]})
    lines = [json.loads(line) for line in data.splitlines()]
    assert status == 200 and lines[-1]['done'] and lines[-1]['done_reason'] == 'stop'
    calls = [c for line in lines for c in line['message'].get('tool_calls', [])]
    assert calls == [{'function': {'name': 'read', 'arguments': {'path': 'README.md'}}}]  # arguments as an object
    model.output = 'hello'
    status, data = call('POST', '/api/generate', {'model': 'test', 'prompt': 'hi', 'stream': False, 'options': {'num_predict': 3}})
    reply = json.loads(data)
    assert reply['response'] == 'hel' and reply['done_reason'] == 'length' and reply['eval_count'] == 3
    assert json.loads(call('GET', '/api/tags')[1])['models'][0]['name'] == 'qwen3.5:4b'
    assert json.loads(call('GET', '/api/ps')[1])['models'][0]['details']['family'] == 'qwen35'
    assert json.loads(call('GET', '/api/version')[1])['version']
  finally: stop()

def test_requests_wait_for_warmup():
  # The server answers status during warmup, and generation once warmup has finished.
  model, tok = Model(), Tokenizer()
  model.output = 'ok'
  template = ChatTemplate({'tokenizer.chat_template': '{% for m in messages %}{{ m.content }}{% endfor %}'}, tok)
  release = threading.Event()
  server, call, stop = serving(model, tok, template, warmup=lambda: release.wait(5))
  try:
    assert json.loads(call('GET', '/api/status')[1])['status'] == 'warming up'
    threading.Timer(0.2, release.set).start()
    status, data = call('POST', '/api/chat', {'model': 'test', 'messages': [{'role': 'user', 'content': 'x'}], 'stream': False})
    assert status == 200 and json.loads(data)['message']['content'] == 'ok'
    assert json.loads(call('GET', '/api/status')[1])['status'] == 'ready'
  finally: stop()

def test_generate_renders_the_chat_template():
  # /api/generate wraps the prompt in the chat template unless `raw` is set.
  model, tok = Model(), Tokenizer()
  model.output = 'ok'
  template = ChatTemplate({'tokenizer.chat_template': '{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}'}, tok)
  server, call, stop = serving(model, tok, template)
  try:
    assert call('POST', '/api/generate', {'model': 'test', 'prompt': 'hi', 'stream': False})[0] == 200
    assert tok.rendered == '<user>hi'
    assert call('POST', '/api/generate', {'model': 'test', 'prompt': 'hi', 'raw': True, 'stream': False})[0] == 200
    assert tok.rendered == 'hi'
    status, data = call('POST', '/api/chat', {'model': 'test', 'messages': [], 'options': {'temperature': 'hot'}})
    assert status == 400 and 'temperature' in json.loads(data)['error']
    assert call('POST', '/api/chat', {'model': 'test', 'messages': [], 'options': {'temperature': None}, 'stream': False})[0] == 200
  finally: stop()

def test_failed_warmup_answers_503():
  model, tok = Model(), Tokenizer()
  template = ChatTemplate({'tokenizer.chat_template': '{% for m in messages %}{{ m.content }}{% endfor %}'}, tok)
  def fail(): raise MemoryError('Allocation of 256.00 MB failed on NV')
  server, call, stop = serving(model, tok, template, warmup=fail)
  try:
    server.ready.wait(5)
    status, data = call('POST', '/api/chat', {'model': 'test', 'messages': [{'role': 'user', 'content': 'x'}]})
    assert status == 503 and 'Allocation of 256.00 MB' in json.loads(data)['error']
    status, _ = call('POST', '/v1/chat/completions', {'model': 'test', 'messages': [{'role': 'user', 'content': 'x'}]})
    assert status == 503
  finally: stop()

