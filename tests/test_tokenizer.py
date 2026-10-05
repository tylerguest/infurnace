import ast
import os
from pathlib import Path
import subprocess

import pytest

from infurnace.gguf import parse_gguf
from infurnace.tokenizer import Tokenizer


@pytest.fixture
def metadata():
  # GPT-2's reversible mapping from every byte to a vocabulary character.
  direct = [*range(33, 127), *range(161, 173), *range(174, 256)]
  mapping = {b: chr(b) for b in direct}
  mapping.update({b: chr(256+i) for i, b in enumerate(b for b in range(256) if b not in direct)})
  return {
    "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "qwen2",
    "tokenizer.ggml.tokens": [mapping[b] for b in range(256)] + ["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
    "tokenizer.ggml.token_type": [1]*256 + [3]*3,
    "tokenizer.ggml.bos_token_id": 258, "tokenizer.ggml.eos_token_id": 258,
    "tokenizer.ggml.add_bos_token": False,
  }


@pytest.mark.parametrize("text", ["", "Hello 123456!", "你好 café é 👩🏽‍💻", " \t\r\n", "<|im_start|>user\nhello<|im_end|>"])
def test_roundtrip_and_streaming(metadata, text):
  tokenizer = Tokenizer.from_gguf(metadata)
  ids = tokenizer.encode(text)
  assert tokenizer.decode(ids) == text
  decoder = tokenizer.stream_decoder()
  assert "".join(decoder(i) for i in ids) + decoder(None) == text


def test_streams_are_independent_and_flush(metadata):
  tokenizer = Tokenizer.from_gguf(metadata)
  a, b = tokenizer.stream_decoder(), tokenizer.stream_decoder()
  emoji = list("🙂".encode())
  assert a(emoji[0]) == ""
  assert b(ord("x")) == "x"
  assert "".join(a(i) for i in emoji[1:]) == "🙂"
  assert b(None) == ""
  incomplete = tokenizer.stream_decoder()
  assert incomplete(emoji[0]) == ""
  assert incomplete(None) == "�"


def test_bos_and_stop_policy(metadata):
  tokenizer = Tokenizer.from_gguf(metadata)
  assert tokenizer.bos_id is None
  assert tokenizer.encode("a") == [97]
  assert tokenizer.is_end(257)  # EOT fallback is im_end.
  assert tokenizer.is_end(258)
  assert not tokenizer.is_end(97)
  metadata["tokenizer.ggml.add_bos_token"] = True
  tokenizer = Tokenizer.from_gguf(metadata)
  assert tokenizer.bos_id == 258
  assert tokenizer.encode("a") == [97]  # Raw encoding never silently prepends BOS.


@pytest.mark.parametrize("key,value,error", [
  ("tokenizer.ggml.model", "llama", "GPT-2"),
  ("tokenizer.ggml.pre", "llama3", "qwen2"),
  ("tokenizer.ggml.tokens", [], "vocabulary"),
  ("tokenizer.ggml.tokens", ["a", "a"], "duplicate"),
  ("tokenizer.ggml.token_type", [1], "token types"),
  ("tokenizer.ggml.eos_token_id", -1, "eos_token_id"),
  ("tokenizer.ggml.bos_token_id", True, "bos_token_id"),
  ("tokenizer.ggml.eot_token_id", 999, "eot_token_id"),
  ("tokenizer.ggml.add_bos_token", 1, "boolean"),
])
def test_invalid_metadata(metadata, key, value, error):
  metadata[key] = value
  with pytest.raises(ValueError, match=error): Tokenizer.from_gguf(metadata)


def test_missing_enabled_bos(metadata):
  metadata["tokenizer.ggml.add_bos_token"] = True
  del metadata["tokenizer.ggml.bos_token_id"]
  with pytest.raises(ValueError, match="BOS"): Tokenizer.from_gguf(metadata)


def test_chat_single_turn(metadata):
  tokenizer = Tokenizer.from_gguf(metadata)
  messages = [{"role": "system", "content": "Be concise."}, {"role": "user", "content": "Hello"}]
  expected = "<|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n"
  assert tokenizer.render_chat(messages) == expected + "<|im_start|>assistant\n"
  assert tokenizer.render_chat(messages, enable_thinking=False) == expected + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
  assert tokenizer.render_chat(messages, add_generation_prompt=False) == expected
  assert messages[0]["content"] == "Be concise."


def test_chat_historical_reasoning(metadata):
  tokenizer = Tokenizer.from_gguf(metadata)
  messages = [{"role": "user", "content": "one"},
              {"role": "assistant", "content": "<think>old secret</think>\n\nanswer"},
              {"role": "user", "content": "two"}]
  rendered = tokenizer.render_chat(messages)
  assert "old secret" not in rendered
  assert "<|im_start|>assistant\nanswer<|im_end|>" in rendered
  messages.append({"role": "assistant", "content": "\nnew answer", "reasoning_content": "\nnew thought\n"})
  assert "<think>\nnew thought\n</think>\n\nnew answer<|im_end|>" in tokenizer.render_chat(messages)


@pytest.mark.parametrize("messages", [[], [None], [{"role": "tool", "content": "x"}],
                                      [{"role": "user", "content": ["x"]}], [{"role": "user", "content": None}],
                                      [{"role": "assistant", "content": "x", "tool_calls": [{"name": "foo"}]}],
                                      [{"role": "assistant", "content": "x", "reasoning_content": 1}],
                                      [{"role": "user", "content": "<tool_response>x</tool_response>"}]])
def test_invalid_chat(metadata, messages):
  with pytest.raises(ValueError): Tokenizer.from_gguf(metadata).render_chat(messages)


def test_invalid_chat_flags(metadata):
  with pytest.raises(ValueError, match="boolean"):
    Tokenizer.from_gguf(metadata).render_chat([{"role": "user", "content": "x"}], enable_thinking=0)


CHAT_CASES = [
  [{"role": "user", "content": "Hello"}],
  [{"role": "system", "content": "Helpful assistant"}, {"role": "user", "content": "你好"}],
  [{"role": "user", "content": "one"}, {"role": "assistant", "content": "<think>old</think>\n\nfirst"},
   {"role": "user", "content": "two"}],
  [{"role": "user", "content": "one"}, {"role": "assistant", "content": "\nanswer", "reasoning_content": "\nthought\n"}],
  [{"role": "user", "content": "one"}, {"role": "assistant", "content": "answer"}],
  [{"role": "system", "content": "system only"}],
  [{"role": "user", "content": ""}, {"role": "assistant", "content": "<think>a</think>\n\nb</think>\nc"},
   {"role": "system", "content": "later system"}, {"role": "user", "content": "next"}],
]


@pytest.mark.skipif(not os.environ.get("INFURNACE_TEST_MODEL"), reason="set INFURNACE_TEST_MODEL for bundled-template parity")
def test_bundled_chat_template_parity():
  from jinja2 import Environment, StrictUndefined
  metadata, _ = parse_gguf(Path(os.environ["INFURNACE_TEST_MODEL"]))
  tokenizer = Tokenizer.from_gguf(metadata)
  template = Environment(undefined=StrictUndefined).from_string(metadata["tokenizer.chat_template"])
  for messages in CHAT_CASES:
    # The template accesses tool_calls on assistant messages; supply the supported empty value.
    reference_messages = [{**message, "tool_calls": []} for message in messages]
    for thinking in (True, False):
      for prefix in (True, False):
        expected = template.render(messages=reference_messages, tools=[], enable_thinking=thinking, add_generation_prompt=prefix)
        actual = tokenizer.render_chat(messages, enable_thinking=thinking, add_generation_prompt=prefix)
        assert actual == expected
        assert tokenizer.encode(actual) == tokenizer.encode(expected)
  ids = tokenizer.encode(tokenizer.render_chat([{"role": "user", "content": "Hello"}]))
  assert ids == [151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198]


# Recorded using llama.cpp llama-tokenize against the bundled Qwen3 GGUF,
# --stdin --ids --no-bos (special token parsing enabled). No network required.
QWEN_CASES = [
  ("", []),
  ("Hello world", [9707, 1879]),
  ("Hello 123456 café 👋", [9707, 220, 16, 17, 18, 19, 20, 21, 51950, 61804, 233]),
  ("2026-09-19 3.1415926535", [17, 15, 17, 21, 12, 15, 24, 12, 16, 24, 220, 18, 13, 16, 19, 16, 20, 24, 17, 21, 20, 18, 20]),
  ("I'm coding; DON'T stop!", [40, 2776, 10822, 26, 44273, 17323, 2936, 0]),
  ("def add(a, b):\n\treturn a + b\n", [750, 912, 2877, 11, 293, 982, 853, 264, 488, 293, 198]),
  ("  hello\tworld\r\n\n", [220, 23811, 76508, 80823]),
  ("你好，世界！こんにちは 한국어", [108386, 3837, 99489, 6313, 89015, 130092, 31079]),
  ("café é naïve", [924, 58858, 384, 53839, 94880, 586]),
  ("👩🏽‍💻🙂🚀", [145233, 145375, 378, 235, 145851, 145080, 145836]),
  ("<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n", [151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198]),
  ("١٢٣ １２３ ²³", [149, 94, 149, 95, 149, 96, 220, 20109, 24918, 33517, 220, 29456, 43201]),
  ("a\u00a0b\u2003c", [64, 4102, 65, 378, 225, 66]),
  ("<think>reason</think>\n\nanswer", [151667, 19895, 151668, 271, 9217]),
]


@pytest.mark.skipif(not os.environ.get("INFURNACE_TEST_MODEL"), reason="set INFURNACE_TEST_MODEL to the bundled Qwen3 GGUF")
def test_qwen_reference_ids():
  metadata, _ = parse_gguf(Path(os.environ["INFURNACE_TEST_MODEL"]))
  tokenizer = Tokenizer.from_gguf(metadata)
  for text, expected in QWEN_CASES:
    ids = tokenizer.encode(text)
    assert ids == expected, repr(text)
    assert tokenizer.decode(ids) == text
    decoder = tokenizer.stream_decoder()
    assert "".join(decoder(i) for i in ids) + decoder(None) == text


@pytest.mark.skipif(not (os.environ.get("INFURNACE_TEST_MODEL") and os.environ.get("INFURNACE_LLAMA_TOKENIZE")),
                    reason="set INFURNACE_TEST_MODEL and INFURNACE_LLAMA_TOKENIZE for live reference parity")
def test_live_llama_parity():
  path = Path(os.environ["INFURNACE_TEST_MODEL"]).resolve()
  metadata, _ = parse_gguf(path)
  tokenizer = Tokenizer.from_gguf(metadata)
  for text, _ in QWEN_CASES:
    result = subprocess.run([os.environ["INFURNACE_LLAMA_TOKENIZE"], "-m", str(path), "--stdin", "--ids", "--no-bos", "--log-disable"],
                            input=text, text=True, capture_output=True, check=True, timeout=30)
    assert tokenizer.encode(text) == ast.literal_eval(result.stdout.strip()), repr(text)
