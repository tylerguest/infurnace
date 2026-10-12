"""DFlash's on-disk graph cache across processes (real model, NV)."""
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest

TARGET = Path('models/qwen35-validation/Qwen3.5-4B-Q4_K_M.gguf')
DRAFT = Path('models/dflash-qwen35/qwen35-4b-dflash-Q4_K_M.gguf')
pytestmark = [pytest.mark.nv, pytest.mark.skipif(not (TARGET.is_file() and DRAFT.is_file()),
                                                  reason='needs the Qwen3.5-4B target and DFlash draft under models/')]

RUN = f'''
import json
from infurnace.loader import load_model as _build
from infurnace.dflash import DFlashDraft, DFlashGenerator
model, tokenizer = _build({str(TARGET)!r}, 1024, 'NV')
generator = DFlashGenerator(model, DFlashDraft.from_gguf({str(DRAFT)!r}, model), verifier='parallel')
how = generator.warmup()
ids = tokenizer.encode(tokenizer.render_chat([{{'role': 'user', 'content': 'What are the pros and cons of remote work?'}}],
                                             enable_thinking=False))
got = list(generator.stream(list(ids), max_new_tokens=32, block_size=16))
plain = list(model.generate(list(ids), 32, reuse_cache=False))
print(json.dumps({{'how': how, 'got': got, 'plain': plain}}))
'''


def run(cache):
  env = dict(os.environ, XDG_CACHE_HOME=str(cache), INFURNACE_GRAPH_CACHE='1')
  result = subprocess.run([sys.executable, '-c', RUN], env=env, capture_output=True, text=True, timeout=280, check=True)
  return json.loads(result.stdout.strip().splitlines()[-1])


def test_second_process_loads_cached_graphs_and_decodes_identically(tmp_path):
  first, second = run(tmp_path), run(tmp_path)
  assert (first['how'], second['how']) == ('captured', 'cached')
  assert first['got'] == second['got'] == first['plain']
