"""Shared test setup.

Tests live in core/, kernels/, dflash/ and eagle/. Some files import helpers from
others by module name (e.g. `from test_qwen35 import checkpoint`), so every group
directory is importable. Tests marked `nv` need an NVIDIA GPU and run only with
INFURNACE_TEST_NV=1; select them with `pytest -m nv` or skip with `-m "not nv"`.
"""
import os
import sys
from pathlib import Path
import pytest

for group in ('core', 'kernels', 'dflash', 'eagle'):
  path = str(Path(__file__).parent / group)
  if path not in sys.path: sys.path.insert(0, path)


def pytest_configure(config):
  config.addinivalue_line('markers', 'nv: needs an NVIDIA GPU; runs only with INFURNACE_TEST_NV=1')


def pytest_collection_modifyitems(config, items):
  if os.environ.get('INFURNACE_TEST_NV'): return
  skip = pytest.mark.skip(reason='set INFURNACE_TEST_NV=1 on an NV GPU')
  for item in items:
    if 'nv' in item.keywords: item.add_marker(skip)


@pytest.fixture(autouse=True)
def _no_installed_dflash_draft(monkeypatch, tmp_path):
  # The CLI enables an installed DFlash draft automatically; tests opt in explicitly.
  monkeypatch.setenv('INFURNACE_DFLASH_DRAFT', str(tmp_path / 'no-draft.gguf'))
