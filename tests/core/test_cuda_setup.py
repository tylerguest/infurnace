import pytest

from infurnace import _cuda

@pytest.fixture
def packages(tmp_path, monkeypatch):
  """Fake nvidia.cuda_nvrtc and nvidia.cuda_runtime pip packages, no toolkit, no variables set."""
  nvrtc, runtime = tmp_path / "cuda_nvrtc", tmp_path / "cuda_runtime"
  (nvrtc / "lib").mkdir(parents=True)
  for name in ("libnvrtc-builtins.so.12.9", "libnvrtc-builtins.alt.so.12.9", "libnvrtc.alt.so.12"): (nvrtc / "lib" / name).touch()
  (nvrtc / "lib" / "libnvrtc.so.12").touch()
  (runtime / "include").mkdir(parents=True)
  (runtime / "include" / "cuda_fp16.h").touch()
  monkeypatch.setattr(_cuda, "_package", {"cuda_nvrtc": nvrtc, "cuda_runtime": runtime}.get)
  monkeypatch.setattr(_cuda, "_toolkit", lambda: False)
  monkeypatch.setattr(_cuda.sys, "platform", "linux")
  for name in ("NVRTC_PATH", "CUDA_PATH"):  # recorded, so what use_pip_compiler sets is undone after the test
    monkeypatch.setenv(name, "")
    monkeypatch.delenv(name)
  return nvrtc, runtime

def test_points_tinygrad_at_the_pip_packages(packages, monkeypatch):
  nvrtc, runtime = packages
  assert _cuda.use_pip_compiler()
  assert _cuda.os.environ["NVRTC_PATH"] == str(nvrtc / "lib" / "libnvrtc.so.12")  # not the builtins or .alt libraries
  assert _cuda.os.environ["CUDA_PATH"] == str(runtime)

@pytest.mark.parametrize("setup", ["toolkit", "variable", "no_runtime", "no_header"])
def test_leaves_other_setups_alone(packages, monkeypatch, setup):
  nvrtc, runtime = packages
  if setup == "toolkit": monkeypatch.setattr(_cuda, "_toolkit", lambda: True)
  if setup == "variable": monkeypatch.setenv("CUDA_PATH", "/opt/cuda")
  if setup == "no_runtime": monkeypatch.setattr(_cuda, "_package", {"cuda_nvrtc": nvrtc}.get)
  if setup == "no_header": (runtime / "include" / "cuda_fp16.h").unlink()
  assert not _cuda.use_pip_compiler()
  assert "NVRTC_PATH" not in _cuda.os.environ
