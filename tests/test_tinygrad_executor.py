import os
import subprocess
import sys
import textwrap
import unittest

import pytest


class TinygradContractTest(unittest.TestCase):
  def run_contract(self, source: str, device: str):
    environment = os.environ.copy()
    environment.update(DEV=device, JIT="1")
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(source)], env=environment, capture_output=True, text=True)
    self.assertEqual(result.returncode, 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")


class TestTinygradCPUContracts(TinygradContractTest):
  def test_device_selection_and_initialized_storage(self):
    self.run_contract("""
      import sys
      from tinygrad import Device, Tensor

      assert Device.DEFAULT == "CPU"
      empty = Tensor.empty(4, device="CPU").realize()
      assert empty.uop.base.realized is None

      initialized = Tensor.zeros(4, device="CPU").contiguous().realize()
      assert initialized.uop.base.realized is not None
      assert initialized.uop.base.realized.is_initialized()
      assert initialized.tolist() == [0.0, 0.0, 0.0, 0.0]
      assert "CPU" in Device._opened_devices
      assert all(not device.startswith("NV") for device in Device._opened_devices)
      assert "tinygrad.runtime.ops_nv" not in sys.modules
    """, "CPU")

  def test_assignment_dependencies_and_backing_storage(self):
    self.run_contract("""
      from tinygrad import Tensor

      value = Tensor.zeros(4).contiguous().realize()
      backing = value.uop.base.realized
      value.assign(Tensor.full((4,), 3.0).contiguous()).realize()
      assert value.uop.base.realized is backing
      assert value.tolist() == [3.0, 3.0, 3.0, 3.0]

      cache = Tensor.zeros(6).contiguous().realize()
      write = cache[2:4].assign(Tensor([5.0, 7.0]).contiguous())
      dependent_read = cache.sum()
      Tensor.realize(write, dependent_read)
      assert cache.tolist() == [0.0, 0.0, 5.0, 7.0, 0.0, 0.0]
      assert dependent_read.item() == 12.0

      chained = Tensor([0.0]).contiguous().realize()
      before = chained + 0
      chained.assign(chained + 1)
      after_one = chained + 0
      chained.assign(chained + 1)
      after_two = chained + 0
      chained.assign(chained + 1)
      assert [before.item(), after_one.item(), after_two.item(), chained.item()] == [0.0, 1.0, 2.0, 3.0]
    """, "CPU")

  def test_tinyjit_lifecycle_replacement_and_output_reuse(self):
    self.run_contract("""
      from tinygrad import Tensor, TinyJit

      @TinyJit
      def add_one(value):
        return (value + 1).contiguous().realize()

      warmup = add_one(Tensor([1.0]).contiguous().realize())
      assert add_one.cnt == 1 and warmup.item() == 2.0
      capture = add_one(Tensor([2.0]).contiguous().realize())
      saved_capture = capture.clone().realize()
      assert add_one.cnt == 2 and capture.item() == 3.0
      replacement = Tensor([3.0]).contiguous().realize()
      replay = add_one(replacement)
      assert add_one.cnt == 3 and replay.item() == 4.0
      assert capture.item() == 4.0
      assert saved_capture.item() == 3.0

      @TinyJit
      def increment(value):
        value.assign(value + 1).realize()
        return value

      first = Tensor.zeros(1).contiguous().realize()
      for _ in range(3): increment(first)
      second = Tensor.zeros(1).contiguous().realize()
      increment(second)
      assert first.item() == 3.0
      assert second.item() == 1.0
    """, "CPU")

  def test_tinyjit_input_compatibility_contract(self):
    self.run_contract("""
      from tinygrad import Tensor, TinyJit, dtypes
      from tinygrad.engine.jit import JitError

      @TinyJit
      def add(a, b): return (a + b).contiguous().realize()

      for _ in range(2):
        add(Tensor.ones(2).contiguous().realize(), Tensor.ones(2).contiguous().realize())

      try: add(Tensor.ones(2, dtype=dtypes.int32).contiguous().realize(), Tensor.ones(2, dtype=dtypes.int32).contiguous().realize())
      except JitError: pass
      else: raise AssertionError("dtype change was accepted")

      @TinyJit
      def identity(value): return (value + 1).contiguous().realize()
      base = Tensor.arange(6).contiguous().realize()
      try: identity(base[1:3])
      except JitError: pass
      else: raise AssertionError("virtual view input was accepted")

      @TinyJit
      def duplicate(a, b): return (a + b).contiguous().realize()
      value = Tensor.ones(2).contiguous().realize()
      try: duplicate(value, value)
      except JitError: pass
      else: raise AssertionError("duplicate backing input was accepted")
    """, "CPU")


@pytest.mark.nv
class TestTinygradNVContracts(TinygradContractTest):
  def test_nv_mutation_jit_and_replacement(self):
    self.run_contract("""
      import sys
      from tinygrad import Device, Tensor, TinyJit

      assert Device.DEFAULT == "NV"
      cache = Tensor.zeros(4).contiguous().realize()
      assert cache.uop.base.realized is not None
      assert cache.uop.base.realized.is_initialized()
      backing = cache.uop.base.realized

      write = cache[1:3].assign(Tensor([2.0, 4.0], device="NV").contiguous())
      dependent_read = cache.sum()
      Tensor.realize(write, dependent_read)
      Device["NV"].synchronize()
      assert cache.uop.base.realized is backing
      assert cache.tolist() == [0.0, 2.0, 4.0, 0.0]
      assert dependent_read.item() == 6.0

      @TinyJit
      def increment(value):
        value.assign(value + 1).realize()
        return value

      first = Tensor.zeros(1).contiguous().realize()
      increment(first)
      increment(first)
      increment(first)
      second = Tensor.zeros(1).contiguous().realize()
      increment(second)
      Device["NV"].synchronize()
      assert increment.cnt == 4
      assert first.item() == 3.0
      assert second.item() == 1.0
      assert "tinygrad.runtime.ops_nv" in sys.modules
    """, "NV")
