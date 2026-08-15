import subprocess
import sys
import unittest
from importlib.metadata import metadata, version

class TestPackage(unittest.TestCase):
  def test_distribution_metadata(self):
    self.assertEqual(metadata("infurnace")["Name"], "infurnace")
    self.assertEqual(version("infurnace"), "0.0.0")

  def test_import_has_no_tinygrad_side_effects(self):
    code = "import sys; import infurnace; assert not any(name == 'tinygrad' or name.startswith('tinygrad.') for name in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    self.assertEqual(result.returncode, 0, result.stderr)
