import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sz

class TestLineCount(unittest.TestCase):
  def test_counts_code_lines_without_comments_blanks_or_docstrings(self):
    source = '''"""module
documentation"""
# comment
x = (
  1
  + 2
)

def value():
  'function documentation'
  return x
'''
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "sample.py"
      path.write_text(source, encoding="utf-8")
      line_count, token_count = sz.count_python_file(path)
    self.assertEqual(line_count, 6)
    self.assertGreater(token_count, line_count)

  def test_collects_only_production_package_and_sorts_output(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      package = root / "src" / "infurnace"
      package.mkdir(parents=True)
      (package / "small.py").write_text("x = 1\n", encoding="utf-8")
      (package / "large.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
      tests = root / "tests"
      tests.mkdir()
      (tests / "test_large.py").write_text("x = 1\ny = 2\nz = 3\n", encoding="utf-8")

      stats = sz.collect_stats(root)
      rendered = sz.render(stats)

    self.assertEqual({item.path for item in stats}, {"src/infurnace/large.py", "src/infurnace/small.py"})
    self.assertLess(rendered.index("src/infurnace/large.py"), rendered.index("src/infurnace/small.py"))
    self.assertIn("src/infurnace", rendered)
    self.assertIn(":      3 in  2 files", rendered)
    self.assertTrue(rendered.endswith("total lines: 3"))

  def test_optional_limit_returns_failure(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      package = root / "src" / "infurnace"
      package.mkdir(parents=True)
      (package / "module.py").write_text("x = 1\n", encoding="utf-8")
      stdout, stderr = io.StringIO(), io.StringIO()
      with patch.dict(os.environ, {"MAX_LINE_COUNT": "0"}), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        result = sz.main([str(root)])

    self.assertEqual(result, 1)
    self.assertIn("total lines: 1", stdout.getvalue())
    self.assertIn("exceeds MAX_LINE_COUNT=0", stderr.getvalue())