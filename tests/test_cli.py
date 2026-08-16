import io
import os
import sys
import unittest
from unittest.mock import patch

from infurnace.cli import main


class TestCLIFake(unittest.TestCase):
    def test_fake_driver_generates_output(self):
        out = io.StringIO()
        with patch.object(sys, "argv", ["infurnace", "--fake", "--prompt", "hello", "--max-tokens", "3"]):
            with patch.object(sys, "stdout", out):
                rc = main()
        self.assertEqual(rc, 0)
        # FakeTokenizer maps chars to ordinals; FakeRunner produces tokens 0,1,2,...
        self.assertEqual(out.getvalue(), "\x00\x01\x02\n")

    def test_fake_driver_stops_at_max_tokens(self):
        out = io.StringIO()
        with patch.object(sys, "argv", ["infurnace", "--fake", "--prompt", "hi", "--max-tokens", "0"]):
            with patch.object(sys, "stdout", out):
                rc = main()
        self.assertEqual(rc, 0)
        self.assertEqual(out.getvalue().strip(), "\x00")


class TestCLIRealBuildPath(unittest.TestCase):
    def test_missing_artifact_exits(self):
        stderr = io.StringIO()
        with patch.object(sys, "argv", [
            "infurnace",
            "--prompt", "hello",
            "--artifact", "does-not-exist.gguf",
            "--manifest", "does-not-exist.json",
        ]):
            with patch.object(sys, "stderr", stderr):
                with self.assertRaises(SystemExit) as cm:
                    main()
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("artifact not found", stderr.getvalue())

    def test_device_env_set(self):
        env = os.environ.copy()
        env.pop("DEV", None)
        out = io.StringIO()
        with patch.dict(os.environ, env, clear=True):
            with patch.object(sys, "argv", ["infurnace", "--fake", "--prompt", "x", "--device", "CPU"]):
                with patch.object(sys, "stdout", out):
                    main()
            self.assertEqual(os.environ.get("DEV"), "CPU")


if __name__ == "__main__":
    unittest.main()
