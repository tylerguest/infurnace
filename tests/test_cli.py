import io
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from infurnace.cli import main


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
        with patch.dict(os.environ, env, clear=True):
            with patch.object(sys, "argv", [
                "infurnace", "--prompt", "x", "--device", "CPU",
                "--artifact", "does-not-exist.gguf",
            ]):
                with self.assertRaises(SystemExit):
                    main()
            self.assertEqual(os.environ.get("DEV"), "CPU")

    def test_import_cli_has_no_tinygrad_side_effects(self):
        # cli.py must not import tinygrad at module level, or --device (which
        # sets os.environ["DEV"] before lazy imports in main) cannot take effect.
        code = ("import sys; import infurnace.cli; "
                "assert not any(n == 'tinygrad' or n.startswith('tinygrad.') for n in sys.modules)")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_device_applied_before_tinygrad_import(self):
        # tinygrad captures DEV once at import; only a clean subprocess can prove
        # the CLI sets DEV before tinygrad is imported. The missing artifact
        # aborts main() after DEV is set but before any model work.
        code = (
            "import os, sys\n"
            "sys.argv = ['infurnace', '--prompt', 'x', '--device', 'CPU', "
            "'--artifact', 'does-not-exist.gguf']\n"
            "from infurnace.cli import main\n"
            "try:\n"
            "    main()\n"
            "except SystemExit:\n"
            "    pass\n"
            "from tinygrad.device import Device\n"
            "print(Device.DEFAULT)"
        )
        env = {k: v for k, v in os.environ.items() if k != "DEV"}
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.strip().endswith("CPU"), result.stdout)


if __name__ == "__main__":
    unittest.main()