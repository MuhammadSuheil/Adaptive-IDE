"""Dependency checks must not confuse installed versions with loaded modules."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))


class RuntimeTest(unittest.TestCase):
    def test_stale_import_remains_blocked_after_package_installation(self):
        from batch02.runtime import ensure_environment

        root = Path(__file__).resolve().parents[1]
        with (
            patch.dict(sys.modules, {"xgboost": SimpleNamespace(__version__="0.0.0")}),
            self.assertRaisesRegex(RuntimeError, "Restart|restart"),
        ):
            ensure_environment(root)


if __name__ == "__main__":
    unittest.main()
