"""Execute notebook configuration cells without mounting Drive or fitting."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))


class NotebookStorageTests(unittest.TestCase):
    def test_every_exp03_to_exp14_uses_workspace_and_distinct_drive_target(self):
        batch = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(3, 15):
                with self.subTest(experiment=index):
                    notebook = json.loads(
                        (
                            batch / f"exp{index:02d}/EyeHeartCognitiveLoad.ipynb"
                        ).read_text(encoding="utf-8")
                    )
                    namespace = {
                        "PROJECT_ROOT": root / "Drive",
                        "BATCH_DIR": root / "Drive/models/batch-02",
                        "WORK_BATCH_DIR": root / "Colab",
                        "ENABLE_TABPFN": False,
                        "os": os,
                    }
                    exec(  # noqa: S102 - execute the repository notebook cell under test.
                        compile(
                            "".join(notebook["cells"][6]["source"]),
                            f"exp{index:02d}-configuration",
                            "exec",
                        ),
                        namespace,
                    )
                    self.assertEqual(
                        namespace["config"].batch, (root / "Colab").resolve()
                    )
                    self.assertEqual(
                        namespace["persistence"].drive_batch,
                        (root / "Drive/models/batch-02").resolve(),
                    )
                    self.assertEqual(
                        namespace["persistence"].experiment, f"exp{index:02d}"
                    )


if __name__ == "__main__":
    unittest.main()
