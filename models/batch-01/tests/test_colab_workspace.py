"""Portable workspace bootstrap checks, using a small frozen-release fixture."""

import hashlib
import importlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("colab_bootstrap"),
            "Colab workspace bootstrap has not been implemented",
        )
        self.module = importlib.import_module("colab_bootstrap")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.drive, self.work = root / "shared-drive/batch-02", root / "colab"
        put(self.drive / "shared/batch02/__init__.py", {})
        put(self.drive / "shared/colab_bootstrap.py", {})
        (self.drive / "requirements.txt").write_text("numpy==2.5.1\n")
        snapshot = self.drive / "inputs/release-one"
        files = {"manifest.json": put(snapshot / "manifest.json", {"current": True})}
        files["window_15s_step_5s/good_acceptable.csv"] = put(
            snapshot / "window_15s_step_5s/good_acceptable.csv", {"row": 1}
        )
        release = {
            "release_id": "release-one",
            "files": files,
            "gates": dict.fromkeys(
                ("integrity", "additional_data", "proportions", "frozen"), "approved"
            ),
        }
        put(self.drive / "registry/dataset_release.json", release)
        put(snapshot / "release.json", release)
        stream_files = {
            f"window_{s}s_step_5s.csv": put(
                snapshot / f"streams/window_{s}s_step_5s.csv", {"sensor": s}
            )
            for s in (15, 25, 45)
        }
        put(
            snapshot / "streams/stream_manifest.json",
            {
                "release_id": "release-one",
                "files": stream_files,
                "dataset_manifest_sha256": files["manifest.json"],
            },
        )
        put(snapshot / "streams/verification.json", {"verified": True})

    def test_copies_verified_release_and_code_once_without_mutating_source(self):
        original = (self.drive / "registry/dataset_release.json").read_bytes()
        local = self.module.prepare_workspace(self.drive, self.work)
        self.assertEqual(local, self.work / "release-one")
        self.assertTrue((local / "shared/batch02/__init__.py").is_file())
        self.assertEqual(
            (local / "inputs/release-one/streams/window_25s_step_5s.csv").read_bytes(),
            b'{"sensor": 25}',
        )
        self.assertEqual(self.module.prepare_workspace(self.drive, self.work), local)
        self.assertEqual(
            (self.drive / "registry/dataset_release.json").read_bytes(), original
        )
        self.assertFalse((self.drive / "outputs").exists())

    def test_rejects_corrupt_source_or_changed_code_without_replacing_workspace(self):
        local = self.module.prepare_workspace(self.drive, self.work)
        before = (local / "requirements.txt").read_bytes()
        (self.drive / "requirements.txt").write_text("changed\n")
        with self.assertRaises(ValueError):
            self.module.prepare_workspace(self.drive, self.work)
        self.assertEqual((local / "requirements.txt").read_bytes(), before)
        (self.drive / "inputs/release-one/manifest.json").write_text("corrupt")
        with self.assertRaises(ValueError):
            self.module.prepare_workspace(self.drive, self.work)

    def test_rejects_low_disk_space_before_copying(self):
        with (
            patch("shutil.disk_usage", return_value=type("Disk", (), {"free": 0})()),
            self.assertRaises(OSError),
        ):
            self.module.prepare_workspace(self.drive, self.work)
        self.assertFalse((self.work / "release-one").exists())

    def test_optional_checkpoint_is_verified_before_copying(self):
        source = self.drive / "model.ckpt"
        source.write_bytes(b"fixture-checkpoint")
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        self.assertTrue(
            hasattr(self.module, "prepare_checkpoint"),
            "Optional model weights must be staged once",
        )
        local = self.module.prepare_checkpoint(source, sha, self.work)
        self.assertEqual(local.read_bytes(), b"fixture-checkpoint")
        with self.assertRaises(ValueError):
            self.module.prepare_checkpoint(source, "0" * 64, self.work)


if __name__ == "__main__":
    unittest.main()
