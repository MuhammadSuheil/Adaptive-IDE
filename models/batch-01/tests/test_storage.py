"""Storage regression tests; no research model is fitted."""

import importlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SHARED = Path(__file__).resolve().parents[1] / "shared"
sys.path.insert(0, str(SHARED))
from batch02.artifacts import read_json, write_json


class JsonWriterTests(unittest.TestCase):
    def test_transient_missing_parent_recovers_without_losing_old_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_json(path, {"value": "old"})
            original = Path.write_text
            failures = [True]

            def flaky(target, *args, **kwargs):
                if target.suffix == ".tmp" and failures:
                    failures.pop()
                    raise FileNotFoundError(2, "mount metadata not visible")
                return original(target, *args, **kwargs)

            with patch.object(Path, "write_text", flaky), patch("time.sleep"):
                write_json(path, {"value": "new"})
            self.assertEqual(read_json(path), {"value": "new"})
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_transient_missing_temporary_during_replace_recreates_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            original = Path.replace
            failures = [True]

            def flaky(source, destination):
                if failures:
                    failures.pop()
                    source.unlink()
                    raise FileNotFoundError(2, "temporary disappeared")
                return original(source, destination)

            with patch.object(Path, "replace", flaky), patch("time.sleep"):
                write_json(path, {"ok": True})
            self.assertEqual(read_json(path), {"ok": True})

    def test_bad_json_is_not_retried_or_written(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_json(path, {"value": "old"})
            with self.assertRaises(TypeError):
                write_json(path, {"value": object()})
            self.assertEqual(read_json(path), {"value": "old"})


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec("batch02.persistence"),
            "Colab checkpoint persistence has not been implemented",
        )
        self.module = importlib.import_module("batch02.persistence")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.work, self.drive = root / "work", root / "drive"
        self.drive.mkdir()
        self.run = self.work / "outputs" / "exp03" / "run-one"
        write_json(self.run / "identity.json", {"hash": "identity-one"})
        write_json(self.run / "status.json", {"status": "running"})

    def storage(self, experiment="exp03"):
        return self.module.RunPersistence(self.work, self.drive, experiment)

    def test_delta_checkpoint_restores_latest_files_and_ignores_uncommitted_parts(self):
        writer = self.storage()
        writer.bind(self.run)
        write_json(self.run / "inner" / "a.json", {"score": 0.4})
        writer.publish(force=True)
        write_json(self.run / "inner" / "b.json", {"score": 0.7})
        write_json(self.run / "progress.json", {"fold": "fold_01"})
        writer.publish(force=True)
        remote = self.drive / "outputs" / "exp03" / self.run.name
        parts = list((remote / "checkpoints" / "parts").glob("*.tar.gz"))
        self.assertEqual(len(parts), 2)
        (remote / "checkpoints" / "parts" / "unfinished.tar.gz").write_bytes(b"partial")
        (remote / "checkpoints" / "commit_9999999999_bad.json").write_text("{")
        other = self.module.RunPersistence(
            self.work.parent / "new-runtime", self.drive, "exp03"
        )
        restored = other.restore(remote)
        self.assertEqual(read_json(restored / "inner/a.json"), {"score": 0.4})
        self.assertEqual(read_json(restored / "inner/b.json"), {"score": 0.7})
        self.assertEqual(read_json(restored / "progress.json"), {"fold": "fold_01"})

    def test_failed_publish_keeps_old_checkpoint_and_can_retry_without_training(self):
        writer = self.storage()
        writer.bind(self.run)
        writer.publish(force=True)
        write_json(self.run / "progress.json", {"fold": "fold_02"})
        with (
            patch.object(
                writer, "_upload", side_effect=FileNotFoundError(2, "Drive unavailable")
            ),
            patch("time.sleep"),
            self.assertRaises(self.module.PersistenceError),
        ):
            writer.publish(force=True)
        remote = self.drive / "outputs" / "exp03" / self.run.name
        other = self.module.RunPersistence(
            self.work.parent / "reader", self.drive, "exp03"
        )
        restored = other.restore(remote)
        self.assertFalse((restored / "progress.json").exists())
        self.assertTrue((self.run / "progress.json").exists())
        writer.retry_publish()
        self.assertEqual(read_json(remote / "progress.json"), {"fold": "fold_02"})

    def test_corrupt_newest_part_falls_back_to_previous_valid_checkpoint(self):
        writer = self.storage()
        writer.bind(self.run)
        writer.publish(force=True)
        write_json(self.run / "progress.json", {"fold": "fold_02"})
        writer.publish(force=True)
        remote = self.drive / "outputs" / "exp03" / self.run.name
        newest = max((remote / "checkpoints" / "parts").glob("*.tar.gz"))
        newest.write_bytes(b"corrupt")
        other = self.module.RunPersistence(
            self.work.parent / "reader", self.drive, "exp03"
        )
        restored = other.restore(remote)
        self.assertFalse((restored / "progress.json").exists())

    def test_completion_is_visible_only_after_verified_reports(self):
        writer = self.storage()
        writer.bind(self.run)
        writer.publish(force=True)
        write_json(self.run / "status.json", {"status": "complete"})
        (self.run / "summary.csv").write_text("accuracy\n0.5\n")
        original = writer._upload

        def fail_report(source, target, expected):
            if target.name == "summary.csv":
                raise PermissionError("sharing permission revoked")
            return original(source, target, expected)

        with (
            patch.object(writer, "_upload", fail_report),
            patch("time.sleep"),
            self.assertRaises(self.module.PersistenceError),
        ):
            writer.finish()
        remote = self.drive / "outputs" / "exp03" / self.run.name
        self.assertNotEqual(read_json(remote / "status.json")["status"], "complete")
        writer.retry_publish()
        self.assertEqual(read_json(remote / "status.json")["status"], "complete")
        self.assertEqual((remote / "summary.csv").read_text(), "accuracy\n0.5\n")

    def test_distinct_experiments_do_not_share_outputs(self):
        first = self.storage()
        first.bind(self.run)
        first.publish(force=True)
        other_run = self.work / "outputs" / "exp04" / self.run.name
        write_json(other_run / "identity.json", {"hash": "identity-two"})
        write_json(other_run / "status.json", {"status": "running"})
        second = self.storage("exp04")
        second.bind(other_run)
        second.publish(force=True)
        self.assertEqual(
            read_json(self.drive / "outputs/exp03/run-one/identity.json")["hash"],
            "identity-one",
        )
        self.assertEqual(
            read_json(self.drive / "outputs/exp04/run-one/identity.json")["hash"],
            "identity-two",
        )

    def test_restore_rejects_wrong_experiment_and_unsafe_member(self):
        writer = self.storage()
        writer.bind(self.run)
        writer.publish(force=True)
        remote = self.drive / "outputs/exp03/run-one"
        with self.assertRaises(ValueError):
            self.storage("exp04").restore(remote)
        with self.assertRaises(ValueError):
            self.module.safe_relative("../../outside")

    def test_same_runtime_resume_preserves_unpublished_local_progress(self):
        writer = self.storage()
        writer.bind(self.run)
        writer.publish(force=True)
        write_json(self.run / "inner/new.json", {"score": 0.9})
        remote = self.drive / "outputs/exp03/run-one"
        other = self.storage()
        restored = other.restore(remote)
        self.assertEqual(read_json(restored / "inner/new.json"), {"score": 0.9})

    def test_transient_upload_is_retried_but_disk_full_stops(self):
        import errno

        writer = self.storage()
        writer.bind(self.run)
        original = __import__("shutil").copyfile
        failures = [True]

        def flaky(source, target):
            if failures:
                failures.pop()
                raise FileNotFoundError(2, "mount stale")
            return original(source, target)

        with patch("shutil.copyfile", flaky), patch("time.sleep"):
            writer.publish(force=True)
        write_json(self.run / "progress.json", {"fold": "fold_01"})
        with (
            patch("shutil.copyfile", side_effect=OSError(errno.ENOSPC, "disk full")),
            patch("time.sleep"),
            self.assertRaises(self.module.PersistenceError),
        ):
            writer.publish(force=True)
        self.assertTrue((self.run / "progress.json").exists())

    def test_disconnected_drive_is_not_recreated_as_a_local_output_folder(self):
        writer = self.storage()
        writer.bind(self.run)
        writer.publish(force=True)
        self.drive.rename(self.drive.with_name("drive-disconnected"))
        with patch("time.sleep"), self.assertRaises(self.module.PersistenceError):
            writer.publish(force=True)
        self.assertFalse(self.drive.exists())

    def test_retry_budget_is_bounded_and_permanent_permission_is_not_retried(self):
        import errno

        writer = self.storage()
        calls = []

        def transient():
            calls.append(1)
            raise FileNotFoundError(2, "temporary mount failure")

        with patch("time.sleep"), self.assertRaises(FileNotFoundError):
            writer._retry(transient)
        self.assertEqual(len(calls), 5)
        calls.clear()

        def denied():
            calls.append(1)
            raise PermissionError(errno.EACCES, "read-only share")

        with self.assertRaises(PermissionError):
            writer._retry(denied)
        self.assertEqual(len(calls), 1)

    def test_restore_downloads_archives_before_opening_them(self):
        import tarfile

        writer = self.storage()
        writer.bind(self.run)
        writer.publish(force=True)
        remote = self.drive / "outputs/exp03/run-one"
        original = tarfile.open

        def local_only(name, *args, **kwargs):
            if Path(name).resolve().is_relative_to(self.drive):
                raise AssertionError(
                    "Archives must be read from VM disk, not the Drive mount"
                )
            return original(name, *args, **kwargs)

        other = self.module.RunPersistence(
            self.work.parent / "reader", self.drive, "exp03"
        )
        with patch("tarfile.open", local_only):
            restored = other.restore(remote)
        self.assertEqual(read_json(restored / "identity.json")["hash"], "identity-one")


if __name__ == "__main__":
    unittest.main()
