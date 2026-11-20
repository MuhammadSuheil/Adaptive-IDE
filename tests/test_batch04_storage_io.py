"""Checkpoint publication under transient Drive-like filesystem failures."""

import ast
import hashlib
import json
import platform
import tempfile
import unittest
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import test_batch04_notebooks as notebooks


class Batch04StorageIOTests(unittest.TestCase):
    namespaces: ClassVar[list[dict[str, Any]]]

    @classmethod
    def setUpClass(cls):
        cls.namespaces = [notebooks.namespace(p) for p in notebooks.NOTEBOOKS]

    def test_copy_recreates_parent_when_first_mkdir_is_not_visible(self):
        for ns in self.namespaces:
            with (
                self.subTest(notebook=ns["IMPLEMENTATION_HASH"]),
                tempfile.TemporaryDirectory() as tmp,
            ):
                store = ns["ArtifactStore"](
                    Path(tmp) / "local", Path(tmp) / "remote", {"data": "synthetic"}
                )
                target = store.remote / "fold" / "model.pkl"
                mkdir = Path.mkdir
                attempts: list[Path] = []

                def delayed_parent(
                    path,
                    *args,
                    expected=target.parent,
                    calls=attempts,
                    original=mkdir,
                    **kwargs,
                ):
                    if path == expected:
                        calls.append(path)
                        if len(calls) == 1:
                            return None
                    return original(path, *args, **kwargs)

                error = None
                with (
                    patch.object(Path, "mkdir", delayed_parent),
                    patch.object(ns["time"], "sleep"),
                ):
                    try:
                        store._copy_verified(target, b"complete-model")
                    except OSError as exc:
                        error = exc
                self.assertIsNone(
                    error, "A missing parent must be recreated during the next retry"
                )
                self.assertEqual(target.read_bytes(), b"complete-model")

    def test_copy_retries_a_transient_mkdir_failure(self):
        for ns in self.namespaces:
            with (
                self.subTest(notebook=ns["IMPLEMENTATION_HASH"]),
                tempfile.TemporaryDirectory() as tmp,
            ):
                store = ns["ArtifactStore"](
                    Path(tmp) / "local", Path(tmp) / "remote", {"data": "synthetic"}
                )
                target = store.remote / "fold" / "model.pkl"
                mkdir = Path.mkdir
                first = True

                def interrupted_mkdir(
                    path, *args, expected=target.parent, original=mkdir, **kwargs
                ):
                    nonlocal first
                    if path == expected and first:
                        first = False
                        raise FileNotFoundError("Drive directory not visible yet")
                    return original(path, *args, **kwargs)

                error = None
                with (
                    patch.object(Path, "mkdir", interrupted_mkdir),
                    patch.object(ns["time"], "sleep"),
                ):
                    try:
                        store._copy_verified(target, b"complete-model")
                    except OSError as exc:
                        error = exc
                self.assertIsNone(
                    error, "Directory creation must be inside the retry boundary"
                )
                self.assertEqual(target.read_bytes(), b"complete-model")

    def test_failed_upload_keeps_complete_local_checkpoint_for_recovery(self):
        for ns in self.namespaces:
            with (
                self.subTest(notebook=ns["IMPLEMENTATION_HASH"]),
                tempfile.TemporaryDirectory() as tmp,
            ):
                store = ns["ArtifactStore"](
                    Path(tmp) / "local", Path(tmp) / "remote", {"data": "synthetic"}
                )
                copy = store._copy_verified

                def offline(target, payload, expected=store.remote, original=copy):
                    if target.is_relative_to(expected):
                        raise OSError("Drive offline")
                    return original(target, payload)

                with (
                    patch.object(store, "_copy_verified", side_effect=offline),
                    self.assertRaises(OSError),
                ):
                    store.commit(
                        "fold",
                        {"id": "fold"},
                        {"model.pkl": b"model", "validation.npz": b"predictions"},
                    )
                self.assertFalse((store.remote / "fold" / "done.json").is_file())
                self.assertTrue(
                    (store.local / "fold" / "done.json").is_file(),
                    "Stage the entire local checkpoint before beginning Drive upload",
                )
                self.assertEqual(
                    (store.local / "fold" / "validation.npz").read_bytes(),
                    b"predictions",
                )
                with patch.object(ns["time"], "sleep"):
                    self.assertEqual(store.read("fold"), {"id": "fold"})
                self.assertEqual(store.artifact("fold", "model.pkl"), b"model")
                self.assertTrue((store.remote / "fold" / "done.json").is_file())

    def test_invalid_artifact_does_not_publish_part_of_a_checkpoint(self):
        for ns in self.namespaces:
            with (
                self.subTest(notebook=ns["IMPLEMENTATION_HASH"]),
                tempfile.TemporaryDirectory() as tmp,
            ):
                store = ns["ArtifactStore"](
                    Path(tmp) / "local", Path(tmp) / "remote", {"data": "synthetic"}
                )
                with self.assertRaises(ValueError):
                    store.commit(
                        "fold",
                        {"id": "fold"},
                        {"model.pkl": b"model", "../invalid": b"bad"},
                    )
                self.assertFalse(
                    (store.remote / "fold").exists(),
                    "Validate all names before uploading any artifact",
                )

    def test_version_two_run_resumes_without_rewriting_training_identity(self):
        implementations = (
            "7aa1c28983ebbf362e9fc665f713dfc4a36bbb20db3980c2252d1464a87ace2d",
            "bb8d769128b074bd94e0288804ffb8a8849e7daf4a2a51116b088d0d36cb478f",
        )
        for i, ns in enumerate(self.namespaces):
            experiment = f"exp0{i + 1}"
            document = json.loads(notebooks.NOTEBOOKS[i].read_text(encoding="utf-8"))
            source = "".join(
                next(
                    c["source"] for c in document["cells"] if c.get("id") == "run-start"
                )
            )
            definitions = ast.Module(
                body=[
                    node
                    for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef)
                ],
                type_ignores=[],
            )
            exec(compile(definitions, "run-start-definitions", "exec"), ns)  # noqa: S102
            with (
                self.subTest(experiment=experiment),
                tempfile.TemporaryDirectory() as tmp,
            ):
                project = Path(tmp)
                run = (
                    project
                    / "models"
                    / "batch-04"
                    / "outputs"
                    / experiment
                    / "legacy_run"
                )
                run.mkdir(parents=True)
                manifest, hashes = {"schema_version": 2}, {"15": "unchanged-dataset"}
                signature = {
                    "code_version": "batch04-2.0.0",
                    "experiment": experiment,
                    "dataset_sha256": hashes,
                    "split_sha256": ns["digest"](manifest),
                    "environment": ns["environment_versions"](experiment),
                    "python": platform.python_version(),
                    "implementation": implementations[i],
                    "protocol": ns["experiment_protocol"](experiment),
                }
                identity = ns["canonical"]({"signature": signature}).encode()
                (run / "run.json").write_bytes(identity)
                fold = run / "old_fold"
                fold.mkdir()
                files = {
                    "record.json": b'{"id":"old_fold"}',
                    "model.pkl": b"unchanged-model",
                }
                for name, data in files.items():
                    (fold / name).write_bytes(data)
                (fold / "done.json").write_text(
                    json.dumps(
                        {
                            "checksums": {
                                name: hashlib.sha256(data).hexdigest()
                                for name, data in files.items()
                            }
                        }
                    )
                )
                with patch.object(
                    ns["tempfile"], "mkdtemp", return_value=str(project / "cache")
                ):
                    store = ns["start_run"](
                        project, experiment, hashes, manifest, str(run)
                    )
                self.assertEqual(
                    store.artifact("old_fold", "model.pkl"), b"unchanged-model"
                )
                self.assertEqual((run / "run.json").read_bytes(), identity)
                self.assertEqual(
                    json.loads((run / "artifact_io.json").read_text())["revision"], 2
                )


if __name__ == "__main__":
    unittest.main()
