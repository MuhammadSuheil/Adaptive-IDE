"""Runner parity and recovery with deterministic fit doubles, never estimators."""

import contextlib
import io
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from batch02.artifacts import read_json, write_json
from batch02.contracts import Config
from batch02.persistence import RunPersistence
from batch02.registry import Recipe
from batch02.runner import run_experiment


class DeterministicBundle:
    feature_names = ("ear_left_mean",)
    epoch = None

    def predict(self, rows, store, seconds):
        return np.eye(3)[rows.ear_left_mean.astype(int)], "probability"


def fixture():
    rows = []
    for person in range(3):
        for value in range(3):
            rows.append(
                {
                    "person_id": f"person-{person}",
                    "session_id": f"session-{person}",
                    "window_id": f"w-{person}-{value}",
                    "window_seconds": 15,
                    "window_end": pd.Timestamp("2026-11-17T00:00:00Z")
                    + pd.Timedelta(seconds=value * 20),
                    "window_start": pd.Timestamp("2026-11-17T00:00:00Z")
                    + pd.Timedelta(seconds=value * 20 - 15),
                    "y": value,
                    "ear_left_mean": float(value),
                }
            )
    frame = pd.DataFrame(rows)
    folds = []
    for person in range(3):
        others = [p for p in range(3) if p != person]
        ids = lambda p: list(range(p * 3, p * 3 + 3))
        folds.append(
            {
                "id": f"fold_{person:02d}",
                "person": f"person-{person}",
                "status": "ready",
                "train": ids(others[0]) + ids(others[1]),
                "test": ids(person),
                "inner": [
                    {"train": ids(others[0]), "valid": ids(others[1])},
                    {"train": ids(others[1]), "valid": ids(others[0])},
                ],
            }
        )
    return (
        {"release_id": "fixture"},
        {15: frame},
        ["ear_left_mean"],
        {"windows": {"15": {"folds": folds}}},
        None,
        None,
    )


def fixture_group_splits(frame, seed=42):
    """Fixed group partitions for storage tests, not research evaluation folds."""
    for person in sorted(frame.person_id.unique()):
        yield (
            np.flatnonzero(frame.person_id.to_numpy() != person),
            np.flatnonzero(frame.person_id.to_numpy() == person),
        )


class RunnerStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "drive").mkdir()

    def setup_runner(self):
        stack = ExitStack()
        stack.enter_context(
            patch("batch02.runner._context", side_effect=lambda c: fixture())
        )
        stack.enter_context(
            patch("batch02.runner._identity", return_value={"fixture": "same-science"})
        )
        stack.enter_context(
            patch("batch02.runner.experiment_windows", return_value=(15,))
        )
        stack.enter_context(
            patch("batch02.runner.recipes_for", return_value=[Recipe()])
        )
        stack.enter_context(patch("batch02.runner.estimate_work", return_value=[]))
        stack.enter_context(patch("batch02.runner.capabilities", return_value={}))
        return stack

    def test_persistence_keeps_predictions_identical_and_resume_skips_finished_fits(
        self,
    ):
        fit_count = []

        def fit(*args, **kwargs):
            fit_count.append(1)
            return DeterministicBundle()

        first = Config(self.root, batch_root=self.root / "plain")
        second = Config(self.root, batch_root=self.root / "workspace")
        persistence = RunPersistence(
            second.batch, self.root / "drive", "exp03", interval=0
        )
        with self.setup_runner():
            plain, _ = run_experiment(first, "exp03", fit=fit)
            durable, _ = run_experiment(
                second, "exp03", fit=fit, persistence=persistence
            )
            pd.testing.assert_frame_equal(
                pd.read_csv(plain / "oof_predictions.csv"),
                pd.read_csv(durable / "oof_predictions.csv"),
            )
            count = len(fit_count)
            third = Config(self.root, batch_root=self.root / "new-runtime")
            other = RunPersistence(third.batch, self.root / "drive", "exp03")
            resumed, _ = run_experiment(
                third, "exp03", resume=persistence.remote, fit=fit, persistence=other
            )
            self.assertEqual(len(fit_count), count)
            pd.testing.assert_frame_equal(
                pd.read_csv(resumed / "oof_predictions.csv"),
                pd.read_csv(plain / "oof_predictions.csv"),
            )
        self.assertEqual(
            read_json(persistence.remote / "status.json")["status"], "complete"
        )

    def test_failed_status_write_does_not_mask_original_fit_error(self):
        config = Config(self.root, batch_root=self.root / "work")
        real_write = write_json

        def fail_status(path, value):
            if Path(path).name == "status.json" and value.get("status") == "failed":
                raise PermissionError("status write failed")
            return real_write(path, value)

        def fail_fit(*args, **kwargs):
            raise ValueError("original-fit-error")

        with (
            self.setup_runner(),
            patch("batch02.runner.write_json", fail_status),
            self.assertRaisesRegex(ValueError, "original-fit-error"),
        ):
            run_experiment(config, "exp03", fit=fail_fit)

    def test_interrupted_run_restores_only_committed_progress_without_duplicate_oof(
        self,
    ):
        from batch02.persistence import PersistenceError

        config = Config(self.root, batch_root=self.root / "interrupted")
        persistence = RunPersistence(
            config.batch, self.root / "drive", "exp03", interval=0
        )
        real_publish = persistence.publish
        outer_fits = []

        def fit(recipe, train, *args, **kwargs):
            if train.person_id.nunique() == 2:
                outer_fits.append(tuple(sorted(train.person_id.unique())))
            return DeterministicBundle()

        def interrupt(**kwargs):
            if persistence.run is not None:
                progress = persistence.run / "progress.json"
                if progress.exists() and read_json(progress)["fold"] == "fold_01":
                    raise PersistenceError("fixture disconnect")
            return real_publish(**kwargs)

        with self.setup_runner():
            with (
                patch.object(persistence, "publish", interrupt),
                self.assertRaises(PersistenceError),
            ):
                run_experiment(config, "exp03", fit=fit, persistence=persistence)
            other_config = Config(
                self.root, batch_root=self.root / "replacement-runtime"
            )
            other = RunPersistence(other_config.batch, self.root / "drive", "exp03")
            run, _ = run_experiment(
                other_config,
                "exp03",
                resume=persistence.remote,
                fit=fit,
                persistence=other,
            )
        predictions = pd.read_csv(run / "oof_predictions.csv")
        self.assertEqual(len(predictions), 9)
        self.assertFalse(predictions.window_id.duplicated().any())
        self.assertEqual(len(outer_fits), 3)

    def test_resume_cannot_disable_identity_check(self):
        from batch02.artifacts import open_run

        root = self.root / "exp03"
        run = open_run(root, {"data": "A", "code": "A"}, None)
        with self.assertRaises(ValueError):
            open_run(root, {"data": "A", "code": "B"}, run)

    def test_exp14_sensitivity_result_cache_skips_completed_fit(self):
        from batch02.finalization import _score_fit

        rows = fixture()[1][15]
        calls = []

        def fit(*args, **kwargs):
            calls.append(1)
            return DeterministicBundle()

        config = Config(self.root)
        path = self.root / "sensitivity/result.json"
        first = _score_fit(
            Recipe(),
            rows.iloc[:6],
            rows.iloc[6:],
            None,
            15,
            config,
            fit,
            cache_path=path,
        )
        second = _score_fit(
            Recipe(),
            rows.iloc[:6],
            rows.iloc[6:],
            None,
            15,
            config,
            fit,
            cache_path=path,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_exp14_deployment_does_not_refit_verified_completed_model(self):
        from batch02.finalization import _save_deployment

        rows = fixture()[1][15]
        calls = []

        def fit(*args, **kwargs):
            calls.append(1)
            return DeterministicBundle()

        record = {"recipe": Recipe(), "score": 0.5}
        directory = self.root / "deployment"
        _save_deployment(Config(self.root), record, rows, None, 15, directory, fit)
        _save_deployment(Config(self.root), record, rows, None, 15, directory, fit)
        self.assertEqual(len(calls), 1)

    def test_exp14_negative_controls_reuse_completed_fold_predictions(self):
        from batch02.diagnostics import negative_controls
        from batch02.features import MissingStream

        context = fixture()
        calls = []

        class Store:
            def __init__(self):
                self.features = ["ear_left_mean"]

            def stream(self, seconds):
                raise MissingStream("fixture has no temporal stream")

        class Control:
            def predict(self, matrix):
                return np.eye(3)[matrix.index.to_numpy() % 3], "probability"

        def fit(*args):
            calls.append(1)
            return Control()

        run = self.root / "controls"
        run.mkdir()
        with patch("batch02.diagnostics.fit_tabular", fit):
            negative_controls(Config(self.root), context[1], Store(), context[3], run)
            first = read_json(run / "negative_controls.json")
            negative_controls(Config(self.root), context[1], Store(), context[3], run)
        self.assertEqual(len(calls), 3)
        self.assertEqual(first, read_json(run / "negative_controls.json"))

    def test_every_remaining_experiment_can_publish_with_fit_doubles(self):
        from batch02 import runner

        capabilities = dict.fromkeys(
            ("mlp", "ft_transformer", "gru", "tabpfn"), "skipped_dependency"
        )
        for index in range(3, 15):
            print(f"Storage fixture exp{index:02d}", file=sys.stderr, flush=True)
            with (
                self.subTest(experiment=index),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                experiment = f"exp{index:02d}"
                config = Config(self.root, batch_root=self.root / experiment)
                persistence = RunPersistence(
                    config.batch, self.root / "drive", experiment
                )
                context = fixture()
                frames = {}
                manifests = {"windows": {}}
                for seconds in (15, 25, 45):
                    frame = context[1][15].copy()
                    frame["window_seconds"] = seconds
                    frame["window_start"] = frame.window_end - pd.Timedelta(
                        seconds=seconds
                    )
                    frame["window_id"] = str(seconds) + "-" + frame.window_id
                    frame["grid_origin"] = frame.window_start.iloc[0]
                    frames[seconds] = frame
                    manifests["windows"][str(seconds)] = context[3]["windows"]["15"]
                replay = (context[0], frames, context[2], manifests, None, None)
                with ExitStack() as stack:
                    stack.enter_context(
                        patch("batch02.runner._context", return_value=replay)
                    )
                    stack.enter_context(
                        patch(
                            "batch02.runner._identity",
                            return_value={"fixture": experiment},
                        )
                    )
                    stack.enter_context(
                        patch("batch02.runner.capabilities", return_value=capabilities)
                    )
                    stack.enter_context(
                        patch("batch02.search.capabilities", return_value=capabilities)
                    )
                    stack.enter_context(
                        patch("batch02.runner.estimate_work", return_value=[])
                    )
                    stack.enter_context(
                        patch("batch02.splits.group_splits", fixture_group_splits)
                    )
                    stack.enter_context(
                        patch("batch02.finalization.group_splits", fixture_group_splits)
                    )
                    stack.enter_context(patch("matplotlib.figure.Figure.savefig"))
                    stack.enter_context(patch("matplotlib.figure.Figure.tight_layout"))
                    stack.enter_context(
                        patch(
                            "batch02.modeling.fit_tabular",
                            side_effect=AssertionError(
                                "Real tabular fitting is forbidden in storage tests"
                            ),
                        )
                    )
                    stack.enter_context(
                        patch(
                            "batch02.modeling.fit_neural",
                            side_effect=AssertionError(
                                "Real neural fitting is forbidden in storage tests"
                            ),
                        )
                    )
                    # Dedicated tests exercise controls and source coverage separately.
                    stack.enter_context(patch("batch02.diagnostics.coverage_report"))
                    stack.enter_context(patch("batch02.diagnostics.negative_controls"))
                    run, _summary = runner.run_experiment(
                        config,
                        experiment,
                        fit=lambda *a, **k: DeterministicBundle(),
                        persistence=persistence,
                    )
                self.assertTrue(
                    read_json(persistence.remote / "status.json")["status"].startswith(
                        "complete"
                    )
                )
                self.assertTrue((persistence.remote / "checkpoints").is_dir())
                if index not in (11, 12, 13):
                    self.assertEqual(
                        set(pd.read_csv(run / "oof_predictions.csv").window_seconds),
                        {15, 25, 45},
                    )


if __name__ == "__main__":
    unittest.main()
