"""End-to-end artifact tests; the classifier is a deterministic test double."""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))


class ProtocolDouble:
    feature_names = ("rmssd_ms", "sdnn_ms")
    epoch = None
    fallback_count = 0

    def predict(self, rows, store, seconds):
        return np.tile([0.5, 0.3, 0.2], (len(rows), 1)), "probability"


def double_fit(recipe, rows, store, seconds, config, epochs=None):
    return ProtocolDouble()


def make_fixture(directory):
    from batch02.artifacts import file_hash, write_json
    from batch02.contracts import Config

    dataset = directory / "dataset"
    batch = directory / "models" / "batch-02"
    batch.mkdir(parents=True)
    shutil.copy2(ROOT / "requirements.txt", batch / "requirements.txt")
    definition = json.loads(
        (
            ROOT.parent.parent
            / "analysis_sessions/dataset_fusion/output/v2_session_phase/window_25s_step_5s/feature_columns.json"
        ).read_text()
    )
    features = definition["features"]
    files = {}
    registry = {
        "sessions": [
            {"session_id": f"s{i}", "person_id": f"p{i}", "confirmed": True}
            for i in range(9)
        ]
    }
    for seconds in (15, 25, 45):
        subdir = dataset / f"window_{seconds}s_step_5s"
        subdir.mkdir(parents=True)
        rows = []
        for person in range(9):
            for label in range(3):
                end = pd.Timestamp("2026-01-01T00:01:00Z") + pd.Timedelta(
                    seconds=label * 5
                )
                start = end - pd.Timedelta(seconds=seconds)
                rows.append(
                    {
                        "session_id": f"s{person}",
                        "participant_id": f"p{person}",
                        "window_id": f"s{person}|{start.isoformat()}|{end.isoformat()}",
                        "window_start": start.isoformat(),
                        "window_end": end.isoformat(),
                        "window_seconds": seconds,
                        "quality_tier": "good",
                        "cognitive_load_label": ["Low", "Medium", "High"][label],
                        **{n: (person * 3 + label + 1) / 30 for n in features},
                    }
                )
        pd.DataFrame(rows).to_csv(subdir / "good_acceptable.csv", index=False)
        write_json(subdir / "feature_columns.json", definition)
        for name in ("good_acceptable.csv", "feature_columns.json"):
            files[f"{subdir.name}/{name}"] = file_hash(subdir / name)
    write_json(
        dataset / "manifest.json",
        {
            "admission_policy": "window_quality_only",
            "source_integrity_verified": True,
            "output_sha256": files,
        },
    )
    return Config(directory, dataset_root=dataset, batch_root=batch), registry


class IntegrationTest(unittest.TestCase):
    def test_final_workflow_exports_joint_and_deployment_using_only_doubles(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest.mock import patch

        from batch02.contracts import freeze_release
        from batch02.registry import Recipe
        from batch02.runner import run_final

        def one_control(evaluator, inner):
            record = evaluator.score(Recipe(), inner)
            return (
                record,
                [record],
                {"effective_recipes": 1, "final_pool": [Recipe().key], "truncated": []},
            )

        with tempfile.TemporaryDirectory() as tmp:
            config, registry = make_fixture(Path(tmp))
            freeze_release(
                config, registry, ["s8"], "test fixture", proportions_approved=True
            )
            with (
                patch("batch02.runner.adaptive_search", side_effect=one_control),
                patch("batch02.finalization.adaptive_search", side_effect=one_control),
                patch(
                    "sklearn.linear_model.LogisticRegression.fit",
                    side_effect=AssertionError(
                        "Real classifier training is forbidden in this test"
                    ),
                ),
                redirect_stdout(StringIO()),
            ):
                run, summary = run_final(config, fit=double_fit)
            self.assertTrue(summary.complete_oof.all())
            self.assertEqual(len(summary), 3)
            joint = pd.read_csv(run / "joint_window_oof.csv")
            self.assertEqual(len(joint), 27)
            self.assertFalse(joint.duplicated(["session_id", "window_end"]).any())
            preferred = json.loads(
                (run / "deployment/preferred_window.json").read_text()
            )
            self.assertTrue((run / "deployment" / preferred["model"]).is_file())
            self.assertTrue((run / "negative_controls.json").is_file())

    def test_runner_exports_complete_oof_and_can_resume_without_fitting_again(self):
        from batch02.contracts import freeze_release
        from batch02.runner import run_experiment

        with tempfile.TemporaryDirectory() as tmp:
            config, registry = make_fixture(Path(tmp))
            freeze_release(
                config, registry, ["s8"], "test fixture", proportions_approved=True
            )
            run, summary = run_experiment(config, "exp01", fit=double_fit)
            self.assertTrue(summary.complete_oof.all())
            self.assertEqual(len(summary), 12)
            predictions = pd.read_csv(run / "oof_predictions.csv")
            self.assertEqual(len(predictions), 324)

            def refuse_fit(*args, **kwargs):
                raise AssertionError("Cached resume must not refit a completed fold")

            resumed, again = run_experiment(config, "exp01", resume=run, fit=refuse_fit)
            self.assertEqual(run, resumed)
            self.assertEqual(again.rows.tolist(), summary.rows.tolist())

    def test_changed_csv_is_rejected_before_release_approval(self):
        from batch02.contracts import inspect_dataset

        with tempfile.TemporaryDirectory() as tmp:
            config, _ = make_fixture(Path(tmp))
            path = config.dataset / "window_25s_step_5s" / "good_acceptable.csv"
            frame = pd.read_csv(path)
            frame.loc[0, "rmssd_ms"] = 1234.0
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Checksum"):
                inspect_dataset(config)

    def test_technical_validator_does_not_start_training(self):
        from batch02.validation import validate_package

        result = validate_package(ROOT)
        self.assertFalse(result["training_performed"])
        self.assertEqual(result["notebooks"], 15)


if __name__ == "__main__":
    unittest.main()
