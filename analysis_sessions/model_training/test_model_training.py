"""Check surrogate-label provenance, modality fusion, isolation and artifacts."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from analysis_sessions.model_training import data, reference, train_models as training, visualize
from analysis_sessions.model_training.predict import predict_windows
from analysis_sessions.model_training.presentation import display_frame, NOTE_ID


def fixture(directory, seconds=25):
    directory.mkdir(parents=True)
    rows = []
    for person in range(4):
        for index in range(24):
            demand = index / 23
            start = pd.Timestamp("2026-11-01T10:00:00+07:00") + pd.Timedelta(seconds=index * 5)
            row = {"window_id": f"person{person}-{index}", "participant_id": f"person{person}", "session_id": f"person{person}-session{index % 2}",
                   "window_start": start.isoformat(), "window_end": (start + pd.Timedelta(seconds=seconds)).isoformat(), "window_seconds": seconds,
                   "difficulty_level": "irrelevant-task-metadata", "quality_tier": "good" if index % 4 < 2 else "acceptable" if index % 4 == 2 else "poor"}
            for family, columns in data.FAMILIES.items():
                for offset, column in enumerate(columns):
                    row[column] = .2 + demand * .1 + offset * .01 + person * .003
            row.update(rmssd_ms=60 - demand * 30, sdnn_ms=70 - demand * 20,
                       rmssd_baseline_ratio=1.3 - demand * .8 + person * .017,
                       blink_count=3 - demand * 2, blink_rate_per_minute=20 - demand * 12 + person * .13,
                       eye_state_fixation_ratio=.2 + demand * .65 + person * .006,
                       iris_size_estimate_std=.03 if person == 0 else np.nan,
                       eye_state_unclassified_ratio=.01, eye_state_other_ratio=0.)
            if index == 23:
                row["rmssd_baseline_ratio"] = np.nan
            rows.append(row)
    frame = pd.DataFrame(rows)
    for name, tiers in zip(data.VARIANTS, (("good",), ("good", "acceptable"), ("good", "acceptable", "poor"))):
        frame.loc[frame.quality_tier.isin(tiers)].to_csv(directory / f"{name}.csv", index=False)
    data.write_json(directory / "feature_columns.json", {"target": "difficulty_level", "group": "participant_id", "features": [column for columns in data.FAMILIES.values() for column in columns] + list(data.DIAGNOSTICS)})
    return frame


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "input"
        fixture(self.path)
        self.config = data.load_config()
        self.dataset = data.load_dataset(self.path, self.config)
        self.good = self.dataset["views"]["good_only"]
        self.ref = reference.fit_reference(self.good, self.config)

    def test_uses_five_load_levels_and_ignores_task_difficulty(self):
        first = reference.label(self.ref, self.good)
        changed = self.good.assign(difficulty_level="some-other-task")
        pd.testing.assert_frame_equal(first, reference.label(self.ref, changed))
        self.assertEqual(set(first.reference_level), set(data.LEVELS))
        self.assertFalse("difficulty_level" in self.dataset["features"])
        self.assertTrue(set(data.DIAGNOSTICS).isdisjoint(self.dataset["features"]))
        self.assertEqual(len(self.dataset["features"]), 24)

    def test_score_directions_and_exact_fusion(self):
        test = self.good.iloc[:1].copy()
        lower = test.assign(rmssd_baseline_ratio=.1)
        higher = test.assign(rmssd_baseline_ratio=2.)
        self.assertGreater(reference.components(self.ref, lower).hrv_reference_score.iloc[0], reference.components(self.ref, higher).hrv_reference_score.iloc[0])
        suppressed = test.assign(blink_rate_per_minute=0.)
        frequent = test.assign(blink_rate_per_minute=100.)
        self.assertGreater(reference.components(self.ref, suppressed).eye_reference_score.iloc[0], reference.components(self.ref, frequent).eye_reference_score.iloc[0])
        fixed = test.assign(eye_state_fixation_ratio=1.)
        not_fixed = test.assign(eye_state_fixation_ratio=0.)
        self.assertGreater(reference.components(self.ref, fixed).eye_reference_score.iloc[0], reference.components(self.ref, not_fixed).eye_reference_score.iloc[0])
        self.assertAlmostEqual(float(reference.fuse(80., 20., .6)), 56.)
        self.assertAlmostEqual(float(reference.fuse(80., 20., .5)), 50.)

    def test_ties_boundaries_and_missingness(self):
        tied = self.good.copy()
        tied[list(data.ANCHORS)] = .5
        with self.assertRaisesRegex(ValueError, "ties"):
            reference.fit_reference(tied, self.config)
        ref = copy.deepcopy(self.ref)
        ref["boundaries"]["0.600000"] = [20., 40., 60., 80.]
        self.assertEqual(reference.levels(ref, [0., 19.9, 20., 40., 60., 80., 100., np.nan]).tolist(), ["Low", "Low", "Mid low", "Mid", "Mid high", "High", "High", ""])
        missing = self.good.iloc[:1].assign(rmssd_baseline_ratio=np.nan)
        self.assertEqual(reference.label(self.ref, missing).reference_level.iloc[0], "")
        zeros = self.good.iloc[:1].assign(rmssd_baseline_ratio=0., blink_rate_per_minute=0., eye_state_fixation_ratio=0.)
        self.assertTrue(data.eligible(zeros).all())

    def test_equal_participant_weight_and_reference_isolation(self):
        weights = reference.participant_weights(self.good)
        totals = pd.Series(weights).groupby(self.good.participant_id.reset_index(drop=True)).sum()
        np.testing.assert_allclose(totals, .25)
        held_out = "person0"
        pool = self.good.loc[self.good.participant_id.ne(held_out)]
        ref = reference.fit_reference(pool, self.config)
        self.assertNotIn(held_out, ref["participants"])
        for case, audit in zip(*training.inner_cases(pool.reset_index(drop=True), pool, self.config)):
            self.assertTrue(set(audit["reference_participants"]).issubset(audit["train_participants"]))
            self.assertFalse(set(audit["train_participants"]) & set(audit["validation_participants"]))
            self.assertNotIn(held_out, audit["reference_participants"])
            train, valid, _, _ = case
            self.assertTrue(set(train.session_id).isdisjoint(valid.session_id))

    def test_nested_data_timestamps_and_hashes(self):
        frame = data.read_csv(self.path / "good_acceptable.csv")
        frame["ear_left_mean"] = pd.to_numeric(frame.ear_left_mean)
        frame.loc[0, "ear_left_mean"] = 999
        frame.to_csv(self.path / "good_acceptable.csv", index=False)
        with self.assertRaisesRegex(ValueError, "identical"):
            data.load_dataset(self.path, self.config)
        other = self.path.parent / "other"
        fixture(other, seconds=40)
        self.assertEqual(data.load_dataset(other, self.config)["window_seconds"], 40.)
        changed = data.read_csv(other / "all_data.csv")
        changed["window_seconds"] = 25
        changed.to_csv(other / "all_data.csv", index=False)
        with self.assertRaisesRegex(ValueError, "durations"):
            data.load_dataset(other, self.config)

    def test_no_rowwise_fallback_when_reference_people_are_insufficient(self):
        frame = self.good.loc[self.good.participant_id.isin(["person0", "person1"])]
        with self.assertRaisesRegex(ValueError, "grouped inner folds"):
            training.inner_cases(frame.reset_index(drop=True), frame, self.config)

    def test_config_rejects_bad_weights(self):
        bad = copy.deepcopy(self.config)
        bad["primary_hrv_weight"] = 2.
        path = self.path / "bad_config.json"
        data.write_json(path, bad)
        with self.assertRaisesRegex(ValueError, "Modality weights"):
            data.load_config(path)

    def test_quality_and_fixed_prediction_weights(self):
        good = self.good.copy()
        groups = self.dataset["groups"]
        hrv = DummyRegressor(strategy="constant", constant=80.).fit(good[groups["HRV"]], np.zeros(len(good)))
        eye_features = [col for family, columns in groups.items() if family != "HRV" for col in columns]
        eye = DummyRegressor(strategy="constant", constant=20.).fit(good[eye_features], np.zeros(len(good)))
        bundle = {"bundle_version": 1, "config": self.config, "reference": self.ref, "groups": groups, "features": self.dataset["features"],
                  "hrv_model": hrv, "eye_model": eye, "window_seconds": 25.}
        inputs = good.iloc[:3].copy()
        inputs.index = [10, 20, 30]
        predicted = predict_windows(bundle, inputs)
        np.testing.assert_allclose(predicted.combined_score, 56.)
        np.testing.assert_allclose(predict_windows(bundle, inputs, .5).combined_score, 50.)
        self.assertTrue(predict_windows(bundle, inputs.assign(quality_tier="poor")).cognitive_load_level.eq("").all())
        self.assertTrue(predict_windows(bundle, inputs.drop(columns=["quality_tier"])).signal_status.eq("quality_unverified").all())
        self.assertTrue(predict_windows(bundle, inputs.assign(rmssd_baseline_ratio=np.nan)).signal_status.eq("insufficient_signal").all())
        with self.assertRaisesRegex(ValueError, "Window size"):
            predict_windows(bundle, inputs.assign(window_seconds=40))


class ArtifactTests(unittest.TestCase):
    def test_visualization_rejects_modified_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "window/models/model.joblib"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"modified artifact")
            data.write_json(root / "run_manifest.json", {"status": "complete", "datasets": {},
                                                        "output_sha256": {"window/models/model.joblib": "wrong"}})
            with self.assertRaisesRegex(ValueError, "Result hash mismatch"):
                visualize.visualize_run(root)


class IntegrationTests(unittest.TestCase):
    def test_training_exports_branches_shared_labels_graphs_and_safe_reload(self):
        original = training.model_grid

        def small(name, search, seed):
            pipeline, grid = original(name, search, seed)
            if name == "random_forest":
                pipeline.set_params(model__n_estimators=5)
            if name == "gradient_boosting":
                pipeline.set_params(model__max_iter=5, model__min_samples_leaf=2)
            return pipeline, grid

        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / "input", Path(temporary) / "run"
            fixture(source)
            before = {p.name: data.digest(p) for p in source.iterdir()}
            with patch.object(training, "model_grid", side_effect=small), redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Skipping features without any observed values")
                manifest = training.run([("window25", source)], output, search="quick")
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(before, {p.name: data.digest(p) for p in source.iterdir()})
            root = output / "window25"
            models = list((root / "models").glob("*.joblib"))
            self.assertEqual(len(models), 9)
            self.assertEqual(len(list((root / "visualizations").glob("*.png"))), 9)
            report = (root / "visualizations/report.html").read_text(encoding="utf-8")
            self.assertIn('lang="id"', report)
            self.assertIn(NOTE_ID, report)
            predictions = pd.read_csv(root / "held_out_predictions.csv")
            selected = [frame.set_index("window_id")["reference_level"].sort_index() for _, frame in predictions.groupby(["variant", "model"])]
            self.assertEqual(len(selected), 9)
            for result in selected[1:]:
                pd.testing.assert_series_equal(result, selected[0])
            np.testing.assert_allclose(predictions.combined_score, .6 * predictions.hrv_score + .4 * predictions.eye_score)
            self.assertTrue(predictions.cognitive_load_level.isin(data.LEVELS).all())
            audits = json.loads((root / "training_audit.json").read_text())
            for audit in audits:
                person = audit["held_out_participant_id"]
                self.assertNotIn(person, audit["train_participants"] + audit["reference_participants"])
                for inner in audit["inner_folds"]:
                    self.assertFalse(set(inner["train_participants"]) & set(inner["validation_participants"]))
                    self.assertTrue(set(inner["reference_participants"]).issubset(inner["train_participants"]))
                    self.assertNotIn(person, inner["train_participants"] + inner["validation_participants"])
            for path in models:
                bundle = joblib.load(path)
                self.assertIn("eye_model", bundle)
                self.assertIn("hrv_model", bundle)
                result = predict_windows(bundle, data.read_csv(source / "good_only.csv").iloc[:3])
                self.assertTrue(result.cognitive_load_level.isin(data.LEVELS).all())
                np.testing.assert_allclose(result.combined_score, .6 * result.hrv_score + .4 * result.eye_score)
            for name, value in manifest["output_sha256"].items():
                self.assertEqual(data.digest(output / name), value)
            with self.assertRaises(FileExistsError):
                training.run([("window25", source)], output, search="quick")
            with self.assertRaisesRegex(ValueError, "separate"):
                training.run([("window25", source)], source / "unsafe", search="quick")


if __name__ == "__main__":
    unittest.main()
