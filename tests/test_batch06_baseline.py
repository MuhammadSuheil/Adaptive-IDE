"""Leakage, baseline provenance, and standalone execution checks for Batch 06."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "models/batch-06"
SPEC = importlib.util.spec_from_file_location("batch06", DIRECTORY / "baseline_experiment.py")
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


def notebook_reference():
    notebook = json.loads((DIRECTORY / "exp01/EyeHeartCognitiveLoad.ipynb").read_text(encoding="utf-8"))
    namespace = {"json": json}
    for cell in notebook["cells"]:
        if "reference" in cell["metadata"].get("tags", []):
            exec(compile("".join(cell["source"]), "trusted_batch06_reference", "exec"), namespace)
    return namespace["REFERENCE"], namespace["SOURCE_EVIDENCE"]


def fixture(window=25, participants=8):
    records, evidence = [], {}
    origin = pd.Timestamp("2026-11-01T00:00:00Z")
    for person in range(participants):
        sid, pid = f"session{person}", f"person{person}"
        evidence[sid] = {
            "participant_id": pid, "baseline_end_ns": int((origin - pd.Timedelta(seconds=100)).value),
            "first_task_ns": int((origin + pd.Timedelta(seconds=40)).value),
            "baseline_rmssd_ms": 20.0 + person,
            "baseline_coverage": 0.95, "baseline_duration_seconds": 120.0,
            "baseline_status": "ready",
        }
        for index, end in enumerate([25] + list(range(100, 160, 5))):
            finish = origin + pd.Timedelta(seconds=end)
            label = 0 if index == 0 else (index - 1) % 3
            row = {
                "window_id": f"{sid}|{window}|{end}", "session_id": sid,
                "participant_id": pid,
                "window_start": (finish - pd.Timedelta(seconds=window)).isoformat(),
                "window_end": finish.isoformat(), "window_seconds": window,
                "study_phase": "pre_task" if index == 0 else "active_task",
                "quality_tier": "acceptable", "cognitive_load_label": batch.LABELS[label],
                "baseline_rmssd_ms": 20.0 + person, "baseline_coverage": 0.95,
                "baseline_usable": True,
                "rmssd_ms": 20.0 + person + label * 3 + index * 0.01,
                "sdnn_ms": 35.0 + person + label * 2,
                "blink_rate_per_minute": float(label * 2),
                "ear_left_mean": 0.3 + person * 0.001 + label * 0.01,
                "ear_left_std": 0.02 + label * 0.001,
                "ear_right_mean": 0.32 + person * 0.001 + label * 0.01,
                "ear_right_std": 0.025 + label * 0.001,
                "iris_size_estimate_mean": 0.2 + person * 0.001 + label * 0.002,
                "iris_size_estimate_std": 0.01,
                "iris_size_delta_mean": 0.0 if index == 0 else label * 0.01,
            }
            records.append(row)
    return batch.validate_frame(pd.DataFrame(records), window), evidence


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.frame, self.evidence = fixture()

    def test_notebook_is_standalone_and_matches_reviewed_implementation(self):
        notebook = json.loads((DIRECTORY / "exp01/EyeHeartCognitiveLoad.ipynb").read_text(encoding="utf-8"))
        implementation = (DIRECTORY / "baseline_experiment.py").read_text(encoding="utf-8")
        expected = hashlib.sha256(implementation.encode()).hexdigest()
        self.assertEqual(notebook["metadata"]["implementation_sha256"], expected)
        definitions = next(c for c in notebook["cells"] if "definitions" in c["metadata"].get("tags", []))
        self.assertTrue("".join(definitions["source"]).startswith(implementation.rstrip()))
        combined = "\n".join("".join(c["source"]) for c in notebook["cells"])
        self.assertIn("drive.mount('/content/drive')", combined)
        self.assertIn("Dataset/v2_session_phase", combined)
        self.assertNotIn("D:\\", combined)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "batch06_notebook", "exec")
                self.assertEqual(cell["outputs"], [])

    def test_unknown_source_or_changed_baseline_is_never_guessed(self):
        evidence = copy.deepcopy(self.evidence)
        del evidence["session0"]
        evidence["session1"]["baseline_rmssd_ms"] += 3
        audit, refs = batch.audit_baselines(self.frame, evidence)
        self.assertNotIn("session0", refs)
        self.assertNotIn("session1", refs)
        self.assertIn("not_audited", audit.set_index("session_id").loc["session0", "hrv_rejection_reason"])
        self.assertIn("differs", audit.set_index("session_id").loc["session1", "hrv_rejection_reason"])

    def test_baseline_during_first_task_and_low_coverage_are_excluded(self):
        evidence = copy.deepcopy(self.evidence)
        evidence["session0"]["baseline_end_ns"] = evidence["session0"]["first_task_ns"] + 1
        frame = self.frame.copy()
        frame.loc[frame.session_id.eq("session1"), "baseline_coverage"] = 0.7
        audit, refs = batch.audit_baselines(frame, evidence)
        self.assertNotIn("session0", refs)
        self.assertNotIn("session1", refs)
        self.assertFalse(audit.set_index("session_id").loc["session0", "joint_eligible"])

    def test_pretask_reference_never_uses_later_task_or_another_person(self):
        frame = self.frame.copy()
        frame.loc[frame.session_id.eq("session0"), "study_phase"] = "active_task"
        frame.loc[frame.session_id.eq("session1") & frame.study_phase.eq("pre_task"), "_end"] = self.evidence["session1"]["first_task_ns"]
        audit, refs = batch.audit_baselines(frame, self.evidence)
        self.assertNotIn("eye_values", refs["session0"])
        self.assertNotIn("eye_values", refs["session1"])
        self.assertEqual(int(audit.joint_eligible.sum()), 6)

    def test_reference_support_and_rounding_margin_are_removed(self):
        audit, refs = batch.audit_baselines(self.frame, self.evidence)
        selected = batch.make_cohort(self.frame, audit, refs, "hrv_eye")
        self.assertFalse(selected.study_phase.eq("pre_task").any())
        for row in selected.itertuples():
            self.assertGreaterEqual(int(selected.loc[row.Index, "_start"]), refs[row.session_id]["eye_available_ns"] + 10**9)

    def test_zero_blink_baseline_does_not_divide_by_zero_and_sdnn_is_unchanged(self):
        audit, refs = batch.audit_baselines(self.frame, self.evidence)
        selected = batch.make_cohort(self.frame, audit, refs, "hrv_eye")
        values = batch.representation(selected, refs, "hrv_eye", "baseline_relative")
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_array_equal(values[:, 1], selected.sdnn_ms)
        np.testing.assert_array_equal(values[:, 2], selected.blink_rate_per_minute)

    def test_heldout_scaling_has_no_effect_on_fitting_statistics(self):
        x = self.frame[batch.CORE].to_numpy(float)
        y = self.frame._y.to_numpy()
        fit = np.flatnonzero(self.frame.participant_id.ne("person0"))
        held = np.flatnonzero(self.frame.participant_id.eq("person0"))
        _, _, first = batch.fit_predict(x, y, fit, held, "logistic_regression")
        changed = x.copy()
        changed[held] += 100000
        _, _, second = batch.fit_predict(changed, y, fit, held, "logistic_regression")
        np.testing.assert_array_equal(first["scaler"].center_, second["scaler"].center_)
        np.testing.assert_array_equal(first["scaler"].scale_, second["scaler"].scale_)
        np.testing.assert_array_equal(first["estimator"].coef_, second["estimator"].coef_)

    def test_repeat_recordings_of_one_person_stay_in_one_group(self):
        frame = self.frame.copy()
        frame.loc[frame.session_id.eq("session1"), "participant_id"] = "person0"
        folds, reason = batch.make_folds(frame, {}, "new_dataset")
        self.assertEqual(reason, "")
        for _, fit, held in folds["participant_held_out"]:
            self.assertFalse(set(frame.iloc[fit].participant_id) & set(frame.iloc[held].participant_id))

    def test_new_dataset_does_not_silently_reuse_old_split(self):
        reference, _ = notebook_reference()
        frame, protocol = batch.split_frame(self.frame, {"25": "different"}, reference)
        self.assertEqual(protocol, "new_dataset_fixed_participant_split_42_43")
        people = [set(frame.loc[frame._split.eq(part), "participant_id"]) for part in ("train", "validation", "test")]
        self.assertFalse(people[0] & people[1] or people[0] & people[2] or people[1] & people[2])

    def test_complete_fixture_run_preserves_sources_and_never_predicts_test(self):
        reference, _ = notebook_reference()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for seconds in (15, 25, 45):
                frame, _ = fixture(seconds)
                directory = root / "dataset" / f"window_{seconds}s_step_5s"
                directory.mkdir(parents=True)
                frame.drop(columns=[c for c in frame if c.startswith("_")]).to_csv(directory / "good_acceptable.csv", index=False)
            parameters = {**batch.MODELS["extra_trees"], "n_estimators": 8}
            with patch.dict(batch.MODELS, {"extra_trees": parameters}):
                result = batch.run_experiment(root / "dataset", root / "outputs", reference, self.evidence, "fixture")
            run = result["run_dir"]
            state = json.loads((run / "status.json").read_text())
            self.assertTrue(state["complete"])
            self.assertFalse(state["test_evaluated"])
            self.assertTrue(state["dataset_unchanged"])
            assignments = pd.read_csv(run / "split_assignments.csv")
            test_ids = set(assignments.loc[assignments._split.eq("test"), "window_id"])
            for filename in ("oof_predictions.csv", "validation_predictions.csv"):
                predicted = pd.read_csv(run / filename)
                self.assertFalse(test_ids & set(predicted.window_id))
            predictions = pd.read_csv(run / "oof_predictions.csv")
            for row in result["leaderboard"].itertuples():
                subset = predictions[
                    predictions.cohort.eq(row.cohort) & predictions.protocol.eq(row.protocol)
                    & predictions.family.eq(row.family) & predictions.representation.eq(row.representation)
                ]
                self.assertEqual(subset.window_id.nunique(), len(subset))
                truth = subset.cognitive_load_label.map(dict(zip(batch.LABELS, range(3))))
                predicted = subset.predicted_label.map(dict(zip(batch.LABELS, range(3))))
                scored = batch.metrics(truth, predicted)
                self.assertAlmostEqual(scored["accuracy"], row.accuracy)
                self.assertAlmostEqual(scored["macro_f1"], row.macro_f1)


if __name__ == "__main__":
    unittest.main()
