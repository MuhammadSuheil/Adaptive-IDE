"""Behavioral checks for the frozen-cohort, standalone Exp02 ablation notebook."""

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
HERE = ROOT / "models/batch-06"


def definitions():
    ns = {"__name__": "batch06_ablation_test"}
    for filename in ("baseline_experiment.py", "ablation_experiment.py"):
        exec(compile((HERE / filename).read_text(encoding="utf-8"), filename, "exec"), ns)
    return ns


def snapshot_from_notebook():
    notebook = json.loads((HERE / "exp02/EyeHeartCognitiveLoad.ipynb").read_text(encoding="utf-8"))
    ns = {"json": json}
    for cell in notebook["cells"]:
        if "reference" in cell["metadata"].get("tags", []):
            exec(compile("".join(cell["source"]), "trusted_exp02_snapshot", "exec"), ns)
    return ns["ABLATION_SNAPSHOT"]


def build_fixture(root, ns):
    helper_spec = importlib.util.spec_from_file_location("exp01_fixture", ROOT / "tests/test_batch06_baseline.py")
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    frames = {}
    hashes = {}
    for window in (15, 25, 45):
        frame, evidence = helper.fixture(window, participants=4)
        for sid, group in frame.groupby("session_id"):
            active = group[group.study_phase.eq("active_task")]
            for index, key in enumerate(active.index):
                end = pd.Timestamp("2026-11-01T00:00:00Z") + pd.Timedelta(seconds=100 + index * 90)
                frame.loc[key, "window_end"] = end.isoformat()
                frame.loc[key, "window_start"] = (end - pd.Timedelta(seconds=window)).isoformat()
        columns = [c for c in frame if not c.startswith("_")]
        frame = ns["validate_frame"](frame[columns], window)
        directory = root / f"window_{window}s_step_5s"
        directory.mkdir(parents=True)
        path = directory / "good_acceptable.csv"
        frame[columns].to_csv(path, index=False)
        frames[window] = frame
        hashes[str(window)] = ns["sha256_file"](path)
    frame = frames[25]
    audit, refs = ns["audit_baselines"](frame, evidence)
    cohort = ns["make_cohort"](frame, audit, refs, "hrv_eye")
    train = pd.concat([g.iloc[:8] for _, g in cohort.groupby("participant_id")]).reset_index(drop=True)
    validation = pd.concat([g.iloc[8:] for _, g in cohort.groupby("participant_id")])
    validation = validation[validation._y.ne(1)].reset_index(drop=True)
    folds = {"participant_held_out": [], "historical_block_cv": []}
    for number, person in enumerate(sorted(train.participant_id.unique())):
        held = train.participant_id.eq(person)
        folds["participant_held_out"].append({"fold": number, "fit_window_ids": train.loc[~held, "window_id"].tolist(), "held_window_ids": train.loc[held, "window_id"].tolist()})
    positions = train.groupby("participant_id").cumcount()
    for number in (0, 1):
        held = positions.mod(2).eq(number)
        folds["historical_block_cv"].append({"fold": number, "fit_window_ids": train.loc[~held, "window_id"].tolist(), "held_window_ids": train.loc[held, "window_id"].tolist()})
    reserved = sorted(set(frame.window_id) - set(train.window_id) - set(validation.window_id))
    snapshot = {
        "source_run": "fixture_not_research", "dataset_sha256": hashes,
        "participants": sorted(train.participant_id.unique().tolist()),
        "train_window_ids": train.window_id.tolist(), "validation_window_ids": validation.window_id.tolist(),
        "test_window_ids": reserved, "folds": folds, "baseline_references": refs,
        "source_evidence": evidence,
        "exp01_controls": [{"protocol": protocol, "representation": mode, "accuracy": 0.0, "macro_f1": 0.0} for protocol in folds for mode in ("raw", "combined")],
    }
    return snapshot


class AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = definitions()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.dataset = Path(self.temporary.name) / "dataset"
        self.snapshot = build_fixture(self.dataset, self.ns)
        self.data = self.ns["prepare_ablation"](self.dataset, self.snapshot)

    def test_standalone_notebook_contains_exact_reviewed_sources(self):
        notebook = json.loads((HERE / "exp02/EyeHeartCognitiveLoad.ipynb").read_text(encoding="utf-8"))
        for tag, filename in (("shared_exp01", "baseline_experiment.py"), ("exp02", "ablation_experiment.py")):
            source = (HERE / filename).read_text(encoding="utf-8")
            key = "shared_exp01" if tag == "shared_exp01" else "exp02"
            self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), notebook["metadata"]["implementation_sha256"][key])
            cell = next(c for c in notebook["cells"] if c["cell_type"] == "code" and tag in c["metadata"].get("tags", []))
            self.assertTrue("".join(cell["source"]).startswith(source.rstrip()))
        text = "\n".join("".join(c["source"]) for c in notebook["cells"])
        self.assertIn("drive.mount('/content/drive')", text)
        self.assertIn("Dataset/v2_session_phase", text)
        self.assertIn("models/batch-06/outputs/exp02", text)
        self.assertNotIn("D:\\", text)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "exp02_notebook", "exec")
                self.assertEqual(cell["outputs"], [])

    def test_transforms_change_only_the_intended_sensor_columns(self):
        frame, refs = self.data["train"], self.data["references"]
        values = {mode: self.ns["ablation_representation"](frame, refs, mode) for mode in self.ns["ABLATION_MODES"]}
        original = frame[self.ns["CORE"]].to_numpy(float)
        np.testing.assert_array_equal(values["raw"], original)
        np.testing.assert_array_equal(values["rmssd_only"][:, 1:], original[:, 1:])
        np.testing.assert_array_equal(values["eye_only"][:, :2], original[:, :2])
        np.testing.assert_array_equal(values["combined"][:, 0], values["rmssd_only"][:, 0])
        np.testing.assert_array_equal(values["combined"][:, 2:], values["eye_only"][:, 2:])
        for mode in values:
            np.testing.assert_array_equal(values[mode][:, 1], original[:, 1])
            self.assertTrue(np.isfinite(values[mode]).all())

    def test_missing_medium_has_unavailable_recall_and_explicit_zero_support(self):
        frame = self.data["validation"]
        result = self.ns["ablation_metrics"](frame, frame._y.to_numpy())
        self.assertEqual(result["support_medium"], 0)
        self.assertIsNone(result["recall_medium"])
        self.assertFalse(result["three_class_evaluable"])
        self.assertAlmostEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["macro_f1"], 2 / 3)

    def test_equal_person_weights_are_not_window_weights(self):
        source = self.data["train"]
        first = source[source.participant_id.eq("person0")].iloc[[0]]
        second = source[source.participant_id.eq("person1")].iloc[[0]]
        frame = pd.concat([first, first, first, second], ignore_index=True)
        prediction = frame._y.to_numpy().copy()
        prediction[-1] = (prediction[-1] + 1) % 3
        result = self.ns["ablation_metrics"](frame, prediction)
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["mean_participant_accuracy"], 0.5)

    def test_changed_dataset_is_rejected_instead_of_resplit(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["dataset_sha256"]["25"] = "changed"
        with self.assertRaisesRegex(ValueError, "Hash CSV"):
            self.ns["prepare_ablation"](self.dataset, snapshot)

    def test_reserved_test_window_cannot_enter_supervised_comparison(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["test_window_ids"].append(snapshot["train_window_ids"][0])
        with self.assertRaisesRegex(ValueError, "Test"):
            self.ns["prepare_ablation"](self.dataset, snapshot)

    def test_reference_values_cannot_be_replaced_after_snapshot(self):
        snapshot = copy.deepcopy(self.snapshot)
        ref = next(iter(snapshot["baseline_references"].values()))
        ref["eye_values"]["ear_left_mean"] += 0.1
        with self.assertRaisesRegex(ValueError, "acuan eye berubah"):
            self.ns["prepare_ablation"](self.dataset, snapshot)

    def test_participant_leakage_is_rejected(self):
        snapshot = copy.deepcopy(self.snapshot)
        case = snapshot["folds"]["participant_held_out"][0]
        case["fit_window_ids"].append(case["held_window_ids"].pop())
        with self.assertRaisesRegex(ValueError, "Peserta bocor"):
            self.ns["prepare_ablation"](self.dataset, snapshot)

    def test_historical_time_overlap_is_rejected(self):
        snapshot = copy.deepcopy(self.snapshot)
        path = self.dataset / "window_25s_step_5s/good_acceptable.csv"
        frame = pd.read_csv(path)
        group = frame[frame.session_id.eq("session0") & frame.study_phase.eq("active_task")]
        source, destination = group.iloc[0], group.index[1]
        end = pd.Timestamp(source.window_end) + pd.Timedelta(seconds=5)
        frame.loc[destination, "window_end"] = end.isoformat()
        frame.loc[destination, "window_start"] = (end - pd.Timedelta(seconds=25)).isoformat()
        frame.to_csv(path, index=False)
        snapshot["dataset_sha256"]["25"] = self.ns["sha256_file"](path)
        with self.assertRaisesRegex(ValueError, "Guard waktu"):
            self.ns["prepare_ablation"](self.dataset, snapshot)

    def test_complete_fixture_run_and_saved_metrics(self):
        model = {**self.ns["MODELS"]["extra_trees"], "n_estimators": 8}
        with patch.dict(self.ns["MODELS"], {"extra_trees": model}):
            result = self.ns["run_ablation"](self.dataset, Path(self.temporary.name) / "outputs", self.snapshot, {"fixture": "not_research"})
        run = result["run_dir"]
        status = json.loads((run / "status.json").read_text())
        self.assertTrue(status["complete"])
        self.assertEqual(status["cv_fits"], 30)
        self.assertEqual(status["validation_fits"], 4)
        self.assertFalse(status["test_evaluated"])
        self.assertFalse(status["validation_three_class_evaluable"])
        self.assertTrue(status["dataset_unchanged"])
        oof = pd.read_csv(run / "oof_predictions.csv")
        validation = pd.read_csv(run / "validation_predictions.csv")
        self.assertFalse(set(self.snapshot["test_window_ids"]) & set(pd.concat([oof, validation]).window_id))
        label_map = dict(zip(self.ns["LABELS"], range(3)))
        for row in result["leaderboard"].itertuples():
            table = oof[oof.protocol.eq(row.protocol) & oof.representation.eq(row.representation)]
            self.assertEqual(table.window_id.nunique(), len(table))
            self.assertEqual(set(table.window_id), set(self.snapshot["train_window_ids"]))
            frame = table.assign(_y=table.cognitive_load_label.map(label_map))
            scored = self.ns["ablation_metrics"](frame, table.predicted_label.map(label_map).to_numpy())
            self.assertAlmostEqual(scored["accuracy"], row.accuracy)
            self.assertAlmostEqual(scored["macro_f1"], row.macro_f1)
        saved_metrics = pd.read_csv(run / "validation_metrics.csv")
        self.assertTrue(saved_metrics.recall_medium.isna().all())
        lock = json.loads((run / "comparison_lock.json").read_text())
        self.assertEqual(set(lock["variants"]), set(self.ns["ABLATION_MODES"]))
        self.assertFalse(lock["validation_evaluated"])
        self.assertIn("N/A", (run / "RESULTS.md").read_text())
        for filename in ("ablation_oof.png", "ablation_contrasts.png", "validation_support.png"):
            self.assertTrue((run / "figures" / filename).is_file())

    def test_research_preflight_reuses_exp01_cohort_without_training(self):
        snapshot = snapshot_from_notebook()
        result = self.ns["prepare_ablation"](ROOT / "analysis_sessions/dataset_fusion/output/v2_session_phase", snapshot)
        self.assertEqual(len(result["train"]), 671)
        self.assertEqual(len(result["validation"]), 99)
        self.assertEqual(result["train"].participant_id.nunique(), 4)
        self.assertEqual(result["validation"]._y.eq(1).sum(), 0)
        self.assertEqual({k: len(v) for k, v in result["folds"].items()}, {"participant_held_out": 4, "historical_block_cv": 5})
        for name, expected in snapshot["source_artifact_sha256"].items():
            path = HERE / "outputs/exp01" / snapshot["source_run"] / name
            self.assertEqual(self.ns["sha256_file"](path), expected)


if __name__ == "__main__":
    unittest.main()
