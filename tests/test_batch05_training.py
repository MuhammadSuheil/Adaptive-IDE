"""Stage 2 training boundaries and persistence, using small synthetic data."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from test_batch05_preparation import definitions


def training_fixture(ns):
    n = 180
    y = np.arange(n) % 3
    frame = pd.DataFrame(
        {
            "window_id": [f"w{i}" for i in range(n)],
            "session_id": ["s"] * n,
            "participant_id": ["p"] * n,
            "_end": np.arange(n, dtype=np.int64) * 60 * 10**9,
            "_block": np.arange(n) // 3,
            "_y": y,
            "cognitive_load_label": np.array(["Low", "Medium", "High"])[y],
            "study_phase": ["active_task"] * n,
            "_split": ["train"] * 90 + ["validation"] * 45 + ["test"] * 45,
        }
    )
    frame["_key"] = frame.session_id + "|" + frame._end.astype(str)
    for i, column in enumerate(ns["CORE"]):
        frame[column] = 0.1 + 0.1 * y + 0.0001 * np.arange(n) + i / 1000
    return frame


class Stage2Availability(unittest.TestCase):
    def test_standalone_notebook_exposes_training_workflow(self):
        self.assertTrue(
            "run_stage2" in definitions(), "Stage 2 has not been implemented"
        )


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = definitions()
        if "run_stage2" not in cls.ns:
            raise unittest.SkipTest("Stage 2 is not implemented")

    def setUp(self):
        self.frame = training_fixture(self.ns)

    def test_test_rows_and_overlap_rejected_before_fitting(self):
        for fit, held in (
            ([0, 1, 2], [150, 151, 152]),
            ([150, 151, 152], [90, 91, 92]),
            ([0, 1, 2], [2, 3, 4]),
        ):
            with self.subTest(fit=fit, held=held), self.assertRaises(ValueError):
                self.ns["validate_training_indices"](self.frame, fit, held)
        changed = self.frame.copy()
        changed.loc[90, "_end"] = changed.loc[89, "_end"] + 5 * 10**9
        with self.assertRaises(ValueError):
            self.ns["validate_training_indices"](
                changed, np.arange(90), np.arange(90, 135)
            )

    def test_metrics_use_all_three_classes_and_reject_nonfinite_output(self):
        result = self.ns["classification_metrics"]([0, 1, 2], [0, 0, 0])
        self.assertAlmostEqual(result["accuracy"], 1 / 3)
        self.assertAlmostEqual(result["macro_f1"], 1 / 6)
        self.assertEqual(result["recall"], [1.0, 0.0, 0.0])
        self.assertEqual(result["predicted_counts"], [3, 0, 0])
        for invalid in ([0, np.nan, 2], [0, 0.5, 2], [0, 1, 3]):
            with self.assertRaises(ValueError):
                self.ns["classification_metrics"]([0, 1, 2], invalid)

    def test_learning_subsets_are_nested_blocks_and_never_include_heldout(self):
        ids = np.arange(90)
        subsets = self.ns["learning_subsets"](self.frame, ids)
        previous = set()
        for fraction in (0.25, 0.5, 0.75, 1.0):
            current = set(subsets[fraction])
            self.assertTrue(previous.issubset(current))
            self.assertTrue(current.issubset(set(ids)))
            for block in self.frame.iloc[list(current)]._block.unique():
                self.assertTrue(
                    set(self.frame.index[self.frame._block.eq(block)]) <= current
                )
            previous = current
        self.assertEqual(previous, set(ids))

    def test_checkpoint_rejects_mismatched_signature_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self.ns["Stage2Store"](root, {"data": "a"})
            job = store.job_dir("one")
            (job / "result.json").write_text('{"value": 17}', encoding="utf-8")
            store.complete("one")
            resumed = self.ns["Stage2Store"](root, {"data": "a"}, store.remote)
            self.assertEqual(
                json.loads((resumed.get("one") / "result.json").read_text())["value"],
                17,
            )
            with self.assertRaises(ValueError):
                self.ns["Stage2Store"](root, {"data": "b"}, store.remote)
            (store.remote / "jobs/one/result.json").write_text(
                "corrupt", encoding="utf-8"
            )
            third = self.ns["Stage2Store"](root, {"data": "a"}, store.remote)
            with self.assertRaises(ValueError):
                third.get("one")

    def test_interrupted_job_is_not_a_completed_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.ns["Stage2Store"](Path(directory), {"data": "a"})
            store.job_dir("partial").joinpath("result.json").write_text("{}")
            self.assertIsNone(store.get("partial"))
            with self.assertRaises(ValueError):
                store.job_dir("../escape")

    def test_real_fit_resume_and_loaded_model_predictions_match(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.ns["Stage2Store"](Path(directory), {"data": "synthetic"})
            result = self.ns["fit_job"](
                self.frame,
                np.arange(90),
                np.arange(90, 135),
                "logistic",
                "global",
                store,
                "final",
                save_model=True,
            )
            self.assertTrue(result["converged"])
            self.assertGreater(result["held_metrics"]["accuracy"], 0.9)
            again = self.ns["fit_job"](
                self.frame,
                np.arange(90),
                np.arange(90, 135),
                "logistic",
                "global",
                store,
                "final",
                save_model=True,
            )
            self.assertEqual(result, again)
            self.assertEqual(len(list((store.remote / "jobs").glob("*/done.json"))), 1)
            model, transform = self.ns["load_saved_model"](store.get("final"))
            p = model.predict(transform.transform(self.frame.iloc[90:135]))
            self.assertTrue(np.array_equal(p, np.arange(90, 135) % 3))

    def test_prediction_table_cannot_be_misaligned_with_truth(self):
        table = pd.DataFrame(
            {
                "window_id": ["w0", "w1", "w2"],
                "target": [0, 1, 2],
                "prediction": [0, 1, 2],
            }
        )
        self.ns["verify_prediction_table"](self.frame, [0, 1, 2], table)
        with self.assertRaises(ValueError):
            self.ns["verify_prediction_table"](self.frame, [0, 1, 2], table.iloc[::-1])
        table.loc[1, "target"] = 0
        with self.assertRaises(ValueError):
            self.ns["verify_prediction_table"](self.frame, [0, 1, 2], table)

    def test_supported_classifier_wrappers_learn_or_provide_dummy_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            store = self.ns["Stage2Store"](Path(directory), {"data": "four_models"})
            for family in ("logistic", "extra_trees", "hist_gradient", "dummy"):
                with self.subTest(family=family):
                    result = self.ns["fit_job"](
                        self.frame,
                        np.arange(90),
                        np.arange(90, 135),
                        family,
                        "global",
                        store,
                        family,
                    )
                    if family == "dummy":
                        self.assertAlmostEqual(
                            result["held_metrics"]["accuracy"], 1 / 3
                        )
                    else:
                        self.assertGreater(result["held_metrics"]["accuracy"], 0.8)


if __name__ == "__main__":
    unittest.main()
