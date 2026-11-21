"""Performance and equivalence contracts for the standalone exp02 notebook."""

import contextlib
import io
import tempfile
import time
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from tests.test_batch04_current import definitions, synthetic


class Exp02Performance(unittest.TestCase):
    def test_committed_checkpoint_is_local_first_and_final_audit_detects_corruption(
        self,
    ):
        ns = definitions("exp02")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ns["ArtifactStore"](root / "local", root / "drive", {"test": 1})
            store.commit("fold_0", {"score": 0.5}, {"data.bin": b"valid"})
            remote_checks = []
            original = store._verified

            def counted(directory, key):
                if directory == store.remote:
                    remote_checks.append(key)
                return original(directory, key)

            store._verified = counted
            for _ in range(3):
                self.assertEqual(store.artifact("fold_0", "data.bin"), b"valid")
            self.assertEqual(remote_checks, [])
            (store.remote / "fold_0/data.bin").write_bytes(b"broken")
            with self.assertRaises(ValueError):
                store.audit_remote()

    def test_fresh_runtime_downloads_once_and_failed_upload_is_recovered(self):
        ns = definitions("exp02")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ns["ArtifactStore"](root / "local", root / "drive", {"test": 1})
            copy = store._copy_verified

            def interrupted(target, data):
                if target.is_relative_to(store.remote):
                    raise OSError("Drive disconnected")
                copy(target, data)

            store._copy_verified = interrupted
            with self.assertRaises(OSError):
                store.commit("fold_0", {"score": 0.5})
            store._copy_verified = copy
            self.assertEqual(store.read("fold_0"), {"score": 0.5})
            fresh = ns["ArtifactStore"](root / "new", root / "drive", {"test": 1})
            self.assertEqual(fresh.read("fold_0"), {"score": 0.5})
            (fresh.remote / "fold_0/record.json").write_bytes(b"broken")
            self.assertEqual(fresh.read("fold_0"), {"score": 0.5})
            with self.assertRaises(ValueError):
                fresh.audit_remote()

    def test_lru_cache_evicts_oldest_and_hits_refresh_recency(self):
        ns = definitions("exp02")
        cache = ns["FeatureCache"](limit_bytes=2000)
        cache["a"] = np.zeros(100)
        cache["b"] = np.ones(100)
        _ = cache["a"]
        cache["c"] = np.ones(100)
        self.assertIn("a", cache)
        self.assertNotIn("b", cache)
        self.assertLessEqual(cache.bytes_used, 2000)

    def test_causal_matrix_is_cached_and_matches_original_without_label_dependency(
        self,
    ):
        original, ns = definitions("exp01"), definitions("exp02")
        frame = ns["validate_frame"](synthetic(sessions=1, duration=900)[15], 15)
        recipe = ns["base_recipe"]("logistic")
        recipe.update(normalization="causal", history=30, representation="history")
        expected = original["SensorTransform"](recipe)._matrix(frame.copy())
        pre = ns["SensorTransform"](recipe)
        first = pre._matrix(frame)
        second = pre._matrix(frame)
        self.assertIs(first, second)
        np.testing.assert_array_equal(first, expected)
        changed = frame.copy()
        changed.loc[20:, "rmssd_ms"] = 900
        changed["cognitive_load_label"] = "High"
        np.testing.assert_array_equal(first[:20], pre._matrix(changed)[:20])
        self.assertFalse(np.array_equal(first[20:], pre._matrix(changed)[20:]))

    def test_heartbeat_reports_and_stops_after_exception(self):
        ns = definitions("exp02")
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            self.assertRaisesRegex(RuntimeError, "stop"),
            ns["FitHeartbeat"]("SVM fold 1", interval=0.01) as heartbeat,
        ):
            time.sleep(0.04)
            raise RuntimeError("stop")
        self.assertIn("SVM fold 1", output.getvalue())
        self.assertIn("elapsed", output.getvalue())
        self.assertFalse(heartbeat.thread.is_alive())

    def test_tabular_predictions_match_original_and_bundle_contains_no_raw_metadata(
        self,
    ):
        old, ns = definitions("exp01"), definitions("exp02")
        frame = ns["validate_frame"](synthetic(sessions=1, duration=900)[15], 15)
        frame["note"] = "PRIVATE_DATA_SHOULD_NOT_BE_PICKLED"
        ids, scoring = np.arange(140), np.arange(140, len(frame))
        for family in ("logistic", "svm", "hist"):
            recipe = ns["base_recipe"](family)
            recipe.update(normalization="causal", representation="history", history=30)
            if family == "hist":
                recipe["params"]["max_iter"] = 5
            before = old["fit_model"](recipe, frame.copy(), ids)
            after = ns["fit_model"](recipe, frame.copy(), ids)
            expected = old["predict_model"](before, frame, scoring)
            actual = ns["predict_model"](after, frame, scoring)
            np.testing.assert_array_equal(expected["labels"], actual["labels"])
            np.testing.assert_allclose(
                expected["scores"], actual["scores"], rtol=1e-6, atol=1e-7
            )
            payload = ns["pack_model"](after)
            self.assertNotIn(b"PRIVATE_DATA_SHOULD_NOT_BE_PICKLED", payload)
            restored = ns["unpack_model"](payload)
            np.testing.assert_array_equal(
                actual["labels"],
                ns["predict_model"](restored, frame, scoring)["labels"],
            )
            self.assertAlmostEqual(
                after["training_accuracy"],
                ns["metrics"](
                    frame.iloc[ids]._y, ns["predict_model"](after, frame, ids)["labels"]
                )["accuracy"],
            )

    def test_svm_convergence_warning_is_recorded_and_still_visible(self):
        ns = definitions("exp02")
        frame = ns["validate_frame"](synthetic(sessions=1, duration=900)[15], 15)
        recipe = ns["base_recipe"]("svm")
        recipe["params"].update(max_iter=1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = ns["fit_model"](recipe, frame, np.arange(140))
        self.assertTrue(
            any(w.category.__name__ == "ConvergenceWarning" for w in caught)
        )
        self.assertEqual(model["convergence"][0]["fit_status"], 1)
        self.assertTrue(model["convergence"][0]["warnings"])
        self.assertTrue(model["convergence"][0]["n_iter"])
        self.assertEqual(ns["make_estimator"](ns["base_recipe"]("svm"), 3).max_iter, -1)

    def test_selection_test_summary_and_resume_are_complete(self):
        ns = definitions("exp02")
        views, _, _ = ns["prepare_data"](synthetic(), candidates=4)
        frame = views[15]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ns["ArtifactStore"](root / "local", root / "drive", {"test": 1})
            records = []
            for family in ("logistic", "hist"):
                recipe = ns["base_recipe"](family)
                if family == "hist":
                    recipe["params"]["max_iter"] = 5
                records.append(
                    ns["evaluate_recipe"](store, frame, 15, recipe, repeats=(0, 1))
                )
            lock = ns["select_models"](store, {15: frame}, {15: records}, "exp02")
            table = ns["evaluate_test"](store, {15: frame}, lock)
            self.assertEqual(int(table.primary.sum()), 1)
            self.assertIsNotNone(store.read("complete"))
            timing = pd.read_csv(store.remote / "runtime_summary.csv")
            self.assertEqual(len(timing[timing.checkpoint.str.startswith("fold_")]), 25)
            self.assertTrue(
                timing.loc[timing.checkpoint.str.startswith("fold_"), "write_seconds"]
                .notna()
                .all()
            )
            fresh = ns["ArtifactStore"](root / "fresh", root / "drive", {"test": 1})
            original_fit = ns["fit_model"]

            def forbidden(*args, **kwargs):
                raise AssertionError("Completed fitting must not repeat")

            ns["fit_model"] = forbidden
            try:
                self.assertEqual(
                    records[0],
                    ns["evaluate_recipe"](
                        fresh, frame, 15, records[0]["recipe"], repeats=(0, 1)
                    ),
                )
                self.assertEqual(
                    lock,
                    ns["select_models"](fresh, {15: frame}, {15: records}, "exp02"),
                )
                ns["evaluate_test"](fresh, {15: frame}, lock)
                resumed_timing = pd.read_csv(store.remote / "runtime_summary.csv")
                self.assertEqual(set(timing.checkpoint), set(resumed_timing.checkpoint))
            finally:
                ns["fit_model"] = original_fit

    def test_tuning_reuses_completed_folds_after_interruption(self):
        ns = definitions("exp02")
        views, _, _ = ns["prepare_data"](synthetic(), candidates=4)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = ns["ArtifactStore"](root / "local", root / "drive", {"test": 1})
            actual = ns["fit_model"]
            calls = []

            def interrupted(*args, **kwargs):
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError("runtime interruption")
                return actual(*args, **kwargs)

            ns["fit_model"] = interrupted
            with self.assertRaisesRegex(RuntimeError, "runtime interruption"):
                ns["optimize"](store, views[15], 15, "resume", ["logistic"], 1, "exp02")
            calls.clear()

            def counted(*args, **kwargs):
                calls.append(1)
                return actual(*args, **kwargs)

            ns["fit_model"] = counted
            result = ns["optimize"](
                store, views[15], 15, "resume", ["logistic"], 1, "exp02"
            )
            self.assertEqual(len(result), 1)
            self.assertEqual(len(calls), 4)
            calls.clear()
            self.assertEqual(
                result,
                ns["optimize"](
                    store, views[15], 15, "resume", ["logistic"], 1, "exp02"
                ),
            )
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
