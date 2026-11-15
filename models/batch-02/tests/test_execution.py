"""Runner and model adapters use doubles instead of expensive training."""

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))


class FixedClassifier:
    def fit(self, x, y, sample_weight=None):
        self.classes_ = np.array([0, 1, 2])
        return self

    def predict_proba(self, x):
        return np.tile([0.6, 0.25, 0.15], (len(x), 1))


class FittingBoundaryTest(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(
            {
                "y": [0, 1, 2, 0, 1, 2],
                "person_id": ["a", "b", "c", "d", "e", "f"],
                "session_id": ["a", "b", "c", "d", "e", "f"],
                "rmssd_ms": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "sdnn_ms": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            }
        )

    def test_fit_predict_does_not_modify_input_or_learn_from_validation(self):
        from batch02.fitting import fit_tabular
        from batch02.registry import Recipe

        frame = self.frame()
        source = frame[["rmssd_ms", "sdnn_ms"]]
        fitted = fit_tabular(
            Recipe(features="F5"),
            source,
            frame,
            factory=lambda recipe, seed, threads, classes: FixedClassifier(),
        )
        p, kind = fitted.predict(source * 100)
        self.assertEqual(kind, "probability")
        np.testing.assert_allclose(p[0], [0.6, 0.25, 0.15])
        self.assertAlmostEqual(fitted.preprocessing.medians_["rmssd_ms"], 3.5)
        self.assertEqual(frame.rmssd_ms.iloc[0], 1.0)

    def test_native_missing_is_not_silently_given_to_logistic(self):
        from batch02.fitting import fit_tabular
        from batch02.registry import Recipe

        frame = self.frame()
        with self.assertRaises(ValueError):
            fit_tabular(
                Recipe(missing="native"),
                frame[["rmssd_ms", "sdnn_ms"]],
                frame,
                factory=lambda *args: FixedClassifier(),
            )

    def test_svm_scores_are_never_called_probabilities(self):
        from batch02.fitting import prediction_values

        class DecisionClassifier:
            classes_ = np.array([0, 1, 2])

            def decision_function(self, x):
                return np.tile([1.0, -1.0, 0.0], (len(x), 1))

        values, kind = prediction_values(DecisionClassifier(), np.ones((2, 1)))
        self.assertEqual(kind, "decision")
        self.assertEqual(values[0].tolist(), [1.0, -1.0, 0.0])


class SearchTest(unittest.TestCase):
    def test_history_cache_cannot_reuse_old_sensor_values(self):
        from batch02.features import FeatureStore
        from batch02.registry import Recipe

        store = FeatureStore(["rmssd_ms"])
        store.loaded[25] = pd.DataFrame(
            {
                "session_id": ["s", "s"],
                "window_end": pd.date_range(
                    "2026-01-01", periods=2, freq="5s", tz="UTC"
                ),
                "rmssd_ms": [1.0, 3.0],
            }
        )
        rows = store.loaded[25].copy()
        first = store.matrix(Recipe(features="F5", history=15), rows, 25)
        store.loaded[25].loc[0, "rmssd_ms"] = 9.0
        second = store.matrix(Recipe(features="F5", history=15), rows, 25)
        self.assertEqual(first.loc[1, "rmssd_ms__history_mean_15s"], 2.0)
        self.assertEqual(second.loc[1, "rmssd_ms__history_mean_15s"], 6.0)

    def test_tie_prefers_simpler_history_without_looking_at_test(self):
        from batch02.registry import Recipe
        from batch02.search import select_record

        plain = Recipe()
        complex_recipe = replace(plain, history=30)
        records = [
            {"recipe": complex_recipe, "score": 0.503, "n_features": 20},
            {"recipe": plain, "score": 0.5, "n_features": 4},
        ]
        self.assertEqual(select_record(records)["recipe"], plain)

    def test_cache_identity_changes_with_sensor_values(self):
        from batch02.features import frame_hash

        frame = pd.DataFrame({"a": [1.0, 2.0]})
        changed = frame.copy()
        changed.loc[0, "a"] = 99.0
        self.assertNotEqual(frame_hash(frame), frame_hash(changed))


class NotebookTest(unittest.TestCase):
    def test_setup_cells_run_locally_without_installation_or_training(self):
        import contextlib
        import io
        import json
        from unittest.mock import patch

        root = Path(__file__).resolve().parents[1]
        with patch(
            "subprocess.run",
            side_effect=AssertionError(
                "Setup lokal tidak boleh memasang paket saat semua versi sesuai"
            ),
        ):
            for path in root.glob("exp*/EyeHeartCognitiveLoad.ipynb"):
                namespace = {}
                nb = json.loads(path.read_text(encoding="utf-8"))
                for cell in nb["cells"]:
                    if cell["cell_type"] == "code" and set(
                        cell["metadata"].get("tags", [])
                    ).intersection({"setup", "environment", "configuration"}):
                        with contextlib.redirect_stdout(io.StringIO()):
                            exec(  # noqa: S102 - only trusted, tagged local setup cells
                                compile("".join(cell["source"]), str(path), "exec"),
                                namespace,
                            )
                self.assertEqual(namespace["environment_status"]["status"], "ready")

    def test_notebooks_are_empty_outputs_with_valid_python_cells(self):
        import ast
        import json

        root = Path(__file__).resolve().parents[1]
        notebooks = list(root.glob("exp*/EyeHeartCognitiveLoad.ipynb"))
        self.assertEqual(len(notebooks), 15)
        for path in notebooks:
            nb = json.loads(path.read_text(encoding="utf-8"))
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    self.assertEqual(cell["outputs"], [])
                    self.assertIsNone(cell["execution_count"])
                    ast.parse("".join(cell["source"]), filename=str(path))


if __name__ == "__main__":
    unittest.main()
