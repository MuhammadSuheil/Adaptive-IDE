"""Regression checks for the read-only result review; no training."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class ReviewTests(unittest.TestCase):
    def review(self):
        path = Path(__file__).with_name("review_results.py")
        self.assertTrue(path.exists(), "Result audit has not been implemented")
        spec = importlib.util.spec_from_file_location("review_results", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_confusion_matrix_keeps_absent_classes_and_real_zero_scores(self):
        module = self.review()
        result = module.metrics([0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 0])
        self.assertEqual(result["confusion_matrix"], [[1, 1, 0], [0, 1, 1], [1, 0, 1]])
        self.assertAlmostEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["macro_f1"], 0.5)
        missing = module.metrics([0, 0], [0, 0])
        self.assertAlmostEqual(missing["macro_f1"], 1 / 3)
        self.assertIsNone(missing["classes"]["High"]["f1"])

    def test_split_downloads_merge_identical_paths_but_reject_conflicts(self):
        module = self.review()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for part in ("outputs-part-001", "outputs-part-002"):
                path = root / part / "outputs/exp01/run/status.json"
                path.parent.mkdir(parents=True)
                path.write_text('{"status":"complete"}')
            index, duplicates = module.logical_files(root)
            self.assertEqual(len(index), 1)
            self.assertEqual(duplicates, 1)
            (root / "outputs-part-002/outputs/exp01/run/status.json").write_text(
                "changed"
            )
            with self.assertRaises(ValueError):
                module.logical_files(root)

    def test_oof_checks_actual_ids_labels_and_duplicate_rows(self):
        module = self.review()
        expected = {"a": 0, "b": 1}
        group = pd.DataFrame(
            {"window_id": ["a", "b"], "y_true": [0, 1], "y_pred": [0, 1]}
        )
        module.validate_oof(group, expected)
        with self.assertRaises(ValueError):
            module.validate_oof(pd.concat([group, group.iloc[[0]]]), expected)
        with self.assertRaises(ValueError):
            module.validate_oof(group.assign(y_true=[1, 1]), expected)


if __name__ == "__main__":
    unittest.main()
