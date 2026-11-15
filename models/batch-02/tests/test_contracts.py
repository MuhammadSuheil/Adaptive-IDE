"""Release and stream contracts; no classifier training in this suite."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))


class PackageBoundaryTest(unittest.TestCase):
    def test_batch02_has_an_independent_importable_runtime(self):
        self.assertIsNotNone(importlib.util.find_spec("batch02"))


class ReleaseTest(unittest.TestCase):
    def setUp(self):
        from batch02.contracts import require_approved_release

        self.check = require_approved_release

    def test_pending_release_cannot_start_training(self):
        with self.assertRaisesRegex(ValueError, "approved"):
            self.check({"gates": {"additional_data": "pending"}})

    def test_approval_requires_all_gates_and_signer(self):
        with self.assertRaises(ValueError):
            self.check(
                {
                    "gates": dict.fromkeys(
                        ["integrity", "additional_data", "proportions", "frozen"],
                        "approved",
                    )
                }
            )


class SensorStreamTest(unittest.TestCase):
    def test_sequence_context_survives_interleaved_session_rows(self):
        from batch02.streams import sequence_indices

        frame = pd.DataFrame(
            {
                "session_id": ["a", "b", "a"],
                "window_end": pd.to_datetime(
                    [
                        "2026-01-01T00:00:00Z",
                        "2026-01-01T00:00:00Z",
                        "2026-01-01T00:00:05Z",
                    ]
                ),
            }
        )
        self.assertEqual(sequence_indices(frame, frame.iloc[[2]]), [[0, 2]])

    def test_irregular_non_grid_endpoint_is_rejected(self):
        from batch02.streams import checked_stream

        frame = self.frame()
        frame.loc[1, "window_end"] += pd.Timedelta(seconds=1)
        with self.assertRaises(ValueError):
            checked_stream(frame)

    def frame(self):
        return pd.DataFrame(
            {
                "session_id": ["s"] * 5,
                "window_end": pd.date_range(
                    "2026-01-01", periods=5, freq="5s", tz="UTC"
                ),
                "rmssd_ms": [1.0, 3.0, 5.0, np.nan, 9.0],
            }
        )

    def test_history_uses_unlabeled_endpoints_before_supervised_mask(self):
        from batch02.streams import causal_history

        result = causal_history(self.frame(), ["rmssd_ms"], 15)
        self.assertEqual(result.loc[2, "rmssd_ms__history_mean_15s"], 3.0)
        self.assertEqual(result.loc[2, "rmssd_ms__delta"], 2.0)
        self.assertTrue(np.isnan(result.loc[4, "rmssd_ms__delta"]))

    def test_appending_future_cannot_change_prefix(self):
        from batch02.streams import causal_history

        frame = self.frame()
        a = causal_history(frame.iloc[:3], ["rmssd_ms"], 15)
        b = causal_history(frame, ["rmssd_ms"], 15).iloc[:3]
        pd.testing.assert_frame_equal(a, b)

    def test_annotations_cannot_reset_history(self):
        from batch02.streams import causal_history

        frame = self.frame()
        frame["puzzle_id"] = ["a", "b", "c", "d", "e"]
        first = causal_history(frame, ["rmssd_ms"], 15)
        frame["puzzle_id"] = "other"
        second = causal_history(frame, ["rmssd_ms"], 15)
        cols = [c for c in first if "__" in c]
        pd.testing.assert_frame_equal(first[cols], second[cols])

    def test_gap_resets_and_duplicate_time_is_rejected(self):
        from batch02.streams import causal_history

        frame = self.frame().iloc[:3].copy()
        frame.loc[2, "window_end"] += pd.Timedelta(seconds=20)
        result = causal_history(frame, ["rmssd_ms"], 30)
        self.assertEqual(result.loc[2, "rmssd_ms__history_mean_30s"], 5.0)
        self.assertTrue(np.isnan(result.loc[2, "rmssd_ms__delta"]))
        frame.loc[2, "window_end"] = frame.loc[1, "window_end"]
        with self.assertRaises(ValueError):
            causal_history(frame, ["rmssd_ms"], 30)


class MetricsTest(unittest.TestCase):
    def test_absent_true_class_is_not_reported_as_a_measured_zero(self):
        from batch02.metrics import classification_metrics

        result = classification_metrics([0, 0, 2], [0, 2, 2])
        self.assertIsNone(result["classes"]["Medium"]["recall"])
        self.assertEqual(result["classes"]["Medium"]["support"], 0)
        self.assertAlmostEqual(result["macro_f1"], 4 / 9)

    def test_invalid_probabilities_are_rejected(self):
        from batch02.metrics import classification_metrics

        with self.assertRaises(ValueError):
            classification_metrics([0], [0], np.array([[0.2, 0.2, 0.2]]))


class ArtifactTest(unittest.TestCase):
    def test_resume_rejects_changed_inputs(self):
        from batch02.artifacts import open_run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = open_run(root, {"release": "one"}, None)
            with self.assertRaisesRegex(ValueError, "hash|identitas"):
                open_run(root, {"release": "two"}, run)

    def test_json_does_not_publish_nonstandard_nan(self):
        from batch02.artifacts import write_json

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_json(path, {"missing": float("nan")})
            self.assertIsNone(json.loads(path.read_text())["missing"])


if __name__ == "__main__":
    unittest.main()
