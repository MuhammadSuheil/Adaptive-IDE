"""Behavior checks for the standalone preparation notebook; no model training."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "models/batch-05/exp01/EyeHeartCognitiveLoad.ipynb"


def definitions():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {"__name__": "batch05_test"}
    for cell in notebook["cells"]:
        if "definitions" in cell.get("metadata", {}).get("tags", []):
            # Execute only definitions from the trusted repository notebook.
            exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)  # noqa: S102  # nosec B102
    return namespace


def fixture(namespace, seconds=3600):
    frames = {}
    for window in (15, 25, 45):
        records = []
        for session in range(6):
            for end in range(60, seconds, 5):
                label = (end // 600 + session) % 3
                finish = pd.Timestamp("2026-11-01T00:00:00Z") + pd.Timedelta(
                    seconds=end
                )
                record = {
                    "window_id": f"s{session}|{window}|{end}",
                    "session_id": f"s{session}",
                    "participant_id": f"p{session}",
                    "window_start": (finish - pd.Timedelta(seconds=window)).isoformat(),
                    "window_end": finish.isoformat(),
                    "window_seconds": window,
                    "study_phase": "active_task" if end % 100 < 50 else "rating",
                    "cognitive_load_label": ("Low", "Medium", "High")[label],
                    "quality_tier": "acceptable",
                }
                for i, column in enumerate(namespace["CORE"]):
                    record[column] = 0.1 + i * 0.01 + end / 1e6 + session / 1e5
                for column in namespace["HEAD"]:
                    record[column] = 1.0
                records.append(record)
        frames[window] = pd.DataFrame(records)
    return frames


class ArtifactTest(unittest.TestCase):
    def test_notebook_is_available_for_the_standalone_workflow(self):
        self.assertTrue(NOTEBOOK.is_file(), "Standalone Stage 1 notebook is missing")


@unittest.skipUnless(NOTEBOOK.is_file(), "Preparation notebook not implemented yet")
class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = definitions()
        cls.frames = fixture(cls.ns)

    def test_notebook_exists_before_loading_definitions(self):
        self.assertTrue(NOTEBOOK.is_file(), "Standalone Stage 1 notebook is missing")

    def test_invalid_sensor_time_duration_and_duplicate_rejected(self):
        for key, value in (
            ("rmssd_ms", float("inf")),
            ("ear_left_mean", -0.1),
            ("window_start", "not-a-date"),
            ("window_start", "2026-11-01T00:00:00"),
            ("window_seconds", 99),
            ("cognitive_load_label", "unknown"),
        ):
            with self.subTest(key=key, value=value):
                frame = self.frames[25].copy()
                frame.loc[0, key] = value
                with self.assertRaises(ValueError):
                    self.ns["validate_frame"](frame, 25)
        duplicate = pd.concat([self.frames[25], self.frames[25].iloc[[0]]])
        with self.assertRaises(ValueError):
            self.ns["validate_frame"](duplicate, 25)
        with self.assertRaises(ValueError):
            self.ns["validate_frame"](self.frames[25].drop(columns="sdnn_ms"), 25)

    def test_conflicting_endpoint_labels_rejected(self):
        frames = {w: f.copy() for w, f in self.frames.items()}
        frames[15].loc[0, "cognitive_load_label"] = "High"
        with self.assertRaisesRegex(ValueError, "antarwindow"):
            self.ns["build_union"](frames)

    def test_guard_preserves_boundary_and_separates_sessions(self):
        frame = pd.DataFrame(
            {
                "session_id": ["a", "a", "a", "b"],
                "_end": np.array([0, 40, 45, 0], dtype=np.int64) * 10**9,
            }
        )
        kept = self.ns["purge"](frame, [1, 2, 3], [0])
        self.assertEqual(kept.tolist(), [2, 3])

    def test_split_preserves_test_and_shared_endpoints_without_overlap(self):
        views, union = self.ns["build_union"](self.frames)
        test = union.loc[union._block.eq(union._block.iloc[0])]
        reference = {
            "rows": [
                {"session_id": r.session_id, "end_ns": int(r._end), "_split": "test"}
                for _, r in test.iterrows()
            ]
        }
        result = self.ns["development_split"](views, union, reference, range(42, 50))
        assigned = result["union"]
        self.assertEqual(
            set(assigned.loc[assigned._split.eq("test"), "_key"]), set(test._key)
        )
        for held in ("validation", "test"):
            held_ids = np.flatnonzero(assigned._split.eq(held))
            fitting = np.flatnonzero(assigned._split.eq("train"))
            self.assertEqual(
                self.ns["purge"](assigned, fitting, held_ids).tolist(), fitting.tolist()
            )
        lookup = assigned.set_index("_key")._split
        for frame in result["views"].values():
            self.assertTrue(frame._split.eq(frame._key.map(lookup)).all())
        cv = self.ns["development_cv"](result["views"], assigned, range(42, 46))
        for fold in cv["folds"]:
            self.assertTrue(set(fold["fit"]).isdisjoint(fold["validation"]))
            self.assertTrue(assigned.iloc[fold["fit"]]._split.eq("train").all())

    def test_missing_class_cannot_be_called_ready(self):
        views, union = self.ns["build_union"](
            {
                w: f[f.cognitive_load_label.ne("High")].copy()
                for w, f in self.frames.items()
            }
        )
        with self.assertRaisesRegex(ValueError, "kelas"):
            self.ns["development_split"](views, union, {"rows": []}, range(42, 44))

    def test_monitor_has_minimum_support_balanced_classes_and_fixed_ids(self):
        frame = self.ns["validate_frame"](self.frames[25], 25)
        frame["_block"] = np.arange(len(frame)) // 48
        ids = np.arange(len(frame))
        monitor = self.ns["choose_monitor"](frame, ids)
        self.assertTrue(monitor["ready"])
        fit, held = np.array(monitor["fit"]), np.array(monitor["monitor"])
        counts = [np.bincount(frame.iloc[v]._y, minlength=3) for v in (fit, held)]
        self.assertTrue(all((v >= 10).all() for v in counts))
        self.assertLessEqual(
            float(
                np.max(np.abs(counts[0] / sum(counts[0]) - counts[1] / sum(counts[1])))
            ),
            0.1,
        )
        self.assertEqual(monitor, self.ns["choose_monitor"](frame, ids))
        self.assertEqual(self.ns["purge"](frame, fit, held).tolist(), fit.tolist())

    def test_split_selection_cannot_read_sensor_values(self):
        views, union = self.ns["build_union"](self.frames)
        first = self.ns["development_split"](views, union, {"rows": []}, range(42, 46))
        changed = {w: f.copy() for w, f in self.frames.items()}
        for frame in changed.values():
            frame["rmssd_ms"] *= 1000
            frame["iris_size_delta_mean"] *= -1234
        views, union = self.ns["build_union"](changed)
        second = self.ns["development_split"](views, union, {"rows": []}, range(42, 46))
        self.assertEqual(first["seed"], second["seed"])
        self.assertEqual(
            first["union"]._split.tolist(), second["union"]._split.tolist()
        )

    def test_missing_frozen_endpoint_cannot_be_replaced(self):
        views, union = self.ns["build_union"](self.frames)
        with self.assertRaisesRegex(ValueError, "Test referensi hilang"):
            self.ns["development_split"](
                views,
                union,
                {"rows": [{"session_id": "missing", "end_ns": 123, "_split": "test"}]},
            )

    def test_impossible_monitor_is_reported_not_relaxed(self):
        frame = self.ns["validate_frame"](self.frames[25].iloc[:20].copy(), 25)
        frame["_block"] = 0
        monitor = self.ns["choose_monitor"](frame, np.arange(len(frame)))
        self.assertFalse(monitor["ready"])
        self.assertEqual(monitor["fit"], [])
        self.assertIn("reason", monitor)

    def test_preprocessing_fits_only_fitting_and_preserves_signed_values(self):
        frame = pd.DataFrame(
            {
                "session_id": ["a", "a", "b", "new"],
                "rmssd_ms": [10.0, 30.0, 50.0, 999.0],
            }
        )
        for mode in ("global", "session_centered", "session_robust"):
            first = self.ns["SensorPreprocessor"](["rmssd_ms"], mode).fit(
                frame, [0, 1, 2]
            )
            mutated = frame.copy()
            mutated.loc[3, "rmssd_ms"] = -1e9
            second = self.ns["SensorPreprocessor"](["rmssd_ms"], mode).fit(
                mutated, [0, 1, 2]
            )
            self.assertEqual(first.state(), second.state())
            self.assertLess(first.transform(frame.iloc[[0]])[0, 0], 0)
            self.assertTrue(np.isfinite(first.transform(frame.iloc[[3]])).all())
        # Session a median 20; global median 30. Unseen session uses 30.
        transform = self.ns["SensorPreprocessor"](["rmssd_ms"], "session_centered").fit(
            frame, [0, 1, 2]
        )
        self.assertAlmostEqual(transform.transform(frame.iloc[[3]])[0, 0], 969 / 20)
        with self.assertRaises(ValueError):
            transform.fit(frame, [0, 0])

    def test_selection_rejects_collapsed_and_nonfinite_candidates(self):
        candidates = [
            {
                "name": "collapsed",
                "finite_predictions": True,
                "macro_f1": 0.8,
                "accuracy": 0.9,
                "recall": [0.9, 0.8, 0],
                "fold_std": 0.1,
                "complexity": 1,
            },
            {
                "name": "valid",
                "finite_predictions": True,
                "macro_f1": 0.5,
                "accuracy": 0.6,
                "recall": [0.5, 0.4, 0.3],
                "fold_std": 0.1,
                "complexity": 1,
            },
            {
                "name": "nan",
                "finite_predictions": True,
                "macro_f1": float("nan"),
                "accuracy": 0.99,
                "recall": [0.9, 0.9, 0.9],
                "fold_std": 0.1,
                "complexity": 1,
            },
        ]
        self.assertEqual(self.ns["select_candidate"](candidates)["name"], "valid")
        self.assertIsNone(self.ns["select_candidate"]([candidates[0], candidates[2]]))
        self.assertIsNone(
            self.ns["select_candidate"]([dict(candidates[1], finite_predictions=False)])
        )
        without_finite_check = {
            k: v for k, v in candidates[1].items() if k != "finite_predictions"
        }
        self.assertIsNone(self.ns["select_candidate"]([without_finite_check]))

    def test_report_writer_roundtrip_and_no_source_mutation(self):
        original = copy.deepcopy(self.frames[25])
        self.ns["validate_frame"](self.frames[25], 25)
        pd.testing.assert_frame_equal(original, self.frames[25])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            self.ns["save_json"](path / "readiness.json", {"ready": True})
            self.assertEqual(
                json.loads((path / "readiness.json").read_text()), {"ready": True}
            )


if __name__ == "__main__":
    unittest.main()
