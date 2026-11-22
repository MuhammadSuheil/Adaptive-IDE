"""Closing audit contracts: boundaries, event linkage, and diagnostic scope."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "models/batch-05/outputs/closing_audit/audit.py"
)


def load_audit():
    spec = importlib.util.spec_from_file_location("closing_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ClosingAvailability(unittest.TestCase):
    def test_audit_is_available(self):
        self.assertTrue(SCRIPT.is_file(), "Closing audit has not been implemented")


@unittest.skipUnless(SCRIPT.is_file(), "Closing audit not implemented")
class ClosingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = load_audit()

    def test_constant_pair_is_not_affine_evidence(self):
        self.assertFalse(self.audit.affine_pair(np.ones(25), np.ones(25))["eligible"])
        result = self.audit.affine_pair(np.arange(25), np.arange(25) - 7)
        self.assertTrue(result["redundant"])
        self.assertAlmostEqual(result["r_squared"], 1)
        self.assertFalse(
            self.audit.affine_pair(np.arange(19), np.arange(19))["eligible"]
        )

    def test_attempt_duration_cannot_cross_puzzle_or_set(self):
        records = []
        for kind, when, puzzle, attempt in [
            ("set_started", 0, "", ""),
            ("puzzle_started", 1000, "a", ""),
            ("puzzle_started", 2000, "b", ""),
            ("attempt_submitted", 3000, "a", "x"),
            ("puzzle_started", 4000, "a", ""),
            ("attempt_submitted", 6500, "a", "y"),
            ("attempt_submitted", 7000, "a", "z"),
        ]:
            records.append(
                {
                    "event_id": str(when),
                    "session_id": "s",
                    "event_type": kind,
                    "event_time_ms": when,
                    "puzzle_set": "easy",
                    "difficulty_level": "easy",
                    "puzzle_id": puzzle,
                    "attempt_id": attempt,
                    "outcome": "correct",
                }
            )
        attempts, visits = self.audit.link_attempts(pd.DataFrame(records))
        self.assertTrue(pd.isna(attempts.iloc[0].active_seconds))
        self.assertEqual(attempts.iloc[1].active_seconds, 2.5)
        self.assertTrue(pd.isna(attempts.iloc[2].active_seconds))
        self.assertEqual(attempts.attempt_number.tolist(), [1, 2, 3])
        self.assertEqual(visits.iloc[0].closed_by, "puzzle_started")

    def test_duplicate_attempt_id_is_ambiguous_not_a_second_measurement(self):
        events = pd.DataFrame(
            [
                {"event_id": "0", "event_type": "set_started", "event_time_ms": 0},
                {
                    "event_id": "1",
                    "event_type": "puzzle_started",
                    "event_time_ms": 1000,
                },
                {
                    "event_id": "2",
                    "event_type": "attempt_submitted",
                    "event_time_ms": 2000,
                },
                {
                    "event_id": "3",
                    "event_type": "attempt_submitted",
                    "event_time_ms": 3000,
                },
            ]
        ).assign(
            session_id="s",
            puzzle_set="easy",
            difficulty_level="easy",
            puzzle_id="a",
            attempt_id="same",
            outcome="incorrect",
        )
        attempts, _ = self.audit.link_attempts(events)
        self.assertTrue(attempts.active_seconds.isna().all())
        self.assertTrue(attempts.link_issue.str.contains("duplicate_attempt_id").all())

    def test_eye_boundary_counter_and_sample_sd(self):
        eye = pd.DataFrame(
            {
                "timestamp_ms": [0, 1000, 2000],
                "total_blinks": [5, 6, 8],
                "face_detected": [True] * 3,
                "EAR_Left": [0.1, 0.2, 0.4],
                "EAR_Right": [0.1, 0.2, 0.4],
                "iris_valid": [True] * 3,
                "is_blinking": [False] * 3,
                "iris_size": [0.1, 0.2, 0.4],
                "iris_size_mode": ["eye_width_ratio"] * 3,
                "iris_baseline_ready": [True] * 3,
                "iris_size_delta": [0, 0.1, 0.3],
            }
        )
        result, _ = self.audit.eye_features(eye, 0, 2000)
        self.assertEqual(result["eye_frames_count"], 2)
        self.assertEqual(result["blink_count"], 3)
        self.assertEqual(result["blink_rate_per_minute"], 90)
        self.assertAlmostEqual(result["ear_left_std"], 0.1414213562373095)

    def test_rr_uses_availability_and_never_pairs_across_segment(self):
        rows = [
            {
                "raw_id": i + 1,
                "segment_id": segment,
                "beat_elapsed_seconds": beat,
                "elapsed_seconds": beat,
                "available_seconds": available,
                "rr_processed_ms": rr,
            }
            for i, (beat, available, rr, segment) in enumerate(
                [
                    (0, 0, 800, 1),
                    (1, 1, 1000, 1),
                    (2, 2, 1100, 1),
                    (3, 3, 900, 2),
                    (4, 5, 1300, 2),
                ]
            )
        ]
        result, selected = self.audit.rr_features(rows, 0, 4)
        self.assertEqual(result["hrv_rr_count"], 3)
        self.assertEqual(result["hrv_rr_pairs"], 1)
        self.assertEqual(result["rmssd_ms"], 100)
        self.assertEqual(result["sdnn_ms"], 100)
        self.assertEqual(len(selected), 3)

    def test_case_selection_uses_existing_confidence_and_distinct_blocks(self):
        rows = []
        for label in range(3):
            for j, (prediction, block, confidence) in enumerate(
                [(label, 1, 0.9), ((label + 1) % 3, 1, 0.99), ((label + 1) % 3, 2, 0.8)]
            ):
                rows.append(
                    {
                        "window_id": f"{label}-{j}",
                        "session_id": f"s{label}",
                        "target": label,
                        "prediction": prediction,
                        "block": block,
                        "probability_Low": confidence,
                        "probability_Medium": 0,
                        "probability_High": 0,
                    }
                )
        cases = self.audit.select_cases(pd.DataFrame(rows))
        self.assertEqual(len(cases), 6)
        self.assertEqual(
            cases.window_id.tolist(), ["0-0", "0-2", "1-0", "1-2", "2-0", "2-2"]
        )

    def test_snapshot_detects_changed_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.txt"
            path.write_text("before")
            before = self.audit.snapshot([path])
            path.write_text("after")
            self.assertNotEqual(before, self.audit.snapshot([path]))

    def test_completed_audit_cannot_silently_repeat_ten_fits(self):
        self.assertTrue(callable(getattr(self.audit, "require_new_audit", None)))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.audit.require_new_audit(output)
            (output / "audit_result.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                self.audit.require_new_audit(output)

    def test_phase_target_check_rejects_wrong_duration_and_unknown_phase(self):
        self.assertTrue(callable(getattr(self.audit, "phase_target_check", None)))
        check = self.audit.phase_target_check
        self.assertTrue(
            check(
                [
                    {
                        "study_phase": "active_task",
                        "difficulty_level": "easy",
                        "duration_seconds": 10,
                    },
                    {
                        "study_phase": "rating",
                        "difficulty_level": "hard",
                        "duration_seconds": 15,
                    },
                ],
                "Low",
                25,
            )
        )
        self.assertFalse(
            check(
                [
                    {
                        "study_phase": "active_task",
                        "difficulty_level": "hard",
                        "duration_seconds": 25,
                    }
                ],
                "Low",
                25,
            )
        )
        self.assertFalse(
            check([{"study_phase": "baseline", "duration_seconds": 25}], "Low", 25)
        )
        self.assertFalse(
            check([{"study_phase": "rest", "duration_seconds": 24}], "Low", 25)
        )


if __name__ == "__main__":
    unittest.main()
