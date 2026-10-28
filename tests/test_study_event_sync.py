import csv
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from fuse_and_analyze import (
    annotate_sensor_csv,
    context_at,
    fuse_session,
    interval_segments,
    load_study_events,
    summarize_segments,
)


def event(timestamp_ms, event_type, puzzle_set="", difficulty="", puzzle_id=""):
    return {
        "event_time_ms": timestamp_ms,
        "event_type": event_type,
        "puzzle_set": puzzle_set,
        "difficulty_level": difficulty,
        "puzzle_id": puzzle_id,
        "puzzle_title": puzzle_id,
        "outcome": "",
    }


class StudyEventSyncTests(unittest.TestCase):
    def setUp(self):
        self.events = [
            event(0, "session_linked"),
            event(1000, "set_started", "easy-1", "easy"),
            event(1100, "puzzle_started", "easy-1", "easy", "q1"),
            event(2000, "puzzle_started", "easy-1", "easy", "q2"),
            event(3000, "set_ended", "easy-1", "easy"),
            event(5000, "set_started", "hard", "hard"),
            event(5100, "puzzle_started", "hard", "hard", "q3"),
            event(8000, "set_ended", "hard", "hard"),
            event(8500, "study_ended"),
        ]

    def test_phase_and_question_context_follow_variable_order(self):
        self.assertEqual(context_at(500, self.events, 100)["study_phase"], "pre_task")
        easy = context_at(2500, self.events, 100)
        self.assertEqual(easy["study_phase"], "active_task")
        self.assertEqual(easy["difficulty_level"], "easy")
        self.assertEqual(easy["puzzle_id"], "q2")
        self.assertEqual(context_at(4000, self.events, 100)["study_phase"], "rest")
        hard = context_at(6000, self.events, 100)
        self.assertEqual(hard["difficulty_level"], "hard")
        self.assertEqual(hard["puzzle_id"], "q3")
        self.assertEqual(context_at(8100, self.events, 100)["study_phase"], "post_task")

    def test_hrv_window_reports_phase_proportions_at_boundaries(self):
        segments = interval_segments(2500, 5500, self.events, 100)
        labels = summarize_segments(segments)
        self.assertEqual(labels["study_phase"], "mixed")
        self.assertEqual(labels["difficulty_level"], "mixed")
        self.assertAlmostEqual(sum(item["duration_seconds"] for item in segments), 3.0)
        self.assertEqual([item["study_phase"] for item in segments],
                         ["active_task", "rest", "feedback", "active_task"])
        self.assertEqual([item["duration_seconds"] for item in segments], [0.5, 2.0, 0.1, 0.4])
        self.assertEqual(json.loads(labels["window_phase_mix"])[1]["duration_seconds"], 2.0)

    def test_unended_last_set_is_interrupted_not_assumed_complete(self):
        events = self.events[:-1]
        self.assertEqual(context_at(4000, events, 100)["study_phase"], "rest")
        self.assertEqual(context_at(9000, events, 100)["study_phase"], "interrupted")

    def test_manual_end_excludes_rest_and_mixed_windows_before_next_set(self):
        ended = event(2000, "set_ended", "easy-1", "easy", "q1")
        ended["outcome"] = "stopped_manually"
        events = [event(1000, "set_started", "easy-1", "easy"),
                  event(1100, "puzzle_started", "easy-1", "easy", "q1"), ended,
                  event(4000, "set_started", "medium", "medium"),
                  event(4100, "puzzle_started", "medium", "medium", "q2")]
        self.assertEqual(context_at(1900, events)["study_phase"], "active_task")
        self.assertEqual(context_at(2000, events)["study_phase"], "rest")
        self.assertEqual(context_at(3500, events)["study_phase"], "rest")
        self.assertEqual(context_at(4200, events)["puzzle_set"], "medium")
        labels = summarize_segments(interval_segments(1500, 4500, events))
        self.assertEqual(labels["study_phase"], "mixed")
        parts = json.loads(labels["window_phase_mix"])
        self.assertEqual([part["duration_seconds"] for part in parts], [.5, 2.0, .1, .4])

    def test_submission_feedback_and_rating_are_not_active_task(self):
        events = [
            event(1000, "set_started", "easy-1", "easy"),
            event(1100, "puzzle_started", "easy-1", "easy", "q1"),
            event(1500, "attempt_submitted", "easy-1", "easy", "q1"),
            event(1600, "rating_started", "easy-1", "easy", "q1"),
            event(1900, "rating_ended", "easy-1", "easy", "q1"),
            event(2200, "puzzle_started", "easy-1", "easy", "q1"),
        ]
        self.assertEqual(context_at(1400, events)["study_phase"], "active_task")
        self.assertEqual(context_at(1550, events)["study_phase"], "feedback")
        self.assertEqual(context_at(1700, events)["study_phase"], "rating")
        self.assertEqual(context_at(2000, events)["study_phase"], "feedback")
        self.assertEqual(context_at(2300, events)["study_phase"], "active_task")

    def test_sensor_rows_include_question_context(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "eye_tracking.csv"
            path.write_text("timestamp_ms,gaze_x\n2500,123\n", encoding="utf-8")
            annotate_sensor_csv(path, "session_1", self.events, 100, "eye")
            with path.open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual((row["difficulty_level"], row["puzzle_set"], row["puzzle_id"]),
                             ("easy", "easy-1", "q2"))

    def test_final_rating_after_set_end_is_separate(self):
        events = [event(1000, "set_started", "hard", "hard"),
                  event(1100, "puzzle_started", "hard", "hard", "q1"),
                  event(1500, "attempt_submitted", "hard", "hard", "q1"),
                  event(1600, "set_ended", "hard", "hard"),
                  event(1700, "rating_started", "hard", "hard", "q1"),
                  event(2000, "rating_ended", "hard", "hard", "q1")]
        self.assertEqual(context_at(1800, events)["study_phase"], "rating")
        self.assertEqual(context_at(2100, events)["study_phase"], "interrupted")

    def test_event_loader_rejects_mismatched_session_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            session_dir = Path(temp) / "session_1"
            session_dir.mkdir()
            with (session_dir / "task_events.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["session_id", "event_time_ms", "event_type"])
                writer.writeheader()
                writer.writerow({"session_id": "another_session", "event_time_ms": 1000, "event_type": "set_started"})
            loaded, status = load_study_events(session_dir)
            self.assertEqual(loaded, [])
            self.assertEqual(status, "unlabeled")

    def test_sensor_csv_keeps_measurements_and_gets_phase_columns(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "eye_tracking.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp_ms", "gaze_x"])
                writer.writeheader()
                writer.writerow({"timestamp_ms": 2500, "gaze_x": 123})
            count = annotate_sensor_csv(path, "session_1", self.events, 100, "eye")
            with path.open("r", newline="", encoding="utf-8") as f:
                row = next(csv.DictReader(f))
            self.assertEqual(count, 1)
            self.assertEqual(row["session_id"], "session_1")
            self.assertEqual(row["study_phase"], "active_task")
            self.assertEqual(row["gaze_x"], "123")

            rr_path = Path(temp) / "rr_raw.csv"
            with rr_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["received_at", "rr_original_ms", "phase"])
                writer.writeheader()
                writer.writerow({"received_at": datetime.fromtimestamp(2.5, timezone.utc).isoformat(),
                                 "rr_original_ms": 900, "phase": "task"})
            annotate_sensor_csv(rr_path, "session_1", self.events, 100, "rr")
            with rr_path.open("r", newline="", encoding="utf-8") as f:
                rr_row = next(csv.DictReader(f))
            self.assertEqual(rr_row["study_phase"], "active_task")
            self.assertEqual(rr_row["rr_original_ms"], "900")

    def test_hrv_csv_gets_mixed_window_details(self):
        def iso(ms):
            return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "hrv.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["phase", "window_start", "window_end", "rmssd_ms"])
                writer.writeheader()
                writer.writerow({"phase": "task", "window_start": iso(2500), "window_end": iso(5500), "rmssd_ms": 42})
            annotate_sensor_csv(path, "session_1", self.events, 100, "hrv")
            with path.open("r", newline="", encoding="utf-8") as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row["study_phase"], "mixed")
            mix = json.loads(row["window_phase_mix"])
            self.assertEqual(sum(part["duration_seconds"] for part in mix), 3.0)
            self.assertEqual({part["difficulty_level"] for part in mix if part["difficulty_level"]},
                             {"easy", "hard"})
            self.assertEqual(row["rmssd_ms"], "42")

    def test_missing_event_file_remains_unlabeled(self):
        with tempfile.TemporaryDirectory() as temp:
            events, status = load_study_events(Path(temp) / "session_1")
            self.assertEqual(events, [])
            self.assertEqual(status, "unlabeled")

    def test_fusion_exports_session_and_mixed_phase_labels(self):
        def iso(ms):
            return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()

        with tempfile.TemporaryDirectory() as temp:
            session_dir = Path(temp) / "session_1"
            session_dir.mkdir()
            (session_dir / "hrv_calib_status.json").write_text(
                json.dumps({"both_calibrated_at_utc": iso(100)}), encoding="utf-8")
            event_rows = [
                {"event_id": "1", "session_id": "session_1", "event_type": "session_linked", "event_time_ms": 0},
                {"event_id": "2", "session_id": "session_1", "event_type": "set_started", "event_time_ms": 1000,
                 "puzzle_set": "easy-1", "difficulty_level": "easy"},
                {"event_id": "3", "session_id": "session_1", "event_type": "puzzle_started", "event_time_ms": 1100,
                 "puzzle_set": "easy-1", "difficulty_level": "easy", "puzzle_id": "even_odd", "puzzle_title": "Even or Odd"},
                {"event_id": "4", "session_id": "session_1", "event_type": "set_ended", "event_time_ms": 3000,
                 "puzzle_set": "easy-1", "difficulty_level": "easy"},
                {"event_id": "5", "session_id": "session_1", "event_type": "study_ended", "event_time_ms": 3500},
            ]
            with (session_dir / "task_events.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["event_id", "session_id", "event_type", "event_time_ms",
                                                       "puzzle_set", "difficulty_level", "puzzle_id", "puzzle_title"])
                writer.writeheader()
                writer.writerows(event_rows)
            with (session_dir / "hrv.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["phase", "window_start", "window_end", "window_seconds",
                                                       "rmssd_ms", "sdnn_ms", "coverage", "status"])
                writer.writeheader()
                writer.writerow({"phase": "baseline", "window_start": iso(0), "window_end": iso(1000),
                                 "window_seconds": 1, "rmssd_ms": 50, "sdnn_ms": 60, "status": "ready"})
                writer.writerow({"phase": "task", "window_start": iso(2500), "window_end": iso(3500),
                                 "window_seconds": 1, "rmssd_ms": 40, "sdnn_ms": 50, "coverage": 1, "status": "ready"})
            with (session_dir / "eye_tracking.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp_ms", "frame_index", "gaze_status", "face_detected"])
                writer.writeheader()
                writer.writerow({"timestamp_ms": 2700, "frame_index": 1, "gaze_status": "on_screen", "face_detected": "True"})

            report = fuse_session(session_dir)
            with (session_dir / "multimodal_timeline.csv").open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            task_row = next(row for row in rows if row["hrv_status"] == "ready")
            self.assertEqual(report["study_session_status"], "completed")
            self.assertEqual(report["excluded_window_count"]["mixed"], 1)
            self.assertEqual(report["task_summary_by_set"], {})
            self.assertEqual(task_row["session_id"], "session_1")
            self.assertEqual(task_row["study_phase"], "mixed")
            self.assertEqual(task_row["difficulty_level"], "mixed")
            self.assertEqual(task_row["puzzle_id"], "mixed")
            self.assertIn("window_phase_mix", task_row)

    def test_mock_multiset_session_keeps_only_pure_task_windows_in_summary(self):
        def iso(ms):
            return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()

        base = 1_800_000_000_000
        with tempfile.TemporaryDirectory() as temp:
            session_dir = Path(temp) / "mock_new_session"
            session_dir.mkdir()
            (session_dir / "hrv_calib_status.json").write_text(
                json.dumps({"both_calibrated_at_utc": iso(base)}), encoding="utf-8")
            events = [
                event(base + 1000, "set_started", "easy-1", "easy"),
                event(base + 2000, "puzzle_started", "easy-1", "easy", "e1"),
                event(base + 32000, "attempt_submitted", "easy-1", "easy", "e1"),
                event(base + 33000, "rating_started", "easy-1", "easy", "e1"),
                event(base + 36000, "rating_ended", "easy-1", "easy", "e1"),
                event(base + 37000, "puzzle_started", "easy-1", "easy", "e1"),
                event(base + 38000, "set_ended", "easy-1", "easy"),
                event(base + 70000, "set_started", "medium", "medium"),
                event(base + 71000, "puzzle_started", "medium", "medium", "m1"),
                event(base + 102000, "attempt_submitted", "medium", "medium", "m1"),
                event(base + 103000, "set_ended", "medium", "medium"),
                event(base + 140000, "set_started", "hard", "hard"),
                event(base + 141000, "puzzle_started", "hard", "hard", "h1"),
                event(base + 172000, "attempt_submitted", "hard", "hard", "h1"),
                event(base + 173000, "set_ended", "hard", "hard"),
                event(base + 174000, "study_ended"),
            ]
            with (session_dir / "task_events.csv").open("w", newline="", encoding="utf-8") as f:
                fields = ["session_id", "event_time_ms", "event_type", "puzzle_set", "difficulty_level", "puzzle_id"]
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows({key: {**item, "session_id": session_dir.name}.get(key, "") for key in fields}
                                 for item in events)
            windows = [(3000, 28000), (30000, 55000), (72000, 97000), (142000, 167000)]
            with (session_dir / "hrv.csv").open("w", newline="", encoding="utf-8") as f:
                fields = ["phase", "window_start", "window_end", "window_seconds", "rmssd_ms", "sdnn_ms", "coverage", "status"]
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"phase": "baseline", "window_start": iso(base - 25000),
                                 "window_end": iso(base), "window_seconds": 25, "rmssd_ms": 50,
                                 "sdnn_ms": 60, "status": "ready"})
                for start, end in windows:
                    writer.writerow({"phase": "task", "window_start": iso(base + start),
                                     "window_end": iso(base + end), "window_seconds": 25,
                                     "rmssd_ms": 40, "sdnn_ms": 50, "coverage": 1, "status": "ready"})
            with (session_dir / "eye_tracking.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp_ms", "frame_index", "gaze_status", "face_detected"])
                writer.writeheader()
                for index, (start, _) in enumerate(windows):
                    writer.writerow({"timestamp_ms": base + start + 5000, "frame_index": index,
                                     "gaze_status": "on_screen", "face_detected": "True"})
            with (session_dir / "rr_raw.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["received_at", "rr_original_ms"])
                writer.writeheader()
                writer.writerow({"received_at": iso(base + 75000), "rr_original_ms": 900})

            report = fuse_session(session_dir)
            with (session_dir / "multimodal_timeline.csv").open(newline="", encoding="utf-8") as f:
                timeline = list(csv.DictReader(f))
            self.assertEqual([row["study_phase"] for row in timeline],
                             ["active_task", "mixed", "active_task", "active_task"])
            self.assertEqual(report["excluded_window_count"]["mixed"], 1)
            self.assertEqual({key: value["window_count"] for key, value in report["task_summary_by_set"].items()},
                             {"easy-1": 1, "medium": 1, "hard": 1})
            with (session_dir / "rr_raw.csv").open(newline="", encoding="utf-8") as f:
                rr = next(csv.DictReader(f))
            self.assertEqual((rr["study_phase"], rr["puzzle_set"], rr["puzzle_id"]),
                             ("active_task", "medium", "m1"))


if __name__ == "__main__":
    unittest.main()
