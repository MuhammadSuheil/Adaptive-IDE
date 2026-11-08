"""Scientific-data invariants: time alignment, missingness, provenance and leakage."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis_sessions.dataset_fusion import build_dataset as fusion


BASE = pd.Timestamp("2026-10-30T09:00:00+07:00").timestamp() * 1000


def iso(offset):
    return pd.Timestamp(BASE + offset, unit="ms", tz="UTC").isoformat()


def eye_source(times):
    size = len(times)
    return pd.DataFrame({
        "timestamp_ms": np.asarray(times) + BASE,
        "session_id": ["source-id"] * size,
        "face_detected": [True] * size, "EAR_Left": [.3] * size, "EAR_Right": [.4] * size,
        "gaze_status": ["on_screen"] * size,
        "gaze_x_raw": [500.] * size, "gaze_y_raw": [400.] * size,
        "gaze_x_smooth": [500.] * size, "gaze_y_smooth": [400.] * size,
        "iris_size": [.5] * size, "iris_size_delta": [.01] * size,
        "iris_valid": [True] * size, "iris_baseline_ready": [True] * size,
        "iris_size_mode": ["eye_width_ratio"] * size, "is_blinking": [False] * size,
        "head_pitch": [2.] * size, "head_yaw": [-1.] * size,
        "Eye_State": ["Fixation"] * size, "capture_fps": [30.] * size,
        "total_blinks": [int(t >= 25000) for t in times],
    })


def fixture(directory):
    directory.mkdir(parents=True)
    session_id = f"source-{directory.name}"
    eye = eye_source(np.arange(0, 90001, 50))
    eye["session_id"] = session_id
    eye.to_csv(directory / "eye_tracking.csv", index=False)
    baseline = {"session_id": session_id, "phase": "baseline", "window_start": iso(-121000), "window_end": iso(-1000),
                "window_seconds": 120, "status": "ready", "coverage": .95, "rmssd_ms": 50., "sdnn_ms": 60.,
                "rr_count": 140, "rr_pairs": 139, "data_age_seconds": 1., "reason": "target_reached"}
    windows = [{**baseline, "phase": "task", "window_start": iso(start), "window_end": iso(start+25000),
                "window_seconds": 25, "coverage": .98, "rmssd_ms": 25., "sdnn_ms": 30., "reason": ""}
               for start in (0, 30000, 60000)]
    pd.DataFrame([baseline] + windows).to_csv(directory / "hrv.csv", index=False)
    events = []
    for start, level in zip((0, 30000, 60000), fusion.LEVELS):
        for kind in ("set_started", "puzzle_started"):
            events.append({"session_id": session_id, "event_time_ms": BASE+start, "event_type": kind,
                           "difficulty_level": level, "puzzle_set": level, "puzzle_id": f"{level}-q", "puzzle_title": level})
    pd.DataFrame(events).to_csv(directory / "task_events.csv", index=False)
    (directory / "hrv_calib_status.json").write_text(json.dumps({"both_calibrated_at_utc": iso(-1000)}), encoding="utf-8")
    (directory / "eye_summary_calibration.json").write_text(json.dumps({
        "config": {"screen": {"width": 1000, "height": 800}},
        "attempts": [{"phase": "held_out_validation", "passed_thresholds": False,
                      "mean_target_error_median_px": 500, "mean_target_error_p95_px": 1200, "max_corner_error_px": 1500}]
    }), encoding="utf-8")
    (directory / "eye_summary.json").write_text(json.dumps({"summary": {"camera": {
        "width": 640, "height": 480, "fps_reported": 60, "capture_fps_observed": 29.8
    }}}), encoding="utf-8")
    return directory


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.config = fusion.load_config()
        self.screen = {"width": 1000, "height": 800}

    def test_right_closed_alignment_and_counter_events(self):
        eye = fusion.prepare_eye(eye_source([0, 50, 24950, 25000, 25050]), self.screen, self.config)
        first = fusion.aggregate_eye(eye, BASE, BASE+25000, self.config)
        second = fusion.aggregate_eye(eye, BASE+25000, BASE+50000, self.config)
        self.assertEqual(first["eye_frames_count"], 3)
        self.assertEqual(second["eye_frames_count"], 1)
        self.assertEqual(first["blink_count"], 1)
        self.assertEqual(second["blink_count"], 0)
        self.assertAlmostEqual(first["blink_rate_per_minute"], 2.4)

    def test_coverage_counts_time_bins_not_frames_or_nominal_fps(self):
        eye = fusion.prepare_eye(eye_source(list(range(1, 101))), self.screen, self.config)
        result = fusion.aggregate_eye(eye, BASE, BASE+25000, self.config)
        self.assertAlmostEqual(result["eye_temporal_coverage"], 1/250)
        self.assertEqual(result["eye_frames_count"], 100)
        end_only = fusion.prepare_eye(eye_source([25000]), self.screen, self.config)
        self.assertAlmostEqual(fusion.aggregate_eye(end_only, BASE, BASE+25000, self.config)["eye_temporal_coverage"], 1/250)

    def test_invalid_features_masked_and_clipping_denominator(self):
        source = eye_source([0, 100, 200, 300])
        source.loc[1, ["gaze_x_raw", "gaze_x_smooth"]] = -100
        source.loc[2, "face_detected"] = False
        source.loc[2, "gaze_x_raw"] = -100
        source.loc[3, "EAR_Left"] = 1.2
        source.loc[3, "iris_size_mode"] = "unrecognized_units"
        eye = fusion.prepare_eye(source, self.screen, self.config)
        result = fusion.aggregate_eye(eye, BASE, BASE+25000, self.config)
        self.assertAlmostEqual(result["clipping_ratio_valid"], .5)
        self.assertTrue(np.isnan(eye.ear_left.iloc[2]))
        self.assertTrue(np.isnan(eye.ear_left.iloc[3]))
        self.assertTrue(np.isnan(eye.gaze_x_norm.iloc[2]))
        self.assertTrue(np.isnan(eye.iris_size_estimate.iloc[3]))
        self.assertAlmostEqual(result["face_ear_ratio"], 1/3)

    def test_empty_windows_remain_unavailable(self):
        eye = fusion.prepare_eye(eye_source([0]), self.screen, self.config)
        result = fusion.aggregate_eye(eye, BASE+1000, BASE+26000, self.config)
        self.assertEqual(result["eye_frames_count"], 0)
        self.assertEqual(result["eye_temporal_coverage"], 0)
        for column in ("blink_count", "ear_left_mean", "clipping_ratio_valid", "eye_state_fixation_ratio"):
            self.assertTrue(np.isnan(result[column]), column)

    def test_counter_reset_does_not_fabricate_blinks(self):
        source = eye_source([0, 100, 200, 300])
        source["total_blinks"] = [5, 6, 0, 1]
        eye = fusion.prepare_eye(source, self.screen, self.config)
        result = fusion.aggregate_eye(eye, BASE, BASE+25000, self.config)
        self.assertEqual(result["blink_counter_discontinuities"], 1)
        self.assertTrue(np.isnan(result["blink_count"]))

    def test_nonmonotonic_or_duplicate_timestamps_rejected(self):
        for times in ([0, 0], [100, 0]):
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                fusion.prepare_eye(eye_source(times), self.screen, self.config)

    def test_quality_exact_boundaries_and_nested_thresholds(self):
        row = {"hrv_usable": True, "eye_temporal_coverage": .95, "face_ear_ratio": .95,
               "valid_gaze_ratio": .9, "iris_available_ratio": .9, "clipping_ratio_valid": .05}
        self.assertEqual(fusion.quality_tier(row, self.config)["quality_tier"], "good")
        row["clipping_ratio_valid"] = .2
        self.assertEqual(fusion.quality_tier(row, self.config)["quality_tier"], "acceptable")
        row["clipping_ratio_valid"] = np.nan
        self.assertEqual(fusion.quality_tier(row, self.config)["quality_tier"], "poor")
        row["clipping_ratio_valid"] = 0
        row["hrv_usable"] = False
        self.assertIn("hrv_unusable", fusion.quality_tier(row, self.config)["acceptable_failed_criteria"])


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = fixture(Path(self.temp.name) / "Faqih")
        self.config = fusion.load_config()

    def test_original_ids_team_annotations_and_calibration_separated(self):
        before = fusion.snapshot(self.directory)
        rows, excluded, report = fusion.build_session(self.directory, self.config)
        self.assertEqual(len(rows), 3)
        self.assertFalse(excluded)
        self.assertEqual({r["team_review_group"] for r in rows}, {"proposed_good"})
        self.assertEqual({r["quality_tier"] for r in rows}, {"good"})
        self.assertFalse(rows[0]["gaze_validation_passed"])
        self.assertEqual(rows[0]["session_id"], "source-Faqih")
        self.assertAlmostEqual(rows[0]["rmssd_baseline_ratio"], .5)
        self.assertEqual(report["camera_observed_capture_fps"], 29.8)
        self.assertEqual(before, fusion.snapshot(self.directory))

    def test_unready_numeric_hrv_masked_and_low_coverage_baseline_rejected(self):
        source = pd.read_csv(self.directory / "hrv.csv", keep_default_na=False)
        source.loc[0, "coverage"] = .6
        source.loc[1, "status"] = "gap"
        source.loc[1, ["rmssd_ms", "sdnn_ms"]] = 999
        source.to_csv(self.directory / "hrv.csv", index=False)
        rows, _, _ = fusion.build_session(self.directory, self.config)
        self.assertEqual(rows[0]["quality_tier"], "poor")
        self.assertTrue(np.isnan(rows[0]["rmssd_ms"]))
        self.assertTrue(np.isnan(rows[0]["sdnn_ms"]))
        self.assertTrue(all(np.isnan(row["rmssd_baseline_ratio"]) for row in rows))
        self.assertEqual(rows[1]["rmssd_ms"], 25)

    def test_same_level_cross_question_kept_but_feedback_crossing_excluded(self):
        source = pd.read_csv(self.directory / "task_events.csv", keep_default_na=False)
        new = source.iloc[1].to_dict()
        new.update(event_time_ms=BASE+10000, puzzle_id="another-q")
        pd.concat([source, pd.DataFrame([new])], ignore_index=True).to_csv(self.directory / "task_events.csv", index=False)
        rows, _, _ = fusion.build_session(self.directory, self.config)
        self.assertEqual(rows[0]["puzzle_id"], "mixed")
        new.update(event_type="attempt_submitted", event_time_ms=BASE+24999.9)
        pd.concat([source, pd.DataFrame([new])], ignore_index=True).to_csv(self.directory / "task_events.csv", index=False)
        rows, excluded, _ = fusion.build_session(self.directory, self.config)
        self.assertEqual(len(rows), 2)
        self.assertEqual(excluded[0]["study_phase"], "mixed")

    def test_mixed_difficulty_and_missing_ledger_excluded(self):
        source = pd.read_csv(self.directory / "task_events.csv", keep_default_na=False)
        source.loc[source.event_time_ms.eq(BASE+30000), "event_time_ms"] = BASE+20000
        source.to_csv(self.directory / "task_events.csv", index=False)
        rows, excluded, _ = fusion.build_session(self.directory, self.config)
        self.assertEqual(len(rows), 2)
        self.assertEqual(excluded[0]["difficulty_level"], "mixed")
        (self.directory / "task_events.csv").unlink()
        rows, excluded, _ = fusion.build_session(self.directory, self.config)
        self.assertFalse(rows)
        self.assertEqual(len(excluded), 3)
        self.assertEqual({r["exclusion_reason"] for r in excluded}, {"no_matching_task_events"})

    def test_repeated_recordings_share_a_participant(self):
        renamed = self.directory.with_name("Ramadhan-2")
        self.directory.rename(renamed)
        rows, _, _ = fusion.build_session(renamed, self.config)
        self.assertEqual({r["participant_id"] for r in rows}, {"Ramadhan"})
        self.assertEqual([r["difficulty_level"] for r in rows if r["repeated_task"]], ["medium"])

    def test_sensor_id_disagreement_and_window_grid_rejected(self):
        source = pd.read_csv(self.directory / "hrv.csv", keep_default_na=False)
        source.loc[1, "session_id"] = "wrong-id"
        source.to_csv(self.directory / "hrv.csv", index=False)
        with self.assertRaisesRegex(ValueError, "IDs disagree"):
            fusion.build_session(self.directory, self.config)
        source.loc[1, "session_id"] = "source-Faqih"
        source.loc[2, "window_start"] = iso(31000)
        source.loc[2, "window_end"] = iso(56000)
        source.to_csv(self.directory / "hrv.csv", index=False)
        with self.assertRaisesRegex(ValueError, "off-grid"):
            fusion.build_session(self.directory, self.config)


class ExportTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows ACL regression")
    def test_published_export_inherits_destination_permissions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "output"
            output.mkdir()
            with tempfile.TemporaryDirectory(dir=root) as staging:
                source = Path(staging) / "dataset.csv"
                content = b'participant_id,value\r\nFaqih,0.3\r\n'
                source.write_bytes(content)
                destination = output / "dataset.csv"
                fusion.publish_artifact(source, destination)
                self.assertEqual(destination.read_bytes(), content)
                acl = subprocess.run(["icacls", str(destination)], capture_output=True, text=True, check=True)
                self.assertIn("(I)", acl.stdout, "Published file must inherit its output folder permissions.")

    def test_nested_csvs_deduplication_source_hashes_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            fixture(source / "Fabian")
            fixture(source / "Athaya")
            fixture(source / "Baithar")
            shutil.copytree(source / "Baithar", source / "Baithar - Copy")
            before = fusion.snapshot(source)
            output = root / "output"
            first = fusion.build_dataset(source, output)
            self.assertEqual(first["selected_folders"], ["Athaya", "Baithar", "Fabian"])
            self.assertEqual(first["datasets"]["all_data"]["windows"], 9)
            self.assertEqual(first["datasets"]["good_only"]["windows"], 3)
            self.assertEqual(first["datasets"]["good_acceptable"]["windows"], 6)
            self.assertEqual(first["source_sha256"], before)
            self.assertEqual(before, fusion.snapshot(source))
            views = {name: pd.read_csv(output / f"{name}.csv", keep_default_na=False) for name in fusion.DATASETS}
            self.assertTrue(set(views["good_only"].window_id) <= set(views["good_acceptable"].window_id) <= set(views["all_data"].window_id))
            for view in views.values():
                self.assertEqual(view.columns.tolist(), fusion.DATA_COLUMNS)
                shared = views["all_data"].set_index("window_id", drop=False).loc[view.window_id].reset_index(drop=True)
                pd.testing.assert_frame_equal(view, shared)
            self.assertEqual(set(views["good_only"].participant_id), {"Fabian"})
            self.assertEqual(set(views["good_acceptable"].participant_id), {"Fabian", "Athaya"})
            summary = pd.read_csv(output / "session_quality_summary.csv").set_index("session_folder")
            self.assertEqual(summary.loc["Baithar", "good_windows"], 3)
            self.assertEqual(summary.loc["Baithar", "good_only_windows"], 0)
            self.assertEqual(summary.loc["Baithar", "good_acceptable_windows"], 0)
            dictionary = pd.read_csv(output / "data_dictionary.csv")
            self.assertEqual(set(dictionary.column), set(fusion.DATA_COLUMNS))
            self.assertNotIn("difficulty_level", fusion.FEATURE_COLUMNS)
            self.assertNotIn("participant_id", fusion.FEATURE_COLUMNS)
            self.assertNotIn("quality_tier", fusion.FEATURE_COLUMNS)
            second = fusion.build_dataset(source, output)
            self.assertEqual(first["output_sha256"], second["output_sha256"])
            self.assertEqual(before, fusion.snapshot(source))

    def test_all_data_retains_missing_sensor_windows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = fixture(root / "source" / "Sullivan")
            hrv = pd.read_csv(directory / "hrv.csv", keep_default_na=False)
            hrv.loc[hrv.phase.eq("task"), "status"] = "insufficient_data"
            hrv.to_csv(directory / "hrv.csv", index=False)
            fusion.build_dataset(root / "source", root / "output")
            frame = pd.read_csv(root / "output" / "all_data.csv")
            self.assertEqual(len(frame), 3)
            self.assertTrue(frame.rmssd_ms.isna().all())
            self.assertTrue(frame.quality_tier.eq("poor").all())
            self.assertTrue(pd.read_csv(root / "output" / "good_only.csv").empty)

    def test_output_cannot_be_inside_or_contain_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root / "source" / "Fabian")
            before = fusion.snapshot(root / "source")
            for output in (root / "source" / "output", root, root / "source"):
                with self.assertRaisesRegex(ValueError, "Output must be separate"):
                    fusion.build_dataset(root / "source", output)
            self.assertEqual(before, fusion.snapshot(root / "source"))

    def test_source_change_during_export_prevents_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = fixture(root / "source" / "Fabian")
            original_writer = fusion.write_artifacts

            def concurrent_source_change(*args, **kwargs):
                result = original_writer(*args, **kwargs)
                (directory / "new-event.json").write_text('{}')
                return result

            with patch.object(fusion, "write_artifacts", side_effect=concurrent_source_change):
                with self.assertRaisesRegex(RuntimeError, "Source files changed during export"):
                    fusion.build_dataset(root / "source", root / "output")
            self.assertFalse((root / "output").exists())

    def test_common_test_windows_and_participant_isolation(self):
        rows = []
        groups = {"Fabian": "proposed_good", "faqih": "proposed_good", "Ramadhan": "unreviewed", "Shakilla": "questionable"}
        for participant, group in groups.items():
            for tier in fusion.TIERS:
                for level in fusion.LEVELS:
                    for session in (("Ramadhan", "Ramadhan-2") if participant == "Ramadhan" else (participant,)):
                        rows.append({"window_id": f"{session}-{tier}-{level}", "participant_id": participant,
                                     "session_id": session, "quality_tier": tier, "difficulty_level": level, "team_review_group": group})
        frame = pd.DataFrame(rows)
        views = fusion.dataset_views(frame)
        folds, tests = fusion.evaluation_contract(frame, views, frame.participant_id)
        self.assertEqual(set(folds.loc[folds.comparison_feasible, "held_out_participant_id"]), {"Fabian", "faqih"})
        self.assertEqual(set(tests.participant_id), {"Fabian", "faqih"})
        self.assertEqual(set(tests.window_id), set(views["good_only"].window_id))
        for fold in folds.to_dict("records"):
            ids = set(tests.loc[tests.fold_id.eq(fold["fold_id"]), "window_id"])
            for view in views.values():
                train = view[view.participant_id.ne(fold["held_out_participant_id"])]
                self.assertFalse(ids & set(train.window_id))
                if fold["held_out_participant_id"] == "Ramadhan":
                    self.assertFalse(train.session_id.isin(["Ramadhan", "Ramadhan-2"]).any())

    def test_eye_cohorts_cannot_be_promoted_by_window_metrics(self):
        frame = pd.DataFrame([
            {"window_id": f"{group}-{tier}", "team_review_group": group, "quality_tier": tier}
            for group in ("proposed_good", "questionable", "unreviewed") for tier in fusion.TIERS
        ])
        views = fusion.dataset_views(frame)
        self.assertEqual(set(views["good_only"].window_id), {"proposed_good-good"})
        self.assertEqual(set(views["good_acceptable"].window_id), {
            "proposed_good-good", "proposed_good-acceptable", "questionable-good", "questionable-acceptable"
        })
        self.assertEqual(len(views["all_data"]), 9)

    def test_config_rejects_non_nested_thresholds_and_duplicate_review_names(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            config = fusion.load_config()
            changed = copy.deepcopy(config)
            changed["quality"]["good"]["min_valid_gaze_ratio"] = .1
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "at least as strict"):
                fusion.load_config(path)
            changed = copy.deepcopy(config)
            changed["participant_review_groups"]["questionable"].append("faqih")
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "unique across groups"):
                fusion.load_config(path)

    def test_config_rejects_typos_and_string_thresholds(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            config = fusion.load_config()
            changed = copy.deepcopy(config)
            changed["quality"]["good"]["min_valid_gaze_ratio"] = "0.9"
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "Invalid quality rules"):
                fusion.load_config(path)
            changed = copy.deepcopy(config)
            changed["window_second"] = changed.pop("window_seconds")
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "Config keys"):
                fusion.load_config(path)


if __name__ == "__main__":
    unittest.main()
