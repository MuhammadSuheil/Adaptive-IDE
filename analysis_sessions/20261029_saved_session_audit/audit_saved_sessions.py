"""Read-only audit of archived sensor data; writes derived files beside this script.

Run from any directory: python analysis_sessions/20261029_saved_session_audit/audit_saved_sessions.py
Requires pandas/numpy. Does not invoke fuse_session (which rewrites source CSVs).
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from fuse_and_analyze import context_at, interval_segments
from analysis_sessions.session_inventory import select_session_directories

SOURCE = ROOT / "sessions" / "saved_session"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}


def read_csv(path):
    return pd.read_csv(path, keep_default_na=False, low_memory=False) if path.exists() else pd.DataFrame()


def numeric(frame, column):
    return pd.to_numeric(frame[column], errors="coerce")


def truth(frame, column):
    return frame[column].astype(str).str.lower().isin(["true", "1"])


def percent(mask):
    return round(float(mask.mean() * 100), 3) if len(mask) else None


def stamps(series):
    parsed = pd.to_datetime(series, utc=True, format="mixed")
    return (parsed - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy() * 1000


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def eye_metrics(frame, screen):
    if frame.empty:
        return {"eye_frames": 0}
    t = numeric(frame, "timestamp_ms")
    valid = frame.gaze_status.isin(["on_screen", "gaze_outside_screen"])
    x, y = numeric(frame, "gaze_x_raw"), numeric(frame, "gaze_y_raw")
    w, h = screen["width"], screen["height"]
    clamped = (np.isclose(x, -.1 * w) | np.isclose(x, 1.1 * w)
               | np.isclose(y, -.1 * h) | np.isclose(y, 1.1 * h))
    return {
        "eye_frames": len(frame),
        "face_detected_pct": percent(truth(frame, "face_detected")),
        "on_screen_pct": percent(frame.gaze_status.eq("on_screen")),
        "outside_screen_pct": percent(frame.gaze_status.eq("gaze_outside_screen")),
        "invalid_gaze_pct": percent(~valid),
        "raw_clamp_pct_of_valid_gaze": percent(pd.Series(clamped, index=frame.index)[valid]),
        "iris_delta_available_pct": percent(numeric(frame, "iris_size_delta").notna()),
        "head_shifted_pct": percent(truth(frame, "head_pose_shifted")),
        "mean_logged_processing_fps": round(numeric(frame, "fps_actual").mean(), 3),
        "mean_capture_fps": round(numeric(frame, "capture_fps").mean(), 3),
        "median_frame_interval_ms": round(t.diff().median(), 3),
        "largest_frame_gap_ms": round(t.diff().max(), 3),
    }


def main():
    directories, inventory = select_session_directories(SOURCE)
    files = [p for d in SOURCE.iterdir() if d.is_dir() for p in d.iterdir()
             if p.is_file() and p.suffix in (".csv", ".json")]
    before = {str(p.relative_to(ROOT)): digest(p) for p in files}
    summaries, phases, sets, targets, hrv_levels, checks, gaps = [], [], [], [], [], [], []
    for directory in directories:
        name = directory.name
        eye = read_csv(directory / "eye_tracking.csv")
        hrv = read_csv(directory / "hrv.csv")
        timeline = read_csv(directory / "multimodal_timeline.csv")
        ledger = read_csv(directory / "task_events.csv").sort_values("event_time_ms")
        rr = read_csv(directory / "rr_raw.csv")
        calibration = read_json(directory / "eye_summary_calibration.json")
        eye_summary = read_json(directory / "eye_summary.json")
        report = read_json(directory / "session_report.json")
        ready_status = read_json(directory / "hrv_calib_status.json")
        config = calibration["config"]
        screen = config["screen"]
        stored_ids = sorted(set(eye.session_id) | set(hrv.session_id) | set(ledger.session_id))
        events = ledger.to_dict("records")
        # Read existing IDs directly: archived folders were renamed after acquisition.
        ready = ready_status.get("both_calibrated_at_utc")
        ready_ms = pd.Timestamp(ready).timestamp() * 1000 if ready else None
        eye_times = numeric(eye, "timestamp_ms").to_numpy()
        boundaries = sorted(set([float(eye_times.min())] +
                                [float(e["event_time_ms"]) for e in events] +
                                ([ready_ms] if ready_ms is not None else [])))
        contexts = [context_at(t, events, ready_ms) for t in boundaries]
        indices = np.searchsorted(boundaries, eye_times, side="right") - 1
        for column in ("study_phase", "difficulty_level", "puzzle_set", "puzzle_id"):
            eye["ledger_" + column] = [contexts[i][column] for i in indices]
        active = eye[eye.ledger_study_phase.eq("active_task")]
        for (phase, level), group in eye.groupby(["ledger_study_phase", "ledger_difficulty_level"]):
            phases.append({"subject_session": name, "phase": phase, "difficulty": level,
                           **eye_metrics(group, screen)})

        attempts = calibration["attempts"]
        gaze_attempts = [a for a in attempts if a["phase"] == "gaze_calibration"]
        validation = [a for a in attempts if a["phase"] == "held_out_validation"]
        final_validation = validation[-1] if validation else {}
        for i, attempt in enumerate(validation):
            for j, point in enumerate(attempt["points"]):
                samples = point.get("samples", [])
                predictions = np.array([s["prediction"] for s in samples])
                mean = predictions.mean(axis=0) if len(predictions) else [np.nan, np.nan]
                targets.append({"subject_session": name, "validation_attempt": i,
                                "point_index": j, "target_x": point["target"][0],
                                "target_y": point["target"][1], "sample_count": len(samples),
                                "mean_prediction_x": mean[0], "mean_prediction_y": mean[1],
                                "mean_target_error_px": point.get("mean_prediction_error_px"),
                                "sample_error_median_px": np.median([s["error_px"] for s in samples]) if samples else None})
        task = hrv[hrv.phase.eq("task")].copy()
        baseline = hrv[hrv.phase.eq("baseline")].iloc[-1]
        for (phase, level), group in task.groupby(["study_phase", "difficulty_level"]):
            hrv_levels.append({"subject_session": name, "stored_phase": phase,
                               "stored_difficulty": level, "windows": len(group),
                               "ready_windows": int(group.status.eq("ready").sum()),
                               "ready_pct": percent(group.status.eq("ready")),
                               "median_coverage": numeric(group, "coverage").median(),
                               "median_ready_rmssd_ms": numeric(group[group.status.eq("ready")], "rmssd_ms").median()})
        intervals = interval_segments(float(eye_times.min()), float(eye_times.max()), events, ready_ms)
        starts = ledger[ledger.event_type.eq("set_started")]
        for _, start in starts.iterrows():
            later = ledger[(ledger.event_time_ms > start.event_time_ms) & ledger.event_type.isin(["set_ended", "set_started"])]
            end = later.iloc[0] if len(later) else None
            closed = end is not None and end.event_type == "set_ended"
            end_ms = float(end.event_time_ms) if end is not None else float(eye_times.max())
            submitted = ledger[(ledger.event_time_ms >= start.event_time_ms) &
                               (ledger.event_time_ms <= end_ms) & ledger.event_type.eq("attempt_submitted")]
            group = active[(active.timestamp_ms >= start.event_time_ms) & (active.timestamp_ms <= end_ms)]
            hs = task[(task.puzzle_set == start.puzzle_set) & task.study_phase.eq("active_task")]
            ready_rmssd = numeric(hs[hs.status.eq("ready")], "rmssd_ms")
            baseline_rmssd = pd.to_numeric(baseline.rmssd_ms, errors="coerce")
            active_s = sum(s["duration_seconds"] for s in intervals
                           if s["study_phase"] == "active_task" and s["puzzle_set"] == start.puzzle_set)
            eye_overlap_s = max(0.0, min(end_ms, eye_times.max()) - max(start.event_time_ms, eye_times.min())) / 1000
            sets.append({"subject_session": name, "puzzle_set": start.puzzle_set,
                         "difficulty": start.difficulty_level, "start_wib": start.event_time_wib,
                         "end_wib": end.event_time_wib if closed else "",
                         "outcome": end.outcome if closed else "no_set_end_recorded",
                         "observed_elapsed_seconds": round((end_ms - start.event_time_ms) / 1000, 3),
                         "eye_recording_overlap_seconds": round(eye_overlap_s, 3),
                         "active_seconds_in_eye_recording": round(active_s, 3),
                         "attempts": len(submitted), "correct_attempts": int(submitted.outcome.eq("correct").sum()),
                         "incorrect_attempts": int(submitted.outcome.eq("incorrect").sum()),
                         "pure_active_hrv_windows": len(hs), "ready_pure_active_hrv_windows": int(hs.status.eq("ready").sum()),
                         "median_ready_pure_active_rmssd_ms": ready_rmssd.median(),
                         "median_ready_pure_active_rmssd_ratio": ready_rmssd.median() / baseline_rmssd if baseline_rmssd > 0 else None,
                         **eye_metrics(group, screen)})

        cam = eye_summary.get("summary", {}).get("camera", {})
        pure = task.study_phase.eq("active_task") & ~task.puzzle_set.isin(["", "mixed"])
        quality = eye_summary.get("summary", {}).get("calibration_quality")
        last_rr_ms = stamps(rr.received_at)[-1] if len(rr) else None
        raw_task = rr[rr.phase.eq("task")].drop_duplicates("packet_id").copy()
        if len(raw_task):
            raw_task["received_ms"] = stamps(raw_task.received_at)
            raw_task["gap_seconds"] = raw_task.received_ms.diff() / 1000
            for idx, row in raw_task[raw_task.gap_seconds > 5].iterrows():
                gaps.append({"subject_session": name, "gap_ends_wib": row.received_at,
                             "gap_seconds": row.gap_seconds})
        summary = {
            "subject_session": name, "stored_session_ids": "|".join(stored_ids),
            "directory_matches_stored_id": len(stored_ids) == 1 and name == stored_ids[0],
            "calibration_quality_pct": round(100 * quality, 3) if quality is not None else None,
            "gaze_calibration_decisions": "|".join(a.get("decision", "") for a in gaze_attempts),
            "held_out_passed": final_validation.get("passed_thresholds"),
            "held_out_decision": final_validation.get("decision"),
            "held_out_target_median_error_px": final_validation.get("mean_target_error_median_px"),
            "held_out_target_p95_error_px": final_validation.get("mean_target_error_p95_px"),
            "held_out_max_corner_error_px": final_validation.get("max_corner_error_px"),
            "held_out_sample_p95_error_px": final_validation.get("sample_error_p95_px"),
            "camera_width": cam.get("width"), "camera_height": cam.get("height"),
            "camera_reported_fps": cam.get("fps_reported"),
            "camera_observed_fps": cam.get("capture_fps_observed"),
            "screen_width": screen["width"], "screen_height": screen["height"],
            "recording_seconds": round((eye_times.max() - eye_times.min()) / 1000, 3),
            "effective_logged_fps": round((len(eye) - 1) / ((eye_times.max() - eye_times.min()) / 1000), 3),
            "active_frames": len(active),
            "active_on_screen_pct": percent(active.gaze_status.eq("on_screen")),
            "active_face_detected_pct": percent(truth(active, "face_detected")),
            "active_outside_screen_pct": percent(active.gaze_status.eq("gaze_outside_screen")),
            "baseline_rmssd_ms": baseline.rmssd_ms, "baseline_coverage": baseline.coverage,
            "baseline_seconds": baseline.window_seconds,
            "hrv_task_windows": len(task), "hrv_ready_windows": int(task.status.eq("ready").sum()),
            "hrv_ready_pct": percent(task.status.eq("ready")),
            "hrv_task_status_counts": json.dumps(task.status.value_counts().to_dict()),
            "pure_active_windows": int(pure.sum()),
            "ready_pure_active_windows": int((pure & task.status.eq("ready")).sum()),
            "timeline_exists": not timeline.empty, "timeline_rows": len(timeline),
            "report_status": report.get("study_session_status"),
            "ledger_study_ended": bool(ledger.event_type.eq("study_ended").any()),
            "report_analyzed_at": report.get("analyzed_at"),
            "last_event_wib": ledger.iloc[-1].event_time_wib,
            "eye_minus_last_rr_seconds": round((eye_times.max() - last_rr_ms) / 1000, 3) if last_rr_ms else None,
            "eye_crops_referenced": int(eye.eye_crop_file.ne("").sum()),
            "eye_crop_directory_exists": (directory / "eye_crops").is_dir(),
            **eye_metrics(eye, screen),
        }
        summaries.append(summary)
        check = {"subject_session": name,
                 "eye_timestamp_regressions": int((np.diff(eye_times) < 0).sum()),
                 "eye_timestamp_duplicates": int(pd.Series(eye_times).duplicated().sum()),
                 "capture_id_duplicates": int(eye.capture_frame_id.duplicated().sum()),
                 "event_id_duplicates": int(ledger.event_id.duplicated().sum()),
                 "event_session_ids_consistent": len(stored_ids) == 1,
                 "eye_stored_phase_vs_final_ledger_mismatch": int(eye.study_phase.ne(eye.ledger_study_phase).sum()),
                 "eye_stored_difficulty_vs_final_ledger_mismatch": int(eye.difficulty_level.ne(eye.ledger_difficulty_level).sum())}
        mismatches = eye[eye.study_phase.ne(eye.ledger_study_phase)]
        check["phase_mismatch_pairs"] = json.dumps({f"{old}->{new}": len(group)
            for (old, new), group in mismatches.groupby(["study_phase", "ledger_study_phase"])})
        if len(timeline):
            t_start, t_end = stamps(timeline.window_start), stamps(timeline.window_end)
            # Match current fusion's inclusive endpoints exactly.
            expected_counts = np.searchsorted(eye_times, t_end, side="right") - np.searchsorted(eye_times, t_start, side="left")
            check["timeline_eye_count_mismatches"] = int((numeric(timeline, "eye_frames_count").to_numpy() != expected_counts).sum())
            check["timeline_task_hrv_row_count_matches"] = len(task) == len(timeline)
            if len(task) == len(timeline):
                check["timeline_hrv_status_mismatches"] = int((timeline.hrv_status.to_numpy() != task.status.to_numpy()).sum())
                check["timeline_rmssd_mismatches"] = int((~np.isclose(numeric(timeline, "rmssd_ms"), numeric(task, "rmssd_ms"), equal_nan=True)).sum())
            check["classified_windows_with_unready_hrv"] = int((timeline.hrv_status.ne("ready") & timeline.cognitive_state.ne("")).sum())
            check["classified_windows_with_onscreen_under_45pct"] = int((numeric(timeline, "on_screen_ratio") < .45).sum())
        checks.append(check)
        print(f"Audited {name}: {len(eye):,} frames, {len(task)} HRV windows", flush=True)

    for filename, rows in [("subject_summary.csv", summaries), ("eye_quality_by_phase.csv", phases),
                           ("task_sets.csv", sets), ("validation_targets.csv", targets),
                           ("hrv_quality_by_phase.csv", hrv_levels), ("integrity_checks.csv", checks),
                           ("rr_packet_gaps_over_5s.csv", gaps)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False)
    pd.DataFrame(inventory).to_csv(OUT / "session_inventory.csv", index=False)
    after = {str(p.relative_to(ROOT)): digest(p) for p in files}
    unchanged = before == after
    (OUT / "source_manifest.json").write_text(json.dumps({
        "source_files_unchanged": unchanged, "source_file_count": len(before),
        "sha256": before}, indent=2), encoding="utf-8")
    assert unchanged, "Source files changed during the audit"
    print(f"Verified {len(before)} source files unchanged.")


if __name__ == "__main__":
    main()
