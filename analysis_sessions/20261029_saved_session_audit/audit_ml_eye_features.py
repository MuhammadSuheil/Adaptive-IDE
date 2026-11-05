"""Read-only review of eye features and label availability for ML experiments.

Writes quality summaries, not a model-ready labelled training set. Run from any
directory with Python, pandas and NumPy. Source files are verified unchanged.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from audit_saved_sessions import SOURCE, OUT, ROOT, digest, read_csv, read_json, numeric, truth, stamps
from analysis_sessions.session_inventory import select_session_directories, participant_group


def pct(mask):
    return round(float(np.mean(mask) * 100), 3) if len(mask) else None


def describe(group):
    face = truth(group, "face_detected")
    left, right = numeric(group, "EAR_Left"), numeric(group, "EAR_Right")
    ear_ok = face & np.isfinite(left) & np.isfinite(right)
    iris_ok = face & truth(group, "iris_valid") & np.isfinite(numeric(group, "iris_size_delta"))
    head_ok = face & np.isfinite(numeric(group, "head_pitch")) & np.isfinite(numeric(group, "head_yaw"))
    return {
        "frames": len(group), "face_present_pct": pct(face),
        "ear_pair_finite_with_face_pct": pct(ear_ok),
        "ear_left_median": left[ear_ok].median(), "ear_right_median": right[ear_ok].median(),
        "ear_left_p01": left[ear_ok].quantile(.01), "ear_left_p99": left[ear_ok].quantile(.99),
        "ear_right_p01": right[ear_ok].quantile(.01), "ear_right_p99": right[ear_ok].quantile(.99),
        "ear_above_one_with_face_pct": pct(face & ((left > 1) | (right > 1))),
        "head_pair_finite_with_face_pct": pct(head_ok),
        "head_pitch_std": numeric(group, "head_pitch")[head_ok].std(),
        "head_yaw_std": numeric(group, "head_yaw")[head_ok].std(),
        "iris_valid_delta_pct": pct(iris_ok),
        "iris_delta_median_valid": numeric(group, "iris_size_delta")[iris_ok].median(),
        "iris_delta_iqr_valid": (numeric(group, "iris_size_delta")[iris_ok].quantile(.75)
                                 - numeric(group, "iris_size_delta")[iris_ok].quantile(.25)),
        "iris_valid_and_gaze_outside_frames": int((iris_ok & group.gaze_status.eq("gaze_outside_screen")).sum()),
        "head_shifted_pct": pct(truth(group, "head_pose_shifted")),
        "blink_closed_frame_pct": pct(truth(group, "is_blinking")),
        "fixation_label_pct": pct(group.Eye_State.eq("Fixation")),
        "saccade_label_pct": pct(group.Eye_State.eq("Saccade")),
        "unclassified_eye_state_pct": pct(group.Eye_State.eq("Unclassified")),
    }


def main():
    original = [p for d in SOURCE.iterdir() if d.is_dir() for p in d.iterdir()
                if p.is_file() and p.suffix in (".csv", ".json")]
    attempts_path = ROOT / "codearrange-web/logs/saved_log/attempts.csv"
    if attempts_path.exists():
        original.append(attempts_path)
    before = {str(p.relative_to(ROOT)): digest(p) for p in original}
    attempt_log = read_csv(attempts_path)
    known_ids = set(attempt_log.attempt_id) if not attempt_log.empty else set()
    summaries, phases, windows, clipping_rows = [], [], [], []
    directories, inventory = select_session_directories(SOURCE)
    for directory in directories:
        eye = read_csv(directory / "eye_tracking.csv")
        hrv = read_csv(directory / "hrv.csv")
        events = read_csv(directory / "task_events.csv")
        calib = read_json(directory / "eye_summary_calibration.json")
        active = eye[eye.study_phase.eq("active_task")]
        submitted = events[events.event_type.eq("attempt_submitted")]
        matched = submitted.attempt_id.isin(known_ids)
        blinks = numeric(eye, "total_blinks")
        increments = blinks.diff().fillna(blinks.iloc[0])
        blink_times = numeric(eye, "timestamp_ms")[increments > 0].to_numpy()
        blink_weights = increments[increments > 0].to_numpy()
        summary = {
            "subject_session": directory.name,
            "participant_group": participant_group(directory.name),
            "all_frames": len(eye), "active_frames": len(active),
            **{"active_" + k: v for k, v in describe(active).items() if k != "frames"},
            "final_blink_count": blinks.iloc[-1],
            "blink_count_decreases": int((increments < 0).sum()),
            "blink_count_jumps_over_one": int((increments > 1).sum()),
            "head_position_gate_enabled": calib["config"]["head_positioning"]["enabled"],
            "blink_threshold": calib["config"]["blink"]["ear_threshold"],
            "submitted_events": len(submitted),
            "submitted_attempt_ids_in_available_log": int(matched.sum()),
            "numeric_rating_column_in_event_file": "rating" in events.columns,
        }
        summaries.append(summary)
        w, h = calib["config"]["screen"]["width"], calib["config"]["screen"]["height"]
        def clipped(column, size):
            value = numeric(active, column)
            return np.isclose(value, -.1*size) | np.isclose(value, 1.1*size)
        raw_clip = clipped("gaze_x_raw", w) | clipped("gaze_y_raw", h)
        smooth_both = clipped("gaze_x_smooth", w) & clipped("gaze_y_smooth", h)
        fixation = active.Eye_State.eq("Fixation").to_numpy()
        count = int(fixation.sum())
        clipping_rows.append({"subject_session": directory.name, "active_frames": len(active),
                              "fixation_label_frames": count,
                              "fixation_frames_with_raw_axis_at_limit": int((fixation & raw_clip).sum()),
                              "pct_of_fixation_frames_with_raw_axis_at_limit": 100*(fixation & raw_clip).sum()/count if count else None,
                              "fixation_frames_with_both_smoothed_axes_at_limits": int((fixation & smooth_both).sum())})
        for (phase, level), group in eye.groupby(["study_phase", "difficulty_level"]):
            phases.append({"subject_session": directory.name, "study_phase": phase,
                           "difficulty_level": level, **describe(group)})
        ts = numeric(eye, "timestamp_ms").to_numpy()
        task = hrv[hrv.phase.eq("task")]
        for (_, row), start, end in zip(task.iterrows(), stamps(task.window_start), stamps(task.window_end)):
            lo, hi = np.searchsorted(ts, [start, end], side="left")
            hi = np.searchsorted(ts, end, side="right")
            group = eye.iloc[lo:hi]
            count = blink_weights[(blink_times >= start) & (blink_times <= end)].sum()
            windows.append({"subject_session": directory.name,
                            "participant_group": summary["participant_group"],
                            "window_start": row.window_start, "window_end": row.window_end,
                            "study_phase": row.study_phase, "difficulty_level": row.difficulty_level,
                            "puzzle_set": row.puzzle_set, "puzzle_id": row.puzzle_id,
                            "hrv_status": row.status,
                            "detected_blink_increments_in_window": count,
                            "eye_timing_first_to_last_seconds": (ts[hi-1] - ts[lo]) / 1000 if hi > lo else 0,
                            **describe(group)})
        print(f"Reviewed eye features: {directory.name} ({len(eye):,} frames)", flush=True)
    pd.DataFrame(summaries).to_csv(OUT / "ml_eye_feature_summary.csv", index=False)
    pd.DataFrame(phases).to_csv(OUT / "ml_eye_features_by_phase.csv", index=False)
    pd.DataFrame(windows).to_csv(OUT / "ml_eye_window_quality.csv", index=False)
    pd.DataFrame(clipping_rows).to_csv(OUT / "gaze_eye_state_clipping_review.csv", index=False)
    assert before == {str(p.relative_to(ROOT)): digest(p) for p in original}
    (OUT / "ml_source_manifest.json").write_text(json.dumps({
        "source_files_unchanged": True, "source_file_count": len(before), "sha256": before}, indent=2), encoding="utf-8")
    print("All source files unchanged; quality exports are not a labelled training dataset.")


if __name__ == "__main__":
    main()
