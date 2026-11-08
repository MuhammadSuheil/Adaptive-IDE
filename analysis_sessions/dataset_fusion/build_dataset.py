"""Read-only, window-aligned feature fusion for the archived coding study."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis_sessions.session_inventory import participant_group, select_session_directories
from fuse_and_analyze import calibration_ready_ms, interval_segments, summarize_segments

DEFAULT_CONFIG = Path(__file__).with_name("config.json")
LEVELS = ("easy", "medium", "hard")
TIERS = ("good", "acceptable", "poor")
DATASETS = {
    "good_only": ("good",),
    "good_acceptable": ("good", "acceptable"),
    "all_data": TIERS,
}
# Participant eye-quality review is a separate admission gate from window metrics.
# Only all_data admits people outside the two configured name lists.
DATASET_REVIEW_GROUPS = {
    "good_only": ("proposed_good",),
    "good_acceptable": ("proposed_good", "questionable"),
    "all_data": None,
}
STATES = ("Fixation", "Saccade", "Transition", "Unclassified", "other")
MEASUREMENTS = {
    "ear_left": ("ratio", "Left eye aspect ratio; face present and value in [0, 1]."),
    "ear_right": ("ratio", "Right eye aspect ratio; face present and value in [0, 1]."),
    "iris_size_estimate": ("eye_width_ratio", "Landmark iris-size estimate, not pupil diameter; valid, nonblinking frames only."),
    "iris_size_delta": ("relative_change", "Recorded iris-size change; requires a ready iris baseline."),
    "head_pitch": ("degrees", "Recorded head-pitch estimate; face present."),
    "head_yaw": ("degrees", "Recorded head-yaw estimate; face present."),
    "gaze_x_norm": ("screen_width", "Smoothed gaze x / screen width; valid finite gaze, including outside-screen/clipped estimates."),
    "gaze_y_norm": ("screen_height", "Smoothed gaze y / screen height; valid finite gaze, including outside-screen/clipped estimates."),
}
FEATURE_INFO = {
    "rmssd_ms": ("ms", "Recorded time-domain HRV; unavailable unless status is ready and both HRV measures are finite and nonnegative."),
    "sdnn_ms": ("ms", "Recorded time-domain HRV; not a cognitive-state label."),
    "rmssd_baseline_ratio": ("ratio", "Window RMSSD / positive ready baseline RMSSD with sufficient baseline coverage."),
    "blink_count": ("events", "Increments in the session blink counter at reopening timestamps; blank if a counter discontinuity is observed."),
    "blink_rate_per_minute": ("events/minute", "Blink count * 60 / full window duration; partial recordings can undercount, see coverage."),
}
for _name, (_unit, _description) in MEASUREMENTS.items():
    for _stat in ("mean", "std"):
        FEATURE_INFO[f"{_name}_{_stat}"] = (_unit, f"{_description} Window {_stat}; std uses ddof=1.")
for _state in STATES:
    FEATURE_INFO[f"eye_state_{_state.lower()}_ratio"] = (
        "ratio", f"Frame proportion of algorithmic {_state} Eye_State among valid-gaze frames; not an independently validated eye event."
    )
FEATURE_COLUMNS = list(FEATURE_INFO)
META_INFO = {
    "window_id": "Unique source session ID and source window boundaries, separated by |.",
    "participant_id": "Participant grouping key; both Ramadhan recordings use Ramadhan.",
    "session_id": "Original sensor session ID, independent of archive folder name.",
    "session_folder": "Selected archive directory name.",
    "team_review_group": "Team eye-quality cohort, matched case-insensitively. proposed_good permits good_only/good_acceptable; questionable permits good_acceptable; unreviewed permits all_data only. Window checks still apply.",
    "participant_note": "Team-provided acquisition context, not a sensor-derived conclusion.",
    "repeated_task": "True for session/difficulty pairs marked as repeated in config.",
    "window_start": "Original timezone-aware HRV window start; excluded eye endpoint.",
    "window_end": "Original timezone-aware HRV window end; included eye endpoint.",
    "window_seconds": "Window duration in seconds; v1 requires 25 seconds.",
    "study_phase": "Final-ledger phase; retained dataset rows are entirely active_task.",
    "difficulty_level": "TARGET: easy/medium/hard task condition; not independently measured experienced cognitive load.",
    "puzzle_set": "Ledger set provenance; mixed if the interval crosses sets.",
    "puzzle_id": "Ledger puzzle provenance; mixed if the interval crosses questions.",
    "window_phase_mix": "Ledger segment provenance; not a predictor.",
    "hrv_status": "Recorded readiness status; unready numbers are masked.",
    "hrv_reason": "Recorded HRV readiness explanation.",
    "hrv_coverage": "Recorded accepted-RR duration / window duration.",
    "hrv_rr_count": "Recorded usable RR count.",
    "hrv_rr_pairs": "Recorded usable adjacent-RR pair count.",
    "hrv_data_age_seconds": "Recorded age of the latest usable RR interval.",
    "hrv_usable": "Ready status and finite, nonnegative RMSSD/SDNN; required for good/acceptable tiers.",
    "baseline_rmssd_ms": "Positive ready session baseline, blank when coverage is below configured minimum.",
    "baseline_coverage": "Recorded baseline accepted-RR coverage, irrespective of baseline eligibility.",
    "baseline_usable": "Whether baseline qualifies for the ratio feature; not required for raw-HRV eligibility.",
    "eye_frames_count": "Observed frames in (start, end].",
    "eye_temporal_coverage": "Fraction of 100-ms bins occupied by at least one eye row, using right-closed bins anchored at window start.",
    "face_ear_ratio": "Fraction of observed rows with a face and both EAR values in [0, 1].",
    "valid_gaze_ratio": "Fraction of observed rows with a face, valid gaze status and finite raw/smoothed gaze coordinates.",
    "iris_available_ratio": "Fraction of observed rows with a face, valid nonblinking iris-size estimate in eye_width_ratio mode.",
    "clipping_ratio_valid": "Fraction of valid-gaze rows at a recorded raw-coordinate clamp, using configured margin; blank without valid gaze.",
    "on_screen_ratio": "Recorded on_screen status fraction; not treated as an accuracy measure.",
    "blink_counter_discontinuities": "Rows with a missing, negative, fractional or decreasing blink counter, including unavailable predecessors.",
    "observed_eye_fps": "Observed window row count / duration; not nominal device FPS.",
    "capture_fps_mean": "Mean logged capture_fps in the window, when available.",
    "screen_width_px": "Screen width saved with calibration; used to normalize x.",
    "screen_height_px": "Screen height saved with calibration; used to normalize y.",
    "camera_width_px": "Actual camera width reported in session summary, not requested width.",
    "camera_height_px": "Actual camera height reported in session summary, not requested height.",
    "camera_reported_fps": "Driver-reported FPS; not proof of actual capture rate.",
    "camera_observed_capture_fps": "Observed capture rate saved in session summary.",
    "gaze_validation_passed": "Original last held-out validation decision; blank if absent. Not an input to window tier assignment.",
    "gaze_validation_median_px": "Last held-out median error of point mean predictions; not a frame-level error quantile.",
    "gaze_validation_p95_px": "Last held-out P95 error of point mean predictions; not a frame-level error quantile.",
    "gaze_validation_max_corner_px": "Last held-out maximum corner error.",
    "quality_tier": "Operational completeness/clipping tier; good does not certify spatial accuracy or mental-state validity.",
    "good_failed_criteria": "Semicolon-delimited failed good rules; empty when good.",
    "acceptable_failed_criteria": "Semicolon-delimited failed acceptable rules; empty when acceptable rules pass.",
}
DATA_COLUMNS = list(META_INFO) + FEATURE_COLUMNS


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig")) if Path(path).exists() else {}


def finite(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else np.nan
    except (ValueError, TypeError):
        return np.nan


def number(frame, column):
    return pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce").replace([np.inf, -np.inf], np.nan)


def truth(frame, column):
    return frame.get(column, pd.Series(False, index=frame.index)).astype(str).str.lower().isin(("true", "1"))


def lookup(mapping, name, default=None):
    return next((value for key, value in mapping.items() if key.casefold() == name.casefold()), default)


def load_config(path=DEFAULT_CONFIG):
    config = read_json(path)
    expected = {"schema_version", "window_seconds", "step_seconds", "eye_coverage_bin_ms", "gaze_clip_margin_fraction",
                "baseline_min_coverage", "quality", "participant_review_groups", "participant_notes", "repeated_difficulties_by_session"}
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object.")
    if set(config) != expected:
        raise ValueError(f"Config keys must be {sorted(expected)}; missing={sorted(expected-set(config))}, unknown={sorted(set(config)-expected)}")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1 or config["window_seconds"] != 25 or config["step_seconds"] != 5:
        raise ValueError("v1 reads recorded 25-second / 5-second-step windows; it does not recompute RR windows.")
    def numeric(value):
        return type(value) in (int, float) and math.isfinite(value)

    if not numeric(config["eye_coverage_bin_ms"]):
        raise ValueError("eye_coverage_bin_ms must be a finite number.")
    bin_ms = finite(config["eye_coverage_bin_ms"])
    if not math.isfinite(bin_ms) or bin_ms <= 0 or not math.isclose(25000 / bin_ms, round(25000 / bin_ms)):
        raise ValueError("eye_coverage_bin_ms must be positive and divide 25000 exactly.")
    if any(not numeric(config[key]) or not 0 <= config[key] <= 1 for key in ("gaze_clip_margin_fraction", "baseline_min_coverage")):
        raise ValueError("Clip margin and baseline coverage must be finite fractions in [0, 1].")
    expected_rules = {"min_eye_temporal_coverage", "min_face_ear_ratio", "min_valid_gaze_ratio", "min_iris_available_ratio", "max_clipping_ratio"}
    if not isinstance(config["quality"], dict) or set(config["quality"]) != {"good", "acceptable"}:
        raise ValueError("quality must contain exactly good and acceptable.")
    for tier, rules in config["quality"].items():
        if not isinstance(rules, dict) or set(rules) != expected_rules or any(not numeric(value) or not 0 <= value <= 1 for value in rules.values()):
            raise ValueError(f"Invalid quality rules for {tier}: expected fractional {sorted(expected_rules)}")
    for key in expected_rules:
        good, acceptable = (config["quality"][tier][key] for tier in ("good", "acceptable"))
        if (key.startswith("min_") and good < acceptable) or (key.startswith("max_") and good > acceptable):
            raise ValueError("Good rules must be at least as strict as acceptable rules.")
    groups = config["participant_review_groups"]
    if not isinstance(groups, dict) or set(groups) != {"proposed_good", "questionable"} or any(not isinstance(group, list) for group in groups.values()):
        raise ValueError("Review groups must be proposed_good and questionable.")
    names = [name.casefold() for group in groups.values() for name in group if isinstance(name, str) and name.strip()]
    if len(names) != sum(len(group) for group in groups.values()) or len(names) != len(set(names)):
        raise ValueError("Review names must be nonempty strings and unique across groups (case-insensitive).")
    if not isinstance(config["participant_notes"], dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in config["participant_notes"].items()):
        raise ValueError("participant_notes must map names to text notes.")
    repeated = config["repeated_difficulties_by_session"]
    if not isinstance(repeated, dict) or any(not isinstance(name, str) or not isinstance(levels, list) or any(level not in LEVELS for level in levels) for name, levels in repeated.items()):
        raise ValueError("Repeated difficulties must use easy/medium/hard.")
    return config


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def snapshot(directory):
    return {str(path.relative_to(directory)).replace("\\", "/"): digest(path)
            for path in sorted(Path(directory).rglob("*")) if path.is_file()}


def epoch_ms(value):
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"Expected a timezone-aware timestamp: {value!r}")
    return timestamp.timestamp() * 1000


def prepare_eye(eye, screen, config):
    eye = eye.copy()
    eye["timestamp_ms"] = number(eye, "timestamp_ms")
    if eye.timestamp_ms.isna().any() or (eye.timestamp_ms.diff().dropna() <= 0).any():
        raise ValueError("Eye timestamps must be finite, unique and strictly increasing.")
    face = truth(eye, "face_detected")
    left, right = number(eye, "EAR_Left"), number(eye, "EAR_Right")
    eye["face_ear"] = face & left.between(0, 1) & right.between(0, 1)
    eye["ear_left"] = left.where(face & left.between(0, 1))
    eye["ear_right"] = right.where(face & right.between(0, 1))
    raw_x, raw_y = number(eye, "gaze_x_raw"), number(eye, "gaze_y_raw")
    smooth_x, smooth_y = number(eye, "gaze_x_smooth"), number(eye, "gaze_y_smooth")
    status = eye.get("gaze_status", pd.Series("", index=eye.index))
    valid = face & status.isin(("on_screen", "gaze_outside_screen")) & raw_x.notna() & raw_y.notna() & smooth_x.notna() & smooth_y.notna()
    eye["valid_gaze"] = valid
    eye["on_screen"] = status.eq("on_screen")
    eye["gaze_x_norm"] = (smooth_x / screen["width"]).where(valid)
    eye["gaze_y_norm"] = (smooth_y / screen["height"]).where(valid)
    margin = config["gaze_clip_margin_fraction"]
    eye["clipped"] = valid & (np.isclose(raw_x, -margin * screen["width"]) | np.isclose(raw_x, (1+margin) * screen["width"])
                              | np.isclose(raw_y, -margin * screen["height"]) | np.isclose(raw_y, (1+margin) * screen["height"]))
    size = number(eye, "iris_size")
    mode = eye.get("iris_size_mode", pd.Series("", index=eye.index))
    iris = face & truth(eye, "iris_valid") & ~truth(eye, "is_blinking") & size.gt(0) & mode.eq("eye_width_ratio")
    eye["iris_available"] = iris
    eye["iris_size_estimate"] = size.where(iris)
    eye["iris_size_delta"] = number(eye, "iris_size_delta").where(iris & truth(eye, "iris_baseline_ready"))
    for column in ("head_pitch", "head_yaw"):
        eye[column] = number(eye, column).where(face)
    counter = number(eye, "total_blinks")
    valid_counter = counter.ge(0) & counter.mod(1).eq(0)
    increment = counter.diff()
    counter_ok = valid_counter & valid_counter.shift(fill_value=False) & increment.ge(0)
    # The first observed count is not a known increment. A window containing it has an unknown count.
    eye["blink_increment"] = increment.where(counter_ok)
    eye["blink_counter_bad"] = ~counter_ok
    eye["capture_fps"] = number(eye, "capture_fps").where(lambda s: s.gt(0))
    eye["Eye_State"] = eye.get("Eye_State", pd.Series("", index=eye.index))
    return eye


def aggregate_eye(eye, start, end, config):
    times = eye.timestamp_ms.to_numpy()
    lo, hi = np.searchsorted(times, [start, end], side="right")
    frame = eye.iloc[lo:hi]
    count = len(frame)
    bin_ms = config["eye_coverage_bin_ms"]
    total_bins = round((end-start) / bin_ms)
    # (start, start+bin], ..., (end-bin, end]; exactly-at-end samples occupy the final bin.
    bins = np.ceil((frame.timestamp_ms.to_numpy()-start) / bin_ms).astype(int)-1
    coverage = len(np.unique(np.clip(bins, 0, total_bins-1))) / total_bins if count else 0.
    result = {
        "eye_frames_count": count, "eye_temporal_coverage": coverage,
        "face_ear_ratio": float(frame.face_ear.mean()) if count else np.nan,
        "valid_gaze_ratio": float(frame.valid_gaze.mean()) if count else np.nan,
        "iris_available_ratio": float(frame.iris_available.mean()) if count else np.nan,
        "clipping_ratio_valid": float(frame.loc[frame.valid_gaze, "clipped"].mean()),
        "on_screen_ratio": float(frame.on_screen.mean()) if count else np.nan,
        "blink_counter_discontinuities": int(frame.blink_counter_bad.sum()),
        "observed_eye_fps": count * 1000 / (end-start),
        "capture_fps_mean": frame.capture_fps.mean(),
    }
    blinks = frame.blink_increment.sum() if count and not frame.blink_counter_bad.any() else np.nan
    result["blink_count"] = blinks
    result["blink_rate_per_minute"] = blinks * 60000 / (end-start)
    for column in MEASUREMENTS:
        result[f"{column}_mean"] = frame[column].mean()
        result[f"{column}_std"] = frame[column].std(ddof=1)
    states = frame.loc[frame.valid_gaze, "Eye_State"]
    for state in STATES:
        match = states.eq(state) if state != "other" else ~states.isin(STATES[:-1])
        result[f"eye_state_{state.lower()}_ratio"] = float(match.mean()) if len(states) else np.nan
    return result


def quality_tier(row, config):
    failed = {}
    for tier in ("good", "acceptable"):
        reasons = [] if row["hrv_usable"] else ["hrv_unusable"]
        for key, threshold in config["quality"][tier].items():
            column = key.removeprefix("min_").removeprefix("max_")
            column = "clipping_ratio_valid" if column == "clipping_ratio" else column
            value = finite(row.get(column))
            passes = math.isfinite(value) and (value >= threshold if key.startswith("min_") else value <= threshold)
            if not passes:
                reasons.append(key)
        failed[f"{tier}_failed_criteria"] = ";".join(reasons)
    tier = "good" if not failed["good_failed_criteria"] else "acceptable" if not failed["acceptable_failed_criteria"] else "poor"
    return {"quality_tier": tier, **failed}


def build_session(directory, config):
    directory = Path(directory)
    eye = pd.read_csv(directory / "eye_tracking.csv", keep_default_na=False, low_memory=False)
    hrv = pd.read_csv(directory / "hrv.csv", keep_default_na=False)
    if eye.empty or hrv.empty:
        raise ValueError(f"{directory.name}: source eye/HRV CSV is empty.")
    ids = set(eye.session_id.astype(str)) | set(hrv.session_id.astype(str))
    if len(ids) != 1 or not next(iter(ids)):
        raise ValueError(f"{directory.name}: sensor session IDs disagree or are empty: {ids}")
    session_id = next(iter(ids))
    participant = participant_group(directory.name)
    calibration = read_json(directory / "eye_summary_calibration.json")
    summary = read_json(directory / "eye_summary.json")
    screen = calibration.get("config", {}).get("screen") or summary.get("config", {}).get("screen", {})
    if not all(finite(screen.get(axis)) > 0 for axis in ("width", "height")):
        raise ValueError(f"{directory.name}: missing/invalid recorded screen dimensions.")
    eye = prepare_eye(eye, screen, config)
    events_path = directory / "task_events.csv"
    events = pd.read_csv(events_path, keep_default_na=False) if events_path.exists() else pd.DataFrame()
    if not events.empty:
        events = events[events.session_id.eq(session_id)].copy()
        events["event_time_ms"] = number(events, "event_time_ms")
        if events.event_time_ms.isna().any():
            raise ValueError(f"{directory.name}: invalid task-event timestamp.")
        events = events.sort_values("event_time_ms", kind="stable")
    event_rows = events.to_dict("records")
    ready_ms = calibration_ready_ms(directory)
    baseline_rows = hrv[hrv.phase.eq("baseline")]
    baseline = baseline_rows.iloc[-1] if len(baseline_rows) else {}
    baseline_rmssd, baseline_coverage = finite(baseline.get("rmssd_ms")), finite(baseline.get("coverage"))
    baseline_usable = baseline.get("status") == "ready" and baseline_rmssd > 0 and baseline_coverage >= config["baseline_min_coverage"]
    validation = [attempt for attempt in calibration.get("attempts", []) if attempt.get("phase") == "held_out_validation"]
    validation = validation[-1] if validation else {}
    camera = summary.get("summary", {}).get("camera", {})
    group = next((name for name, members in config["participant_review_groups"].items() if participant.casefold() in {v.casefold() for v in members}), "unreviewed")
    metadata = {
        "participant_id": participant, "session_id": session_id, "session_folder": directory.name,
        "team_review_group": group, "participant_note": lookup(config["participant_notes"], participant, ""),
        "baseline_rmssd_ms": baseline_rmssd if baseline_usable else np.nan,
        "baseline_coverage": baseline_coverage, "baseline_usable": bool(baseline_usable),
        "screen_width_px": screen["width"], "screen_height_px": screen["height"],
        "camera_width_px": finite(camera.get("width")), "camera_height_px": finite(camera.get("height")),
        "camera_reported_fps": finite(camera.get("fps_reported")),
        "camera_observed_capture_fps": finite(camera.get("capture_fps_observed")),
        "gaze_validation_passed": validation.get("passed_thresholds", ""),
        "gaze_validation_median_px": finite(validation.get("mean_target_error_median_px")),
        "gaze_validation_p95_px": finite(validation.get("mean_target_error_p95_px")),
        "gaze_validation_max_corner_px": finite(validation.get("max_corner_error_px")),
    }
    rows, excluded = [], []
    windows = hrv[hrv.phase.eq("task")]
    previous_end = None
    for source in windows.to_dict("records"):
        start, end = epoch_ms(source["window_start"]), epoch_ms(source["window_end"])
        duration = (end-start)/1000
        if not math.isclose(duration, config["window_seconds"], abs_tol=.001):
            raise ValueError(f"{directory.name}: unexpected window duration {duration}; cannot relabel it as 25 s.")
        if not math.isclose(finite(source.get("window_seconds")), duration, abs_tol=.001):
            raise ValueError(f"{directory.name}: recorded duration disagrees with timestamps.")
        if previous_end is not None:
            steps = (end-previous_end)/(1000*config["step_seconds"])
            if steps <= 0 or not math.isclose(steps, round(steps), abs_tol=1e-5):
                raise ValueError(f"{directory.name}: duplicated, reordered, or off-grid HRV windows.")
        previous_end = end
        window_id = f"{session_id}|{source['window_start']}|{source['window_end']}"
        context = summarize_segments(interval_segments(start, end, event_rows, ready_ms))
        if context["study_phase"] != "active_task" or context["difficulty_level"] not in LEVELS:
            excluded.append({**metadata, "window_id": window_id, "window_start": source["window_start"], "window_end": source["window_end"],
                             **context, "exclusion_reason": "not_pure_active_difficulty" if event_rows else "no_matching_task_events"})
            continue
        rmssd, sdnn = finite(source.get("rmssd_ms")), finite(source.get("sdnn_ms"))
        usable = source.get("status") == "ready" and rmssd >= 0 and sdnn >= 0
        row = {
            **metadata, **context, "window_id": window_id,
            "window_start": source["window_start"], "window_end": source["window_end"], "window_seconds": duration,
            "repeated_task": context["difficulty_level"] in lookup(config["repeated_difficulties_by_session"], directory.name, []),
            "hrv_status": source.get("status", ""), "hrv_reason": source.get("reason", ""),
            "hrv_coverage": finite(source.get("coverage")), "hrv_rr_count": finite(source.get("rr_count")),
            "hrv_rr_pairs": finite(source.get("rr_pairs")), "hrv_data_age_seconds": finite(source.get("data_age_seconds")),
            "hrv_usable": bool(usable), "rmssd_ms": rmssd if usable else np.nan, "sdnn_ms": sdnn if usable else np.nan,
            "rmssd_baseline_ratio": rmssd / baseline_rmssd if usable and baseline_usable else np.nan,
            **aggregate_eye(eye, start, end, config),
        }
        row.update(quality_tier(row, config))
        rows.append(row)
    counts = {tier: sum(row["quality_tier"] == tier for row in rows) for tier in TIERS}
    report = {**metadata, **{f"{tier}_windows": count for tier, count in counts.items()},
              "good_only_windows": counts["good"] if group in DATASET_REVIEW_GROUPS["good_only"] else 0,
              "good_acceptable_windows": counts["good"] + counts["acceptable"] if group in DATASET_REVIEW_GROUPS["good_acceptable"] else 0,
              "all_windows": len(rows), "hrv_usable_windows": sum(row["hrv_usable"] for row in rows),
              "excluded_windows": len(excluded), "source_task_windows": len(windows)}
    return rows, excluded, report


def dataset_views(frame):
    if frame.window_id.duplicated().any():
        raise ValueError("Duplicate window IDs; refuse to double-count acquisitions.")
    views = {}
    for name, tiers in DATASETS.items():
        eligible = frame.quality_tier.isin(tiers)
        groups = DATASET_REVIEW_GROUPS[name]
        if groups is not None:
            eligible &= frame.team_review_group.isin(groups)
        views[name] = frame[eligible].copy()
    return views


def evaluation_contract(frame, views, participants):
    folds, tests = [], []
    for index, participant in enumerate(sorted(set(participants), key=str.casefold), 1):
        fold_id = f"F{index:03d}"
        test = views["good_only"][views["good_only"].participant_id.eq(participant)]
        fold = {"fold_id": fold_id, "held_out_participant_id": participant, "test_windows": len(test)}
        reasons = [] if len(test) else ["no_good_test_windows"]
        for level in LEVELS:
            fold[f"test_{level}_windows"] = int(test.difficulty_level.eq(level).sum())
        for name, view in views.items():
            train = view[view.participant_id.ne(participant)]
            fold[f"{name}_train_windows"] = len(train)
            missing = [level for level in LEVELS if not train.difficulty_level.eq(level).any()]
            fold[f"{name}_missing_train_classes"] = ";".join(missing)
            if missing:
                reasons.append(f"{name}_missing_training_classes")
        fold["comparison_feasible"] = not reasons
        fold["reason"] = ";".join(reasons)
        folds.append(fold)
        for row in test[["window_id", "participant_id", "session_id", "difficulty_level"]].to_dict("records"):
            tests.append({"fold_id": fold_id, **row})
    return pd.DataFrame(folds), pd.DataFrame(tests, columns=["fold_id", "window_id", "participant_id", "session_id", "difficulty_level"])


def write_artifacts(destination, frame, excluded, reports, inventory, config):
    views = dataset_views(frame)
    for name, view in views.items():
        view.to_csv(destination / f"{name}.csv", index=False)
    pd.DataFrame(inventory).to_csv(destination / "session_inventory.csv", index=False)
    report = pd.DataFrame(reports)
    report.to_csv(destination / "session_quality_summary.csv", index=False)
    excluded_columns = ["participant_id", "session_id", "session_folder", "window_id", "window_start", "window_end", "study_phase", "difficulty_level", "puzzle_set", "puzzle_id", "window_phase_mix", "exclusion_reason"]
    pd.DataFrame(excluded, columns=excluded_columns).to_csv(destination / "excluded_windows.csv", index=False)
    index = pd.MultiIndex.from_product([report.session_folder.tolist(), LEVELS, TIERS], names=["session_folder", "difficulty_level", "quality_tier"])
    counts = frame.groupby(["session_folder", "difficulty_level", "quality_tier"]).size().reindex(index, fill_value=0).rename("windows").reset_index()
    counts = counts.merge(report[["session_folder", "participant_id", "team_review_group"]], on="session_folder", validate="many_to_one")
    counts.to_csv(destination / "counts_by_session_difficulty_tier.csv", index=False)
    folds, tests = evaluation_contract(frame, views, report.participant_id.tolist())
    folds.to_csv(destination / "evaluation_folds.csv", index=False)
    tests.to_csv(destination / "evaluation_test_windows.csv", index=False)
    dictionary = [{"column": column, "role": "target" if column == "difficulty_level" else "metadata", "unit": "", "description": description}
                  for column, description in META_INFO.items()]
    dictionary += [{"column": column, "role": "feature", "unit": unit, "description": description} for column, (unit, description) in FEATURE_INFO.items()]
    for row in dictionary:
        if row["column"] == "eye_temporal_coverage":
            row["description"] = row["description"].replace("100-ms", f"{config['eye_coverage_bin_ms']}-ms")
    pd.DataFrame(dictionary).to_csv(destination / "data_dictionary.csv", index=False)
    (destination / "feature_columns.json").write_text(json.dumps({"target": "difficulty_level", "group": "participant_id", "features": FEATURE_COLUMNS}, indent=2)+"\n", encoding="utf-8")
    lines = ["# Fused dataset quality report", "", "Target: easy/medium/hard task condition, a proxy for cognitive demand. No independent cognitive-state labels were collected.", "",
             "Participant eye-quality admission uses the team's configured name lists: good_only admits proposed_good; good_acceptable admits proposed_good and questionable; everyone else appears only in all_data. Passing window checks never promotes an unlisted participant into a curated dataset.", "",
             "Window tiers describe completeness and clipping, not spatial accuracy. Original gaze validation is reported separately and is not used to assign tiers.", "",
             "All features are exported together. Iris estimates are not measured pupil diameter; Eye_State is an algorithmic estimate. Missing values remain blank.", "",
             "| Dataset | Windows | Participants |", "|---|---:|---:|"]
    lines += [f"| {name}.csv | {len(view)} | {view.participant_id.nunique()} |" for name, view in views.items()]
    lines += ["", "## Session review", "", "Counts below are actual dataset membership after both the team's eye-quality admission gate and the window checks. The CSV summary also retains raw window-tier counts for diagnostics.", "",
              "| Session | Eye review group | good_only rows | good_acceptable rows | all_data rows | Original gaze validation |", "|---|---|---:|---:|---:|---|"]
    for row in reports:
        validation = "pass" if row["gaze_validation_passed"] is True else "fail" if row["gaze_validation_passed"] is False else "unavailable"
        lines.append(f"| {row['session_folder']} | {row['team_review_group']} | {row['good_only_windows']} | {row['good_acceptable_windows']} | {row['all_windows']} | {validation} |")
    lines += ["", "## Comparison contract", "", "Use evaluation_folds.csv and the identical evaluation_test_windows.csv for every training variant. For each feasible fold, test only its listed good windows; train the chosen dataset on participant_id != held_out_participant_id. Both Ramadhan recordings must be excluded together. Zero-test and missing-training-class folds are explicitly marked infeasible.", "",
              "Fit imputers, scalers, feature selection and model tuning on training participants only. Handle entirely missing features explicitly; do not treat blanks as zeros. Do not feed metadata, task IDs or heuristic cognitive labels into the model. Report macro-F1, balanced accuracy, per-class recall, confusion matrices, and participant-level results in subsequent training work.", "",
              "Training variants differ in both quality and quantity; scores alone cannot separate those effects. Difficulty may also be confounded with task order, experience, and device conditions. Thousands of overlapping windows are not thousands of independent participants.", "",
              "## Rules used", "", "```json", json.dumps(config, indent=2), "```", ""]
    (destination / "QUALITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return {name: {"windows": len(view), "participants": int(view.participant_id.nunique())} for name, view in views.items()}


def publish_artifact(source, destination):
    """Publish bytes atomically, then inherit the destination folder's Windows ACL.

    Windows preserves the private TemporaryDirectory ACL across os.replace.
    Reset only this generated file to its destination parent's normal permissions,
    otherwise an export made by the sandbox user is unreadable in desktop Excel.
    """
    os.replace(source, destination)
    if os.name == "nt":
        subprocess.run(["icacls", str(destination), "/reset", "/q"], check=True, capture_output=True)


def build_dataset(input_dir, output_dir, config_path=DEFAULT_CONFIG, subjects=None):
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    config_path = Path(config_path).resolve()
    if output_dir.is_relative_to(input_dir) or input_dir.is_relative_to(output_dir) or Path(__file__).resolve().is_relative_to(output_dir):
        raise ValueError("Output must be separate from source sessions and must not contain the builder itself.")
    config = load_config(config_path)
    sources_before = snapshot(input_dir)
    directories, inventory = select_session_directories(input_dir, subjects)
    rows, excluded, reports = [], [], []
    for directory in directories:
        session_rows, session_excluded, report = build_session(directory, config)
        rows.extend(session_rows)
        excluded.extend(session_excluded)
        reports.append(report)
        print(f"{directory.name}: good_only={report['good_only_windows']}, good_acceptable={report['good_acceptable_windows']}, all={report['all_windows']}, excluded={report['excluded_windows']}", flush=True)
    frame = pd.DataFrame(rows, columns=DATA_COLUMNS)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    if snapshot(input_dir) != sources_before:
        raise RuntimeError("Source files changed during the build; outputs were not published.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dataset-build-", dir=output_dir.parent) as staging:
        destination = Path(staging)
        counts = write_artifacts(destination, frame, excluded, reports, inventory, config)
        code_paths = [Path(__file__).resolve(), ROOT / "analysis_sessions/session_inventory.py", ROOT / "fuse_and_analyze.py"]
        manifest = {
            "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_directory": str(input_dir), "selected_folders": [p.name for p in directories],
            "configuration": config, "configuration_sha256": digest(config_path),
            "dataset_eye_review_groups": DATASET_REVIEW_GROUPS,
            "source_integrity_verified": True, "source_sha256": sources_before,
            "code_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in code_paths},
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
            "datasets": counts, "excluded_windows": len(excluded), "feature_columns": FEATURE_COLUMNS,
            "output_sha256": {p.name: digest(p) for p in sorted(destination.iterdir())},
        }
        # Recheck after staging too, so the published integrity claim covers the entire computation.
        if snapshot(input_dir) != sources_before:
            raise RuntimeError("Source files changed during export; outputs were not published.")
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(destination.iterdir(), key=lambda p: (p.name == "manifest.json", p.name)):
            publish_artifact(path, output_dir / path.name)
    print(json.dumps(counts, indent=2), flush=True)
    print(f"Source integrity verified ({len(sources_before)} files). Output: {output_dir}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "sessions/saved_session")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("output"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--subjects", nargs="+", help="Optional exact archive folder names; default includes every unique acquisition.")
    args = parser.parse_args()
    build_dataset(args.input, args.output, args.config, args.subjects)


if __name__ == "__main__":
    main()
