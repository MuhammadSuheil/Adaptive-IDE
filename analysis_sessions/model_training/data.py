"""Read fused exports without reusing their task-difficulty target."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "analysis_sessions/dataset_fusion/output"
VARIANTS = ("good_only", "good_acceptable", "all_data")
LEVELS = ("Low", "Mid low", "Mid", "Mid high", "High")
ANCHORS = ("rmssd_baseline_ratio", "blink_rate_per_minute", "eye_state_fixation_ratio")
FAMILIES = {
    "HRV": ["rmssd_ms", "sdnn_ms", "rmssd_baseline_ratio"],
    "Blink": ["blink_count", "blink_rate_per_minute"],
    "EAR": ["ear_left_mean", "ear_left_std", "ear_right_mean", "ear_right_std"],
    "Iris geometry": ["iris_size_estimate_mean", "iris_size_estimate_std", "iris_size_delta_mean", "iris_size_delta_std"],
    "Gaze": ["gaze_x_norm_mean", "gaze_x_norm_std", "gaze_y_norm_mean", "gaze_y_norm_std"],
    "Eye state": ["eye_state_fixation_ratio", "eye_state_saccade_ratio", "eye_state_transition_ratio"],
    "Head context": ["head_pitch_mean", "head_pitch_std", "head_yaw_mean", "head_yaw_std"],
}
DIAGNOSTICS = ("eye_state_unclassified_ratio", "eye_state_other_ratio")
IDENTITY = ("window_id", "participant_id", "session_id")
NOTE = "Provisional relative HRV/eye estimate; not independently validated cognitive load."


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def numeric(frame, columns):
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"Missing sensor columns: {sorted(missing)}")
    values = frame.loc[:, columns].replace("", np.nan)
    result = values.apply(pd.to_numeric, errors="coerce")
    if (values.notna() & result.isna()).any().any():
        raise ValueError("Nonnumeric sensor values")
    return result.astype(float).replace([np.inf, -np.inf], np.nan)


def eligible(frame):
    """Missing/invalid anchors cannot define the provisional training target."""
    values = numeric(frame, list(ANCHORS))
    valid = values.notna().all(axis=1)
    valid &= values.rmssd_baseline_ratio.ge(0)
    valid &= values.blink_rate_per_minute.ge(0)
    valid &= values.eye_state_fixation_ratio.between(0, 1)
    return valid


def load_config(path=None):
    path = Path(path) if path else Path(__file__).with_name("config.json")
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    expected = {"schema_version", "primary_hrv_weight", "sensitivity_hrv_weight", "reference_eye_weights", "band_quantiles", "include_head_pose_context", "seed"}
    if set(config) != expected or config["schema_version"] != 1:
        raise ValueError("Unsupported scoring configuration")
    for key in ("primary_hrv_weight", "sensitivity_hrv_weight"):
        if type(config[key]) not in (int, float) or not np.isfinite(config[key]) or not 0 < config[key] < 1:
            raise ValueError("Modality weights must be finite and between 0 and 1")
    eyes = config["reference_eye_weights"]
    if set(eyes) != {"blink_suppression", "fixation_proportion"} or any(type(v) not in (int, float) or not np.isfinite(v) or v <= 0 for v in eyes.values()) or not np.isclose(sum(eyes.values()), 1):
        raise ValueError("Positive eye reference weights must sum to 1")
    if config["band_quantiles"] != [.2, .4, .6, .8] or type(config["include_head_pose_context"]) is not bool or type(config["seed"]) is not int:
        raise ValueError("Expected five quintile bands, a boolean context option and an integer seed")
    return config


def feature_groups(config):
    return {key: value for key, value in FAMILIES.items() if key != "Head context" or config["include_head_pose_context"]}


def check_unchanged(dataset):
    if any(digest(dataset["directory"] / name) != value for name, value in dataset["hashes"].items()):
        raise RuntimeError("Input exports changed; this run is invalid")


def load_dataset(directory, config):
    directory = Path(directory).resolve()
    filenames = ["feature_columns.json", *(f"{variant}.csv" for variant in VARIANTS)]
    optional = ("data_dictionary.csv", "manifest.json", "evaluation_test_windows.csv", "evaluation_folds.csv")
    filenames += [name for name in optional if (directory / name).exists()]
    before = {name: digest(directory / name) for name in filenames}
    spec = json.loads((directory / "feature_columns.json").read_text(encoding="utf-8-sig"))
    groups = feature_groups(config)
    predictors = [column for columns in groups.values() for column in columns]
    if not set(predictors).issubset(spec.get("features", [])):
        raise ValueError("Feature specification must include the configured HRV and eye predictors")
    if "data_dictionary.csv" in filenames:
        dictionary = read_csv(directory / "data_dictionary.csv")
        if not set(predictors).issubset(set(dictionary.loc[dictionary.role.eq("feature"), "column"])):
            raise ValueError("Predictors must have sensor-feature roles in the dictionary")
    views = {}
    for variant in VARIANTS:
        frame = read_csv(directory / f"{variant}.csv")
        required = {*IDENTITY, "window_start", "window_end", "window_seconds", *predictors}
        if not required.issubset(frame) or frame.empty:
            raise ValueError(f"{variant}: missing required columns or empty dataset")
        if any(frame[key].eq("").any() for key in IDENTITY) or frame.window_id.duplicated().any():
            raise ValueError(f"{variant}: empty IDs or duplicate windows")
        if frame.groupby("session_id").participant_id.nunique().gt(1).any():
            raise ValueError("A session cannot belong to multiple participants")
        frame[predictors] = numeric(frame, predictors)
        frame["window_seconds"] = pd.to_numeric(frame.window_seconds, errors="raise").astype(float)
        start, end = (pd.to_datetime(frame[column], utc=True, format="mixed", errors="raise") for column in ("window_start", "window_end"))
        duration = (end - start).dt.total_seconds()
        if not np.isfinite(frame.window_seconds).all() or frame.window_seconds.nunique() != 1 or frame.window_seconds.le(0).any() or not np.allclose(duration, frame.window_seconds, atol=.001, rtol=0):
            raise ValueError("Window durations must be real, consistent and one size per export group")
        views[variant] = frame
    for smaller, larger in zip(VARIANTS, VARIANTS[1:]):
        left, right = (views[key].set_index("window_id") for key in (smaller, larger))
        if not set(left.index).issubset(right.index):
            raise ValueError(f"{smaller} must be nested in {larger}")
        try:
            pd.testing.assert_frame_equal(left.sort_index(), right.loc[left.index].sort_index(), check_dtype=False)
        except AssertionError as exc:
            raise ValueError("Shared windows must have identical sensor data and provenance") from exc
    # Reuse the common TEST WINDOWS, never the old difficulty-class feasibility flags.
    good = views["good_only"]
    if "evaluation_test_windows.csv" in filenames:
        tests = read_csv(directory / "evaluation_test_windows.csv")
        if not {"window_id", "participant_id", "session_id"}.issubset(tests) or tests.window_id.duplicated().any() or set(tests.window_id) != set(good.window_id):
            raise ValueError("Shared test contract must match the good windows exactly")
        original = good.set_index("window_id").loc[tests.window_id]
        if any(original[column].tolist() != tests[column].tolist() for column in ("participant_id", "session_id")):
            raise ValueError("Shared test contract has stale participant/session identity")
    if "manifest.json" in filenames:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8-sig"))
        for name, expected in manifest.get("output_sha256", {}).items():
            if name in before and before[name] != expected:
                raise ValueError(f"Fusion manifest hash mismatch: {name}")
        step = manifest.get("configuration", {}).get("step_seconds")
    else:
        step = None
    dataset = {"directory": directory, "views": views, "groups": groups, "features": predictors, "hashes": before,
               "window_seconds": float(views["all_data"].window_seconds.iloc[0]), "step_seconds": step}
    check_unchanged(dataset)
    return dataset
