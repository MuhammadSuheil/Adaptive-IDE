"""Prediction from independently fitted HRV and eye branches."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analysis_sessions.model_training.data import IDENTITY, eligible, numeric, read_csv
from analysis_sessions.model_training.reference import fuse, levels, weight_key


def estimate(bundle, frame, weight=None):
    """Internal numerical prediction; caller determines eligibility and quality."""
    weight = bundle["config"]["primary_hrv_weight"] if weight is None else weight
    if weight_key(weight) not in bundle["reference"]["boundaries"]:
        raise ValueError("No saved reference bands for that HRV weight")
    hrv_features = bundle["groups"]["HRV"]
    eye_features = [column for family, columns in bundle["groups"].items() if family != "HRV" for column in columns]
    with threadpool_limits(limits=1):
        hrv = np.clip(bundle["hrv_model"].predict(numeric(frame, hrv_features)), 0, 100)
        eye = np.clip(bundle["eye_model"].predict(numeric(frame, eye_features)), 0, 100)
    score = fuse(hrv, eye, weight)
    return hrv, eye, score, levels(bundle["reference"], score, weight)


def predict_windows(bundle, frame, weight=None):
    if bundle.get("bundle_version") != 1:
        raise ValueError("Unsupported five-level model bundle")
    # Require the full feature schema; blanks in non-anchor features use fitted medians.
    numeric(frame, bundle["features"])
    if "window_seconds" in frame:
        duration = pd.to_numeric(frame.window_seconds, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(duration).all() or not np.allclose(duration, bundle["window_seconds"], atol=.001, rtol=0):
            raise ValueError("Window size does not match this model")
    valid = eligible(frame).to_numpy()
    poor = frame.quality_tier.eq("poor").to_numpy() if "quality_tier" in frame else np.zeros(len(frame), dtype=bool)
    admitted = valid & ~poor
    known_quality = frame.quality_tier.isin(("good", "acceptable")).to_numpy() if "quality_tier" in frame else np.zeros(len(frame), dtype=bool)
    status = np.where(valid, np.where(known_quality, "ok", "quality_unverified"), "insufficient_signal")
    status = np.where(valid & poor, "poor_signal_quality", status)
    result = frame.loc[:, [column for column in IDENTITY if column in frame]].reset_index(drop=True).copy()
    result["signal_status"] = status
    result["cognitive_load_level"] = ""
    for column in ("hrv_score", "eye_score", "combined_score"):
        result[column] = np.nan
    selected_weight = bundle["config"]["primary_hrv_weight"] if weight is None else weight
    if weight_key(selected_weight) not in bundle["reference"]["boundaries"]:
        raise ValueError("No saved reference bands for that HRV weight")
    result["hrv_weight"] = selected_weight
    result["eye_weight"] = 1 - selected_weight
    result["label_source"] = "provisional_relative_hrv_eye"
    if admitted.any():
        hrv, eye, score, label = estimate(bundle, frame.iloc[np.flatnonzero(admitted)], selected_weight)
        positions = np.flatnonzero(admitted)
        for column, values in zip(("hrv_score", "eye_score", "combined_score", "cognitive_load_level"), (hrv, eye, score, label)):
            result.loc[positions, column] = values
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hrv-weight", type=float, help="Default 0.6; 0.5 uses the saved sensitivity reference")
    args = parser.parse_args()
    try:
        result = predict_windows(joblib.load(args.model), read_csv(args.input), args.hrv_weight)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False, mode="x")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Prediction failed: {exc}\n")
    print(f"Five-level predictions: {args.output}")


if __name__ == "__main__":
    main()
