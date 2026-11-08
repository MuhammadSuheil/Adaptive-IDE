"""Transparent, training-only provisional labels; no task-difficulty input."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import ANCHORS, LEVELS, eligible, numeric


def participant_weights(frame):
    counts = frame.participant_id.value_counts()
    return frame.participant_id.map(lambda participant: 1. / (len(counts) * counts[participant])).to_numpy()


def weighted_quantiles(values, weights, quantiles):
    order = np.argsort(values, kind="stable")
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    positions = (np.cumsum(weights) - .5 * weights) / weights.sum()
    return np.interp(quantiles, positions, values)


def fit_cdf(values, weights):
    points, inverse = np.unique(values, return_inverse=True)
    mass = np.bincount(inverse, weights=weights)
    mass = mass / mass.sum()
    return {"points": points.tolist(), "mass": mass.tolist()}


def percentile(cdf, values):
    points, mass = np.asarray(cdf["points"]), np.asarray(cdf["mass"])
    prefix = np.r_[0., np.cumsum(mass)]
    left, right = np.searchsorted(points, values, "left"), np.searchsorted(points, values, "right")
    # Midrank for ties; below/above the reference saturates at 0/1.
    return np.clip(.5 * (prefix[left] + prefix[right]), 0, 1)


def components(reference, frame):
    values = numeric(frame, list(ANCHORS))
    valid = eligible(frame)
    result = pd.DataFrame(np.nan, index=frame.index, columns=["hrv_reference_score", "eye_reference_score"])
    if valid.any():
        values = values.loc[valid]
        hrv = 1 - percentile(reference["cdf"][ANCHORS[0]], values[ANCHORS[0]].to_numpy())
        blink = 1 - percentile(reference["cdf"][ANCHORS[1]], values[ANCHORS[1]].to_numpy())
        fixation = percentile(reference["cdf"][ANCHORS[2]], values[ANCHORS[2]].to_numpy())
        weights = reference["config"]["reference_eye_weights"]
        result.loc[valid, "hrv_reference_score"] = 100 * hrv
        result.loc[valid, "eye_reference_score"] = 100 * (weights["blink_suppression"] * blink + weights["fixation_proportion"] * fixation)
    return result


def weight_key(weight):
    return f"{weight:.6f}"


def fuse(hrv_score, eye_score, weight):
    return weight * np.clip(hrv_score, 0, 100) + (1 - weight) * np.clip(eye_score, 0, 100)


def levels(reference, scores, weight=None):
    weight = reference["config"]["primary_hrv_weight"] if weight is None else weight
    boundaries = reference["boundaries"][weight_key(weight)]
    values = np.asarray(scores, dtype=float)
    result = np.full(values.shape, "", dtype=object)
    valid = np.isfinite(values)
    result[valid] = np.asarray(LEVELS)[np.searchsorted(boundaries, values[valid], side="right")]
    return result


def label(reference, frame, weight=None):
    weight = reference["config"]["primary_hrv_weight"] if weight is None else weight
    result = components(reference, frame)
    result["reference_score"] = fuse(result.hrv_reference_score, result.eye_reference_score, weight)
    result["reference_level"] = levels(reference, result.reference_score, weight)
    result["signal_status"] = np.where(result.reference_score.notna(), "scorable", "insufficient_signal")
    return result


def fit_reference(frame, config):
    frame = frame.loc[eligible(frame)].copy()
    if frame.participant_id.nunique() < 2 or len(frame) < 10:
        raise ValueError("Reference requires at least two participants and ten scorable good windows")
    weights = participant_weights(frame)
    values = numeric(frame, list(ANCHORS))
    reference = {"version": 1, "config": config, "participants": sorted(frame.participant_id.unique()),
                 "window_ids": frame.window_id.tolist(), "windows": len(frame),
                 "cdf": {column: fit_cdf(values[column].to_numpy(), weights) for column in ANCHORS}, "boundaries": {}}
    scores = components(reference, frame)
    for weight in {config["primary_hrv_weight"], config["sensitivity_hrv_weight"]}:
        fused = fuse(scores.hrv_reference_score, scores.eye_reference_score, weight)
        boundaries = weighted_quantiles(fused, weights, config["band_quantiles"])
        if not np.isfinite(boundaries).all() or np.any(np.diff(boundaries) <= 1e-9):
            raise ValueError("Reference score ties prevent five distinct bands; do not force equal values into different levels")
        reference["boundaries"][weight_key(weight)] = boundaries.tolist()
    return reference
