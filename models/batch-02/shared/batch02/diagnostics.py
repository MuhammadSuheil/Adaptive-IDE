"""Acquisition-pattern controls and descriptive audits, excluded from selection."""

import json

import pandas as pd

from .artifacts import read_json, write_json
from .features import MissingStream
from .fitting import fit_tabular
from .metrics import classification_metrics
from .persistence import checkpoint
from .preprocessing import CandidateInvalid
from .registry import Recipe


def low_activity_group(row):
    text = getattr(row, "window_phase_mix", "")
    segments = json.loads(text) if isinstance(text, str) and text.strip() else []
    phases = {s["study_phase"] for s in segments if s.get("duration_seconds", 0) > 0}
    if not phases:
        phases = {getattr(row, "study_phase", "unlabeled")}
    if phases.intersection({"unlabeled", "mixed"}):
        return "unknown"
    active = "active_task" in phases
    return (
        "active_and_nonactive"
        if active and len(phases) > 1
        else "active_only"
        if active
        else "nonactive_only"
    )


def audit_predictions(predictions, directory):
    audits = {}
    low = predictions[predictions.y_true == 0].copy()
    if len(low):
        low["low_activity_group"] = [
            low_activity_group(row) for row in low.itertuples()
        ]
        audits["low_activity"] = {
            name: classification_metrics(part.y_true, part.y_pred)
            for name, part in low.groupby("low_activity_group")
        }
    for field in ("baseline_usable", "gaze_validation_passed", "collection_wave"):
        if field in predictions:
            audits[field] = {
                str(name): classification_metrics(part.y_true, part.y_pred)
                for name, part in predictions.groupby(field, dropna=False)
            }
    write_json(directory / "audit_subsets.json", audits)


def coverage_report(frames, store, snapshot, destination):
    manifest = read_json(snapshot / "manifest.json")
    report = {}
    for seconds, frame in frames.items():
        try:
            stream = store.stream(seconds)
        except MissingStream:
            report[seconds] = {
                "status": "skipped_missing_stream",
                "sensor_coverage": None,
            }
            continue
        counts = manifest["datasets"][f"window_{seconds}s_step_5s"]["all_data"]
        report[seconds] = {
            "status": "verified_stream",
            "grid_windows": len(stream),
            "quality_eligible_windows": int(
                stream.quality_tier.isin(["good", "acceptable"]).sum()
            ),
            "labelable_windows": counts["windows"],
            "supervised_windows": len(frame),
            "labelable_fraction": counts["windows"] / len(stream),
            "sensor_eligible_fraction": float(
                stream.quality_tier.isin(["good", "acceptable"]).mean()
            ),
        }
    write_json(destination / "coverage.json", report)


def warmup_matrix(stream, endpoints):
    rows = []
    for _, group in stream.groupby("session_id", sort=False):
        gaps = group.window_end.diff().dt.total_seconds()
        blocks = gaps.gt(7.5).cumsum()
        for _, block in group.groupby(blocks, sort=False):
            for position, row in enumerate(block.itertuples()):
                rows.append(
                    {
                        "session_id": row.session_id,
                        "window_end": row.window_end,
                        "history_count": min(position + 1, 7),
                        "stream_age_seconds": (
                            row.window_end - block.window_end.iloc[0]
                        ).total_seconds(),
                    }
                )
    merged = endpoints[["session_id", "window_end"]].merge(
        pd.DataFrame(rows), on=["session_id", "window_end"], validate="one_to_one"
    )
    if len(merged) != len(endpoints):
        raise ValueError("Warm-up control kehilangan endpoint.")
    return merged[["history_count", "stream_age_seconds"]]


def negative_controls(config, frames, store, manifests, run):
    result = {}
    for seconds, frame in frames.items():
        matrices = {"missing_only": frame[store.features].isna().astype(float)}
        try:
            matrices["warmup_only"] = warmup_matrix(store.stream(seconds), frame)
        except MissingStream:
            result[f"{seconds}/warmup_only"] = {"status": "skipped_missing_stream"}
        for name, matrix in matrices.items():
            outputs = []
            for fold in manifests["windows"][str(seconds)]["folds"]:
                if fold["status"] == "no_test_support":
                    continue
                train, test = fold["train"], fold["test"]
                cached = (
                    run / "control_cache" / f"{seconds}s" / name / f"{fold['id']}.json"
                )
                if cached.exists():
                    saved = read_json(cached)
                    if saved["window_id"] != frame.iloc[test].window_id.tolist():
                        raise ValueError("Endpoint cache kontrol berbeda.")
                    outputs.append(pd.DataFrame(saved))
                    continue
                try:
                    fitted = fit_tabular(
                        Recipe(),
                        matrix.iloc[train],
                        frame.iloc[train],
                        config.seed,
                        config.threads,
                    )
                    probabilities, _ = fitted.predict(matrix.iloc[test])
                except CandidateInvalid as error:
                    outputs = []
                    result[f"{seconds}/{name}"] = {
                        "status": "invalid",
                        "reason": str(error),
                    }
                    break
                output = pd.DataFrame(
                    {
                        "window_id": frame.iloc[test].window_id.to_numpy(),
                        "y_true": frame.iloc[test].y.to_numpy(),
                        "y_pred": probabilities.argmax(1),
                    }
                )
                write_json(cached, output.to_dict(orient="list"))
                outputs.append(output)
                checkpoint()
            if outputs:
                combined = pd.concat(outputs)
                combined.to_csv(run / f"diagnostic_{seconds}_{name}.csv", index=False)
                result[f"{seconds}/{name}"] = {
                    "status": "diagnostic_only",
                    **classification_metrics(combined.y_true, combined.y_pred),
                }
            checkpoint(force=True)
    write_json(run / "negative_controls.json", result)
