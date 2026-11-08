"""Train five-level provisional load models; never predict task difficulty."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analysis_sessions.model_training.data import DEFAULT_INPUT, IDENTITY, LEVELS, NOTE, VARIANTS, check_unchanged, digest, eligible, load_config, load_dataset, write_json
from analysis_sessions.model_training.predict import estimate
from analysis_sessions.model_training.reference import components, fit_reference, label, levels, fuse

MODELS = ("svm", "random_forest", "gradient_boosting")
RUNS = Path(__file__).with_name("runs")


def model_grid(name, search, seed):
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=False))]
    if name == "svm":
        steps.extend([("scale", StandardScaler()), ("model", SVR(kernel="rbf", epsilon=1.))])
        grid = {"model__C": [10., 100., 1000.]}
    elif name == "random_forest":
        steps.append(("model", RandomForestRegressor(n_estimators=120, n_jobs=1, random_state=seed)))
        grid = {"model__max_depth": [6, None], "model__min_samples_leaf": [2, 5]}
    elif name == "gradient_boosting":
        steps.append(("model", HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=7, min_samples_leaf=10,
                                                             l2_regularization=1., early_stopping=False, random_state=seed)))
        grid = {"model__learning_rate": [.05, .1]}
    else:
        raise ValueError(f"Unknown model: {name}")
    if search == "quick":
        grid = {key: [values[0]] for key, values in grid.items()}
    return Pipeline(steps), list(ParameterGrid(grid))


def group_maes(truth, predicted, people):
    people = np.asarray(people)
    return [float(np.abs(np.asarray(truth)[people == person] - np.asarray(predicted)[people == person]).mean()) for person in np.unique(people)]


def reference_group_splits(frame, reference_pool, count):
    """Keep people intact and spread good-reference people across validation folds."""
    counts = frame.participant_id.value_counts()
    reference_people = set(reference_pool.participant_id) & set(counts.index)
    ordered = sorted(reference_people, key=lambda person: (-counts[person], person.casefold()))
    assignments, sizes = [set() for _ in range(count)], [0] * count
    for index, person in enumerate(ordered):
        selected = index % count
        assignments[selected].add(person)
        sizes[selected] += int(counts[person])
    for person in sorted(set(counts.index) - reference_people, key=lambda person: (-counts[person], person.casefold())):
        selected = min(range(count), key=lambda index: (sizes[index], len(assignments[index]), index))
        assignments[selected].add(person)
        sizes[selected] += int(counts[person])
    for people in assignments:
        if not people:
            raise ValueError("Empty grouped validation fold")
        valid = frame.participant_id.isin(people).to_numpy()
        yield np.flatnonzero(~valid), np.flatnonzero(valid)


def inner_cases(frame, good_reference_pool, config):
    """Rebuild pseudo-label references inside each inner training fold."""
    for count in range(min(3, frame.participant_id.nunique()), 1, -1):
        cases, audit = [], []
        try:
            for train_index, validation_index in reference_group_splits(frame, good_reference_pool, count):
                train, validation = frame.iloc[train_index], frame.iloc[validation_index]
                people = set(train.participant_id)
                reference = fit_reference(good_reference_pool.loc[good_reference_pool.participant_id.isin(people)], config)
                targets = label(reference, train)
                if set(targets.reference_level) != set(LEVELS):
                    raise ValueError("Inner training data cannot support all five reference levels")
                cases.append((train, validation, targets, label(reference, validation)))
                audit.append({"train_participants": sorted(people), "validation_participants": sorted(set(validation.participant_id)),
                              "reference_participants": reference["participants"], "reference_windows": reference["windows"],
                              "reference_boundaries": reference["boundaries"]})
            return cases, audit
        except ValueError:
            continue
    raise ValueError("Insufficient grouped inner folds: each needs a two-person good reference and five supported training levels")


def fit_branch(name, branch, features, frame, targets, cases, config, search):
    target_column = f"{branch}_reference_score"
    _, parameters = model_grid(name, search, config["seed"])
    scores = []
    for settings in parameters:
        errors = []
        for train, valid, train_targets, valid_targets in cases:
            pipeline, _ = model_grid(name, search, config["seed"])
            pipeline.set_params(**settings)
            pipeline.fit(train[features], train_targets[target_column])
            prediction = np.clip(pipeline.predict(valid[features]), 0, 100)
            errors.extend(group_maes(valid_targets[target_column], prediction, valid.participant_id))
        scores.append(float(np.mean(errors)))
    best = int(np.argmin(scores))
    pipeline, _ = model_grid(name, search, config["seed"])
    pipeline.set_params(**parameters[best])
    pipeline.fit(frame[features], targets[target_column])
    return pipeline, {"features": features, "best_parameters": parameters[best], "inner_participant_mae": scores[best],
                      "search": [{"parameters": settings, "participant_mae": score} for settings, score in zip(parameters, scores)],
                      "entirely_missing_training_features": [column for column in features if frame[column].isna().all()]}


def fit_bundle(name, frame, reference, cases, dataset, config, search):
    targets = label(reference, frame)
    if set(targets.reference_level) != set(LEVELS):
        raise ValueError("Training variant does not contain all five provisional levels")
    hrv_features = dataset["groups"]["HRV"]
    eye_features = [column for family, columns in dataset["groups"].items() if family != "HRV" for column in columns]
    with threadpool_limits(limits=1):
        hrv, hrv_audit = fit_branch(name, "hrv", hrv_features, frame, targets, cases, config, search)
        eye, eye_audit = fit_branch(name, "eye", eye_features, frame, targets, cases, config, search)
    bundle = {"bundle_version": 1, "target": "cognitive_load_level", "target_note": NOTE, "levels": list(LEVELS),
              "model": name, "config": config, "groups": dataset["groups"], "features": dataset["features"],
              "reference": reference, "hrv_model": hrv, "eye_model": eye, "window_seconds": dataset["window_seconds"],
              "step_seconds": dataset["step_seconds"], "input_hashes": dataset["hashes"], "sklearn_version": sklearn.__version__,
              "training_windows": len(frame), "training_participants": int(frame.participant_id.nunique())}
    return bundle, {"hrv": hrv_audit, "eye": eye_audit}


def agreement(truth_levels, predicted_levels, truth_score, predicted_score):
    truth_levels, predicted_levels = np.asarray(truth_levels), np.asarray(predicted_levels)
    ranks = {level: index for index, level in enumerate(LEVELS)}
    distance = np.abs(np.array([ranks[value] for value in truth_levels]) - np.array([ranks[value] for value in predicted_levels]))
    recall = recall_score(truth_levels, predicted_levels, labels=list(LEVELS), average=None, zero_division=0)
    supported = np.array([np.any(truth_levels == level) for level in LEVELS])
    return {"macro_f1_agreement": float(f1_score(truth_levels, predicted_levels, labels=list(LEVELS), average="macro", zero_division=0)),
            "exact_agreement": float(np.mean(distance == 0)), "within_one_level": float(np.mean(distance <= 1)),
            "ordinal_mae": float(distance.mean()), "score_mae": float(np.abs(np.asarray(truth_score) - np.asarray(predicted_score)).mean()),
            "balanced_recall": float(recall[supported].mean()), "recall": {level: float(value) if present else None for level, value, present in zip(LEVELS, recall, supported)},
            "confusion_matrix": confusion_matrix(truth_levels, predicted_levels, labels=list(LEVELS)).tolist()}


def family_sensitivity(bundle, test, targets, seed):
    """Grouped circular perturbation measures dependence, not physiological validity."""
    _, _, baseline_score, _ = estimate(bundle, test)
    baseline_error = np.abs(baseline_score - targets.reference_score.to_numpy()).mean()
    rng = np.random.default_rng(seed)
    rows = []
    for family, columns in bundle["groups"].items():
        changes = []
        if len(test) > 1:
            for _ in range(3):
                changed = test.copy()
                shift = int(rng.integers(1, len(test)))
                changed[columns] = np.roll(test[columns].to_numpy(), shift, axis=0)
                _, _, score, _ = estimate(bundle, changed)
                changes.append(float(np.abs(score - targets.reference_score.to_numpy()).mean() - baseline_error))
        rows.append({"family": family, "mae_increase": float(np.mean(changes)) if changes else None})
    return rows


def evaluate(dataset_id, dataset, destination, config, search, export=True, diagnostics=True):
    destination.mkdir(parents=True, exist_ok=False)
    views = dataset["views"]
    good = views["good_only"]
    good_scorable = good.loc[eligible(good)]
    eligible_views = {name: frame.loc[eligible(frame)].reset_index(drop=True) for name, frame in views.items()}
    coverage = [{"variant": name, "total_windows": len(views[name]), "scorable_windows": len(frame),
                 "unscorable_windows": len(views[name]) - len(frame), "total_participants": int(views[name].participant_id.nunique()),
                 "scorable_participants": int(frame.participant_id.nunique())} for name, frame in eligible_views.items()]
    predictions, metrics, folds, audits, sensitivities = [], [], [], [], []
    people = sorted(views["all_data"].participant_id.unique(), key=str.casefold)
    for index, person in enumerate(people, 1):
        fold_id = f"F{index:03d}"
        test = good_scorable.loc[good_scorable.participant_id.eq(person)].reset_index(drop=True)
        fold = {"fold_id": fold_id, "held_out_participant_id": person, "test_windows": len(test),
                "unscorable_good_test_windows": int((good.participant_id.eq(person) & ~eligible(good)).sum()), "comparison_feasible": False, "reason": ""}
        if test.empty:
            fold["reason"] = "no_scorable_good_test_windows"
            folds.append(fold)
            continue
        pool = good_scorable.loc[good_scorable.participant_id.ne(person)]
        try:
            reference = fit_reference(pool, config)
            targets = label(reference, test)
            cases_by_variant = {}
            for variant, frame in eligible_views.items():
                train = frame.loc[frame.participant_id.ne(person)].reset_index(drop=True)
                if set(label(reference, train).reference_level) != set(LEVELS):
                    raise ValueError(f"{variant}: missing provisional training levels")
                cases_by_variant[variant] = (train, *inner_cases(train, pool, config))
            fold.update(comparison_feasible=True, **{f"test_{level.replace(' ', '_').lower()}": int(targets.reference_level.eq(level).sum()) for level in LEVELS})
        except ValueError as exc:
            fold["reason"] = str(exc)
            folds.append(fold)
            continue
        folds.append(fold)
        for variant, (train, cases, split_audit) in cases_by_variant.items():
            for name in MODELS:
                started = time.perf_counter()
                bundle, details = fit_bundle(name, train, reference, cases, dataset, config, search)
                elapsed = time.perf_counter() - started
                hrv, eye, score, predicted = estimate(bundle, test)
                alternate_weight = config["sensitivity_hrv_weight"]
                alternate_target = label(reference, test, alternate_weight)
                alternate_score = fuse(hrv, eye, alternate_weight)
                alternate_level = levels(reference, alternate_score, alternate_weight)
                primary_weight = config["primary_hrv_weight"]
                prefix = {"dataset_id": dataset_id, "variant": variant, "model": name, "fold_id": fold_id, "held_out_participant_id": person}
                for row_index, source in enumerate(test.to_dict("records")):
                    target = targets.iloc[row_index]
                    predictions.append({**prefix, **{key: source[key] for key in (*IDENTITY, "window_start", "window_end")},
                                        **target.to_dict(), "hrv_score": float(hrv[row_index]), "eye_score": float(eye[row_index]),
                                        "hrv_contribution": float(primary_weight * hrv[row_index]), "eye_contribution": float((1 - primary_weight) * eye[row_index]),
                                        "combined_score": float(score[row_index]), "cognitive_load_level": str(predicted[row_index]),
                                        "sensitivity_score": float(alternate_score[row_index]), "sensitivity_level": str(alternate_level[row_index]),
                                        "sensitivity_reference_level": str(alternate_target.reference_level.iloc[row_index])})
                estimate(bundle, test.iloc[:1])
                began = time.perf_counter()
                for _ in range(5):
                    estimate(bundle, test.iloc[:1])
                timing = (time.perf_counter() - began) * 200
                record = {**prefix, "train_windows": len(train), "train_participants": int(train.participant_id.nunique()),
                          "test_windows": len(test), "fit_seconds": elapsed, "single_window_ms": timing,
                          "hrv_mae": float(np.abs(hrv - targets.hrv_reference_score.to_numpy()).mean()),
                          "eye_mae": float(np.abs(eye - targets.eye_reference_score.to_numpy()).mean()),
                          "weight_label_change_fraction": float(np.mean(predicted != alternate_level)),
                          "reference_weight_label_change_fraction": float(np.mean(targets.reference_level.to_numpy() != alternate_target.reference_level.to_numpy())),
                          **agreement(targets.reference_level, predicted, targets.reference_score, score)}
                metrics.append(record)
                audits.append({**prefix, "train_participants": sorted(set(train.participant_id)), "reference_participants": reference["participants"],
                               "reference_boundaries": reference["boundaries"], "inner_folds": split_audit, "branches": details})
                if diagnostics:
                    sensitivities.extend({**prefix, **row} for row in family_sensitivity(bundle, test, targets, config["seed"]))
                print(f"{dataset_id} | {person} | {variant} | {name}: F1 kesesuaian lima tingkat={record['macro_f1_agreement']:.3f}, MAE skor={record['score_mae']:.2f}", flush=True)
    pd.DataFrame(folds).to_csv(destination / "evaluation_folds.csv", index=False)
    pd.DataFrame(coverage).to_csv(destination / "data_coverage.csv", index=False)
    if not predictions:
        raise ValueError("No feasible common five-level evaluation folds; see evaluation_folds.csv")
    prediction_frame = pd.DataFrame(predictions)
    prediction_frame.to_csv(destination / "held_out_predictions.csv", index=False)
    pd.DataFrame([{key: value for key, value in row.items() if key not in ("recall", "confusion_matrix")} for row in metrics]).to_csv(destination / "participant_metrics.csv", index=False)
    write_json(destination / "participant_metrics.json", metrics)
    write_json(destination / "training_audit.json", audits)
    pd.DataFrame(sensitivities, columns=["dataset_id", "variant", "model", "fold_id", "held_out_participant_id", "family", "mae_increase"]).to_csv(destination / "family_sensitivity.csv", index=False)
    summaries = []
    for variant in VARIANTS:
        for name in MODELS:
            selected = prediction_frame.loc[prediction_frame.variant.eq(variant) & prediction_frame.model.eq(name)]
            records = [row for row in metrics if row["variant"] == variant and row["model"] == name]
            summaries.append({"dataset_id": dataset_id, "variant": variant, "model": name,
                              "participant_macro_f1": float(np.mean([row["macro_f1_agreement"] for row in records])),
                              "participant_score_mae": float(np.mean([row["score_mae"] for row in records])),
                              "test_participants": len(records), "test_windows": len(selected),
                              "mean_single_window_ms": float(np.mean([row["single_window_ms"] for row in records])),
                              "weight_label_change_fraction": float(np.mean([row["weight_label_change_fraction"] for row in records])),
                              **agreement(selected.reference_level, selected.cognitive_load_level, selected.reference_score, selected.combined_score)})
    summaries.sort(key=lambda row: (-row["participant_macro_f1"], row["participant_score_mae"]))
    write_json(destination / "summary.json", summaries)
    pd.DataFrame([{key: value for key, value in row.items() if key not in ("recall", "confusion_matrix")} for row in summaries]).to_csv(destination / "comparison.csv", index=False)
    # A shared final reference prevents variant-specific definitions of High.
    final_reference = fit_reference(good_scorable, config)
    write_json(destination / "reference.json", final_reference)
    scored = []
    for variant, frame in views.items():
        values = label(final_reference, frame)
        scored.append(pd.concat([frame[list(IDENTITY)].reset_index(drop=True), values.reset_index(drop=True)], axis=1).assign(variant=variant))
    pd.concat(scored, ignore_index=True).to_csv(destination / "reference_labels.csv", index=False)
    if export:
        model_directory = destination / "models"
        model_directory.mkdir()
        for variant, frame in eligible_views.items():
            cases, split_audit = inner_cases(frame, good_scorable, config)
            for name in MODELS:
                print(f"Melatih model akhir {dataset_id} / {variant} / {name}: cabang HRV dan mata terpisah", flush=True)
                bundle, details = fit_bundle(name, frame, final_reference, cases, dataset, config, search)
                bundle.update(dataset_id=dataset_id, variant=variant, search=search)
                joblib.dump(bundle, model_directory / f"{variant}__{name}.joblib")
                metadata = {key: value for key, value in bundle.items() if key not in ("hrv_model", "eye_model")}
                write_json(model_directory / f"{variant}__{name}.json", {**metadata, "branches": details, "inner_folds": split_audit})
    check_unchanged(dataset)
    from analysis_sessions.model_training.visualize import render_report
    render_report(destination, dataset, config)
    return summaries


def parse_dataset(value):
    name, separator, path = value.partition("=")
    if not separator or not path or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise argparse.ArgumentTypeError("Use ID=DIRECTORY, e.g. window25_step5=analysis_sessions/dataset_fusion/output")
    return name, Path(path).resolve()


def run(datasets, output, config=None, search="compact", export=True, diagnostics=True):
    config = config or load_config()
    if search not in ("quick", "compact") or not datasets or len(set(name.casefold() for name, _ in datasets)) != len(datasets):
        raise ValueError("Use unique dataset IDs and quick/compact search")
    loaded = [(name, load_dataset(path, config)) for name, path in datasets]
    output = Path(output).resolve()
    for _, dataset in loaded:
        source = dataset["directory"]
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Run output must be separate from source exports")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(), "target": "cognitive_load_level", "target_note": NOTE,
                "config": config, "search": search, "export_models": export, "family_diagnostics": diagnostics,
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
                "datasets": {name: {"directory": str(dataset["directory"]), "input_hashes": dataset["hashes"], "window_seconds": dataset["window_seconds"], "step_seconds": dataset["step_seconds"]} for name, dataset in loaded},
                "code_sha256": {path.name: digest(path) for path in sorted(Path(__file__).parent.glob("*.py"))}}
    write_json(output / "run_manifest.json", manifest)
    try:
        results = []
        for name, dataset in loaded:
            results.extend(evaluate(name, dataset, output / name, config, search, export, diagnostics))
        for _, dataset in loaded:
            check_unchanged(dataset)
        pd.DataFrame([{key: value for key, value in row.items() if key not in ("recall", "confusion_matrix")} for row in results]).to_csv(output / "all_comparisons.csv", index=False)
        manifest.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat(), source_integrity_verified=True)
        manifest["output_sha256"] = {str(path.relative_to(output)).replace("\\", "/"): digest(path) for path in sorted(output.rglob("*")) if path.is_file() and path.name != "run_manifest.json"}
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        write_json(output / "run_manifest.json", manifest)
        raise
    write_json(output / "run_manifest.json", manifest)
    print(f"Eksperimen lima tingkat selesai: {output}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=parse_dataset, action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--search", choices=("quick", "compact"), default="compact")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--no-family-diagnostics", action="store_true")
    args = parser.parse_args()
    output = args.output or RUNS / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    try:
        run(args.dataset or [("window25_step5", DEFAULT_INPUT)], output, load_config(args.config), args.search, not args.no_export, not args.no_family_diagnostics)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Training failed: {exc}\n")


if __name__ == "__main__":
    main()
