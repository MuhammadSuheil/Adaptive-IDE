"""Nested joint-window selection and final refits remain training-time actions."""

from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from .artifacts import digest, file_hash, read_json, write_json
from .metrics import classification_metrics, paired_bootstrap
from .modeling import fit_candidate
from .neural import Ineligible
from .persistence import checkpoint
from .registry import Recipe, recipe_from_dict
from .search import Evaluator, adaptive_search, select_record
from .splits import group_splits, paired_endpoints, restrict_paired


def joint_inner_splits(native, common_training, seed=42, exclude_people=()):
    if common_training.y.nunique() != 3:
        raise Ineligible(
            "skipped_insufficient_support: intersection fitting tidak memuat tiga kelas."
        )
    result = []
    for _, valid in group_splits(common_training.reset_index(drop=True), seed):
        validation = common_training.iloc[valid]
        valid_people = set(validation.person_id)
        training = np.flatnonzero(
            ~native.person_id.isin(valid_people | set(exclude_people))
        )
        keys = pd.MultiIndex.from_frame(validation[["session_id", "window_end"]])
        valid_indices = np.flatnonzero(
            pd.MultiIndex.from_frame(native[["session_id", "window_end"]]).isin(keys)
        )
        if native.iloc[training].y.nunique() != 3 or len(valid_indices) != len(
            validation
        ):
            raise Ineligible(
                "skipped_insufficient_support: matched inner split tidak feasible."
            )
        result.append({"train": training.tolist(), "valid": valid_indices.tolist()})
    return result


def joint_window_evaluation(config, frames, store, run, fit=fit_candidate):
    common = restrict_paired(frames[25], paired_endpoints(frames))
    if common.empty:
        return [], {}, {"status": "skipped_empty_intersection"}
    people = sorted(common.person_id.unique())
    prepared = {}
    try:
        for person in people:
            training = common[common.person_id != person].reset_index(drop=True)
            for seconds, frame in frames.items():
                prepared[(person, seconds)] = joint_inner_splits(
                    frame, training, config.seed, (person,)
                )
    except (Ineligible, ValueError) as error:
        return [], {}, {"status": "skipped_insufficient_support", "reason": str(error)}
    predictions = []
    from .runner import _outer_result, _save_search

    for i, person in enumerate(people):
        selected = []
        for seconds, frame in frames.items():
            folder = run / "joint_window" / f"fold_{i:02d}" / f"window_{seconds}s"
            evaluator = Evaluator(frame, store, seconds, config, folder / "inner", fit)
            winner, records, budget = adaptive_search(
                evaluator, prepared[(person, seconds)]
            )
            _save_search(folder / "search_results.json", records)
            write_json(folder / "budget.json", budget)
            selected.append({**winner, "window": seconds, "evaluator": evaluator})
        best = select_record(selected)
        evaluator, seconds = best["evaluator"], best["window"]
        train = np.flatnonzero(evaluator.frame.person_id != person)
        test_common = common[common.person_id == person]
        keys = pd.MultiIndex.from_frame(test_common[["session_id", "window_end"]])
        test = np.flatnonzero(
            pd.MultiIndex.from_frame(
                evaluator.frame[["session_id", "window_end"]]
            ).isin(keys)
        )
        folder = run / "joint_window" / f"fold_{i:02d}" / "selected"
        predictions.append(
            _outer_result(
                evaluator, best, train, test, "nested_window_selected", "paired", folder
            )
        )
        write_json(
            folder / "selection.json",
            {"window": seconds, "recipe": asdict(best["recipe"])},
        )
        checkpoint(force=True)
    combined = pd.concat(predictions, ignore_index=True)
    if (
        len(combined) != len(common)
        or combined.duplicated(["session_id", "window_end"]).any()
    ):
        raise ValueError("Joint-window OOF tidak mencakup intersection tepat sekali.")
    combined.to_csv(run / "joint_window_oof.csv", index=False)
    metrics = classification_metrics(
        combined.y_true,
        combined.y_pred,
        combined[[f"prob_{k}" for k in ("Low", "Medium", "High")]].to_numpy()
        if all(
            f"prob_{k}" in combined and combined[f"prob_{k}"].notna().all()
            for k in ("Low", "Medium", "High")
        )
        else None,
    )
    write_json(
        run / "joint_window_metrics.json",
        {
            **metrics,
            "rows": len(common),
            "selected_windows": combined.window_seconds.value_counts().to_dict(),
        },
    )
    return (
        predictions,
        {},
        {"status": "complete", "rows": len(common), "people": len(people)},
    )


def _save_deployment(config, record, rows, store, seconds, directory, fit):
    import joblib

    directory.mkdir(parents=True, exist_ok=True)
    card_path = directory / "model_card.json"
    if card_path.exists() and (directory / "transform_metadata.json").exists():
        card = read_json(card_path)
        # JSON converts tuples to lists; compare canonical digests instead.
        if digest(card["recipe"]) != digest(asdict(record["recipe"])):
            raise ValueError("Recipe deployment tersimpan berbeda.")
        if file_hash(directory / "pipeline.joblib") != card["pipeline_sha256"]:
            raise ValueError("Model deployment tersimpan korup.")
        return
    fitted = fit(
        record["recipe"], rows, store, seconds, config, epochs=record.get("epoch")
    )
    joblib.dump(fitted, directory / "pipeline.joblib")
    from .reporting import transform_metadata

    write_json(directory / "transform_metadata.json", transform_metadata(fitted))
    restored = joblib.load(directory / "pipeline.joblib")  # nosec B301
    before = fitted.predict(rows.iloc[:4], store, seconds)[0]
    after = restored.predict(rows.iloc[:4], store, seconds)[0]
    if not np.allclose(before, after, rtol=1e-5, atol=1e-5):
        raise ValueError("Deployment persist/load parity gagal.")
    write_json(
        directory / "model_card.json",
        {
            "window_seconds": seconds,
            "recipe": asdict(record["recipe"]),
            "features": fitted.feature_names,
            "epoch": fitted.epoch,
            "inner_selection_macro_f1": record["score"],
            "selection_score_is_test_score": False,
            "pipeline_sha256": file_hash(directory / "pipeline.joblib"),
            "stream_required": bool(
                record["recipe"].history
                or record["recipe"].normalization == "N3"
                or record["recipe"].model == "gru"
            ),
        },
    )
    checkpoint(force=True)


def refit_release(config, frames, store, run, joint_status, fit=fit_candidate):
    winners = {}
    from .runner import _save_search

    for seconds, frame in frames.items():
        inner = [
            {"train": a.tolist(), "valid": b.tolist()}
            for a, b in group_splits(frame, config.seed)
        ]
        evaluator = Evaluator(
            frame,
            store,
            seconds,
            config,
            run / "deployment" / f"window_{seconds}s" / "inner",
            fit,
        )
        best, records, budget = adaptive_search(evaluator, inner)
        winners[seconds] = best
        directory = run / "deployment" / f"window_{seconds}s"
        _save_search(directory / "selection_results.json", records)
        write_json(directory / "budget.json", budget)
        _save_deployment(config, best, frame, store, seconds, directory, fit)
    preferred_window, reason = 25, "compatibility_default_joint_not_eligible"
    if joint_status["status"] == "complete":
        common = restrict_paired(frames[25], paired_endpoints(frames))
        choices = []
        for seconds, frame in frames.items():
            inner = joint_inner_splits(frame, common, config.seed)
            evaluator = Evaluator(
                frame,
                store,
                seconds,
                config,
                run / "deployment" / "preferred" / f"window_{seconds}s" / "inner",
                fit,
            )
            best, _, _ = adaptive_search(evaluator, inner)
            choices.append({**best, "window": seconds})
        best = select_record(choices)
        preferred_window, reason = (
            best["window"],
            "group_validation_on_common_endpoints",
        )
        _save_deployment(
            config,
            best,
            frames[preferred_window],
            store,
            preferred_window,
            run / "deployment" / "preferred",
            fit,
        )
    write_json(
        run / "deployment" / "preferred_window.json",
        {
            "window_seconds": preferred_window,
            "reason": reason,
            "model": "preferred/pipeline.joblib"
            if reason == "group_validation_on_common_endpoints"
            else "window_25s/pipeline.joblib",
        },
    )


def _score_fit(
    recipe, training, test, store, seconds, config, fit, epoch=None, cache_path=None
):
    identity = digest(
        {
            "recipe": asdict(recipe),
            "train": training.window_id.tolist(),
            "test": test.window_id.tolist(),
            "seed": config.seed,
            "threads": config.threads,
            "epoch": epoch,
        }
    )
    if cache_path is not None and cache_path.exists():
        cached = read_json(cache_path)
        if cached["identity"] != identity:
            raise ValueError("Identitas cache sensitivity berbeda.")
        return cached["metrics"], np.asarray(cached["predicted"], dtype=int)
    fitted = fit(recipe, training, store, seconds, config, epochs=epoch)
    values, kind = fitted.predict(test, store, seconds)
    metrics = classification_metrics(
        test.y, values.argmax(1), values if kind == "probability" else None
    )
    predicted = values.argmax(1)
    if cache_path is not None:
        write_json(
            cache_path,
            {"identity": identity, "metrics": metrics, "predicted": predicted},
        )
        checkpoint()
    return metrics, predicted


def sensitivity_checks(config, frames, store, manifests, run, fit=fit_candidate):
    from .search import thin_training

    rows, baseline_predictions = [], []
    for seconds, frame in frames.items():
        for fold in manifests["windows"][str(seconds)]["folds"]:
            if fold["status"] == "no_test_support":
                continue
            folder = run / f"window_{seconds}s" / fold["id"]
            chosen = recipe_from_dict(read_json(folder / "selection.json")["recipe"])
            epoch = read_json(folder / "selected" / "features.json")["epoch"]
            train, test = frame.iloc[fold["train"]], frame.iloc[fold["test"]]
            for seed in (17, 2026):
                local = replace(config, seed=seed)
                metrics, _ = _score_fit(
                    chosen,
                    train,
                    test,
                    store,
                    seconds,
                    local,
                    fit,
                    epoch,
                    cache_path=run
                    / "sensitivity_cache"
                    / f"{seconds}s"
                    / fold["id"]
                    / f"seed_{seed}.json",
                )
                rows.append(
                    {
                        "window": seconds,
                        "fold": fold["id"],
                        "kind": "seed",
                        "seed": seed,
                        **metrics,
                    }
                )
            search = read_json(folder / "search_results.json")
            static = [
                {**r, "recipe": recipe_from_dict(r["recipe"])}
                for r in search
                if r.get("status") == "ok"
                and not r["recipe"]["history"]
                and r["recipe"]["model"]
                in (
                    "logistic",
                    "svm",
                    "xgboost",
                    "hgb",
                    "catboost",
                    "random_forest",
                    "extra_trees",
                )
            ]
            best_static = select_record(static)["recipe"]
            people = sorted(train.person_id.unique())
            ordered = np.random.default_rng(config.seed).permutation(people)
            for recipe in (Recipe(), best_static):
                for fraction in (0.5, 0.75, 1.0):
                    count = max(1, int(np.ceil(fraction * len(people))))
                    while (
                        count < len(people)
                        and train[train.person_id.isin(ordered[:count])].y.nunique()
                        != 3
                    ):
                        count += 1
                    selected = train[train.person_id.isin(ordered[:count])]
                    if recipe.thin == 10:
                        selected = thin_training(selected)
                    metrics, prediction = _score_fit(
                        recipe,
                        selected,
                        test,
                        store,
                        seconds,
                        config,
                        fit,
                        cache_path=run
                        / "sensitivity_cache"
                        / f"{seconds}s"
                        / fold["id"]
                        / f"{recipe.key}_{fraction}.json",
                    )
                    rows.append(
                        {
                            "window": seconds,
                            "fold": fold["id"],
                            "kind": "learning_curve",
                            "recipe": recipe.key,
                            "fraction": fraction,
                            "actual_people": count,
                            **metrics,
                        }
                    )
                    if fraction == 1.0 and recipe == Recipe():
                        baseline = test[["window_id", "person_id", "y"]].copy()
                        baseline["baseline"], baseline["window_seconds"] = (
                            prediction,
                            seconds,
                        )
                        baseline_predictions.append(baseline)
            # Persist cumulative results so a checkpoint contains the finished fold.
            write_json(run / "sensitivity.json", rows)
            checkpoint(force=True)
    write_json(run / "sensitivity.json", rows)
    if baseline_predictions:
        baseline = pd.concat(baseline_predictions).drop_duplicates(
            ["window_id", "window_seconds"]
        )
        selected = pd.read_csv(run / "oof_predictions.csv")
        selected = selected[
            (selected.comparison == "nested_selected") & (selected.support == "native")
        ]
        comparisons = {}
        for seconds, part in selected.groupby("window_seconds"):
            joined = part.merge(
                baseline[baseline.window_seconds == seconds][["window_id", "baseline"]],
                on="window_id",
                validate="one_to_one",
            )
            comparisons[int(seconds)] = paired_bootstrap(joined, "y_pred", "baseline")
        write_json(run / "paired_bootstrap.json", comparisons)
