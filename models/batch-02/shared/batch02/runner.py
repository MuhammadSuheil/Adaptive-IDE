"""Notebook entry points. Training requires an approved immutable release."""

from dataclasses import asdict
from time import perf_counter

import numpy as np
import pandas as pd

from .artifacts import (
    code_hash,
    digest,
    environment,
    file_hash,
    open_run,
    read_json,
    write_json,
)
from .contracts import inspect_dataset, load_release
from .features import FeatureStore
from .metrics import classification_metrics
from .modeling import fit_candidate
from .registry import experiment_windows, model_bank, recipes_for
from .reporting import (
    prediction_frame,
    summarize_predictions,
    transform_metadata,
    write_run_notes,
)
from .runtime import capabilities, require_core_environment
from .search import Evaluator, adaptive_search, thin_training, vote_from_records
from .splits import make_manifests, paired_endpoints, restrict_paired
from .workload import estimate_work


def readiness(config):
    inspected = inspect_dataset(config)
    path = config.batch / "registry" / "dataset_release.json"
    release = (
        read_json(path)
        if path.exists()
        else {
            "gates": {
                "additional_data": "pending",
                "proportions": "pending",
                "frozen": "pending",
            }
        }
    )
    return {
        "dataset": inspected["summary"],
        "release": release,
        "capabilities": capabilities(config),
        "status": "validation_only",
        "training_performed": False,
    }


def registry_draft(config):
    inspected = inspect_dataset(config)
    sessions = pd.concat(
        [
            frame[["session_id", "participant_id"]]
            for frame in inspected["frames"].values()
        ]
    ).drop_duplicates()
    return {
        "sessions": [
            {
                "session_id": row.session_id,
                "person_id": row.participant_id,
                "confirmed": False,
            }
            for row in sessions.itertuples(index=False)
        ]
    }


def describe_experiment(experiment):
    recipes = recipes_for(experiment) if experiment not in ("exp00", "exp14") else []
    return pd.DataFrame(
        [
            {
                "recipe_id": r.key,
                **{k: v for k, v in asdict(r).items() if k != "children"},
            }
            for r in recipes
        ]
    )


def _context(config):
    release, frames, features, manifests = load_release(config)
    require_core_environment(config.batch)
    root = config.batch / "inputs" / release["release_id"] / "streams"
    store = FeatureStore(features, root, release["release_id"])
    cache_identity = None
    if (root / "stream_manifest.json").exists():
        store.verify_against(frames)
        cache_identity = file_hash(root / "stream_manifest.json")
        for seconds, frame in frames.items():
            stream = store.stream(seconds)
            origins = (
                stream.groupby("session_id").window_start.first().map(pd.Timestamp)
            )
            frame["grid_origin"] = pd.to_datetime(
                frame.session_id.map(origins), utc=True
            )
    return release, frames, features, manifests, store, cache_identity


def _identity(config, experiment, release, cache_identity):
    return {
        "release": digest(release),
        "experiment": experiment,
        "code": code_hash(),
        "environment": environment(),
        "seed": config.seed,
        "threads": config.threads,
        "stream": cache_identity,
        "checkpoint": config.tabpfn_checkpoint_sha256,
        "checkpoint_license": config.tabpfn_license_confirmed,
        "capabilities": capabilities(config),
        "requirements_sha256": file_hash(config.batch / "requirements.txt"),
        "protocol": "batch02_v2_three_windows_colab_stopping34_pool24",
    }


def _outer_result(
    evaluator, record, train_indices, test_indices, comparison, support, folder
):
    recipe, frame, store = record["recipe"], evaluator.frame, evaluator.store
    metadata_path = folder / "metrics.json"
    predictions_path = folder / "predictions.csv"
    if metadata_path.exists() and predictions_path.exists():
        return pd.read_csv(predictions_path)
    train, test = frame.iloc[train_indices].copy(), frame.iloc[test_indices].copy()
    if recipe.thin == 10:
        train = thin_training(train)
    fitted = evaluator.fit(
        recipe,
        train,
        store,
        evaluator.seconds,
        evaluator.config,
        epochs=record.get("epoch"),
    )
    values, kind = fitted.predict(test, store, evaluator.seconds)
    fallback_count = getattr(fitted, "fallback_count", 0)
    test_metrics = classification_metrics(
        test.y, values.argmax(1), values if kind == "probability" else None
    )
    train_values, train_kind = fitted.predict(train, store, evaluator.seconds)
    train_metrics = classification_metrics(
        train.y,
        train_values.argmax(1),
        train_values if train_kind == "probability" else None,
    )
    predictions = prediction_frame(
        test, values, kind, comparison, recipe, evaluator.seconds, support
    )
    folder.mkdir(parents=True, exist_ok=True)
    import joblib

    model_path = folder / "pipeline.joblib"
    joblib.dump(fitted, model_path)
    restored = joblib.load(model_path)  # nosec B301
    reference = restored.predict(
        test.iloc[: min(4, len(test))], store, evaluator.seconds
    )[0]
    if not np.allclose(reference, values[: len(reference)], atol=1e-5, rtol=1e-5):
        raise ValueError(
            "Persist/load atau batch-versus-endpoint inference parity gagal."
        )
    predictions.to_csv(predictions_path, index=False)
    write_json(folder / "recipe.json", asdict(recipe))
    write_json(
        folder / "features.json",
        {
            "features": fitted.feature_names,
            "epoch": fitted.epoch,
            "transforms": transform_metadata(fitted),
        },
    )
    write_json(
        metadata_path,
        {
            "train": train_metrics,
            "test": test_metrics,
            "inner": record["score"],
            "test_fallback_rows": fallback_count,
            "pipeline_sha256": file_hash(model_path),
        },
    )
    return predictions


def _save_search(path, records):
    write_json(
        path,
        [
            {
                **{
                    k: v
                    for k, v in r.items()
                    if k not in ("values", "predicted", "y_true")
                },
                "recipe": asdict(r["recipe"]),
            }
            for r in records
        ],
    )


def run_experiment(config, experiment, resume=None, fit=fit_candidate):
    if experiment == "exp14":
        return run_final(config, resume, fit=fit)
    release, frames, _features, manifests, store, cache_identity = _context(config)
    if experiment not in [f"exp{i:02d}" for i in range(1, 14)]:
        raise ValueError(
            "Gunakan exp00 untuk penerimaan data; training tersedia pada exp01–exp14."
        )
    run = open_run(
        config.batch / "outputs" / experiment,
        _identity(config, experiment, release, cache_identity),
        resume,
    )
    from filelock import FileLock

    with FileLock(str(run / "writer.lock"), timeout=0):
        paired = paired_endpoints(frames)
        paired_manifests = (
            make_manifests(
                {s: restrict_paired(f, paired) for s, f in frames.items()}, config.seed
            )
            if experiment == "exp04"
            else None
        )
        workload = estimate_work(
            experiment,
            manifests,
            paired_manifests=paired_manifests,
            capabilities=capabilities(config),
        )
        write_json(run / "workload.json", workload)
        print("Rencana pekerjaan:", workload)
        write_json(run / "status.json", {"status": "running", "experiment": experiment})
        write_json(run / "capabilities.json", capabilities(config))
        all_predictions, expected, events = [], {}, []
        try:
            for seconds in experiment_windows(experiment):
                for support in (
                    ("native", "paired") if experiment == "exp04" else ("native",)
                ):
                    frame = (
                        restrict_paired(frames[seconds], paired)
                        if support == "paired"
                        else frames[seconds]
                    )
                    if frame.empty:
                        events.append(
                            {
                                "window": seconds,
                                "support": support,
                                "status": "no_test_support",
                            }
                        )
                        continue
                    folds = (
                        make_manifests({seconds: frame}, config.seed)["windows"][
                            str(seconds)
                        ]["folds"]
                        if support == "paired"
                        else manifests["windows"][str(seconds)]["folds"]
                    )
                    for fold in folds:
                        if fold["status"] == "no_test_support":
                            events.append(fold)
                            continue
                        folder = run / f"window_{seconds}s" / support / fold["id"]
                        print(
                            f"{experiment} / {seconds}s / {support} / {fold['id']} / {fold['person']}",
                            flush=True,
                        )
                        evaluator = Evaluator(
                            frame, store, seconds, config, folder / "inner", fit
                        )
                        recipes = [
                            r for r in recipes_for(experiment) if r.support == support
                        ]
                        vote_label = None
                        if experiment == "exp09":
                            vote_label = next(
                                r.key for r in recipes if r.model == "vote"
                            )
                            scored = [
                                evaluator.score(r, fold["inner"])
                                for r in model_bank()
                                if r.model != "svm"
                            ]
                            recipes = [r for r in recipes if r.model != "vote"] + [
                                vote_from_records(scored)
                            ]
                        records = []
                        for recipe in recipes:
                            started = perf_counter()
                            write_json(
                                run / "progress.json",
                                {
                                    "window": seconds,
                                    "support": support,
                                    "fold": fold["id"],
                                    "recipe": recipe.key,
                                    "stage": "inner_selection",
                                },
                            )
                            record = evaluator.score(recipe, fold["inner"])
                            records.append(record)
                            comparison = (
                                vote_label if recipe.model == "vote" else recipe.key
                            )
                            expected[(seconds, comparison, support)] = len(frame)
                            if record["status"] != "ok":
                                events.append(
                                    {
                                        "window": seconds,
                                        "fold": fold["id"],
                                        "recipe": recipe.key,
                                        "status": record["status"],
                                        "reason": record.get("reason"),
                                    }
                                )
                                continue
                            result = _outer_result(
                                evaluator,
                                record,
                                fold["train"],
                                fold["test"],
                                comparison,
                                support,
                                folder / recipe.key,
                            )
                            all_predictions.append(result)
                            print(
                                f"  {recipe.model} / {recipe.features} / {perf_counter() - started:.1f}s",
                                flush=True,
                            )
                        _save_search(folder / "search_results.json", records)
            if all_predictions:
                summary = summarize_predictions(
                    pd.concat(all_predictions, ignore_index=True), run, expected
                )
            else:
                summary = pd.DataFrame()
            write_json(run / "events.json", events)
            if not summary.empty:
                write_run_notes(run, summary, experiment)
            write_json(
                run / "status.json",
                {
                    "status": "complete_with_skips" if events else "complete",
                    "training_performed": bool(all_predictions),
                    "summary_rows": len(summary),
                },
            )
        except Exception as error:
            write_json(
                run / "status.json",
                {"status": "failed", "error": f"{type(error).__name__}: {error}"},
            )
            raise
    return run, summary


def run_final(config, resume=None, fit=fit_candidate):
    release, frames, _features, manifests, store, cache_identity = _context(config)
    run = open_run(
        config.batch / "outputs" / "exp14",
        _identity(config, "exp14", release, cache_identity),
        resume,
    )
    from filelock import FileLock

    with FileLock(str(run / "writer.lock"), timeout=0):
        workload = estimate_work("exp14", manifests)
        write_json(run / "workload.json", workload)
        print("Rencana pekerjaan:", workload)
        write_json(run / "status.json", {"status": "running"})
        predictions, expected = [], {}
        try:
            for seconds, frame in frames.items():
                for fold in manifests["windows"][str(seconds)]["folds"]:
                    if fold["status"] == "no_test_support":
                        continue
                    directory = run / f"window_{seconds}s" / fold["id"]
                    print(
                        f"exp14 / {seconds}s / {fold['id']} / {fold['person']}",
                        flush=True,
                    )
                    write_json(
                        run / "progress.json",
                        {
                            "window": seconds,
                            "fold": fold["id"],
                            "stage": "adaptive_search",
                        },
                    )
                    evaluator = Evaluator(
                        frame, store, seconds, config, directory / "inner", fit
                    )
                    chosen, records, budget = adaptive_search(evaluator, fold["inner"])
                    _save_search(directory / "search_results.json", records)
                    write_json(
                        directory / "selection.json",
                        {"recipe": asdict(chosen["recipe"]), "budget": budget},
                    )
                    result = _outer_result(
                        evaluator,
                        chosen,
                        fold["train"],
                        fold["test"],
                        "nested_selected",
                        "native",
                        directory / "selected",
                    )
                    predictions.append(result)
                expected[(seconds, "nested_selected", "native")] = len(frame)
            # Joint-window selection is evaluated on an identical validation/test endpoint set.
            from .finalization import (
                joint_window_evaluation,
                refit_release,
                sensitivity_checks,
            )

            _joint, _joint_expected, joint_status = joint_window_evaluation(
                config, frames, store, run, fit
            )
            summary = summarize_predictions(
                pd.concat(predictions, ignore_index=True), run, expected
            )
            refit_release(config, frames, store, run, joint_status, fit)
            sensitivity_checks(config, frames, store, manifests, run, fit)
            from .diagnostics import coverage_report, negative_controls

            coverage_report(
                frames, store, config.batch / "inputs" / release["release_id"], run
            )
            negative_controls(config, frames, store, manifests, run)
            write_run_notes(run, summary, "exp14")
            write_json(
                run / "status.json",
                {
                    "status": "complete",
                    "training_performed": True,
                    "joint": joint_status,
                },
            )
        except Exception as error:
            write_json(
                run / "status.json",
                {"status": "failed", "error": f"{type(error).__name__}: {error}"},
            )
            raise
    return run, summary
