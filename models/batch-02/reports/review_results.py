"""Audit exported OOF results and build report tables/figures, without training."""

import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

BATCH = Path(__file__).resolve().parents[1]
PROJECT = BATCH.parents[1]
sys.path.insert(0, str(BATCH / "shared"))
from batch02.registry import feature_group, recipes_for

LABELS = ("Low", "Medium", "High")


def sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def logical_files(root):
    files, duplicates = {}, 0
    for part in sorted(Path(root).glob("outputs-*/outputs")):
        for path in part.rglob("*"):
            if not path.is_file():
                continue
            key = path.relative_to(part).as_posix()
            if key in files:
                if sha256(path) != sha256(files[key]):
                    raise ValueError(f"Conflicting exported file: {key}")
                duplicates += 1
            else:
                files[key] = path
    if not files:
        raise ValueError("No split output export found")
    return files, duplicates


def matrix_metrics(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    correct = matrix.diagonal()
    f1 = np.divide(
        2 * correct,
        support + predicted,
        out=np.zeros(3),
        where=(support + predicted) > 0,
    )
    recall = np.divide(correct, support, out=np.zeros(3), where=support > 0)
    precision = np.divide(correct, predicted, out=np.zeros(3), where=predicted > 0)
    return {
        "macro_f1": float(f1.mean()),
        "accuracy": float(correct.sum() / matrix.sum()),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "confusion_matrix": matrix.tolist(),
        "classes": {
            label: {
                "precision": float(precision[i]) if support[i] else None,
                "recall": float(recall[i]) if support[i] else None,
                "f1": float(f1[i]) if support[i] else None,
                "support": int(support[i]),
            }
            for i, label in enumerate(LABELS)
        },
    }


def metrics(truth, predicted):
    truth, predicted = np.asarray(truth, dtype=int), np.asarray(predicted, dtype=int)
    if (
        not len(truth)
        or truth.shape != predicted.shape
        or not np.isin(truth, (0, 1, 2)).all()
        or not np.isin(predicted, (0, 1, 2)).all()
    ):
        raise ValueError("Invalid prediction labels")
    matrix = np.bincount(3 * truth + predicted, minlength=9).reshape(3, 3)
    return matrix_metrics(matrix)


def validate_oof(group, expected):
    if group.window_id.duplicated().any() or set(group.window_id) != set(expected):
        raise ValueError("Missing, additional, or duplicated OOF endpoint")
    if not np.array_equal(
        group.window_id.map(expected).to_numpy(), group.y_true.to_numpy()
    ):
        raise ValueError("OOF labels disagree with the dataset")


def nonoverlap(group):
    selected = []
    for _, session in group.groupby("session_id", sort=False):
        previous = None
        for index, row in session.sort_values("window_end").iterrows():
            start, end = pd.Timestamp(row.window_start), pd.Timestamp(row.window_end)
            if previous is None or start >= previous:
                selected.append(index)
                previous = end
    return group.loc[selected]


def close_metrics(first, second):
    for name in ("macro_f1", "accuracy", "balanced_accuracy"):
        if not np.isclose(first[name], second[name], atol=1e-12, rtol=0):
            raise ValueError(
                f"Stored {name} differs from OOF: {first[name]} != {second[name]}"
            )
    if (
        "confusion_matrix" in second
        and first["confusion_matrix"] != second["confusion_matrix"]
    ):
        raise ValueError("Stored confusion matrix differs from OOF")


def prediction_digest(group):
    value = group.sort_values("window_id")[["window_id", "y_true", "y_pred"]].to_csv(
        index=False
    )
    return hashlib.sha256(value.encode()).hexdigest()


def bootstrap(groups, seed=42, repetitions=2000):
    people = sorted(groups[0].person_id.unique())
    if any(set(group.person_id) != set(people) for group in groups):
        raise ValueError("Participant sets differ for paired bootstrap")
    weights = np.random.default_rng(seed).multinomial(
        len(people), np.full(len(people), 1 / len(people)), size=repetitions
    )
    results = []
    for group in groups:
        per_person = np.array(
            [
                metrics(part.y_true, part.y_pred)["confusion_matrix"]
                for _, part in group.groupby("person_id", sort=True)
            ]
        )
        matrices = np.einsum("rp,pij->rij", weights, per_person)
        correct = matrices.diagonal(axis1=1, axis2=2)
        denominator = matrices.sum(1) + matrices.sum(2)
        f1 = np.divide(
            2 * correct,
            denominator,
            out=np.zeros_like(correct, dtype=float),
            where=denominator > 0,
        ).mean(1)
        accuracy = correct.sum(1) / matrices.sum((1, 2))
        results.append({"f1": f1, "accuracy": accuracy})
    return results


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def run():
    files, duplicates = logical_files(BATCH / "outputs")
    dataset = PROJECT / "analysis_sessions/dataset_fusion/output/v2_session_phase"
    release_key = next(
        k for k in files if k.startswith("exp00/") and k.endswith("/release.json")
    )
    release = read_json(files[release_key])
    expected, dataset_rows = {}, []
    features = read_json(dataset / "window_25s_step_5s/feature_columns.json")[
        "features"
    ]
    for name, checksum in release["files"].items():
        if (
            name not in ("session_registry.json", "splits.json")
            and sha256(dataset / name) != checksum
        ):
            raise ValueError(f"Current dataset differs from frozen release: {name}")
    for seconds in (15, 25, 45):
        frame = pd.read_csv(
            dataset / f"window_{seconds}s_step_5s/good_acceptable.csv",
            usecols=[
                "window_id",
                "session_id",
                "window_end",
                "participant_id",
                "cognitive_load_label",
            ],
        )
        frame["y"] = frame.cognitive_load_label.map(
            {label: i for i, label in enumerate(LABELS)}
        )
        expected[seconds] = dict(zip(frame.window_id, frame.y))
        dataset_rows.append(
            {
                "window_seconds": seconds,
                "rows": len(frame),
                "people": frame.participant_id.nunique(),
                **frame.cognitive_load_label.value_counts().to_dict(),
            }
        )
    # Window IDs contain duration, so paired endpoints must use session/time/target.
    endpoint_sets = []
    for seconds in (15, 25, 45):
        data = pd.read_csv(
            dataset / f"window_{seconds}s_step_5s/good_acceptable.csv",
            usecols=["window_id", "session_id", "window_end", "cognitive_load_label"],
        )
        endpoint_sets.append(
            set(zip(data.session_id, data.window_end, data.cognitive_load_label))
        )
    shared_endpoints = set.intersection(*endpoint_sets)
    paired_expected = {}
    for seconds in (15, 25, 45):
        data = pd.read_csv(
            dataset / f"window_{seconds}s_step_5s/good_acceptable.csv",
            usecols=["window_id", "session_id", "window_end", "cognitive_load_label"],
        )
        keep = [
            (s, t, y) in shared_endpoints
            for s, t, y in zip(
                data.session_id, data.window_end, data.cognitive_load_label
            )
        ]
        paired_expected[seconds] = {
            r.window_id: LABELS.index(r.cognitive_load_label)
            for r in data[keep].itertuples()
        }
    runs = sorted(
        {
            "/".join(k.split("/")[:2])
            for k in files
            if k.endswith("/status.json")
            and k.count("/") == 2
            and k.split("/")[0] in {f"exp{i:02d}" for i in range(1, 14)}
        }
    )
    records, run_records, people_records, candidates = [], [], [], {}
    recipe_definitions = {}
    stored_groups = {}
    required = {
        "window_id",
        "session_id",
        "person_id",
        "window_start",
        "window_end",
        "y_true",
        "y_pred",
        "comparison",
        "recipe",
        "window_seconds",
        "support",
    }
    for prefix in runs:
        experiment, run_id = prefix.split("/")
        status, identity = (
            read_json(files[prefix + "/status.json"]),
            read_json(files[prefix + "/identity.json"]),
        )
        events = read_json(files[prefix + "/events.json"])
        source = identity["identity"]
        if (
            source["release"]
            != hashlib.sha256(
                json.dumps(release, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
        ):
            raise ValueError("Run release identity differs")
        definitions = {recipe.key: recipe for recipe in recipes_for(experiment)}
        recipe_definitions.update(
            {key: asdict(value) for key, value in definitions.items()}
        )
        record = {
            "experiment": experiment,
            "run_id": run_id,
            "status": status["status"],
            "training_performed": status["training_performed"],
            "reported_summary_rows": status["summary_rows"],
            "event_reasons": dict(
                Counter(e.get("reason", e.get("status")) for e in events)
            ),
            "code": source["code"],
            "release": source["release"],
            "stream": source["stream"],
            "seed": source["seed"],
            "environment": source["environment"],
            "capabilities": source["capabilities"],
            "source_status": files[prefix + "/status.json"]
            .relative_to(BATCH)
            .as_posix(),
            "checkpoint_files_local": sum(
                k.startswith(prefix + "/checkpoints/") for k in files
            ),
            "local_model_files": sum(
                k.startswith(prefix + "/") and k.endswith(".joblib") for k in files
            ),
        }
        run_records.append(record)
        summary_path = files.get(prefix + "/summary.csv")
        if summary_path is None:
            if status["training_performed"] or status["summary_rows"]:
                raise ValueError("A trained run is missing summary.csv")
            continue
        summary = pd.read_csv(summary_path)
        if len(summary) != status["summary_rows"]:
            raise ValueError("Summary length differs from status")
        stored_metrics = read_json(files[prefix + "/metrics.json"])
        predictions = pd.read_csv(
            files[prefix + "/oof_predictions.csv"], usecols=lambda c: c in required
        )
        if predictions.y_true.isna().any() or predictions.y_pred.isna().any():
            raise ValueError("Missing OOF labels")
        for (seconds, comparison, support), group in predictions.groupby(
            ["window_seconds", "comparison", "support"], sort=True
        ):
            seconds = int(seconds)
            validate_oof(
                group,
                paired_expected[seconds] if support == "paired" else expected[seconds],
            )
            recalculated = metrics(group.y_true, group.y_pred)
            row = summary[
                (summary.window_seconds == seconds)
                & (summary.comparison == comparison)
                & (summary.support == support)
            ]
            if (
                len(row) != 1
                or not bool(row.iloc[0].complete_oof)
                or row.iloc[0].rows != len(group)
            ):
                raise ValueError("OOF/summary completeness mismatch")
            close_metrics(recalculated, row.iloc[0].to_dict())
            close_metrics(
                recalculated, stored_metrics[f"{seconds}/{support}/{comparison}"]
            )
            recipe = definitions[comparison]
            calculated = {
                "experiment": experiment,
                "run_id": run_id,
                "window_seconds": seconds,
                "support": support,
                "comparison": comparison,
                "model": recipe.model,
                "feature_group": recipe.features,
                "feature_count": len(
                    feature_group(recipe.features, features, recipe.baseline_ratio)
                ),
                "baseline_ratio": recipe.baseline_ratio,
                "normalization": recipe.normalization,
                "history": recipe.history,
                "selector": recipe.selector,
                "balance": recipe.balance,
                "thin": recipe.thin,
                "params": json.dumps(recipe.params, sort_keys=True),
                "rows": len(group),
                "people": group.person_id.nunique(),
                "macro_f1": recalculated["macro_f1"],
                "accuracy": recalculated["accuracy"],
                "balanced_accuracy": recalculated["balanced_accuracy"],
                "complete_oof": True,
                "prediction_sha256": prediction_digest(group),
                "source_summary": summary_path.relative_to(BATCH).as_posix(),
            }
            for label in LABELS:
                for metric, value in recalculated["classes"][label].items():
                    calculated[f"{label}_{metric}"] = value
            records.append(calculated)
            path = f"{prefix}/window_{seconds}s/{support}/{comparison}/nonoverlap_metrics.json"
            if path in files:
                selected = nonoverlap(group)
                close_metrics(
                    metrics(selected.y_true, selected.y_pred), read_json(files[path])
                )
            for person, rows in group.groupby("person_id"):
                person_metric = metrics(rows.y_true, rows.y_pred)
                people_records.append(
                    {
                        "experiment": experiment,
                        "window_seconds": seconds,
                        "comparison": comparison,
                        "support": support,
                        "person_id": person,
                        "rows": len(rows),
                        "true_classes": rows.y_true.nunique(),
                        "accuracy": person_metric["accuracy"],
                        "macro_f1_fixed_three_classes": person_metric["macro_f1"],
                        "balanced_accuracy_present_classes": person_metric[
                            "balanced_accuracy"
                        ],
                    }
                )
            if support == "native" and (
                (experiment == "exp02" and comparison == "logistic_499f533391d1cfa0")
                or (
                    experiment == "exp09"
                    and comparison == "late_fusion_eae7b1017659a2c1"
                )
                or (
                    experiment == "exp01"
                    and comparison
                    in ("logistic_b7381697047dac41", "dummy_6b62e62e812efb0e")
                )
            ):
                candidates[(experiment, seconds, comparison)] = group.copy()
                stored_groups[(experiment, seconds, comparison)] = recalculated
        print(f"Verified {experiment}: {len(summary)} OOF configurations", flush=True)
    table = pd.DataFrame(records).sort_values(["macro_f1", "accuracy"], ascending=False)
    native = table[table.support == "native"]
    best = native.iloc[0]
    if (
        best.experiment != "exp02"
        or best.comparison != "logistic_499f533391d1cfa0"
        or best.window_seconds != 25
    ):
        raise ValueError(
            "Review's focal comparison needs updating for a different winner"
        )
    winner = candidates[("exp02", 25, best.comparison)]
    late = candidates[("exp09", 25, "late_fusion_eae7b1017659a2c1")]
    baseline = candidates[("exp01", 25, "logistic_b7381697047dac41")]
    simulations = bootstrap([winner, late, baseline])
    intervals = {
        name: {
            metric: np.quantile(values[metric], [0.025, 0.975]).tolist()
            for metric in ("f1", "accuracy")
        }
        for name, values in zip(
            ("winner", "late_fusion", "baseline_fusion"), simulations
        )
    }
    intervals["winner_minus_late_fusion_f1"] = np.quantile(
        simulations[0]["f1"] - simulations[1]["f1"], [0.025, 0.975]
    ).tolist()
    intervals["winner_minus_baseline_fusion_f1"] = np.quantile(
        simulations[0]["f1"] - simulations[2]["f1"], [0.025, 0.975]
    ).tolist()
    winner_metrics = metrics(winner.y_true, winner.y_pred)
    winner_metrics["nonoverlap"] = metrics(
        nonoverlap(winner).y_true, nonoverlap(winner).y_pred
    )
    winner_metrics["nonoverlap_rows"] = len(nonoverlap(winner))
    winner_metrics["bootstrap"] = {
        "unit": "person",
        "method": "paired multinomial cluster bootstrap of fixed OOF predictions",
        "seed": 42,
        "repetitions": 2000,
        "selection_adjusted": False,
        "intervals": intervals,
    }
    winner_metrics["configuration"] = recipe_definitions[best.comparison]
    winner_metrics["feature_names"] = feature_group("F6", features, False)
    person_table = pd.DataFrame(people_records)
    person_best = person_table[
        (person_table.experiment == "exp02")
        & (person_table.window_seconds == 25)
        & (person_table.comparison == best.comparison)
    ]
    winner_metrics["participant_summary"] = {
        "count": len(person_best),
        "full_three_class_count": int(person_best.true_classes.eq(3).sum()),
        "accuracy_min": float(person_best.accuracy.min()),
        "accuracy_max": float(person_best.accuracy.max()),
        "accuracy_median": float(person_best.accuracy.median()),
        "accuracy_mean_equal_person": float(person_best.accuracy.mean()),
    }
    controls = native[native.comparison == "logistic_b7381697047dac41"]
    control_consistency = all(
        group.prediction_sha256.nunique() == 1
        for _, group in controls.groupby("window_seconds")
    )
    if not control_consistency:
        raise ValueError("Shared control predictions differ between experiments")
    notebook_audit = []
    for i in range(1, 14):
        notebook_path = BATCH / f"exp{i:02d}/EyeHeartCognitiveLoad.ipynb"
        notebook = read_json(notebook_path)
        errors = [
            o.get("ename")
            for c in notebook["cells"]
            for o in c.get("outputs", [])
            if o.get("output_type") == "error"
        ]
        output = "\n".join(
            "".join(o.get("text", []))
            for c in notebook["cells"]
            for o in c.get("outputs", [])
            if o.get("output_type") == "stream"
        )
        run_id = next(
            r["run_id"] for r in run_records if r["experiment"] == f"exp{i:02d}"
        )
        if errors or run_id not in output:
            raise ValueError(f"Notebook outputs do not match exported run: exp{i:02d}")
        notebook_audit.append(
            {"experiment": f"exp{i:02d}", "errors": errors, "run_id_matches": True}
        )
    report = BATCH / "reports"
    report.mkdir(exist_ok=True)
    table.to_csv(report / "leaderboard.csv", index=False)
    native.groupby("experiment", sort=True).head(1).sort_values("experiment").to_csv(
        report / "best_per_experiment.csv", index=False
    )
    person_best.to_csv(report / "best_per_participant.csv", index=False)
    pd.DataFrame(dataset_rows).to_csv(report / "dataset_counts.csv", index=False)
    audit = {
        "training_performed": False,
        "logical_files": len(files),
        "duplicate_identical_paths": duplicates,
        "parts": [
            p.relative_to(BATCH).as_posix()
            for p in sorted((BATCH / "outputs").glob("outputs-*/outputs"))
        ],
        "release": release,
        "runs": run_records,
        "notebooks": notebook_audit,
        "verified_summary_rows": len(table),
        "native_summary_rows": len(native),
        "paired_summary_rows": int(table.support.eq("paired").sum()),
        "distinct_native_config_window_pairs": len(
            native.drop_duplicates(["window_seconds", "comparison"])
        ),
        "paired_endpoints": len(shared_endpoints),
        "common_control_predictions_identical": control_consistency,
        "winner": best.to_dict(),
        "highest_accuracy_native": native.sort_values("accuracy", ascending=False)
        .iloc[0]
        .to_dict(),
    }
    write_json(report / "audit.json", audit)
    write_json(report / "winner_metrics.json", winner_metrics)
    write_json(report / "recipes.json", recipe_definitions)
    make_figures(table, winner_metrics, report)
    print(
        json.dumps(
            {
                "summary_rows": len(table),
                "winner": best[
                    [
                        "experiment",
                        "comparison",
                        "window_seconds",
                        "macro_f1",
                        "accuracy",
                    ]
                ].to_dict(),
                "intervals": intervals,
                "participant_summary": winner_metrics["participant_summary"],
            },
            indent=2,
        )
    )


def make_figures(table, winner, report):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    folder = report / "figures"
    folder.mkdir(exist_ok=True)
    native = table[table.support == "native"]
    best = native.groupby("experiment", sort=True).head(1).sort_values("experiment")
    figure = Figure(figsize=(9, 6.2))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    bars = axis.barh(
        best.experiment,
        100 * best.macro_f1,
        color=[
            {15: "#315f9b", 25: "#29836d", 45: "#a46532"}[s]
            for s in best.window_seconds
        ],
    )
    axis.invert_yaxis()
    axis.set(
        xlim=(0, 55),
        xlabel="Macro-F1 (%)",
        title="Hasil native tertinggi per eksperimen",
    )
    for bar, row in zip(bars, best.itertuples()):
        axis.text(
            bar.get_width() + 0.6,
            bar.get_y() + bar.get_height() / 2,
            f"{row.macro_f1 * 100:.2f}% | {row.window_seconds}s",
            va="center",
            fontsize=9,
        )
    axis.spines[["top", "right"]].set_visible(False)
    figure.text(
        0.1,
        0.015,
        "Exp12: TabPFN dilewati. Exp14: tidak diselesaikan. Cakupan test antarwindow berbeda.",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(folder / "best_per_experiment.png", dpi=160)
    figure = Figure(figsize=(6.7, 5.1))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    matrix = np.asarray(winner["confusion_matrix"])
    normalized = matrix / matrix.sum(1, keepdims=True)
    axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set(
        xticks=range(3),
        yticks=range(3),
        xticklabels=LABELS,
        yticklabels=LABELS,
        xlabel="Prediksi",
        ylabel="Label dataset",
        title="Logistic Regression eye-only, window 25 detik",
    )
    for i in range(3):
        for j in range(3):
            axis.text(
                j,
                i,
                f"{matrix[i, j]}\n{100 * normalized[i, j]:.1f}%",
                ha="center",
                va="center",
                color="white" if normalized[i, j] > 0.65 else "black",
                fontsize=11,
            )
    figure.tight_layout()
    figure.savefig(folder / "winner_confusion_matrix.png", dpi=160)
    figure = Figure(figsize=(8.5, 4.8))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    for i, (group, label, color) in enumerate(
        (
            ("F1", "Fusion", "#315f9b"),
            ("F5", "HRV saja", "#a46532"),
            ("F6", "Eye saja", "#29836d"),
        )
    ):
        rows = table[
            (table.experiment == "exp02")
            & (table.model == "logistic")
            & (table.feature_group == group)
            & ~table.baseline_ratio
        ].sort_values("window_seconds")
        axis.bar(
            np.arange(3) + (i - 1) * 0.25,
            rows.macro_f1 * 100,
            width=0.25,
            label=label,
            color=color,
        )
    axis.set(
        xticks=range(3),
        xticklabels=["15 detik", "25 detik", "45 detik"],
        ylim=(0, 55),
        ylabel="Macro-F1 (%)",
        title="Perbandingan sensor pada Logistic Regression yang sama",
    )
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(folder / "sensor_ablation.png", dpi=160)


if __name__ == "__main__":
    run()
