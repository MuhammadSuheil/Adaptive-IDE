"""Independent score/role audit and a no-refitting resume of Stage 3."""

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "models/batch-05/outputs/exp01/stage3_20261122T115353575899Z_933c7b4bcb"
STAGE2 = ROOT / "models/batch-05/outputs/exp01/stage2_20261122T104602264199Z_f8aacc8829"
sys.path.insert(0, str(ROOT / "tests"))
from test_batch05_preparation import definitions


def metric_check(y, predictions, stored):
    y, predictions = np.asarray(y), np.asarray(predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predictions, labels=[0, 1, 2], zero_division=0
    )
    assert np.isclose(stored["accuracy"], accuracy_score(y, predictions))
    assert np.isclose(
        stored["balanced_accuracy"], balanced_accuracy_score(y, predictions)
    )
    assert np.isclose(stored["macro_f1"], f1.mean())
    assert np.allclose(stored["precision"], precision)
    assert np.allclose(stored["recall"], recall)
    assert np.allclose(stored["f1"], f1)
    assert (
        stored["confusion_matrix"]
        == confusion_matrix(y, predictions, labels=[0, 1, 2]).tolist()
    )


def file_snapshot():
    return {
        str(path.relative_to(RUN)): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        )
        for path in (RUN / "jobs").rglob("*")
        if path.is_file()
    }


def main():
    ns = definitions()
    ns["stage3_device"]()
    data_root = ROOT / "analysis_sessions/dataset_fusion/output/v2_session_phase"
    data = ns["locked_data"](
        data_root,
        ns["DEVELOPMENT"],
        ns["REFERENCE"],
        ns["EXPECTED_HASHES"],
        ns["MANIFEST_HASH"],
    )
    frame = data["frame"]
    lookup = {v: i for i, v in enumerate(frame.window_id)}
    truth = frame.set_index("window_id")._y
    test_ids = set(frame.loc[frame._split.eq("test"), "window_id"])
    scores, reloaded, neural_jobs, tree_jobs = 0, 0, 0, 0
    for path in sorted((RUN / "jobs").glob("*/result.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        indices = json.loads((path.parent / "indices.json").read_text(encoding="utf-8"))
        for values in indices.values():
            assert not set(values) & test_ids
        ids = {
            role: np.array([lookup[v] for v in values])
            for role, values in indices.items()
        }
        fold_match = re.search(r"_f([0-4])(?:_|$)", path.parent.name)
        context = "cv_" + fold_match[1] if fold_match else "final"
        pool, expected_held = (
            data["folds"][int(fold_match[1])]
            if fold_match
            else (data["train"], data["validation"])
        )
        assert set(indices["held"]) == set(frame.iloc[expected_held].window_id)
        if value["kind"] == "neural" or "matched" in path.parent.name:
            assert (
                indices["fitting"] == ns["STAGE3_MONITORS"][context]["fit_window_ids"]
            )
        else:
            assert set(indices["fitting"]) == set(frame.iloc[pool].window_id)
        if value["kind"] == "neural":
            assert (
                indices["monitor"]
                == ns["STAGE3_MONITORS"][context]["monitor_window_ids"]
            )
        ns["validate_training_indices"](frame, ids["fitting"], ids["held"])
        internal = json.loads(
            (path.parent / "internal_predictions.json").read_text(encoding="utf-8")
        )
        if value["kind"] == "neural":
            neural_jobs += 1
            assert len(indices["monitor"]) >= 30
            ns["validate_neural_roles"](
                frame, ids["fitting"], ids["monitor"], ids["held"]
            )
            history = pd.read_csv(path.parent / "history.csv")
            assert history.epoch.tolist() == list(range(1, 121))
            assert history.learning_rate.eq(0.001).all()
            loss_epoch = int(
                history.sort_values(["monitor_loss", "epoch"]).iloc[0].epoch
            )
            macro_epoch = int(
                history.sort_values(
                    ["monitor_macro_f1", "monitor_min_recall", "monitor_loss", "epoch"],
                    ascending=[False, False, True, True],
                )
                .iloc[0]
                .epoch
            )
            assert value["policies"]["loss"]["epoch"] == loss_epoch
            assert value["policies"]["macro"]["epoch"] == macro_epoch
            assert value["policies"]["last"]["epoch"] == 120
            pre_state = json.loads(
                (path.parent / "preprocessing.json").read_text(encoding="utf-8")
            )
            raw_fit = frame.iloc[ids["fitting"]][ns["CORE"]].to_numpy(float)
            assert np.allclose(
                pre_state["center"], np.median(raw_fit, axis=0), atol=1e-12
            )
            q = np.quantile(raw_fit, 0.75, axis=0) - np.quantile(raw_fit, 0.25, axis=0)
            assert np.allclose(
                pre_state["scale"], np.where(q > 1e-8, q, 1.0), atol=1e-12
            )
            pre = ns["restore_transform"](pre_state)
            for policy, record in value["policies"].items():
                table = pd.read_csv(path.parent / f"predictions_{policy}.csv")
                ns["verify_prediction_table"](frame, ids["held"], table)
                ns["table_probabilities"](table)
                metric_check(table.target, table.prediction, record["held_metrics"])
                metric_check(
                    frame.iloc[ids["fitting"]]._y,
                    internal["fitting"][policy],
                    record["fit_metrics"],
                )
                metric_check(
                    frame.iloc[ids["monitor"]]._y,
                    internal["monitor"][policy],
                    record["monitor_metrics"],
                )
                model = ns["build_stage3_mlp"]()
                model.load_state_dict(
                    torch.load(
                        path.parent / f"{policy}.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                )
                actual = ns["neural_probabilities_stage3"](
                    model,
                    pre.transform(frame.iloc[ids["held"]]).astype(np.float32),
                    torch.device("cpu"),
                )
                assert np.allclose(
                    actual, table[["probability_" + c for c in ns["LABELS"]]], atol=1e-7
                )
                assert np.array_equal(actual.argmax(1), table.prediction)
                scores += 3
                reloaded += 1
        else:
            tree_jobs += 1
            table = pd.read_csv(path.parent / "predictions.csv")
            ns["verify_prediction_table"](frame, ids["held"], table)
            ns["table_probabilities"](table)
            metric_check(table.target, table.prediction, value["held_metrics"])
            metric_check(
                frame.iloc[ids["fitting"]]._y, internal["fitting"], value["fit_metrics"]
            )
            scores += 2
    assert neural_jobs == 48 and tree_jobs == 24
    selection = json.loads((RUN / "selection.json").read_text(encoding="utf-8"))
    oof = pd.read_csv(RUN / "oof_predictions.csv")
    assert oof.candidate.nunique() == 13
    for name, rows in oof.groupby("candidate"):
        assert len(rows) == 2724 and not rows.window_id.duplicated().any()
        assert set(rows.window_id) == set(frame.iloc[data["train"]].window_id)
        assert rows.target.tolist() == rows.window_id.map(truth).tolist()
        candidate = next(c for c in selection["candidates"] if c["name"] == name)
        metric_check(rows.target, rows.prediction, candidate)
    eligible = [c for c in selection["candidates"] if min(c["recall"]) > 0]
    independent = min(
        eligible,
        key=lambda c: (
            -c["macro_f1"],
            -c["accuracy"],
            -min(c["recall"]),
            c["fold_std"],
            c["estimators"],
            ["global", "session_centered", "session_robust"].index(c["normalization"]),
            {"full": 0, "matched": 1, "neural": 2}[c["source"]],
            ["loss", "macro", "last"].index(c["policy"])
            if c["source"] == "neural"
            else 0,
        ),
    )
    assert selection["winner"] == independent
    baseline_old = (
        pd.read_csv(STAGE2 / "oof_predictions.csv")
        .query("candidate == 'extra_trees__session_robust'")
        .set_index("window_id")
    )
    baseline_new = oof.query(
        "candidate == 'extra_trees_full__session_robust'"
    ).set_index("window_id")
    assert np.array_equal(
        baseline_old.loc[baseline_new.index].prediction, baseline_new.prediction
    )
    reference = pd.read_csv(RUN / "test_reference_predictions.csv")
    assert len(reference) == 683 and set(reference.window_id) == test_ids
    test_record = json.loads((RUN / "test_reference.json").read_text(encoding="utf-8"))
    metric_check(reference.target, reference.prediction, test_record["metrics"])
    assert reference.target.tolist() == reference.window_id.map(truth).tolist()
    before = file_snapshot()
    test_before = (RUN / "test_reference.json").read_bytes()
    # The completed run must resume without any optimizer fitting or Test re-evaluation.
    ns["run_stage3"](
        data_root,
        RUN.parent,
        ns["DEVELOPMENT"],
        ns["REFERENCE"],
        ns["EXPECTED_HASHES"],
        ns["MANIFEST_HASH"],
        ns["STAGE3_MONITORS"],
        ns["MONITOR_HASH"],
        ns["STAGE3_CODE_HASH"],
        RUN,
    )
    assert before == file_snapshot()
    assert test_before == (RUN / "test_reference.json").read_bytes()
    evidence = {
        "fit_jobs": neural_jobs + tree_jobs,
        "neural_jobs": neural_jobs,
        "tree_jobs": tree_jobs,
        "independently_recomputed_metric_tables": scores,
        "neural_checkpoint_reload_checks": reloaded,
        "all_training_roles_exclude_test": True,
        "baseline_oof_reproduces_stage2": True,
        "resume_preserved_job_hashes_and_mtimes": True,
        "test_evaluations": 1,
        "winner": independent["name"],
        "dataset_sha256": data["hashes"],
    }
    ns["save_json"](RUN / "audit.json", evidence)
    print(json.dumps(evidence, indent=2), flush=True)


if __name__ == "__main__":
    main()
