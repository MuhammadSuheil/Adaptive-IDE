"""Independent score/partition audit and completed-run resume; no new fitting."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "models/batch-05/outputs/exp01/stage2_20261122T104602264199Z_f8aacc8829"
sys.path.insert(0, str(ROOT / "tests"))
from test_batch05_preparation import definitions


def main():
    ns = definitions()
    development = ns["DEVELOPMENT"]
    reference = ns["REFERENCE"]
    dataset = ROOT / "analysis_sessions/dataset_fusion/output/v2_session_phase"
    locked = ns["locked_data"](
        dataset, development, reference, ns["EXPECTED_HASHES"], ns["MANIFEST_HASH"]
    )
    frame = locked["frame"]
    truth = frame.set_index("window_id")._y
    test = set(frame.loc[frame._split.eq("test"), "window_id"])
    new_train = set(frame.iloc[locked["train"]].window_id)
    old_train = set(frame.iloc[locked["old_train"]].window_id)
    old_val = set(frame.iloc[locked["old_validation"]].window_id)
    new_val = set(frame.iloc[locked["validation"]].window_id)
    lookup = {v: i for i, v in enumerate(frame.window_id)}
    jobs = list((RUN / "jobs").glob("*/result.json"))
    assert len(jobs) == 71
    snapshots = {
        str(path.relative_to(RUN)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (RUN / "jobs").rglob("*")
        if path.is_file()
    }
    checked_scores = 0
    for path in jobs:
        record = json.loads(path.read_text())
        assert record["converged"] and not record["warnings"]
        fitting = pd.read_csv(path.parent / "fit_predictions.csv")
        held = pd.read_csv(path.parent / "held_predictions.csv")
        for table, metric in ((fitting, "fit_metrics"), (held, "held_metrics")):
            assert not set(table.window_id) & test
            assert table.target.tolist() == table.window_id.map(truth).tolist()
            y, p = table.target.to_numpy(), table.prediction.to_numpy()
            precision, recall, f1, _ = precision_recall_fscore_support(
                y, p, labels=[0, 1, 2], zero_division=0
            )
            expected = record[metric]
            assert np.isclose(accuracy_score(y, p), expected["accuracy"])
            assert np.isclose(
                balanced_accuracy_score(y, p), expected["balanced_accuracy"]
            )
            assert np.isclose(f1.mean(), expected["macro_f1"])
            assert np.allclose(precision, expected["precision"])
            assert np.allclose(recall, expected["recall"])
            assert np.allclose(f1, expected["f1"])
            assert (
                confusion_matrix(y, p, labels=[0, 1, 2]).tolist()
                == expected["confusion_matrix"]
            )
            checked_scores += 1
        fit_ids = np.array([lookup[v] for v in fitting.window_id])
        held_ids = np.array([lookup[v] for v in held.window_id])
        is_reference = path.parent.name.startswith("reference_")
        ns["validate_training_indices"](
            frame, fit_ids, held_ids, 105 if is_reference else 45
        )
        if is_reference:
            assert set(fitting.window_id) == old_train
            assert set(held.window_id) == old_val
        elif path.parent.name.startswith("final_"):
            assert set(fitting.window_id) == new_train
            assert set(held.window_id) == new_val
        else:
            assert set(fitting.window_id) <= new_train
            assert set(held.window_id) <= new_train
    selection = json.loads((RUN / "selection.json").read_text())
    winner = ns["select_candidate"](
        [v for v in selection["candidates"] if v["eligible"]]
    )
    assert winner == selection["winner"]
    result = ns["run_stage2"](
        dataset,
        RUN.parent,
        development,
        reference,
        ns["EXPECTED_HASHES"],
        ns["MANIFEST_HASH"],
        ns["STAGE2_CODE_HASH"],
        RUN,
    )
    assert str(result["run_dir"]) == str(RUN)
    after = {
        str(path.relative_to(RUN)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (RUN / "jobs").rglob("*")
        if path.is_file()
    }
    assert snapshots == after
    notebook = json.loads(
        (ROOT / "models/batch-05/exp01/EyeHeartCognitiveLoad.ipynb").read_text(
            encoding="utf-8"
        )
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    evidence = {
        "verified_fit_jobs": len(jobs),
        "independently_recomputed_score_tables": checked_scores,
        "test_used_for_fitting_or_evaluation": False,
        "resume_changed_job_artifacts": False,
        "winner_selected_from_oof": winner["name"],
        "dataset_sha256": locked["hashes"],
        "manifest_sha256": ns["MANIFEST_HASH"],
    }
    ns["save_json"](RUN / "audit.json", evidence)
    print(json.dumps(evidence, indent=2), flush=True)


if __name__ == "__main__":
    main()
