"""Definitions embedded in the standalone Batch 06 Colab notebook.

Only sensor CSVs are runtime inputs. Source timing evidence is supplied as a
small, separately checksummed reference, never as predictor columns.
"""

import hashlib
import importlib.metadata
import json
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.preprocessing import RobustScaler
from threadpoolctl import threadpool_limits


VERSION = "batch06-baseline-v1"
LABELS = ("Low", "Medium", "High")
CORE = [
    "rmssd_ms", "sdnn_ms", "blink_rate_per_minute", "ear_left_mean",
    "ear_left_std", "ear_right_mean", "ear_right_std",
    "iris_size_estimate_mean", "iris_size_estimate_std", "iris_size_delta_mean",
]
FAMILIES = ("logistic_regression", "extra_trees", "dummy")
MODELS = {
    "logistic_regression": {"C": 1.0, "max_iter": 3000},
    "extra_trees": {
        "n_estimators": 200, "min_samples_leaf": 5,
        "max_features": 0.8, "random_state": 42, "n_jobs": 2,
    },
    "dummy": {"strategy": "most_frequent"},
}


def sha256_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def bool_values(values):
    return values.astype(str).str.lower().isin(("true", "1"))


def validate_frame(frame, window):
    required = CORE + [
        "window_id", "session_id", "participant_id", "window_start", "window_end",
        "window_seconds", "cognitive_load_label", "study_phase", "quality_tier",
        "baseline_rmssd_ms", "baseline_coverage", "baseline_usable",
    ]
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f"CSV {window}s kekurangan kolom: {sorted(missing)}")
    result = frame.copy()
    if result.empty or result.window_id.duplicated().any():
        raise ValueError(f"CSV {window}s kosong atau window_id berulang.")
    for col in ("window_id", "session_id", "participant_id"):
        if result[col].isna().any() or result[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Identitas {col} kosong.")
        result[col] = result[col].astype(str)
    if not set(result.cognitive_load_label).issubset(LABELS):
        raise ValueError("Target harus Low, Medium, atau High.")
    if not set(result.quality_tier).issubset(("good", "acceptable")):
        raise ValueError("Gunakan good_acceptable.csv, bukan all_data.csv.")
    for col in ("window_start", "window_end"):
        if not result[col].map(lambda v: pd.Timestamp(v).tzinfo is not None).all():
            raise ValueError("Timestamp harus menyimpan zona waktu.")
    result["_start"] = pd.to_datetime(result.window_start, utc=True).dt.as_unit("ns").astype("int64")
    result["_end"] = pd.to_datetime(result.window_end, utc=True).dt.as_unit("ns").astype("int64")
    if not np.allclose((result._end - result._start) / 1e9, window, rtol=0, atol=1e-6):
        raise ValueError("Durasi timestamp berbeda dari panjang window.")
    if not pd.to_numeric(result.window_seconds, errors="raise").eq(window).all():
        raise ValueError("window_seconds tidak sesuai folder.")
    x = result[CORE].to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError("Core berisi missing/infinity; tidak ada imputasi.")
    nonnegative = [c for c in CORE if c != "iris_size_delta_mean"]
    if result[nonnegative].lt(0).any().any() or result.iris_size_estimate_mean.le(0).any():
        raise ValueError("Fitur sensor berada di luar rentang yang diizinkan.")
    if not result[["ear_left_mean", "ear_right_mean"]].le(1).all().all():
        raise ValueError("EAR lebih dari satu.")
    if result.groupby("session_id").participant_id.nunique().gt(1).any():
        raise ValueError("Satu sesi dipetakan ke beberapa orang.")
    result["_key"] = result.session_id + "|" + result._end.astype(str)
    if result._key.duplicated().any():
        raise ValueError("Endpoint sesi berulang.")
    result["_y"] = result.cognitive_load_label.map(dict(zip(LABELS, range(3)))).astype(int)
    return result.sort_values(["session_id", "_end"]).reset_index(drop=True)


def load_views(dataset_root):
    frames, hashes, paths = {}, {}, {}
    for window in (15, 25, 45):
        path = Path(dataset_root) / f"window_{window}s_step_5s/good_acceptable.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
        hashes[str(window)] = sha256_file(path)
        paths[str(window)] = str(path)
        frames[window] = validate_frame(pd.read_csv(path, float_precision="round_trip"), window)
    joined = pd.concat([f[["_key", "session_id", "participant_id", "_y"]] for f in frames.values()])
    if joined.groupby("_key")._y.nunique().gt(1).any():
        raise ValueError("Target endpoint berbeda antarwindow.")
    if joined.groupby("session_id").participant_id.nunique().gt(1).any():
        raise ValueError("Grup peserta berbeda antarwindow.")
    return frames, hashes, paths


def split_frame(frame, hashes, reference):
    """Reuse historical assignments only for exactly matching dataset hashes."""
    result = frame.copy()
    if hashes == reference["dataset_sha256"]:
        records = reference["development"]["window_ids"]["25"]
        mapping = {r["window_id"]: r["split"] for r in records}
        if set(mapping) != set(result.window_id):
            raise ValueError("Snapshot split tidak cocok dengan window_id.")
        result["_split"] = result.window_id.map(mapping)
        protocol = "batch05_train_only_historical_reference"
    else:
        # A new dataset gets a new, fixed, participant-separated development
        # assignment. Test is reserved and is never evaluated by this notebook.
        groups = result.participant_id.to_numpy()
        if len(set(groups)) < 8:
            raise ValueError("Dataset baru memerlukan >=8 orang untuk split grup tiga bagian.")
        outer, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(result, groups=groups))
        fit_pos, val_pos = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=43).split(result.iloc[outer], groups=groups[outer]))
        result["_split"] = "test"
        result.loc[outer[fit_pos], "_split"] = "train"
        result.loc[outer[val_pos], "_split"] = "validation"
        protocol = "new_dataset_fixed_participant_split_42_43"
        a = set(result.loc[result._split.eq("train"), "participant_id"])
        b = set(result.loc[result._split.eq("validation"), "participant_id"])
        c = set(result.loc[result._split.eq("test"), "participant_id"])
        if a & b or a & c or b & c:
            raise ValueError("Grup orang bocor antarbagian.")
    if set(result.loc[result._split.eq("train"), "_y"]) != {0, 1, 2}:
        raise ValueError("Train tidak mempunyai tiga kelas; split tidak dicari ulang.")
    return result, protocol


def audit_baselines(frame, source_evidence):
    """Choose references using time/quality metadata, never prediction scores.

    Existing source evidence contains first-task boundaries from canonical event
    ledgers and baseline completion from hrv.csv. The Drive CSV remains the
    authority for feature values. Unknown sources are reported, not guessed.
    """
    records, references = [], {}
    for sid, group in frame.groupby("session_id", sort=True):
        record = {"session_id": str(sid), "participant_id": str(group.participant_id.iloc[0])}
        source = source_evidence.get(str(sid))
        hrv_reasons, eye_reasons = [], []
        reference = {}
        if source is None:
            hrv_reasons.append("source_timing_not_audited")
            eye_reasons.append("source_timing_not_audited")
        else:
            if str(source["participant_id"]) != str(group.participant_id.iloc[0]):
                raise ValueError(f"Grup orang berbeda dari audit sumber: {sid}")
            first_task_ns = int(source["first_task_ns"])
            end_ns = int(source["baseline_end_ns"])
            record.update({
                "baseline_end_ns": end_ns, "first_task_ns": first_task_ns,
                "baseline_duration_seconds": source["baseline_duration_seconds"],
                "source_evidence": "embedded_canonical_source_audit",
            })
            usable = bool_values(group.baseline_usable)
            base = pd.to_numeric(group.baseline_rmssd_ms, errors="coerce")
            coverage = pd.to_numeric(group.baseline_coverage, errors="coerce")
            if not usable.all() or not np.isfinite(base).all() or not base.gt(0).all():
                hrv_reasons.append("baseline_unusable_in_csv")
            if not coverage.ge(0.8 - 1e-9).all():
                hrv_reasons.append("baseline_coverage_below_80_percent")
            if source["baseline_status"] != "ready":
                hrv_reasons.append("source_baseline_not_ready")
            if end_ns > first_task_ns:
                hrv_reasons.append("baseline_overlaps_first_task")
            if usable.all() and (
                not np.allclose(base, source["baseline_rmssd_ms"], rtol=1e-9, atol=1e-8)
                or not np.allclose(coverage, source["baseline_coverage"], rtol=1e-9, atol=1e-8)
            ):
                hrv_reasons.append("csv_baseline_differs_from_source_audit")
            if not hrv_reasons:
                reference.update({"rmssd": float(base.iloc[0]), "hrv_available_ns": end_ns})
                record["baseline_rmssd_ms"] = float(base.iloc[0])
            # A 25-second sensor window is a pre-task reference, not a verified
            # resting measurement. It must precede even practice/first set.
            candidates = group[
                group.study_phase.eq("pre_task")
                & group._end.lt(first_task_ns)
                & group._start.ge(end_ns)
            ].sort_values("_end")
            if candidates.empty:
                eye_reasons.append("no_eligible_initial_25s_pretask_window")
            else:
                row = candidates.iloc[0]
                if row[["ear_left_mean", "ear_right_mean", "iris_size_estimate_mean"]].le(0).any():
                    eye_reasons.append("nonpositive_eye_reference")
                else:
                    reference.update({
                        "eye_values": {col: float(row[col]) for col in CORE},
                        "eye_window_id": str(row.window_id),
                        "eye_available_ns": int(row._end),
                    })
                    record.update({"eye_window_id": str(row.window_id), "eye_reference_end_ns": int(row._end)})
            if hrv_reasons:
                eye_reasons.append("hrv_reference_unusable")
        record.update({
            "hrv_eligible": not hrv_reasons,
            "hrv_rejection_reason": ";".join(hrv_reasons),
            "joint_eligible": not eye_reasons,
            "joint_rejection_reason": ";".join(eye_reasons),
        })
        if not hrv_reasons:
            references[str(sid)] = reference
        records.append(record)
    return pd.DataFrame(records), references


def make_cohort(frame, audit, references, cohort):
    column = "hrv_eligible" if cohort == "hrv_rmssd" else "joint_eligible"
    eligible = set(audit.loc[audit[column], "session_id"])
    result = frame[frame.session_id.isin(eligible)].copy()
    kept = []
    for index, row in result.iterrows():
        ref = references[row.session_id]
        available = ref["hrv_available_ns"]
        if cohort == "hrv_eye":
            available = max(available, ref["eye_available_ns"])
        # Separate the complete feature support from the calibration interval.
        # The 1-second margin also avoids endpoint-rounding overlap.
        if int(row._start) >= available + 10**9:
            kept.append(index)
    return result.loc[kept].reset_index(drop=True)


def representation(frame, references, cohort, mode):
    x = frame[CORE].to_numpy(float).copy()
    if mode == "raw":
        return x
    if mode != "baseline_relative":
        raise ValueError("Representasi tidak dikenal.")
    for pos, row in enumerate(frame.itertuples(index=False)):
        ref = references[str(row.session_id)]
        # Only recorded RMSSD has an independent baseline in the supplied CSVs.
        x[pos, 0] = x[pos, 0] / ref["rmssd"] - 1.0
        if cohort == "hrv_eye":
            b = ref["eye_values"]
            x[pos, 2] -= b["blink_rate_per_minute"]  # zero blink baseline is valid
            for mean, std in ((3, 4), (5, 6), (7, 8)):
                denominator = b[CORE[mean]]
                if denominator <= 0:
                    raise ValueError("Baseline mean harus positif.")
                x[pos, mean] = x[pos, mean] / denominator - 1.0
                x[pos, std] /= denominator
            # Recorded delta is dimensionless; centering preserves its units.
            x[pos, 9] -= b["iris_size_delta_mean"]
    if not np.isfinite(x).all():
        raise ValueError("Normalisasi menghasilkan missing/infinity.")
    return x


def metrics(y, prediction):
    return {
        "rows": int(len(y)),
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, labels=[0, 1, 2], average="macro", zero_division=0)),
        "recall": recall_score(y, prediction, labels=[0, 1, 2], average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1, 2]).tolist(),
    }


def estimator(family):
    if family == "logistic_regression":
        return LogisticRegression(**MODELS[family])
    if family == "extra_trees":
        return ExtraTreesClassifier(**MODELS[family])
    if family == "dummy":
        return DummyClassifier(**MODELS[family])
    raise ValueError(f"Keluarga model tidak dikenal: {family}")


def fit_predict(x, y, fit_ids, held_ids, family):
    if set(y[fit_ids]) != {0, 1, 2}:
        raise ValueError("Fitting fold tidak mempunyai tiga kelas.")
    scaler = RobustScaler().fit(x[fit_ids])
    model = estimator(family)
    with threadpool_limits(limits=2):
        model.fit(scaler.transform(x[fit_ids]), y[fit_ids])
        prediction = model.predict(scaler.transform(x[held_ids]))
        fitting = model.predict(scaler.transform(x[fit_ids]))
    return prediction, fitting, {"scaler": scaler, "estimator": model}


def make_folds(train, reference, protocol):
    folds = {"participant_held_out": []}
    groups = train.participant_id.to_numpy()
    if len(set(groups)) < 4:
        return {}, "fewer_than_four_train_participants"
    y = train._y.to_numpy()
    for number, (fit_ids, held_ids) in enumerate(LeaveOneGroupOut().split(train, y, groups)):
        if set(y[fit_ids]) != {0, 1, 2}:
            return {}, "participant_fold_training_missing_class"
        if set(groups[fit_ids]) & set(groups[held_ids]):
            raise ValueError("Peserta bocor ke fold fitting.")
        folds["participant_held_out"].append((number, fit_ids, held_ids))
    if protocol.startswith("batch05"):
        historical = []
        for fold in reference["development"]["cv"]["folds"]:
            a, b = set(fold["fit_endpoint_keys"]), set(fold["validation_endpoint_keys"])
            fit_ids = np.flatnonzero(train._key.isin(a))
            held_ids = np.flatnonzero(train._key.isin(b))
            if not len(fit_ids) or not len(held_ids) or set(y[fit_ids]) != {0, 1, 2}:
                historical = []
                break
            # Independently verify the historical time separation after subset selection.
            for sid in set(train.iloc[fit_ids].session_id) & set(train.iloc[held_ids].session_id):
                left = train.iloc[fit_ids].loc[lambda f: f.session_id.eq(sid), "_end"].to_numpy()
                right = train.iloc[held_ids].loc[lambda f: f.session_id.eq(sid), "_end"].to_numpy()
                if np.abs(left[:, None] - right).min() < 45 * 10**9:
                    raise ValueError("Guard fold historis kurang dari 45 detik.")
            historical.append((int(fold["fold"]), fit_ids, held_ids))
        if historical:
            covered = np.concatenate([r[2] for r in historical])
            if sorted(covered.tolist()) != list(range(len(train))):
                raise ValueError("Fold historis tidak mencakup Train tepat sekali.")
            folds["historical_block_cv"] = historical
    return folds, ""


def prediction_table(frame, prediction, cohort, protocol, family, mode, fold):
    result = frame[["window_id", "participant_id", "session_id", "study_phase", "cognitive_load_label"]].copy()
    result["predicted_label"] = [LABELS[int(v)] for v in prediction]
    result["cohort"], result["protocol"] = cohort, protocol
    result["family"], result["representation"], result["fold"] = family, mode, fold
    return result


def evaluate_cohort(frame, references, cohort, reference, protocol, run):
    train = frame[frame._split.eq("train")].reset_index(drop=True)
    folds, reason = make_folds(train, reference, protocol)
    if reason:
        return {"status": "skipped", "reason": reason}, [], [], []
    y = train._y.to_numpy()
    summaries, fold_records, predictions = [], [], []
    # Freeze endpoint assignments and model budget before any fitting.
    manifest = {
        name: [{"fold": n, "fit_window_ids": train.iloc[a].window_id.tolist(), "held_window_ids": train.iloc[b].window_id.tolist()} for n, a, b in cases]
        for name, cases in folds.items()
    }
    save_json(run / f"{cohort}_fold_manifest.json", manifest)
    maximum_fits = sum(len(cases) for cases in folds.values()) * 5
    print(f"{cohort}: Train={len(train)}, orang={train.participant_id.nunique()}, batas CV={maximum_fits} fitting", flush=True)
    count = 0
    for name, cases in folds.items():
        for family in FAMILIES:
            modes = ("raw",) if family == "dummy" else ("raw", "baseline_relative")
            for mode in modes:
                x = representation(train, references, cohort, mode)
                oof = np.full(len(train), -1, dtype=int)
                for number, fit_ids, held_ids in cases:
                    prediction, fitting, _ = fit_predict(x, y, fit_ids, held_ids, family)
                    if (oof[held_ids] != -1).any():
                        raise ValueError("Prediksi OOF ditulis dua kali.")
                    oof[held_ids] = prediction
                    scored = metrics(y[held_ids], prediction)
                    fit_scored = metrics(y[fit_ids], fitting)
                    record = {
                        "cohort": cohort, "protocol": name, "family": family,
                        "representation": mode, "fold": number,
                        "held_participants": ";".join(sorted(set(train.iloc[held_ids].participant_id))),
                        "rows": len(held_ids), "fit_rows": len(fit_ids),
                        "accuracy": scored["accuracy"], "macro_f1": scored["macro_f1"],
                        "train_accuracy": fit_scored["accuracy"], "train_macro_f1": fit_scored["macro_f1"],
                    }
                    fold_records.append(record)
                    table = prediction_table(train.iloc[held_ids], prediction, cohort, name, family, mode, number)
                    predictions.append(table)
                    job = run / "jobs" / f"{cohort}_{name}_{family}_{mode}_{number}.json"
                    save_json(job, {**record, "metrics": scored})
                    count += 1
                    if count % 5 == 0:
                        print(f"  {cohort}: {count}/{maximum_fits} fitting CV selesai", flush=True)
                if (oof < 0).any():
                    raise ValueError("OOF tidak lengkap.")
                scored = metrics(y, oof)
                current_folds = [r for r in fold_records if r["protocol"] == name and r["family"] == family and r["representation"] == mode]
                summaries.append({
                    "cohort": cohort, "protocol": name, "family": family, "representation": mode,
                    "rows": len(train), "participants": int(train.participant_id.nunique()),
                    "accuracy": scored["accuracy"], "macro_f1": scored["macro_f1"],
                    "recall_low": scored["recall"][0], "recall_medium": scored["recall"][1], "recall_high": scored["recall"][2],
                    "mean_participant_accuracy": float(np.mean([r["accuracy"] for r in current_folds])),
                    "fold_macro_f1_std": float(np.std([r["macro_f1"] for r in current_folds], ddof=1)),
                })
    return {"status": "complete", "cv_fits": count, "train_rows": len(train), "train_participants": int(train.participant_id.nunique())}, summaries, fold_records, predictions


def paired_results(leaderboard, folds):
    records = []
    for keys, group in leaderboard[leaderboard.family.ne("dummy")].groupby(["cohort", "protocol", "family"]):
        a = group[group.representation.eq("raw")].iloc[0]
        b = group[group.representation.eq("baseline_relative")].iloc[0]
        matched = folds[(folds.cohort.eq(keys[0])) & (folds.protocol.eq(keys[1])) & (folds.family.eq(keys[2]))]
        pivot = matched.pivot(index="fold", columns="representation", values="accuracy")
        delta = pivot.baseline_relative - pivot.raw
        records.append({
            "cohort": keys[0], "protocol": keys[1], "family": keys[2],
            "raw_accuracy": float(a.accuracy), "relative_accuracy": float(b.accuracy),
            "accuracy_delta_pp": float(100 * (b.accuracy - a.accuracy)),
            "raw_macro_f1": float(a.macro_f1), "relative_macro_f1": float(b.macro_f1),
            "macro_f1_delta_pp": float(100 * (b.macro_f1 - a.macro_f1)),
            "folds_improved_accuracy": int((delta > 1e-12).sum()),
            "folds_worse_accuracy": int((delta < -1e-12).sum()), "folds": int(len(delta)),
        })
    return pd.DataFrame(records)


def lock_candidates(leaderboard):
    locks = {}
    main = leaderboard[(leaderboard.protocol.eq("participant_held_out")) & leaderboard.family.ne("dummy")]
    for cohort, group in main.groupby("cohort"):
        locks[cohort] = {}
        for mode, rows in group.groupby("representation"):
            rows = rows[(rows[["recall_low", "recall_medium", "recall_high"]] > 0).all(axis=1)]
            if rows.empty:
                continue
            selected = rows.sort_values(["macro_f1", "accuracy", "family"], ascending=[False, False, True]).iloc[0]
            locks[cohort][mode] = {"family": str(selected.family), "oof_macro_f1": float(selected.macro_f1)}
    return locks


def evaluate_validation(cohorts, references, locks, run):
    records, tables = [], []
    for cohort, candidates in locks.items():
        frame = cohorts[cohort].reset_index(drop=True)
        fit_ids = np.flatnonzero(frame._split.eq("train"))
        held_ids = np.flatnonzero(frame._split.eq("validation"))
        if not len(held_ids):
            continue
        for mode, chosen in candidates.items():
            x = representation(frame, references, cohort, mode)
            y = frame._y.to_numpy()
            predicted, fitting, bundle = fit_predict(x, y, fit_ids, held_ids, chosen["family"])
            # Serialize only the selected model. No Test features are transformed.
            bundle.update({
                "version": VERSION, "cohort": cohort, "representation": mode,
                "features": CORE, "labels": LABELS,
                "baseline_contract": "New sessions supply audited initial reference; no session-ID lookup learned by the classifier.",
            })
            path = run / "models" / f"{cohort}_{mode}.joblib"
            path.parent.mkdir(exist_ok=True)
            joblib.dump(bundle, path)
            scored = metrics(y[held_ids], predicted)
            records.append({
                "cohort": cohort, "representation": mode, "family": chosen["family"],
                "rows": len(held_ids), "accuracy": scored["accuracy"], "macro_f1": scored["macro_f1"],
                "recall_low": scored["recall"][0], "recall_medium": scored["recall"][1], "recall_high": scored["recall"][2],
                "train_accuracy": metrics(y[fit_ids], fitting)["accuracy"],
                "model_sha256": sha256_file(path),
            })
            tables.append(prediction_table(frame.iloc[held_ids], predicted, cohort, "validation", chosen["family"], mode, -1))
    return pd.DataFrame(records), tables


def export_figures(run, leaderboard, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = run / "figures"
    directory.mkdir(exist_ok=True)
    view = leaderboard[leaderboard.protocol.eq("participant_held_out")].copy()
    view["name"] = view.cohort + " / " + view.family + " / " + view.representation
    fig, ax = plt.subplots(figsize=(11, max(4, len(view) * 0.42)))
    ax.barh(view.name, view.macro_f1 * 100, color=np.where(view.representation.eq("raw"), "#68859b", "#159b88"))
    ax.set(xlabel="Macro-F1 OOF (%)", title="Batch 06: participant held out, Train only", xlim=(0, 100))
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(directory / "participant_oof.png", dpi=170)
    plt.close(fig)
    view = paired[paired.protocol.eq("participant_held_out")]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(view.cohort + " / " + view.family, view.macro_f1_delta_pp, color="#159b88")
    ax.axvline(0, color="black", linewidth=1)
    ax.set(xlabel="Macro-F1 relative minus raw (percentage points)", title="Paired baseline comparison on identical windows")
    fig.tight_layout()
    fig.savefig(directory / "paired_change.png", dpi=170)
    plt.close(fig)


def write_report(run, audit, cohort_status, leaderboard, paired, validation, protocol):
    lines = [
        "# Hasil eksperimen baseline Batch 06", "",
        "Window 25 detik, 10 fitur core, model umum Logistic Regression dan Extra Trees. "
        "Pembanding raw dan baseline-relative memakai baris serta fold identik. "
        "Test tidak dievaluasi. Hasil ini merupakan evaluasi development eksploratif.", "",
        f"Protokol split: `{protocol}`.", "",
        "## Ketersediaan acuan awal", "",
        f"Sesi dalam CSV: {len(audit)}; baseline RMSSD layak: {int(audit.hrv_eligible.sum())}; "
        f"baseline RMSSD dan acuan eye awal layak: {int(audit.joint_eligible.sum())}.", "",
        "`hrv_rmssd` mengganti RMSSD dengan RMSSD / baseline - 1; sembilan fitur lain tetap. "
        "`hrv_eye` juga menormalkan statistik eye terhadap window pre_task murni 25 detik. "
        "SDNN tetap mentah karena CSV tidak menyimpan baseline SDNN independen. "
        "Pre_task adalah acuan sensor sebelum tugas, bukan bukti kondisi istirahat fisiologis.", "",
        "Batas awal tugas dan akhir baseline berasal dari audit sumber kanonis yang disertakan. "
        "Nilai baseline RMSSD dan coverage harus cocok dengan CSV Drive. Sumber baru yang belum "
        "diaudit dikeluarkan dan dilaporkan. Notebook tidak mengklaim telah membaca ulang sumber mentah di Drive.", "",
        "## Perbandingan peserta yang ditahan dari fitting", "",
        "| Subset | Model | Representasi | Accuracy OOF | Macro-F1 OOF | Recall Low/Medium/High |", "|---|---|---|---:|---:|---|",
    ]
    for row in leaderboard[leaderboard.protocol.eq("participant_held_out")].itertuples():
        recalls = "/".join(f"{getattr(row, 'recall_' + label.lower()):.2%}" for label in LABELS)
        lines.append(f"| {row.cohort} | {row.family} | {row.representation} | {row.accuracy:.2%} | {row.macro_f1:.2%} | {recalls} |")
    lines.extend(["", "## Perubahan berpasangan", "", "| Subset | Model | Delta accuracy | Delta Macro-F1 | Fold accuracy naik/total |", "|---|---|---:|---:|---:|"])
    for row in paired[paired.protocol.eq("participant_held_out")].itertuples():
        lines.append(f"| {row.cohort} | {row.family} | {row.accuracy_delta_pp:+.2f} pp | {row.macro_f1_delta_pp:+.2f} pp | {row.folds_improved_accuracy}/{row.folds} |")
    lines.extend(["", "OOF menggabungkan prediksi held-out setiap window Train tepat sekali. "
                  "Fold seorang peserta dapat tidak mempunyai ketiga kelas; Macro-F1 per fold memakai tiga kelas tetap. "
                  "Window overlap bukan sampel independen. Selisih deskriptif bukan bukti signifikansi statistik.", "",
                  "## Validation kandidat terkunci", "",
                  "Model dipilih menurut OOF peserta sebelum validation. Tidak ada pemilihan ulang menurut validation.", "",
                  "| Subset | Representasi | Model | Baris | Accuracy | Macro-F1 |", "|---|---|---|---:|---:|---:|"])
    for row in validation.itertuples():
        lines.append(f"| {row.cohort} | {row.representation} | {row.family} | {row.rows} | {row.accuracy:.2%} | {row.macro_f1:.2%} |")
    lines.extend(["", "## Interpretasi", "",
                  "Manfaat baseline dinilai dari perbandingan raw-relative pada subset dan protokol yang sama. "
                  "Skor subset kecil tidak dibandingkan langsung dengan seluruh data batch sebelumnya. "
                  "Subset eye dengan hanya beberapa orang merupakan kontrol eksploratif dengan ketidakpastian besar. "
                  "Validation historis masih berisi sesi yang dikenal, sedangkan OOF peserta menahan seluruh rekaman orang tersebut. "
                  "Dataset baru memakai split berbasis orang untuk Train, validation, dan Test.", "",
                  "Baseline RMSSD recorder umumnya lebih panjang daripada window tugas 25 detik; "
                  "rasio ini menguji representasi relatif, bukan kesetaraan pengukuran HRV dua durasi. "
                  "Belum ada rating beban mental independen. Perbaikan skor tidak memvalidasi label difficulty secara fisiologis.", "",
                  "## Status subset", "", "```json", json.dumps(cohort_status, indent=2), "```", ""])
    (run / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def run_experiment(dataset_root, output_root, reference, source_evidence, implementation_sha256):
    frames, hashes, paths = load_views(dataset_root)
    frame, protocol = split_frame(frames[25], hashes, reference)
    # Test rows are retained only in the read-only split ledger. All baseline
    # extraction and fitting below operate on Train/validation or pre-task refs.
    audit, references = audit_baselines(frame, source_evidence)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = Path(output_root) / f"baseline_{stamp}_{uuid.uuid4().hex[:8]}"
    run.mkdir(parents=True, exist_ok=False)
    audit.to_csv(run / "baseline_audit.csv", index=False)
    save_json(run / "baseline_references.json", references)
    frame[["window_id", "participant_id", "session_id", "_split"]].to_csv(run / "split_assignments.csv", index=False)
    plan = {
        "version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": hashes, "implementation_sha256": implementation_sha256,
        "reference_sha256": object_hash(reference), "source_evidence_sha256": object_hash(source_evidence),
        "split_protocol": protocol, "window_seconds": 25, "features": CORE,
        "cohorts": ["hrv_rmssd", "hrv_eye"], "models": MODELS,
        "selection": "participant_held_out_oof_macro_f1_then_accuracy_positive_recall_all_classes",
        "test_evaluation": False, "search": False, "source_root_required": False,
        "minimum_train_participants": 4, "reference_margin_seconds": 1,
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib", "matplotlib")},
        "python": platform.python_version(),
    }
    save_json(run / "plan.json", plan)
    save_json(run / "status.json", {"stage": "fitting", "complete": False})
    print("Run:", run, flush=True)
    print(audit[["participant_id", "hrv_eligible", "joint_eligible", "hrv_rejection_reason", "joint_rejection_reason"]].to_string(index=False), flush=True)
    all_summaries, all_folds, all_predictions = [], [], []
    statuses, cohorts = {}, {}
    for cohort in ("hrv_rmssd", "hrv_eye"):
        # Never transform or fit Test, including when constructing representations.
        pool = frame[frame._split.isin(("train", "validation"))].copy()
        cohort_frame = make_cohort(pool, audit, references, cohort)
        cohorts[cohort] = cohort_frame
        status, summaries, folds, predictions = evaluate_cohort(cohort_frame, references, cohort, reference, protocol, run)
        statuses[cohort] = status
        all_summaries.extend(summaries)
        all_folds.extend(folds)
        all_predictions.extend(predictions)
    if not all_summaries:
        save_json(run / "status.json", {"stage": "baseline_unavailable", "complete": False, "cohorts": statuses})
        raise ValueError(f"Tidak ada subset layak. Baca {run / 'baseline_audit.csv'}; jangan membuat baseline dari tugas.")
    leaderboard, fold_table = pd.DataFrame(all_summaries), pd.DataFrame(all_folds)
    leaderboard.to_csv(run / "cv_leaderboard.csv", index=False)
    fold_table.to_csv(run / "fold_metrics.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(run / "oof_predictions.csv", index=False)
    paired = paired_results(leaderboard, fold_table)
    paired.to_csv(run / "paired_comparison.csv", index=False)
    locks = lock_candidates(leaderboard)
    save_json(run / "model_lock.json", {"selected": locks, "plan_sha256": object_hash(plan), "validation_evaluated": False, "test_evaluated": False})
    validation, validation_predictions = evaluate_validation(cohorts, references, locks, run)
    validation.to_csv(run / "validation_metrics.csv", index=False)
    if validation_predictions:
        pd.concat(validation_predictions, ignore_index=True).to_csv(run / "validation_predictions.csv", index=False)
    for key, path in paths.items():
        if sha256_file(path) != hashes[key]:
            raise ValueError("CSV sumber berubah selama run.")
    export_figures(run, leaderboard, paired)
    write_report(run, audit, statuses, leaderboard, paired, validation, protocol)
    save_json(run / "status.json", {
        "stage": "complete", "complete": True, "cohorts": statuses,
        "validation_fits": len(validation), "test_evaluated": False,
        "dataset_unchanged": True, "plan_sha256": object_hash(plan),
    })
    print("\nPerbandingan utama:", flush=True)
    print(paired[paired.protocol.eq("participant_held_out")].to_string(index=False), flush=True)
    print("\nValidation:", flush=True)
    print(validation.to_string(index=False), flush=True)
    print("\nLaporan:", run / "RESULTS.md", flush=True)
    return {"run_dir": run, "leaderboard": leaderboard, "paired": paired, "validation": validation, "baseline_audit": audit}
