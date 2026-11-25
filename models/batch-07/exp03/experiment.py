"""Batch 07 Exp03: V4 threshold calibration (Dev OOF & nested LOPO) and cross-representation model comparison."""

import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import sklearn
from scipy.special import expit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
HRV2 = ["rmssd_ms", "sdnn_ms"]
EYE7 = [
    "blink_rate_per_minute",
    "ear_left_mean",
    "ear_left_std",
    "ear_right_mean",
    "ear_right_std",
    "iris_size_estimate_mean",
    "iris_size_estimate_std",
]
CORE9 = HRV2 + EYE7
CORE10 = CORE9 + ["iris_size_delta_mean"]
HEAD4 = ["head_pitch_mean", "head_pitch_std", "head_yaw_mean", "head_yaw_std"]
BASE1 = ["rmssd_baseline_ratio"]
SENSORS14 = CORE10 + HEAD4

FEATURE_SETS = {
    "core9": CORE9,
    "core9_head4_base": CORE9 + HEAD4 + BASE1,
    "sensors14": SENSORS14,
}

FROZEN_V4_SPLITS = {
    "Andrean": "Dev", "Andreas": "Dev", "Athaya": "Test", "Baithar": "Test",
    "Fabian": "Dev", "Rafi": "Dev", "Ramadhan": "Dev", "Savitri": "Dev",
    "Siti": "Dev", "Sullivan": "Test", "Tio": "Test", "faqih": "Dev",
    "ikhsan": "Dev", "irfan": "Dev", "kaka": "Dev", "raina": "Dev", "sam": "Dev",
}

THRESHOLD_GRID = tuple(np.round(np.linspace(0.20, 0.80, 25), 3))
POLICIES = (
    "fixed_050",
    "nested_youden",
    "nested_recall75_min_fa",
    "nested_fa_cap25_max_rec",
    "nested_macro_f1",
)

CANDIDATES = (
    "win_logreg_core9_global",
    "win_logreg_full14_global",
    "win_logreg_full14_causal",
    "win_et_full14_global",
    "ep_svm_rbf_prior14",
    "ep_et_prior14",
)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def snapshot_lf_hash(path, expected_lf_hash):
    data = path.read_bytes()
    if hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest() != expected_lf_hash:
        raise ValueError(f"Hash LF tidak cocok untuk {path.name}")
    return hashlib.sha256(data).hexdigest()


def history_streams_v2(path):
    keys = ["participant_id", "session_id"]
    times = ["window_start", "window_end"]
    bank = pd.read_csv(path, usecols=keys + times + ["quality_tier"] + SENSORS14)
    bank = bank[bank.quality_tier.isin(["good", "acceptable"]) & np.isfinite(bank[SENSORS14]).all(axis=1)].copy()
    for field in times:
        bank[field] = pd.to_datetime(bank[field], utc=True, format="mixed", errors="raise")
    return dict(tuple(bank.sort_values("window_end", ascending=False).groupby(keys)))


def prior_ref_median(stream, start_ts):
    chosen = []
    limit = start_ts
    if stream is not None:
        for idx, left, right in stream[["window_start", "window_end"]].itertuples():
            if right <= limit:
                chosen.append(idx)
                limit = left
                if len(chosen) == 12:
                    break
    if len(chosen) < 3:
        return pd.Series(np.nan, index=SENSORS14), len(chosen)
    return stream.loc[chosen, SENSORS14].median(), len(chosen)


def load_v4_cohort(root):
    v4_dir = root / "v4_self_reported_rating"
    hashes = {}
    for rel in ("window_15s_step_5s/good_acceptable.csv", "window_15s_step_5s/window_assignments.csv", "episode_ratings.csv"):
        p = v4_dir / rel
        hashes[f"v4/{rel}"] = hashlib.sha256(p.read_bytes()).hexdigest()
    v2_all = root / "v2_session_phase/window_15s_step_5s/all_data.csv"
    hashes["v2_all"] = snapshot_lf_hash(v2_all, "acbef9a60f4114de460ee3c57ca94ff5cae0d48b67229d8b14283d173fef1355")

    raw_df = pd.read_csv(v4_dir / "window_15s_step_5s/good_acceptable.csv")
    wa_df = pd.read_csv(v4_dir / "window_15s_step_5s/window_assignments.csv")
    ev_df = pd.read_csv(v4_dir / "episode_ratings.csv")

    df = raw_df.merge(
        wa_df[["window_id", "episode_id", "difficulty_rating"]].rename(columns={"difficulty_rating": "wa_rating"}),
        on="window_id", how="left", validate="1:1",
    )
    if not (df["difficulty_rating"] == df["wa_rating"]).all():
        raise ValueError("Ketidakcocokan difficulty_rating pada window_assignments.")
    df = df.drop(columns=["wa_rating"])
    df["window_start_dt"] = pd.to_datetime(df["window_start"], utc=True, format="mixed")
    df = df.sort_values(["session_id", "window_start_dt", "window_id"]).reset_index(drop=True)
    df["split"] = df["participant_id"].map(FROZEN_V4_SPLITS)
    if df["split"].isna().any():
        raise ValueError("Peserta V4 tidak terdaftar pada split beku.")
    df["y"] = (df["difficulty_rating"] >= 7).astype(int)
    df["_row_idx"] = np.arange(len(df), dtype=int)

    # Precompute strictly causal within-session statistics for causal_session_robust
    causal_cols = FEATURE_SETS["core9_head4_base"]
    causal_med, causal_iqr = {}, {}
    causal_pos = np.zeros(len(df), dtype=int)
    for col in causal_cols:
        c_med = np.full(len(df), np.nan, dtype=float)
        c_iqr = np.full(len(df), np.nan, dtype=float)
        vals = df[col].to_numpy(dtype=float)
        for _, idx in df.groupby("session_id").indices.items():
            for pos, ri in enumerate(idx):
                causal_pos[ri] = pos
                if pos >= 3:
                    hist = vals[idx[:pos]]
                    hist = hist[np.isfinite(hist)]
                    if len(hist) >= 3:
                        c_med[ri] = np.median(hist)
                        q75, q25 = np.percentile(hist, [75, 25])
                        c_iqr[ri] = q75 - q25
        causal_med[col] = c_med
        causal_iqr[col] = c_iqr
    causal_stats = (causal_pos, causal_med, causal_iqr)

    # Build episode-level table + clean prior-history references from v2 all_data.csv
    streams = history_streams_v2(v2_all)
    ep_medians = df.groupby("episode_id")[SENSORS14].median()
    ep_meta = df.groupby("episode_id").agg(
        participant_id=("participant_id", "first"),
        session_id=("session_id", "first"),
        difficulty_rating=("difficulty_rating", "first"),
        y=("y", "first"),
        split=("split", "first"),
        first_window_start=("window_start_dt", "min"),
        n_windows=("window_id", "count"),
    ).loc[ep_medians.index]
    # Use episode start_ms from episode_ratings if available, else first_window_start
    ev_starts = ev_df.set_index("episode_id")
    refs = []
    hist_counts = []
    for ep_id, row in ep_meta.iterrows():
        if ep_id in ev_starts.index and "start_ms" in ev_starts.columns and pd.notna(ev_starts.at[ep_id, "start_ms"]):
            start_ts = pd.to_datetime(float(ev_starts.at[ep_id, "start_ms"]), unit="ms", utc=True)
        else:
            start_ts = row.first_window_start
        ref, n_h = prior_ref_median(streams.get((row.participant_id, row.session_id)), start_ts)
        refs.append(ref)
        hist_counts.append(n_h)
    ep_meta["history_windows"] = hist_counts
    ep_refs = pd.DataFrame(refs, index=ep_medians.index)
    return df, causal_stats, ep_medians, ep_refs, ep_meta, hashes


def episode_balanced_window_weights(sub_df, y_bin, clip_max=5.0):
    n = len(y_bin)
    w = np.ones(n, dtype=float)
    for cls in (0, 1):
        mask = y_bin == cls
        if not mask.any():
            continue
        g = sub_df.loc[mask]
        n_ep = g["episode_id"].nunique()
        ep_sz = g.groupby("episode_id")["window_id"].transform("count").to_numpy(dtype=float)
        wc = (n / (2.0 * n_ep)) / ep_sz
        wc = np.minimum(wc, clip_max)
        wc = wc * ((n / 2.0) / wc.sum())
        w[mask] = wc
    return w


def transform_windows(fit_df, eval_df, cols, mode, causal_stats):
    fit_vals = fit_df[cols].to_numpy(dtype=float)
    g_med = np.nanmedian(fit_vals, axis=0)
    filled = np.where(np.isfinite(fit_vals), fit_vals, g_med)
    q75, q25 = np.percentile(filled, [75, 25], axis=0)
    g_iqr = np.where((q75 - q25) > 1e-9, q75 - q25, 1.0)

    def _apply(target_df):
        vals = target_df[cols].to_numpy(dtype=float)
        vals = np.where(np.isfinite(vals), vals, g_med)
        if mode == "global_robust":
            return np.clip((vals - g_med) / g_iqr, -10.0, 10.0)
        if mode == "causal_session_robust":
            causal_pos, causal_med, causal_iqr = causal_stats
            ridx = target_df["_row_idx"].to_numpy(dtype=int)
            out = (vals - g_med) / g_iqr
            use_c = causal_pos[ridx] >= 3
            if use_c.any():
                c_med = np.column_stack([causal_med[c][ridx] for c in cols])
                c_iqr = np.column_stack([causal_iqr[c][ridx] for c in cols])
                valid_m = np.isfinite(c_med) & use_c[:, None]
                eff_med = np.where(valid_m, c_med, g_med)
                eff_iqr = np.where(valid_m & np.isfinite(c_iqr) & (c_iqr > 1e-9), c_iqr, g_iqr)
                out = (vals - eff_med) / eff_iqr
            return np.clip(out, -10.0, 10.0)
        raise ValueError(mode)

    return _apply(fit_df), _apply(eval_df)


def score_fold_candidates(df, causal_stats, ep_medians, ep_refs, ep_meta, fit_people, held_people):
    """Fit all 6 V4 candidates strictly on `fit_people` and return episode-level probability predictions on `held_people`."""
    if fit_people & held_people:
        raise ValueError("Peserta fitting dan evaluasi tumpang tindih.")
    w_fit = df[df.participant_id.isin(fit_people)].reset_index(drop=True)
    w_held = df[df.participant_id.isin(held_people)].reset_index(drop=True)
    y_w_fit = w_fit["y"].to_numpy(dtype=int)
    sw_fit = episode_balanced_window_weights(w_fit, y_w_fit)

    ep_fit_ids = ep_meta.index[ep_meta.participant_id.isin(fit_people)]
    ep_held_ids = ep_meta.index[ep_meta.participant_id.isin(held_people)]
    held_base = ep_meta.loc[ep_held_ids, ["participant_id", "difficulty_rating", "y", "n_windows", "history_windows"]].copy()

    out = {}

    # 1-4: Per-window models -> episode mean probability
    win_specs = [
        ("win_logreg_core9_global", "core9", "global_robust", "logreg"),
        ("win_logreg_full14_global", "core9_head4_base", "global_robust", "logreg"),
        ("win_logreg_full14_causal", "core9_head4_base", "causal_session_robust", "logreg"),
        ("win_et_full14_global", "core9_head4_base", "global_robust", "et"),
    ]
    tx_cache = {}
    for cname, fset, norm, mkind in win_specs:
        key = (fset, norm)
        if key not in tx_cache:
            tx_cache[key] = transform_windows(w_fit, w_held, FEATURE_SETS[fset], norm, causal_stats)
        X_tr, X_te = tx_cache[key]
        if mkind == "logreg":
            clf = LogisticRegression(C=0.5, max_iter=5000, random_state=SEED).fit(X_tr, y_w_fit, sample_weight=sw_fit)
        else:
            clf = ExtraTreesClassifier(
                n_estimators=200, max_depth=4, min_samples_leaf=8, max_features=0.7, random_state=SEED, n_jobs=1
            ).fit(X_tr, y_w_fit, sample_weight=sw_fit)
        p_win = clf.predict_proba(X_te)[:, 1]
        ep_prob = pd.Series(p_win, index=w_held["episode_id"]).groupby(level=0).mean().reindex(ep_held_ids)
        out[cname] = ep_prob.to_numpy(dtype=float)

    # 5-6: Episode-median prior-history models (Batch 07 Exp01 architecture on V4)
    y_ep_fit = ep_meta.loc[ep_fit_ids, "y"]
    people_ep = ep_meta.loc[ep_fit_ids, "participant_id"]
    w_ep = 1.0 / (people_ep.groupby(y_ep_fit).transform("nunique") * y_ep_fit.groupby([y_ep_fit, people_ep]).transform("size"))
    w_ep /= w_ep.mean()
    rel = ep_medians - ep_refs.fillna(ep_medians.loc[ep_fit_ids].median())
    scaler = StandardScaler().fit(rel.loc[ep_fit_ids], sample_weight=w_ep)
    X_ep_tr = scaler.transform(rel.loc[ep_fit_ids])
    X_ep_te = scaler.transform(rel.loc[ep_held_ids])

    svm = SVC(C=1, kernel="rbf", gamma="scale").fit(X_ep_tr, y_ep_fit.to_numpy(), sample_weight=w_ep.to_numpy())
    svm_margin = svm.decision_function(X_ep_te) * (1 if svm.classes_[1] == 1 else -1)
    out["ep_svm_rbf_prior14"] = expit(svm_margin)

    et_ep = ExtraTreesClassifier(n_estimators=300, max_depth=3, min_samples_leaf=3, random_state=SEED, n_jobs=1).fit(
        X_ep_tr, y_ep_fit.to_numpy(), sample_weight=w_ep.to_numpy()
    )
    out["ep_et_prior14"] = et_ep.predict_proba(X_ep_te)[:, list(et_ep.classes_).index(1)]

    return held_base, out


def choose_thresholds_from_oof(y_true, probs):
    """Select thresholds for all 5 pre-specified operational policies strictly from fitting inner-LOPO episodes."""
    y_b = np.asarray(y_true, dtype=int)
    p_a = np.asarray(probs, dtype=float)
    chosen = {"fixed_050": 0.50}

    best_youden = None
    best_mf1 = None
    feas_rec75 = []
    feas_fa25 = []

    for thr in THRESHOLD_GRID:
        pred = (p_a >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_b, pred, labels=[0, 1]).ravel()
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        far = fp / (tn + fp) if (tn + fp) else 0.0
        bal = 0.5 * (rec + spec)
        mf1 = float(f1_score(y_b, pred, labels=[0, 1], average="macro", zero_division=0))

        cand_y = (-bal, fp, abs(thr - 0.50), thr)
        if best_youden is None or cand_y < best_youden:
            best_youden = cand_y

        cand_f = (-mf1, fp, abs(thr - 0.50), thr)
        if best_mf1 is None or cand_f < best_mf1:
            best_mf1 = cand_f

        if rec >= 0.75:
            feas_rec75.append((fp, -tp, -thr, thr))
        if far <= 0.25:
            feas_fa25.append((-tp, -bal, abs(thr - 0.50), thr))

    chosen["nested_youden"] = float(best_youden[3])
    chosen["nested_macro_f1"] = float(best_mf1[3])
    chosen["nested_recall75_min_fa"] = float(min(feas_rec75)[3]) if feas_rec75 else float(best_youden[3])
    chosen["nested_fa_cap25_max_rec"] = float(min(feas_fa25)[3]) if feas_fa25 else 0.65
    return chosen


def bootstrap_auc_ci(y_arr, p_arr, pid_arr, n_boot=1000):
    rng = np.random.RandomState(SEED)
    uniq = np.unique(pid_arr)
    idx_map = {p: np.where(pid_arr == p)[0] for p in uniq}
    vals = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_map[p] for p in samp])
        if len(np.unique(y_arr[idx])) < 2:
            continue
        vals.append(roc_auc_score(y_arr[idx], p_arr[idx]))
    if len(vals) < 50:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summarize_v4(pred):
    rows = []
    ci_cache = {}
    for (protocol, candidate, policy), g in pred.groupby(["protocol", "candidate", "policy"]):
        y_t = g["y"].to_numpy(dtype=int)
        y_p = g["predicted"].to_numpy(dtype=int)
        scores = g["prob_overload"].to_numpy(dtype=float)
        pids = g["participant_id"].to_numpy()
        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
        ck = (protocol, candidate)
        if ck not in ci_cache:
            ci_cache[ck] = (
                float(roc_auc_score(y_t, scores)),
                float(average_precision_score(y_t, scores)),
                *bootstrap_auc_ci(y_t, scores, pids),
            )
        auc_v, ap_v, ci_lo, ci_hi = ci_cache[ck]
        rows.append({
            "protocol": protocol,
            "candidate": candidate,
            "policy": policy,
            "participants": int(g["participant_id"].nunique()),
            "episodes": int(len(g)),
            "overload_total": int(tp + fn),
            "non_overload_total": int(tn + fp),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision_overload": float(precision_score(y_t, y_p, zero_division=0)),
            "recall_overload": float(recall_score(y_t, y_p, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
            "false_alarm_rate": float(fp / (tn + fp)) if (tn + fp) else 0.0,
            "accuracy": float(accuracy_score(y_t, y_p)),
            "balanced_accuracy": float(balanced_accuracy_score(y_t, y_p)),
            "macro_f1": float(f1_score(y_t, y_p, labels=[0, 1], average="macro", zero_division=0)),
            "auc": auc_v,
            "average_precision": ap_v,
            "auc_ci95_lo": ci_lo,
            "auc_ci95_hi": ci_hi,
            "mean_threshold": float(g["threshold"].mean()),
        })
    return pd.DataFrame(rows)


def build_dev_threshold_curve(dev_fixed_pred):
    rows = []
    for candidate, g in dev_fixed_pred.groupby("candidate"):
        y_t = g["y"].to_numpy(dtype=int)
        p_a = g["prob_overload"].to_numpy(dtype=float)
        for thr in THRESHOLD_GRID:
            y_p = (p_a >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
            rows.append({
                "candidate": candidate,
                "threshold": float(thr),
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                "precision_overload": float(precision_score(y_t, y_p, zero_division=0)),
                "recall_overload": float(recall_score(y_t, y_p, zero_division=0)),
                "false_alarm_rate": float(fp / (tn + fp)) if (tn + fp) else 0.0,
                "accuracy": float(accuracy_score(y_t, y_p)),
                "balanced_accuracy": float(balanced_accuracy_score(y_t, y_p)),
                "macro_f1": float(f1_score(y_t, y_p, labels=[0, 1], average="macro", zero_division=0)),
            })
    return pd.DataFrame(rows)


def run_lopo_with_nested_thresholds(protocol_name, cohort_people, df, causal_stats, ep_medians, ep_refs, ep_meta):
    """Compute outer-LOPO scores AND strictly nested inner-LOPO thresholds without leakage."""
    # First get all pairwise 1-held-out predictions needed? Wait:
    # For outer LOPO on cohort_people (size K):
    # - Outer fold holds out p, fits on (cohort_people - {p}).
    # - Inner LOPO for outer fold p holds out q in (cohort_people - {p}), fits on (cohort_people - {p, q}).
    people_sorted = sorted(cohort_people)
    outer_scores = {}
    for p in people_sorted:
        outer_scores[p] = score_fold_candidates(df, causal_stats, ep_medians, ep_refs, ep_meta, cohort_people - {p}, {p})

    pred_rows = []
    thr_rows = []
    for p in people_sorted:
        fit_set = cohort_people - {p}
        # Inner LOPO on fit_set
        inner_y = []
        inner_probs = {c: [] for c in CANDIDATES}
        for q in sorted(fit_set):
            inner_fit = fit_set - {q}
            # Ensure inner_fit has both classes
            if ep_meta.loc[ep_meta.participant_id.isin(inner_fit), "y"].nunique() < 2:
                continue
            hbase_q, sc_q = score_fold_candidates(df, causal_stats, ep_medians, ep_refs, ep_meta, inner_fit, {q})
            inner_y.append(hbase_q["y"].to_numpy(dtype=int))
            for c in CANDIDATES:
                inner_probs[c].append(sc_q[c])
        y_in_all = np.concatenate(inner_y)
        hbase_p, sc_p = outer_scores[p]
        for c in CANDIDATES:
            thrs = choose_thresholds_from_oof(y_in_all, np.concatenate(inner_probs[c]))
            for pol, thr in thrs.items():
                pred_rows.append(hbase_p.assign(
                    protocol=protocol_name,
                    fold=p,
                    candidate=c,
                    policy=pol,
                    threshold=thr,
                    prob_overload=sc_p[c],
                    predicted=(sc_p[c] >= thr).astype(int),
                ).reset_index())
                thr_rows.append({"protocol": protocol_name, "fold": p, "candidate": c, "policy": pol, "threshold": thr})
    return pd.concat(pred_rows, ignore_index=True), pd.DataFrame(thr_rows), outer_scores


def run(root, output_root):
    df, causal_stats, ep_medians, ep_refs, ep_meta, hashes = load_v4_cohort(root)
    dev_people = frozenset(p for p, s in FROZEN_V4_SPLITS.items() if s == "Dev")
    test_people = frozenset(p for p, s in FROZEN_V4_SPLITS.items() if s == "Test")
    all_people = dev_people | test_people

    # 1. Dev_LOPO (13 participants, 99 episodes) with nested 12-person inner-LOPO thresholds
    dev_pred, dev_thrs, dev_outer_scores = run_lopo_with_nested_thresholds(
        "Dev_LOPO_13p", dev_people, df, causal_stats, ep_medians, ep_refs, ep_meta
    )

    # 2. Historical Test (4 participants, 27 episodes) using thresholds chosen strictly from Dev 13p OOF!
    dev_oof_y = np.concatenate([dev_outer_scores[p][0]["y"].to_numpy(dtype=int) for p in sorted(dev_people)])
    test_base, test_scores = score_fold_candidates(df, causal_stats, ep_medians, ep_refs, ep_meta, dev_people, test_people)
    test_rows, test_thrs = [], []
    for c in CANDIDATES:
        dev_oof_p = np.concatenate([dev_outer_scores[p][1][c] for p in sorted(dev_people)])
        thrs = choose_thresholds_from_oof(dev_oof_y, dev_oof_p)
        for pol, thr in thrs.items():
            test_rows.append(test_base.assign(
                protocol="Test_historical_4p",
                fold="Test_historical",
                candidate=c,
                policy=pol,
                threshold=thr,
                prob_overload=test_scores[c],
                predicted=(test_scores[c] >= thr).astype(int),
            ).reset_index())
            test_thrs.append({"protocol": "Test_historical_4p", "fold": "Dev_13p_OOF", "candidate": c, "policy": pol, "threshold": thr})
    test_pred = pd.concat(test_rows, ignore_index=True)

    # 3. Full_LOPO (17 participants, 126 episodes) with nested 16-person inner-LOPO thresholds
    full_pred, full_thrs, _ = run_lopo_with_nested_thresholds(
        "Full_LOPO_17p", all_people, df, causal_stats, ep_medians, ep_refs, ep_meta
    )

    all_pred = pd.concat([dev_pred, test_pred, full_pred], ignore_index=True)
    all_thrs = pd.concat([dev_thrs, pd.DataFrame(test_thrs), full_thrs], ignore_index=True)
    metrics = summarize_v4(all_pred)
    dev_curve = build_dev_threshold_curve(dev_pred[dev_pred.policy.eq("fixed_050")])

    # Lock top Dev_LOPO candidate by AUC (tie-break Balanced Accuracy under fixed_050)
    dev_ranking = metrics[(metrics.protocol == "Dev_LOPO_13p") & (metrics.policy == "fixed_050")].sort_values(
        ["auc", "balanced_accuracy", "candidate"], ascending=[False, False, True]
    ).reset_index(drop=True)

    out = output_root / ("exp03_v4_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    for name, frame in [
        ("episode_predictions", all_pred),
        ("metrics", metrics),
        ("dev_ranking", dev_ranking),
        ("dev_oof_threshold_curve", dev_curve),
        ("selected_thresholds", all_thrs),
    ]:
        frame.to_csv(out / f"{name}.csv", index=False)

    # Visualization
    import matplotlib
    if "ipykernel" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    # Panel A: Dev 13p OOF threshold trade-off on win_logreg_full14_global
    c_focus = "win_logreg_full14_global"
    sub_c = dev_curve[dev_curve.candidate == c_focus]
    axes[0].plot(sub_c.threshold, sub_c.recall_overload * 100, label="Recall Overload (%)", color="#dc2626", lw=2)
    axes[0].plot(sub_c.threshold, sub_c.false_alarm_rate * 100, label="Alarm Palsu Non-Overload (%)", color="#f59e0b", lw=2)
    axes[0].plot(sub_c.threshold, sub_c.precision_overload * 100, label="Presisi Overload (%)", color="#2563eb", lw=2)
    axes[0].plot(sub_c.threshold, sub_c.balanced_accuracy * 100, label="Balanced Accuracy (%)", color="#10b981", ls="--", lw=1.8)
    axes[0].axvline(0.50, color="#64748b", ls=":", label="Ambang 0,50")
    axes[0].set_title("Dev 13p OOF: Kurva Trade-Off Ambang\n(LogReg L2 · core9+head4+base)")
    axes[0].set_xlabel("Ambang Probabilitas Overload")
    axes[0].set_ylabel("Persentase (%)")
    axes[0].legend(fontsize=7.8)
    axes[0].grid(alpha=0.2)

    # Panel B: Model comparison AUC (Dev 13p vs Full 17p)
    m_dev = metrics[(metrics.protocol == "Dev_LOPO_13p") & (metrics.policy == "fixed_050")].set_index("candidate").reindex(CANDIDATES)
    m_full = metrics[(metrics.protocol == "Full_LOPO_17p") & (metrics.policy == "fixed_050")].set_index("candidate").reindex(CANDIDATES)
    xp = np.arange(len(CANDIDATES))
    axes[1].bar(xp - 0.18, m_dev.auc, width=0.35, label="Dev LOPO (13p, 99 ep)", color="#3b82f6")
    axes[1].bar(xp + 0.18, m_full.auc, width=0.35, label="Full LOPO (17p, 126 ep)", color="#10b981")
    axes[1].axhline(0.50, color="#94a3b8", ls="--")
    axes[1].set_xticks(xp, CANDIDATES, rotation=25, ha="right", fontsize=8)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("Perbandingan 6 Kandidat Model & Representasi (ROC AUC)")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.2)

    # Panel C: Threshold policies on Full_LOPO_17p (win_logreg_full14_global)
    p_sub = metrics[(metrics.protocol == "Full_LOPO_17p") & (metrics.candidate == c_focus)].set_index("policy").reindex(POLICIES)
    xp2 = np.arange(len(POLICIES))
    axes[2].bar(xp2 - 0.2, p_sub.recall_overload * 100, width=0.2, label="Recall Overload (%)", color="#dc2626")
    axes[2].bar(xp2, p_sub.false_alarm_rate * 100, width=0.2, label="Alarm Palsu (%)", color="#f59e0b")
    axes[2].bar(xp2 + 0.2, p_sub.accuracy * 100, width=0.2, label="Akurasi (%)", color="#2563eb")
    axes[2].set_xticks(xp2, POLICIES, rotation=25, ha="right", fontsize=8)
    axes[2].set_ylim(0, 100)
    axes[2].set_title(" Kebijakan Ambang Nested pada Full LOPO (17p)")
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", alpha=0.2)

    fig.savefig(out / "v4_exp03_threshold_calibration.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    save_json(out / "manifest.json", {
        "experiment": "batch-07/exp03",
        "target": "v4_self_reported_rating: 1-6 (Non-Overload, 0) vs 7-10 (Overload, 1)",
        "inputs": hashes,
        "candidates": list(CANDIDATES),
        "policies": list(POLICIES),
        "threshold_grid": list(THRESHOLD_GRID),
        "locked_dev_winner": str(dev_ranking.iloc[0]["candidate"]),
        "historical_test_note": "4 Test participants were previously evaluated in Batch 06; reported here as historical reference under locked Dev thresholds.",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    })

    print("=== PERINGKAT KANDIDAT PADA DEV LOPO (13 PESERTA, 99 EPISODE) ===")
    print(dev_ranking[["candidate", "auc", "auc_ci95_lo", "auc_ci95_hi", "balanced_accuracy", "accuracy", "tp", "fp"]].to_string(index=False))
    print("\n=== HASIL KEBIJAKAN AMBANG PADA WIN_LOGREG_FULL14_GLOBAL & WIN_LOGREG_CORE9_GLOBAL ===")
    focus_m = metrics[metrics.candidate.isin(["win_logreg_full14_global", "win_logreg_core9_global"])]
    print(focus_m[["protocol", "candidate", "policy", "mean_threshold", "precision_overload", "recall_overload",
                   "false_alarm_rate", "balanced_accuracy", "accuracy", "auc", "tp", "fp"]].to_string(index=False))
    archive = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    print("ZIP:", archive)
    return Path(archive)
