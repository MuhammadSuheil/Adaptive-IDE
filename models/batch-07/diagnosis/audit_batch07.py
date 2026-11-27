"""Audit ulang hasil Batch 07 dari prediksi episode yang tersimpan (tanpa training ulang).

Hanya memerlukan numpy, pandas, dan scipy. Jalankan dari root repo:

    python models/batch-07/diagnosis/audit_batch07.py

Keluaran: ringkasan di terminal dan `diagnosis_checks.json` di folder ini.
"""

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[3]
B7 = ROOT / "models/batch-07"
DATA = ROOT / "analysis_sessions/dataset_fusion/output"
RNG = np.random.default_rng(20261127)


def auc(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    n1 = y.sum()
    n0 = len(y) - n1
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bal_acc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))


def logit_ll(X, y):
    X = np.column_stack([np.ones(len(X)), X])
    y = np.asarray(y, float)

    def nll(b):
        z = X @ b
        return np.sum(np.logaddexp(0, z) - y * z)

    res = minimize(nll, np.zeros(X.shape[1]), method="BFGS")
    return -res.fun, 1 / (1 + np.exp(-(X @ res.x)))


def lr_test(base, extra, y):
    ll0, p0 = logit_ll(base, y)
    ll1, p1 = logit_ll(np.column_stack([base, extra]), y)
    stat = 2 * (ll1 - ll0)
    return dict(chi2=round(stat, 2), p=float(chi2.sf(stat, 1)), auc_base=round(auc(y, p0), 3),
                auc_base_plus_sensor=round(auc(y, p1), 3))


def within_person(g, score):
    aucs, conc, pairs = [], 0.0, 0
    for _, h in g.groupby("participant_id"):
        if h.y.nunique() == 2:
            aucs.append(auc(h.y, h[score]))
            pos, neg = h.loc[h.y == 1, score].to_numpy(), h.loc[h.y == 0, score].to_numpy()
            conc += (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
            pairs += len(pos) * len(neg)
    return dict(people_with_both_classes=len(aucs), mean_within_auc=round(float(np.mean(aucs)), 3) if aucs else None,
                pair_concordance=round(conc / pairs, 3) if pairs else None, pairs=int(pairs))


def cluster_ci(g, fn, n=1000):
    ids = g.participant_id.unique()
    groups = {i: g[g.participant_id == i] for i in ids}
    vals = []
    for _ in range(n):
        h = pd.concat([groups[i] for i in RNG.choice(ids, len(ids))])
        if h.y.nunique() == 2:
            vals.append(fn(h))
    return [round(float(v), 3) for v in np.percentile(vals, [2.5, 97.5])]


def difficulty_code(puzzle_set):
    return puzzle_set.str.extract(r"^(easy|medium|hard|demo)")[0].map({"easy": 0, "demo": 0, "medium": 1, "hard": 2})


def audit_v4():
    pred = pd.read_csv(glob.glob(str(B7 / "exp03/outputs/*/episode_predictions.csv"))[0])
    ep = pd.read_csv(DATA / "v4_self_reported_rating/episode_ratings.csv")
    session_start = ep.groupby("session_id").start_ms.min()
    ep["minutes_into_session"] = (ep.start_ms - ep.session_id.map(session_start)) / 60000
    pred = pred.merge(ep[["episode_id", "puzzle_set", "active_seconds", "manual_errors", "timeout_submissions",
                          "minutes_into_session"]], on="episode_id", how="left")
    out = {"protocols": {}, "selected_policies": {}}
    for proto in ["Dev_LOPO_13p", "Full_LOPO_17p", "Test_historical_4p"]:
        for cand in ["win_logreg_core9_global", "win_logreg_full14_global"]:
            g = pred[(pred.protocol == proto) & (pred.candidate == cand) & (pred.policy == "fixed_050")]
            out["protocols"][f"{proto}|{cand}"] = dict(
                episodes=len(g), positives=int(g.y.sum()), majority_accuracy=round(1 - g.y.mean(), 3),
                pooled_auc=round(auc(g.y, g.prob_overload), 3),
                pooled_auc_ci95=cluster_ci(g, lambda h: auc(h.y, h.prob_overload)),
                **within_person(g, "prob_overload"))
    for proto, cand, pol in [("Dev_LOPO_13p", "win_logreg_core9_global", "fixed_050"),
                             ("Dev_LOPO_13p", "win_logreg_core9_global", "nested_fa_cap25_max_rec"),
                             ("Full_LOPO_17p", "win_logreg_core9_global", "nested_fa_cap25_max_rec"),
                             ("Test_historical_4p", "win_logreg_full14_global", "nested_fa_cap25_max_rec"),
                             ("Test_historical_4p", "win_logreg_core9_global", "nested_fa_cap25_max_rec")]:
        g = pred[(pred.protocol == proto) & (pred.candidate == cand) & (pred.policy == pol)]
        out["selected_policies"][f"{proto}|{cand}|{pol}"] = dict(
            accuracy=round(float((g.y == g.predicted).mean()), 3), majority_accuracy=round(1 - g.y.mean(), 3),
            balanced_accuracy=round(bal_acc(g.y, g.predicted), 3),
            balanced_accuracy_ci95=cluster_ci(g, lambda h: bal_acc(h.y, h.predicted)))
    g = pred[(pred.protocol == "Full_LOPO_17p") & (pred.candidate == "win_logreg_core9_global")
             & (pred.policy == "fixed_050")].copy()
    g["difficulty"] = difficulty_code(g.puzzle_set)
    log_time = np.log(g.active_seconds)
    out["context_baselines_full_lopo"] = dict(
        auc_active_seconds=round(auc(g.y, g.active_seconds), 3),
        auc_n_windows=round(auc(g.y, g.n_windows), 3),
        auc_difficulty_set=round(auc(g.y, g.difficulty), 3),
        auc_minutes_into_session=round(auc(g.y, g.minutes_into_session), 3),
        auc_timeout=round(auc(g.y, g.timeout_submissions > 0), 3),
        sensor_over_difficulty=lr_test(g[["difficulty"]].to_numpy(), g.prob_overload, g.y),
        sensor_over_difficulty_time_session=lr_test(
            np.column_stack([g.difficulty, log_time, g.minutes_into_session]), g.prob_overload, g.y),
        within_label_spearman_prob_vs_time={
            str(k): round(float(spearmanr(h.prob_overload, h.active_seconds)[0]), 3) for k, h in g.groupby("y")},
        within_difficulty_auc={k: (round(auc(h.y, h.prob_overload), 3) if h.y.nunique() == 2 else None)
                               for k, h in g.groupby(g.puzzle_set.str.extract(r"^(easy|medium|hard)")[0])},
        person_mean_prob_vs_overload_rate_spearman=round(float(spearmanr(
            g.groupby("participant_id").prob_overload.mean(), g.groupby("participant_id").y.mean())[0]), 3),
        positives_per_person=g.groupby("participant_id").y.sum().astype(int).to_dict())
    dev = pred[(pred.protocol == "Dev_LOPO_13p")]
    rank = (dev.groupby(["candidate", "policy"])
            .apply(lambda h: bal_acc(h.y, h.predicted), include_groups=False).sort_values(ascending=False))
    out["dev_balanced_accuracy_rank_top8"] = {f"{c}|{p}": round(v, 3) for (c, p), v in rank.head(8).items()}
    return out


def audit_v3():
    pred = pd.read_csv(glob.glob(str(B7 / "exp02/outputs/*/episode_predictions.csv"))[0])
    ep = pd.read_csv(DATA / "v3_behavioral_proxy/episode_labels.csv")[["episode_id", "puzzle_set", "active_seconds"]]
    pred = pred.merge(ep, on="episode_id", how="left")
    pred["y"] = pred.actual.eq("High").astype(int)
    out = {"protocols": {}}
    clean = pred[pred.source_history.eq("clean_all_data") & pred.policy.eq("fixed_default")]
    for (proto, rep, rec), g in clean.groupby(["protocol", "representation", "recipe"]):
        score = "raw_score" if rec == "svm_rbf" else "prob_high"
        out["protocols"][f"{proto}|{rep}|{rec}"] = dict(
            episodes=len(g), positives=int(g.y.sum()), majority_accuracy=round(1 - g.y.mean(), 3),
            pooled_auc=round(auc(g.y, g.prob_high), 3), **within_person(g, score))
    g = clean[clean.protocol.eq("Train_LOPO") & clean.representation.eq("center_only") & clean.recipe.eq("svm_rbf")]
    out["duration_leakage_train_lopo_svm"] = dict(
        auc_active_seconds_alone=round(auc(g.y, g.active_seconds), 3),
        spearman_score_vs_active_seconds=round(float(spearmanr(g.raw_score, g.active_seconds)[0]), 3),
        within_label_spearman_score_vs_time={k: round(float(spearmanr(h.raw_score, h.active_seconds)[0]), 3)
                                             for k, h in g.groupby("actual")},
        sensor_over_log_time=lr_test(np.log(g[["active_seconds"]].to_numpy()), g.raw_score, g.y),
        sensor_over_difficulty=lr_test(difficulty_code(g.puzzle_set).to_numpy()[:, None], g.raw_score, g.y))
    return out


def main():
    result = {"v4_exp03": audit_v4(), "v3_exp02": audit_v3()}
    path = Path(__file__).with_name("diagnosis_checks.json")
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Tersimpan:", path)


if __name__ == "__main__":
    main()
