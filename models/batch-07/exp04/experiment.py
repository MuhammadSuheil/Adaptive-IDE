"""Batch 07 Exp04: Unified cross-target evaluation of V3 (behavioral) and V4 (self-reported) models on shared episodes."""

from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
HEAD4 = ["head_pitch_mean", "head_pitch_std", "head_yaw_mean", "head_yaw_std"]
BASE1 = ["rmssd_baseline_ratio"]
V4_FULL14 = CORE9 + HEAD4 + BASE1


def history_streams(path):
    keys = ["participant_id", "session_id"]
    times = ["window_start", "window_end"]
    bank = pd.read_csv(path, usecols=keys + times + ["quality_tier"] + SENSORS)
    bank = bank[bank.quality_tier.isin(["good", "acceptable"]) & np.isfinite(bank[SENSORS]).all(axis=1)].copy()
    for field in times:
        bank[field] = pd.to_datetime(bank[field], utc=True, format="mixed", errors="raise")
    return dict(tuple(bank.sort_values("window_end", ascending=False).groupby(keys)))


def load_unified_cohort(root):
    official_v3, _, hashes_v3 = read_bundle(root, "v3")
    official_v4, _, hashes_v4 = read_bundle(root, "v4")
    hashes = {f"v3/{k}": v for k, v in hashes_v3.items()}
    hashes.update({f"v4/{k}": v for k, v in hashes_v4.items()})

    v3_pkg = root / "v3_behavioral_proxy"
    ev_v3 = pd.read_csv(v3_pkg / "episode_labels.csv", keep_default_na=False, float_precision="round_trip")
    ev_v3["intervals"] = ev_v3.intervals.map(json.loads)
    records = ev_v3.to_dict("records")
    v3_splits = pd.read_csv(v3_pkg / "participant_splits.csv")
    v3_train_people = frozenset(v3_splits.loc[v3_splits.split.eq("Train"), "participant_id"])

    v2_all = root / "v2_session_phase/window_15s_step_5s/all_data.csv"
    hashes["v2_all"] = snapshot_hash(v2_all, "acbef9a60f4114de460ee3c57ca94ff5cae0d48b67229d8b14283d173fef1355")

    # Map all quality-eligible active_task windows in v2_all to episodes (covers both V3 and V4 episodes)
    streams = history_streams(v2_all)
    frame = pd.read_csv(
        v2_all,
        usecols=["window_id", "session_id", "participant_id", "window_start", "window_end", "window_phase_mix", "quality_tier"] + SENSORS,
    )
    frame = frame[frame.quality_tier.isin(["good", "acceptable"])]
    matcher = WindowIndex(records)
    matches = []
    for row in frame.itertuples():
        episode, _ = matcher.match(row.session_id, epoch_ms(row.window_start), epoch_ms(row.window_end))
        active = all(p["study_phase"] == "active_task" for p in json.loads(row.window_phase_mix) if p["duration_seconds"] > 0)
        if episode and active:
            matches.append((row.Index, episode["episode_id"]))
    mapped = frame.loc[[i for i, _ in matches]].copy()
    mapped["episode_id"] = [e for _, e in matches]
    mapped = mapped[np.isfinite(mapped[SENSORS]).all(axis=1)].copy()

    known = pd.DataFrame(records).set_index("episode_id")
    ep_x = mapped.groupby("episode_id")[SENSORS].median()
    ep_meta = known.loc[ep_x.index, [
        "participant_id", "session_id", "start_ms", "split", "cognitive_load_label",
        "active_seconds", "manual_errors", "completed",
    ]].copy()

    refs = []
    hist_cnt = []
    for ep_id, row in ep_meta.iterrows():
        start = pd.to_datetime(row.start_ms, unit="ms", utc=True)
        ref, n_h, _ = prior_reference(streams.get((row.participant_id, row.session_id)), start)
        refs.append(ref)
        hist_cnt.append(n_h)
    ep_refs = pd.DataFrame(refs, index=ep_x.index)
    ep_meta["history_windows"] = hist_cnt

    # Load V4 windows with full columns (for V4 per-window models)
    v4_dir = root / "v4_self_reported_rating"
    v4_raw = pd.read_csv(v4_dir / "window_15s_step_5s/good_acceptable.csv")
    v4_wa = pd.read_csv(v4_dir / "window_15s_step_5s/window_assignments.csv", usecols=["window_id", "episode_id", "difficulty_rating"])
    v4_win = v4_raw.merge(v4_wa.rename(columns={"difficulty_rating": "wa_r"}), on="window_id", how="left", validate="1:1")
    if not (v4_win["difficulty_rating"] == v4_win["wa_r"]).all():
        raise ValueError("Rating window V4 tidak konsisten.")
    v4_win = v4_win.drop(columns=["wa_r"])
    v4_win["y_overload"] = (v4_win["difficulty_rating"] >= 7).astype(int)

    # Official V3 and V4 episode sets
    v3_ep_ids = set(official_v3["episode_id"].unique())
    v4_ep_ids = set(v4_win["episode_id"].unique())
    shared_ep_ids = sorted(v3_ep_ids & v4_ep_ids)

    v4_ep_ratings = v4_win.groupby("episode_id")["difficulty_rating"].first()
    ep_meta["difficulty_rating"] = ep_meta.index.map(v4_ep_ratings)
    ep_meta["in_v3_official"] = ep_meta.index.isin(v3_ep_ids)
    ep_meta["in_v4_official"] = ep_meta.index.isin(v4_ep_ids)
    ep_meta["in_shared"] = ep_meta.index.isin(shared_ep_ids)

    return ep_x, ep_refs, ep_meta, records, v3_train_people, v4_win, shared_ep_ids, hashes


def v4_window_weights(sub_df, y_bin, clip_max=5.0):
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


def fit_v4_window_model(fit_win, eval_win, cols, kind="logreg"):
    fit_vals = fit_win[cols].to_numpy(dtype=float)
    med = np.nanmedian(fit_vals, axis=0)
    filled = np.where(np.isfinite(fit_vals), fit_vals, med)
    q75, q25 = np.percentile(filled, [75, 25], axis=0)
    iqr = np.where((q75 - q25) > 1e-9, q75 - q25, 1.0)
    X_tr = np.clip((np.where(np.isfinite(fit_vals), fit_vals, med) - med) / iqr, -10.0, 10.0)
    ev_vals = eval_win[cols].to_numpy(dtype=float)
    X_te = np.clip((np.where(np.isfinite(ev_vals), ev_vals, med) - med) / iqr, -10.0, 10.0)
    y_tr = fit_win["y_overload"].to_numpy(dtype=int)
    w_tr = v4_window_weights(fit_win, y_tr)
    if kind == "logreg":
        clf = LogisticRegression(C=0.5, max_iter=5000, random_state=SEED).fit(X_tr, y_tr, sample_weight=w_tr)
    else:
        clf = ExtraTreesClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=8, max_features=0.7, random_state=SEED, n_jobs=1
        ).fit(X_tr, y_tr, sample_weight=w_tr)
    p_win = clf.predict_proba(X_te)[:, 1]
    return pd.Series(p_win, index=eval_win["episode_id"]).groupby(level=0).mean()


def fit_v3_episode_models(ep_x, ep_refs, ep_meta, records, fit_people, eval_ep_ids):
    labels = fold_labels(records, fit_people).reindex(ep_x.index)
    fit_ids = labels.index[labels.ne("") & ep_meta.participant_id.isin(fit_people)]
    y = binary(labels.loc[fit_ids])
    people = ep_meta.loc[fit_ids, "participant_id"]
    w = 1.0 / (people.groupby(y).transform("nunique") * y.groupby([y, people]).transform("size"))
    w /= w.mean()
    rel = ep_x - ep_refs.fillna(ep_x.loc[fit_ids].median())
    scaler = StandardScaler().fit(rel.loc[fit_ids], sample_weight=w)
    X_tr = scaler.transform(rel.loc[fit_ids])
    X_te = scaler.transform(rel.loc[eval_ep_ids])

    svm = SVC(C=1, kernel="rbf", gamma="scale").fit(X_tr, y.to_numpy(), sample_weight=w.to_numpy())
    if svm.fit_status_ != 0:
        raise ValueError("SVM V3 belum konvergen.")
    svm_prob = expit(svm.decision_function(X_te) * (1 if svm.classes_[1] == "High" else -1))

    et = ExtraTreesClassifier(n_estimators=300, max_depth=3, min_samples_leaf=3, random_state=SEED, n_jobs=1).fit(
        X_tr, y.to_numpy(), sample_weight=w.to_numpy()
    )
    et_prob = et.predict_proba(X_te)[:, list(et.classes_).index("High")]
    blend_prob = 0.5 * svm_prob + 0.5 * et_prob
    return {
        "V3_SVM_RBF": pd.Series(svm_prob, index=eval_ep_ids),
        "V3_ExtraTrees": pd.Series(et_prob, index=eval_ep_ids),
        "V3_Blend_SVM_ET": pd.Series(blend_prob, index=eval_ep_ids),
    }, int(len(fit_ids)), int(people.nunique())


def bootstrap_ci(y_arr, p_arr, pid_arr, n_boot=1000):
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


def evaluate_shared(pred_df):
    targets = {
        "behavioral_high_v3": pred_df["y_v3_high"].to_numpy(dtype=int),
        "self_report_7_10_v4": pred_df["y_v4_overload"].to_numpy(dtype=int),
        "concordant_both_high": ((pred_df["y_v3_high"] == 1) & (pred_df["y_v4_overload"] == 1)).astype(int).to_numpy(),
        "either_high_union": ((pred_df["y_v3_high"] == 1) | (pred_df["y_v4_overload"] == 1)).astype(int).to_numpy(),
    }
    rows = []
    for (regime, model_name), g in pred_df.groupby(["training_regime", "model"]):
        probs = g["prob_high"].to_numpy(dtype=float)
        preds = (probs >= 0.50).astype(int)
        pids = g["participant_id"].to_numpy()
        rho_rating = float(spearmanr(probs, g["difficulty_rating"]).statistic)
        rho_time = float(spearmanr(probs, g["active_seconds"]).statistic)
        for tname, _ in targets.items():
            if tname == "behavioral_high_v3":
                y_t = g["y_v3_high"].to_numpy(dtype=int)
            elif tname == "self_report_7_10_v4":
                y_t = g["y_v4_overload"].to_numpy(dtype=int)
            elif tname == "concordant_both_high":
                y_t = ((g["y_v3_high"] == 1) & (g["y_v4_overload"] == 1)).astype(int).to_numpy()
            else:
                y_t = ((g["y_v3_high"] == 1) | (g["y_v4_overload"] == 1)).astype(int).to_numpy()
            tn, fp, fn, tp = confusion_matrix(y_t, preds, labels=[0, 1]).ravel()
            ci_lo, ci_hi = bootstrap_ci(y_t, probs, pids)
            rows.append({
                "training_regime": regime,
                "target": tname,
                "model": model_name,
                "participants": int(g["participant_id"].nunique()),
                "episodes": int(len(g)),
                "positive_episodes": int(y_t.sum()),
                "negative_episodes": int((1 - y_t).sum()),
                "auc": float(roc_auc_score(y_t, probs)),
                "auc_ci95_lo": ci_lo,
                "auc_ci95_hi": ci_hi,
                "accuracy": float((tp + tn) / len(g)),
                "balanced_accuracy": float(balanced_accuracy_score(y_t, preds)),
                "macro_f1": float(f1_score(y_t, preds, labels=[0, 1], average="macro", zero_division=0)),
                "precision": float(precision_score(y_t, preds, zero_division=0)),
                "recall": float(recall_score(y_t, preds, zero_division=0)),
                "false_alarm_rate": float(fp / (tn + fp)) if (tn + fp) else 0.0,
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                "spearman_rating_1_10": rho_rating,
                "spearman_active_seconds": rho_time,
            })
    return pd.DataFrame(rows)


def run(root, output_root):
    ep_x, ep_refs, ep_meta, records, v3_train_people, v4_win, shared_ep_ids, hashes = load_unified_cohort(root)
    shared_meta = ep_meta.loc[shared_ep_ids].copy()
    shared_meta["y_v3_high"] = shared_meta["cognitive_load_label"].eq("High").astype(int)
    shared_meta["y_v4_overload"] = (shared_meta["difficulty_rating"] >= 7).astype(int)

    shared_people = sorted(shared_meta["participant_id"].unique())
    all_v4_people = frozenset(v4_win["participant_id"].unique())
    all_cohort_people = frozenset(ep_meta["participant_id"].unique())

    # Concordance summary between V3 behavioral proxy and V4 self-reported rating on the 40 shared episodes
    ct = confusion_matrix(shared_meta["y_v3_high"], shared_meta["y_v4_overload"], labels=[0, 1])
    concordance_summary = {
        "shared_episodes": int(len(shared_meta)),
        "shared_participants": int(shared_meta["participant_id"].nunique()),
        "v3_high_count": int(shared_meta["y_v3_high"].sum()),
        "v4_overload_7_10_count": int(shared_meta["y_v4_overload"].sum()),
        "both_high_count": int(ct[1, 1]),
        "v3_high_only_count": int(ct[1, 0]),
        "v4_overload_only_count": int(ct[0, 1]),
        "both_non_high_count": int(ct[0, 0]),
        "agreement_rate": float((ct[0, 0] + ct[1, 1]) / len(shared_meta)),
        "spearman_active_sec_vs_rating": float(spearmanr(shared_meta["active_seconds"], shared_meta["difficulty_rating"]).statistic),
        "spearman_errors_vs_rating": float(spearmanr(shared_meta["manual_errors"], shared_meta["difficulty_rating"]).statistic),
    }

    pred_rows = []
    # Regime 1: Native_HeldOut_LOPO (each model trained on its official non-held-out pool: V3 on v3_train-{p}, V4 on v4_all-{p})
    # Regime 2: Matched_Participants_LOPO (both V3 and V4 trained on all_cohort_people-{p} so participant pool is identical)
    for regime in ("Native_HeldOut_LOPO", "Matched_Participants_LOPO"):
        for held_p in shared_people:
            held_eps = shared_meta.index[shared_meta["participant_id"] == held_p].tolist()
            if regime == "Native_HeldOut_LOPO":
                v3_fit_p = v3_train_people - {held_p}
                v4_fit_p = all_v4_people - {held_p}
            else:
                v3_fit_p = all_cohort_people - {held_p}
                v4_fit_p = all_cohort_people - {held_p}

            v3_scores, n_v3_ep, n_v3_p = fit_v3_episode_models(ep_x, ep_refs, ep_meta, records, v3_fit_p, held_eps)
            fit_w4 = v4_win[v4_win["participant_id"].isin(v4_fit_p)].reset_index(drop=True)
            held_w4 = v4_win[v4_win["episode_id"].isin(held_eps)].reset_index(drop=True)

            v4_lr_full = fit_v4_window_model(fit_w4, held_w4, V4_FULL14, kind="logreg").reindex(held_eps)
            v4_lr_core9 = fit_v4_window_model(fit_w4, held_w4, CORE9, kind="logreg").reindex(held_eps)
            v4_et_full = fit_v4_window_model(fit_w4, held_w4, V4_FULL14, kind="et").reindex(held_eps)

            model_map = {
                **v3_scores,
                "V4_LogReg_Full14": v4_lr_full,
                "V4_LogReg_Core9": v4_lr_core9,
                "V4_ExtraTrees_Full14": v4_et_full,
                "Dual_V3ET_V4LR": 0.5 * v3_scores["V3_ExtraTrees"] + 0.5 * v4_lr_full,
                "Dual_V3SVM_V4LR": 0.5 * v3_scores["V3_SVM_RBF"] + 0.5 * v4_lr_full,
            }
            base = shared_meta.loc[held_eps, [
                "participant_id", "cognitive_load_label", "difficulty_rating",
                "active_seconds", "manual_errors", "y_v3_high", "y_v4_overload",
            ]]
            for mname, s_prob in model_map.items():
                pred_rows.append(base.assign(
                    training_regime=regime,
                    model=mname,
                    prob_high=s_prob.to_numpy(dtype=float),
                    v3_fit_episodes=n_v3_ep,
                    v4_fit_episodes=int(fit_w4["episode_id"].nunique()),
                ).reset_index())

    pred_df = pd.concat(pred_rows, ignore_index=True)
    metrics = evaluate_shared(pred_df)

    out = output_root / ("exp04_unified_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    shared_meta.reset_index().to_csv(out / "shared_episodes_concordance.csv", index=False)
    pred_df.to_csv(out / "episode_predictions.csv", index=False)
    metrics.to_csv(out / "metrics.csv", index=False)

    # Visualization
    import matplotlib
    if "ipykernel" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), layout="constrained")
    nat = metrics[metrics.training_regime.eq("Native_HeldOut_LOPO")]
    model_order = ["V3_SVM_RBF", "V3_ExtraTrees", "V4_LogReg_Full14", "V4_LogReg_Core9", "Dual_V3ET_V4LR", "Dual_V3SVM_V4LR"]
    t_list = [
        ("behavioral_high_v3", "Target: Perilaku V3 High", "#2563eb"),
        ("self_report_7_10_v4", "Target: Rating V4 ≥7", "#10b981"),
        ("concordant_both_high", "Target: Konkordan (Keduanya High)", "#dc2626"),
    ]
    xp = np.arange(len(model_order))
    width = 0.25
    for idx_t, (tkey, tlabel, col) in enumerate(t_list):
        sub_t = nat[nat.target.eq(tkey)].set_index("model").reindex(model_order)
        axes[0].bar(xp + (idx_t - 1) * width, sub_t.auc, width=width, label=tlabel, color=col)
    axes[0].axhline(0.50, color="#94a3b8", ls="--")
    axes[0].set_xticks(xp, model_order, rotation=22, ha="right", fontsize=8.5)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("ROC AUC (40 Episode Irisan)")
    axes[0].set_title("Perbandingan Silang Model V3 vs V4 pada 40 Episode yang Sama")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)

    # Scatter of V3_ExtraTrees vs V4_LogReg_Full14 probabilities colored by joint class
    p_v3 = pred_df[(pred_df.training_regime == "Native_HeldOut_LOPO") & (pred_df.model == "V3_ExtraTrees")].set_index("episode_id")
    p_v4 = pred_df[(pred_df.training_regime == "Native_HeldOut_LOPO") & (pred_df.model == "V4_LogReg_Full14")].set_index("episode_id").loc[p_v3.index]
    cats = [
        ((p_v3.y_v3_high == 0) & (p_v3.y_v4_overload == 0), "Non-High pada Keduanya", "#94a3b8", "o"),
        ((p_v3.y_v3_high == 1) & (p_v3.y_v4_overload == 0), "Hanya V3 Perilaku High", "#2563eb", "^"),
        ((p_v3.y_v3_high == 0) & (p_v3.y_v4_overload == 1), "Hanya V4 Rating ≥7", "#10b981", "s"),
        ((p_v3.y_v3_high == 1) & (p_v3.y_v4_overload == 1), "Konkordan High (V3 & V4)", "#dc2626", "D"),
    ]
    for mask, lbl, col, mk in cats:
        if mask.any():
            axes[1].scatter(p_v3.loc[mask, "prob_high"], p_v4.loc[mask, "prob_high"], label=f"{lbl} (n={int(mask.sum())})",
                            color=col, marker=mk, s=55, alpha=0.88, edgecolors="white", linewidth=0.6)
    axes[1].axvline(0.50, color="#cbd5e1", ls=":")
    axes[1].axhline(0.50, color="#cbd5e1", ls=":")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Skor Out-of-Sample V3_ExtraTrees")
    axes[1].set_ylabel("Skor Out-of-Sample V4_LogReg_Full14")
    axes[1].set_title("Sebaran Skor Sensor V3 vs V4 pada 40 Episode Irisan")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].grid(alpha=0.2)

    fig.savefig(out / "exp04_v3_vs_v4_cross_evaluation.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    save_json(out / "manifest.json", {
        "experiment": "batch-07/exp04",
        "inputs": hashes,
        "concordance_summary": concordance_summary,
        "training_regimes": ["Native_HeldOut_LOPO", "Matched_Participants_LOPO"],
        "targets": ["behavioral_high_v3", "self_report_7_10_v4", "concordant_both_high", "either_high_union"],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    })

    print("=== KONKORDANSI TARGET V3 vs V4 PADA 40 EPISODE IRISAN ===")
    print(json.dumps(concordance_summary, indent=2))
    print("\n=== HASIL EVALUASI SILANG (NATIVE_HELDOUT_LOPO) ===")
    print(metrics[metrics.training_regime.eq("Native_HeldOut_LOPO")][
        ["target", "model", "participants", "episodes", "positive_episodes", "auc", "auc_ci95_lo", "auc_ci95_hi",
         "balanced_accuracy", "tp", "fp", "spearman_rating_1_10"]
    ].to_string(index=False))
    archive = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    print("ZIP:", archive)
    return Path(archive)
