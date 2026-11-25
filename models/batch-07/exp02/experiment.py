"""Batch 07 Exp02: V3 clean-history personal IQR scaling, SVM+ET blend, and nested threshold calibration."""

from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

LABELS = ["Non-High", "High"]
MIN_IQR_WINDOWS = 4
THRESHOLD_GRID = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
MIN_RECALL_TARGET = 0.75

MODEL_PARAMS = {
    "svm_rbf": {"C": 1, "kernel": "rbf", "gamma": "scale"},
    "extra_trees": {"n_estimators": 300, "max_depth": 3, "min_samples_leaf": 3, "random_state": 42},
}
REPRESENTATIONS = ("center_only", "personal_iqr")
RECIPES = ("svm_rbf", "extra_trees", "blend_svm_et")
POLICIES = ("fixed_default", "nested_recall75", "nested_bal_acc")


def history_streams(path):
    keys = ["participant_id", "session_id"]
    times = ["window_start", "window_end"]
    bank = pd.read_csv(path, usecols=keys + times + ["quality_tier"] + SENSORS)
    bank = bank[bank.quality_tier.isin(["good", "acceptable"]) & np.isfinite(bank[SENSORS]).all(axis=1)].copy()
    for field in times:
        bank[field] = pd.to_datetime(bank[field], utc=True, format="mixed", errors="raise")
    if bank[keys + times].isna().any().any() or not bank.window_end.gt(bank.window_start).all():
        raise ValueError("Interval sensor historis tidak valid.")
    return dict(tuple(bank.sort_values("window_end", ascending=False).groupby(keys)))


def build_full_pool(bank_path, all_windows_path, records, official):
    """Build episode medians and prior-session references (median + IQR) for Train, validation, and Test."""
    streams = history_streams(bank_path)
    frame = pd.read_csv(
        all_windows_path,
        usecols=["window_id", "session_id", "participant_id", "window_start", "window_end", "window_phase_mix", "quality_tier"] + SENSORS,
    )
    frame = frame[frame.quality_tier.isin(["good", "acceptable"])]
    matcher = WindowIndex(records)
    matches = []
    for row in frame.itertuples():
        episode, _ = matcher.match(row.session_id, epoch_ms(row.window_start), epoch_ms(row.window_end))
        active = all(p["study_phase"] == "active_task" for p in json.loads(row.window_phase_mix) if p["duration_seconds"] > 0)
        if episode and active:
            if episode["participant_id"] != row.participant_id:
                raise ValueError("Pemetaan peserta berbeda.")
            matches.append((row.Index, episode["episode_id"]))
    mapped = frame.loc[[i for i, _ in matches]].copy()
    mapped["episode_id"] = [e for _, e in matches]
    known = pd.DataFrame(records).set_index("episode_id")
    published = mapped[mapped.episode_id.map(known.cognitive_load_label).ne("")]
    cols = ["window_id", "participant_id", "episode_id"]
    pd.testing.assert_frame_equal(
        published[cols].sort_values(cols).reset_index(drop=True),
        official[cols].sort_values(cols).reset_index(drop=True),
        check_dtype=False,
    )
    eligible = mapped[mapped.episode_id.map(known.split).isin(["Train", "validation", "Test"])]
    if not np.isfinite(eligible[SENSORS]).all().all():
        raise ValueError("Fitur sensor tidak lengkap.")
    x = eligible.groupby("episode_id")[SENSORS].median()
    meta = known.loc[x.index, ["participant_id", "session_id", "start_ms", "split"]]
    refs = []
    spreads = pd.DataFrame(np.nan, index=x.index, columns=SENSORS)
    hist_audit = []
    for episode_id, row in meta.iterrows():
        stream = streams.get((row.participant_id, row.session_id))
        start = pd.to_datetime(row.start_ms, unit="ms", utc=True)
        chosen = []
        limit = start
        if stream is not None:
            for idx, left, right in stream[["window_start", "window_end"]].itertuples():
                if right <= limit:
                    chosen.append(idx)
                    limit = left
                    if len(chosen) == 12:
                        break
        if len(chosen) >= 3:
            hist_block = stream.loc[chosen, SENSORS]
            ref = hist_block.median()
            last_end = stream.at[chosen[0], "window_end"]
            if len(chosen) >= MIN_IQR_WINDOWS:
                spreads.loc[episode_id] = hist_block.quantile(0.75) - hist_block.quantile(0.25)
        else:
            ref = pd.Series(np.nan, index=SENSORS)
            last_end = pd.NaT
        refs.append(ref)
        hist_audit.append({
            "episode_id": episode_id,
            "history_windows": len(chosen),
            "has_center_ref": int(len(chosen) >= 3),
            "has_iqr_ref": int(len(chosen) >= MIN_IQR_WINDOWS),
            "last_history_end": str(last_end) if pd.notna(last_end) else "",
        })
    return x, pd.DataFrame(refs, index=x.index), spreads, pd.DataFrame(hist_audit).set_index("episode_id"), meta


def load_v3_all_sources(root):
    official, _, hashes = read_bundle(root, "v3")
    package = root / "v3_behavioral_proxy"
    evidence = pd.read_csv(package / "episode_labels.csv", keep_default_na=False, float_precision="round_trip")
    evidence["intervals"] = evidence.intervals.map(json.loads)
    records = evidence.to_dict("records")
    people = pd.read_csv(package / "participant_splits.csv")
    train = frozenset(people.loc[people.split.eq("Train"), "participant_id"])
    val = frozenset(people.loc[people.split.eq("validation"), "participant_id"])
    test = frozenset(people.loc[people.split.eq("Test"), "participant_id"])
    if not evidence.episode_id.map(fold_labels(records, train)).eq(evidence.cognitive_load_label).all():
        raise ValueError("Label split resmi berbeda.")
    v2_dir = root / "v2_session_phase/window_15s_step_5s"
    clean_pool = v2_dir / "all_data.csv"
    curated_bank = v2_dir / "good_acceptable.csv"
    hashes["v2_all"] = snapshot_hash(clean_pool, "acbef9a60f4114de460ee3c57ca94ff5cae0d48b67229d8b14283d173fef1355")
    hashes["v2_curated"] = snapshot_hash(curated_bank, "39e72fd8504bb57d83e2edd91a371e904ba6203ad5a3df90fa560a3eaa715722")
    rating_path = root / "v4_self_reported_rating/episode_ratings.csv"
    hashes["v4_episode_ratings"] = hashlib.sha256(rating_path.read_bytes()).hexdigest()
    ratings = pd.read_csv(rating_path, usecols=["episode_id", "participant_id", "difficulty_rating"])
    clean = build_full_pool(clean_pool, clean_pool, records, official)
    curated = build_full_pool(curated_bank, clean_pool, records, official)
    return clean, curated, records, train, val, test, ratings, hashes


def prepare_fold_matrices(x, refs, spreads, meta, records, fitting, held, representation):
    if fitting & held:
        raise ValueError("Peserta fitting dan evaluasi tumpang tindih.")
    labels = fold_labels(records, fitting).reindex(x.index)
    fit_ids = labels.index[labels.ne("") & meta.participant_id.isin(fitting)]
    held_ids = labels.index[labels.ne("") & meta.participant_id.isin(held)]
    if not len(held_ids):
        return None
    y = binary(labels.loc[fit_ids])
    if set(y) != set(LABELS):
        raise ValueError("Fold fitting harus memuat kedua kelas.")
    people = meta.participant_id.loc[fit_ids]
    w = 1.0 / (people.groupby(y).transform("nunique") * y.groupby([y, people]).transform("size"))
    w /= w.mean()
    delta = x - refs.fillna(x.loc[fit_ids].median())
    if representation == "center_only":
        features = delta
        scale_cnt = pd.Series(0, index=held_ids, dtype=int)
    elif representation == "personal_iqr":
        by_person = spreads.loc[fit_ids].groupby(people).median()
        typical = by_person.median()
        ratio = spreads.div(typical.where(typical.gt(0)))
        multiplier = (2.0 / (1.0 + ratio)).fillna(1.0)
        features = delta * multiplier
        scale_cnt = ratio.loc[held_ids].count(axis=1).astype(int)
    else:
        raise ValueError(representation)
    scaler = StandardScaler().fit(features.loc[fit_ids], sample_weight=w)
    fit_x = pd.DataFrame(scaler.transform(features.loc[fit_ids]), index=fit_ids, columns=x.columns)
    held_x = pd.DataFrame(scaler.transform(features.loc[held_ids]), index=held_ids, columns=x.columns)
    endpoint = meta.loc[held_ids, ["participant_id"]].assign(
        original=labels.loc[held_ids],
        actual=binary(labels.loc[held_ids]),
        personal_scale_features=scale_cnt,
    )
    return fit_x, y, w, held_x, endpoint


def score_models(fit_x, y, w, held_x):
    svm = SVC(**MODEL_PARAMS["svm_rbf"]).fit(fit_x, y, sample_weight=w)
    if svm.fit_status_ != 0:
        raise ValueError("SVM belum konvergen.")
    svm_margin = np.asarray(high_score(svm, held_x), dtype=float)
    svm_prob = expit(svm_margin)
    et = ExtraTreesClassifier(**MODEL_PARAMS["extra_trees"]).fit(fit_x, y, sample_weight=w)
    et_prob = np.asarray(et.predict_proba(held_x)[:, list(et.classes_).index("High")], dtype=float)
    blend_prob = 0.5 * svm_prob + 0.5 * et_prob
    return {
        "svm_rbf": (svm_prob, svm_margin),
        "extra_trees": (et_prob, et_prob),
        "blend_svm_et": (blend_prob, blend_prob),
    }


def select_inner_thresholds(x, refs, spreads, meta, records, fitting, representation):
    """Run inner LOPO strictly inside `fitting` participants to choose calibrated probability thresholds."""
    inner_rows = []
    for p in sorted(fitting):
        inner_fit = fitting - {p}
        data = prepare_fold_matrices(x, refs, spreads, meta, records, inner_fit, {p}, representation)
        if data is None:
            continue
        fit_x, y, w, held_x, endpoint = data
        scored = score_models(fit_x, y, w, held_x)
        for recipe, (prob, _) in scored.items():
            inner_rows.append(endpoint.assign(recipe=recipe, prob=prob).reset_index())
    inner_df = pd.concat(inner_rows, ignore_index=True)
    selected = {}
    for recipe, g in inner_df.groupby("recipe"):
        y_high = g.actual.eq("High").to_numpy()
        p_arr = g.prob.to_numpy()
        # Policy 1: fixed_default
        selected[(recipe, "fixed_default")] = 0.50
        # Policy 2: nested_recall75 (minimize FP subject to High recall >= 0.75)
        feasible = []
        best_bal = None
        for thr in THRESHOLDS_PROB:
            pred_h = p_arr > thr
            tp = int((pred_h & y_high).sum())
            fp = int((pred_h & ~y_high).sum())
            fn = int((~pred_h & y_high).sum())
            tn = int((~pred_h & ~y_high).sum())
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            spec = tn / (tn + fp) if (tn + fp) else 0.0
            bal = 0.5 * (rec + spec)
            if rec >= MIN_RECALL_TARGET:
                feasible.append((fp, -tp, abs(thr - 0.50), thr))
            cand_bal = (-bal, fp, abs(thr - 0.50), thr)
            if best_bal is None or cand_bal < best_bal:
                best_bal = cand_bal
        selected[(recipe, "nested_recall75")] = min(feasible)[3] if feasible else 0.50
        selected[(recipe, "nested_bal_acc")] = best_bal[3]
    return selected


THRESHOLDS_PROB = THRESHOLD_GRID


def summarize_v3(pred):
    rows, pair_rows = [], []
    group_cols = ["source_history", "protocol", "representation", "recipe", "policy"]
    for keys, g in pred.groupby(group_cols):
        source_hist, protocol, rep, recipe, policy = keys
        tn, fp, fn, tp = confusion_matrix(g.actual, g.predicted, labels=LABELS).ravel()
        y_high = g.actual.eq("High")
        aucs, weights = [], []
        for pid, part in g.groupby("participant_id"):
            ph = part.actual.eq("High")
            n_pairs = int(ph.sum() * (~ph).sum())
            if n_pairs:
                p_auc = float(roc_auc_score(ph, part.raw_score))
                aucs.append(p_auc)
                weights.append(n_pairs)
                if policy == "fixed_default":
                    pair_rows.append({
                        "source_history": source_hist, "protocol": protocol, "representation": rep,
                        "recipe": recipe, "participant_id": pid, "pairs": n_pairs, "concordance": p_auc,
                    })
        rows.append({
            "source_history": source_hist,
            "protocol": protocol,
            "representation": rep,
            "recipe": recipe,
            "policy": policy,
            "participants": int(g.participant_id.nunique()),
            "episodes": int(len(g)),
            "accuracy": float((tp + tn) / len(g)),
            "balanced_accuracy": float(balanced_accuracy_score(g.actual, g.predicted)),
            "macro_f1": float(f1_score(g.actual, g.predicted, labels=LABELS, average="macro", zero_division=0)),
            "precision_high": float(precision_score(y_high, g.predicted.eq("High"), zero_division=0)),
            "recall_high": float(recall_score(y_high, g.predicted.eq("High"), zero_division=0)),
            "false_alarm_rate": float(fp / (tn + fp)) if (tn + fp) else 0.0,
            "auc": float(roc_auc_score(y_high, g.raw_score)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "within_person_pairs": int(sum(weights)),
            "concordance_pair_mean": float(np.average(aucs, weights=weights)) if weights else np.nan,
        })
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def align_with_v4_ratings(pred_fixed, ratings):
    rows = []
    for (source_hist, protocol, rep, recipe), g in pred_fixed.groupby(["source_history", "protocol", "representation", "recipe"]):
        merged = g.merge(ratings, on=["episode_id", "participant_id"], how="left", validate="one_to_one")
        rated = merged.dropna(subset=["difficulty_rating"])
        high = rated.difficulty_rating.ge(7)
        rho = float(spearmanr(rated.raw_score, rated.difficulty_rating).statistic) if len(rated) > 2 else np.nan
        auc7 = float(roc_auc_score(high, rated.raw_score)) if high.nunique() == 2 else np.nan
        rows.append({
            "source_history": source_hist, "protocol": protocol, "representation": rep, "recipe": recipe,
            "eligible_episodes": int(len(g)), "rated_episodes": int(len(rated)),
            "rated_participants": int(rated.participant_id.nunique()),
            "overload_7_10": int(high.sum()), "spearman_rating": rho, "auc_rating_7_10": auc7,
        })
    return pd.DataFrame(rows)


def run(root, output_root):
    clean, curated, records, train, val, test, ratings, hashes = load_v3_all_sources(root)
    x_c, refs_c, spreads_c, hist_c, meta_c = clean
    x_g, refs_g, spreads_g, hist_g, meta_g = curated

    # Outer splits: Train_LOPO (15 active folds), validation (3p), and locked historical Test audit (3p)
    outer_splits = [("Train_LOPO", p, train - {p}, {p}) for p in sorted(train)]
    outer_splits.append(("validation", "validation", train, val))
    outer_splits.append(("Test_historical", "Test_historical", train, test))

    pred_frames = []
    thr_records = []

    # 1) Full clean_all_data evaluation across representations x recipes x threshold policies
    for rep in REPRESENTATIONS:
        # Cache inner-LOPO threshold selection per unique fitting set (`train - {p}` and full `train`)
        thr_cache = {}
        for protocol, fold_name, fitting, held in outer_splits:
            data = prepare_fold_matrices(x_c, refs_c, spreads_c, meta_c, records, fitting, held, rep)
            if data is None:
                continue
            fit_key = tuple(sorted(fitting))
            if fit_key not in thr_cache:
                thr_cache[fit_key] = select_inner_thresholds(x_c, refs_c, spreads_c, meta_c, records, fitting, rep)
            selected_thrs = thr_cache[fit_key]
            fit_x, y, w, held_x, endpoint = data
            scored = score_models(fit_x, y, w, held_x)
            for recipe, (prob, raw_score) in scored.items():
                for policy in POLICIES:
                    thr = selected_thrs[(recipe, policy)]
                    pred_label = np.where(prob > thr, "High", "Non-High")
                    pred_frames.append(endpoint.assign(
                        source_history="clean_all_data",
                        protocol=protocol,
                        fold=fold_name,
                        representation=rep,
                        recipe=recipe,
                        policy=policy,
                        threshold=thr,
                        prob_high=prob,
                        raw_score=raw_score,
                        predicted=pred_label,
                    ).reset_index())
                    thr_records.append({
                        "protocol": protocol, "fold": fold_name, "representation": rep,
                        "recipe": recipe, "policy": policy, "threshold": thr,
                    })

    # 2) Direct paired curated_v2 audit for SVM RBF & Extra Trees under fixed_default (completes EVALUASI_V3_V4 Step 2)
    for rep in REPRESENTATIONS:
        for protocol, fold_name, fitting, held in outer_splits:
            data = prepare_fold_matrices(x_g, refs_g, spreads_g, meta_g, records, fitting, held, rep)
            if data is None:
                continue
            fit_x, y, w, held_x, endpoint = data
            scored = score_models(fit_x, y, w, held_x)
            for recipe in ("svm_rbf", "extra_trees", "blend_svm_et"):
                prob, raw_score = scored[recipe]
                pred_frames.append(endpoint.assign(
                    source_history="curated_good_acceptable",
                    protocol=protocol,
                    fold=fold_name,
                    representation=rep,
                    recipe=recipe,
                    policy="fixed_default",
                    threshold=0.50,
                    prob_high=prob,
                    raw_score=raw_score,
                    predicted=np.where(prob > 0.50, "High", "Non-High"),
                ).reset_index())

    pred = pd.concat(pred_frames, ignore_index=True)
    metrics, within_person = summarize_v3(pred)
    fixed_pred = pred[pred.policy.eq("fixed_default")].copy()
    rating_alignment = align_with_v4_ratings(fixed_pred, ratings)

    # Compute exact episode-by-episode curation impact table (curated vs clean under fixed_default)
    piv_cols = ["protocol", "episode_id", "participant_id", "actual", "representation", "recipe"]
    cur_sub = fixed_pred[fixed_pred.source_history.eq("curated_good_acceptable")].set_index(piv_cols)
    cln_sub = fixed_pred[fixed_pred.source_history.eq("clean_all_data")].set_index(piv_cols)
    joined = cur_sub[["raw_score", "predicted"]].join(
        cln_sub[["raw_score", "predicted"]], lsuffix="_curated", rsuffix="_clean", how="inner"
    ).reset_index()
    curation_rows = []
    for (protocol, rep, recipe), g in joined.groupby(["protocol", "representation", "recipe"]):
        cur_m = metrics[(metrics.source_history == "curated_good_acceptable") & (metrics.protocol == protocol)
                        & (metrics.representation == rep) & (metrics.recipe == recipe) & (metrics.policy == "fixed_default")].iloc[0]
        cln_m = metrics[(metrics.source_history == "clean_all_data") & (metrics.protocol == protocol)
                        & (metrics.representation == rep) & (metrics.recipe == recipe) & (metrics.policy == "fixed_default")].iloc[0]
        curation_rows.append({
            "protocol": protocol,
            "representation": rep,
            "recipe": recipe,
            "episodes": int(len(g)),
            "score_changed_episodes": int((np.abs(g.raw_score_clean - g.raw_score_curated) > 1e-9).sum()),
            "decision_changed_episodes": int((g.predicted_clean != g.predicted_curated).sum()),
            "auc_curated": float(cur_m.auc),
            "auc_clean": float(cln_m.auc),
            "delta_auc": float(cln_m.auc - cur_m.auc),
            "acc_curated": float(cur_m.accuracy),
            "acc_clean": float(cln_m.accuracy),
            "fp_curated": int(cur_m.fp),
            "fp_clean": int(cln_m.fp),
            "tp_curated": int(cur_m.tp),
            "tp_clean": int(cln_m.tp),
        })
    curation_audit = pd.DataFrame(curation_rows)

    out = output_root / ("exp02_v3_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    for name, frame in [
        ("episode_predictions", pred),
        ("metrics", metrics),
        ("curation_impact_audit", curation_audit),
        ("within_person", within_person),
        ("rating_alignment", rating_alignment),
        ("inner_thresholds", pd.DataFrame(thr_records).drop_duplicates()),
        ("history_coverage_clean", hist_c.reset_index()),
    ]:
        frame.to_csv(out / f"{name}.csv", index=False)

    # Save overview figure
    import matplotlib
    if "ipykernel" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained")
    lopo_clean = metrics[(metrics.source_history == "clean_all_data") & (metrics.protocol == "Train_LOPO") & (metrics.policy == "fixed_default")]
    labels_bar = [f"{r.recipe}\n({r.representation})" for r in lopo_clean.itertuples()]
    x_pos = np.arange(len(lopo_clean))
    axes[0].bar(x_pos - 0.18, lopo_clean.auc, width=0.35, label="LOPO ROC AUC", color="#2563eb")
    axes[0].bar(x_pos + 0.18, lopo_clean.balanced_accuracy, width=0.35, label="LOPO Balanced Acc", color="#10b981")
    axes[0].axhline(0.5, color="#94a3b8", linestyle="--", linewidth=1)
    axes[0].set_xticks(x_pos, labels_bar, rotation=20, ha="right", fontsize=8.5)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("V3 LOPO (37 ep): Representasi & Model (Riwayat Bersih)")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(axis="y", alpha=0.2)

    pol_sub = metrics[(metrics.source_history == "clean_all_data") & (metrics.protocol == "Train_LOPO") & (metrics.representation == "center_only")]
    for idx_m, rec in enumerate(RECIPES):
        sub_r = pol_sub[pol_sub.recipe == rec].set_index("policy").reindex(POLICIES)
        axes[1].plot(POLICIES, sub_r.false_alarm_rate * 100, marker="o", label=f"{rec} (% Alarm Palsu)")
    axes[1].set_ylabel("Tingkat Alarm Palsu Non-High (%)")
    axes[1].set_ylim(0, 70)
    axes[1].set_title("Dampak Kalibrasi Ambang Dalam-Fold (LOPO)")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(axis="y", alpha=0.2)
    fig.savefig(out / "v3_exp02_overview.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    save_json(out / "manifest.json", {
        "experiment": "batch-07/exp02",
        "inputs": hashes,
        "models": MODEL_PARAMS,
        "representations": list(REPRESENTATIONS),
        "recipes": list(RECIPES),
        "threshold_policies": list(POLICIES),
        "threshold_grid": list(THRESHOLD_GRID),
        "min_iqr_windows": MIN_IQR_WINDOWS,
        "features": SENSORS,
        "selection_splits": ["Train_LOPO", "validation"],
        "historical_audit_split": "Test_historical (frozen post-hoc comparison against batch-06 exp11/exp13)",
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    })
    print("=== LOPO & VALIDATION (CLEAN HISTORY, FIXED DEFAULT) ===")
    print(metrics[(metrics.source_history == "clean_all_data") & (metrics.policy == "fixed_default")][
        ["protocol", "representation", "recipe", "accuracy", "balanced_accuracy", "auc", "tp", "fp", "fn", "tn"]
    ].to_string(index=False))
    print("\n=== NESTED THRESHOLD CALIBRATION (CLEAN HISTORY, CENTER_ONLY) ===")
    print(metrics[(metrics.source_history == "clean_all_data") & (metrics.representation == "center_only")][
        ["protocol", "recipe", "policy", "accuracy", "balanced_accuracy", "recall_high", "false_alarm_rate", "tp", "fp"]
    ].to_string(index=False))
    print("\n=== DAMPAK KURASI GLOBAL vs RIWAYAT BERSIH ===")
    print(curation_audit.to_string(index=False))
    archive = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    print("ZIP:", archive)
    return Path(archive)
