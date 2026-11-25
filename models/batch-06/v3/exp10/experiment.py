"""Fixed v4 window recipes, evaluated against behavioral High on exp09 episode endpoints."""

RATIO = "rmssd_baseline_ratio"
CORE9 = [c for c in CORE if c != "iris_size_delta_mean"]
PROFILES = {"lr_core9": CORE9, "lr_core10": CORE, "lr_core9_head": CORE9 + HEAD,
            "lr_core9_head_base": CORE9 + HEAD + [RATIO], "lr_head": HEAD,
            "lr_core9_head_prior_session": CORE9 + HEAD}


def window_pool(root, records, meta):
    pool = root / "v2_session_phase/window_15s_step_5s/all_data.csv"
    fields = ["window_id", "participant_id", "session_id", "window_start", "window_end", "window_phase_mix",
              "quality_tier", "baseline_usable", RATIO] + SENSORS
    data = pd.read_csv(pool, usecols=fields)
    data = data[data.quality_tier.isin(["good", "acceptable"]) & data.participant_id.isin(meta.participant_id)]
    index = WindowIndex(records)
    matches = []
    for row in data.itertuples():
        ep, _ = index.match(row.session_id, epoch_ms(row.window_start), epoch_ms(row.window_end))
        if ep and ep["episode_id"] in meta.index and all(
                p["study_phase"] == "active_task" for p in json.loads(row.window_phase_mix) if p["duration_seconds"] > 0):
            if ep["participant_id"] != row.participant_id:
                raise ValueError("Peserta window tidak cocok dengan episode.")
            matches.append((row.Index, ep["episode_id"]))
    windows = data.loc[[i for i, _ in matches]].copy()
    windows["episode_id"] = [ep for _, ep in matches]
    windows.loc[~windows.baseline_usable.eq(True), RATIO] = np.nan
    return windows


def global_robust(fitting, held, features):
    # Same fitting median, IQR after imputation, zero-IQR fallback and clipping as exp03.
    x, x_held = fitting[features].to_numpy(float), held[features].to_numpy(float)
    median = np.nanmedian(x, axis=0)
    if not np.isfinite(median).all():
        raise ValueError("Fitur tanpa median pada fitting.")
    filled = np.where(np.isfinite(x), x, median)
    q25, q75 = np.percentile(filled, [25, 75], axis=0)
    iqr = np.where(q75 - q25 > 1e-9, q75 - q25, 1.0)
    return (np.clip((filled - median) / iqr, -10, 10),
            np.clip((np.where(np.isfinite(x_held), x_held, median) - median) / iqr, -10, 10))


def run(dataset_root, output_root):
    x, refs, meta, records, train, validation, _, hashes = load_cohort(dataset_root)
    windows = window_pool(dataset_root, records, meta)
    pd.testing.assert_frame_equal(windows.groupby("episode_id")[SENSORS].median().sort_index(), x.sort_index())
    parts, window_parts = [], []
    folds = [("Train_LOPO", p, train - {p}, {p}) for p in sorted(train)]
    folds.append(("validation", "validation", train, validation))
    for protocol, fold, fitting, held in folds:
        data = fold_data(x, refs, meta, records, fitting, held)
        if data is None:
            continue
        fit_x, y, weights, held_x, endpoint = data
        model = SVC(C=1, kernel="rbf", gamma="scale").fit(fit_x, y, sample_weight=weights)
        parts.append(endpoint.assign(protocol=protocol, fold=fold, model="exp09_rbf", score=high_score(model, held_x),
                                     predicted=model.predict(held_x)).reset_index())
        fit_windows = windows[windows.episode_id.isin(y.index)]
        held_windows = windows[windows.episode_id.isin(endpoint.index)]
        target = fit_windows.episode_id.map(y).eq("High").to_numpy(dtype=int)
        weights_window = episode_balanced_weights(fit_windows, target)
        for recipe, features in PROFILES.items():
            fit_view, held_view = fit_windows, held_windows
            if recipe == "lr_core9_head_prior_session":
                # Controlled follow-up: same logistic, weights and endpoints as lr_core9_head.
                # Each episode reference was collected before it started. Cold starts use fitting episodes only.
                ref = refs[features].fillna(x.loc[y.index, features].median())
                fit_view = fit_windows[features] - ref.loc[fit_windows.episode_id].to_numpy()
                held_view = held_windows[features] - ref.loc[held_windows.episode_id].to_numpy()
            a, b = global_robust(fit_view, held_view, features)
            lr = LogisticRegression(C=0.5, max_iter=5000, random_state=42).fit(a, target, sample_weight=weights_window)
            prob = lr.predict_proba(b)[:, 1]
            win = held_windows[["window_id", "episode_id", "participant_id"]].assign(prob_high=prob, model=recipe,
                         fold=fold, protocol=protocol)
            window_parts.append(win)
            score = win.groupby("episode_id").prob_high.mean().reindex(endpoint.index)
            if score.isna().any():
                raise ValueError("Episode evaluasi tidak lengkap.")
            parts.append(endpoint.assign(protocol=protocol, fold=fold, model=recipe, score=score,
                                         predicted=np.where(score >= 0.5, "High", "Non-High")).reset_index())
    pred = pd.concat(parts, ignore_index=True)
    rows = []
    for (protocol, model), g in pred.groupby(["protocol", "model"]):
        if g.episode_id.duplicated().any():
            raise ValueError("Endpoint berulang.")
        high = g.actual.eq("High")
        tn, fp, fn, tp = confusion_matrix(g.actual, g.predicted, labels=TWO).ravel()
        pairs = g[high].merge(g[~high], on="participant_id", suffixes=("_hi", "_lo"))
        diff = pairs.score_hi - pairs.score_lo
        rows.append(dict(protocol=protocol, model=model, episodes=len(g), participants=g.participant_id.nunique(),
                         auc=roc_auc_score(high, g.score), balanced_accuracy=balanced_accuracy_score(g.actual, g.predicted),
                         accuracy=g.actual.eq(g.predicted).mean(), tp=tp, fn=fn, fp=fp, tn=tn,
                         pairs=len(diff), pair_concordance=(diff.gt(0).sum()+0.5*diff.eq(0).sum())/len(diff)
                         if len(diff) else np.nan))
    summary = pd.DataFrame(rows)
    out = output_root / ("transfer_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    summary.to_csv(out / "metrics.csv", index=False)
    pred.to_csv(out / "episode_predictions.csv", index=False)
    pd.concat(window_parts, ignore_index=True).to_csv(out / "window_predictions.csv", index=False)
    save_json(out / "manifest.json", {"inputs": hashes, "profiles": PROFILES, "primary": "lr_core9_head_base",
              "C": 0.5, "window_weight_clip": 5.0, "threshold": 0.5, "sklearn": sklearn.__version__,
              "pandas": pd.__version__, "numpy": np.__version__})
    print(summary.to_string(index=False))
    archive = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    print("ZIP:", archive)
    return Path(archive)
