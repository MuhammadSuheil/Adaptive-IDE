"""Episode-level comparison for the paired Colab notebook. Shared CSV loader is embedded at build time."""

from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler


def episode_views(root, case, windows):
    package = "v3_behavioral_proxy" if case == "v3" else "v4_self_reported_rating"
    source = root / package / "window_15s_step_5s/good_acceptable.csv"
    times = pd.read_csv(source, usecols=["window_id", "window_start"])
    times["window_start"] = pd.to_datetime(times.window_start, utc=True, format="mixed", errors="raise")
    data = windows.merge(times, on="window_id", how="left", validate="one_to_one")
    if data.window_start.isna().any() or data.duplicated(["episode_id", "window_start"]).any():
        raise ValueError("Awal window kosong atau ambigu dalam episode.")
    if case == "v3":
        data = data[data.split.isin(["Train", "validation"])]
    ordered = data.sort_values(["episode_id", "window_start"])
    meta = ordered.drop_duplicates("episode_id").set_index("episode_id")
    keys = ["participant_id", "actual"] + (["split"] if case == "v3" else [])
    episodes = meta[keys].copy()
    groups = ordered.groupby("episode_id")
    if groups[keys].nunique().gt(1).any().any():
        raise ValueError("Metadata episode tidak konsisten.")
    if case == "v3":
        # Published weights allocate mass hierarchically down to windows.
        episodes["weight"] = groups.sample_weight.sum(min_count=1)
    return episodes, {"first": meta[CORE], "median": groups[CORE].median().loc[meta.index]}


def evaluate_fold(case, episodes, views, fit, held, labels, fold):
    train, test = episodes.loc[fit], episodes.loc[held]
    if set(train.participant_id) & set(test.participant_id):
        raise ValueError("Peserta evaluasi muncul pada fitting.")
    y = train.actual
    weights = train.weight / train.weight.mean() if case == "v3" else np.ones(len(train))
    predictions = []

    def record(name, predicted, estimate=None):
        rows = test[["participant_id", "actual"]].reset_index()
        rows["case"], rows["fold"], rows["model"] = case, fold, name
        rows["predicted"] = predicted
        if estimate is not None:
            rows["continuous_rating"] = estimate
        predictions.append(rows)

    counts = y.value_counts().reindex(labels, fill_value=0)
    record("dummy_mode", counts.idxmax())
    if case == "v4":
        record("dummy_median", y.quantile(0.5, interpolation="lower"))
    for view, x in views.items():
        scaler = StandardScaler().fit(x.loc[fit], sample_weight=weights)
        x_train = scaler.transform(x.loc[fit])
        x_held = scaler.transform(x.loc[held])
        classifier = LogisticRegression(C=0.1, max_iter=3000)
        classifier.fit(x_train, y, sample_weight=weights)
        record("logistic_" + view, classifier.predict(x_held))
        if case == "v4":
            reg = QuantileRegressor(quantile=0.5, alpha=0.10, solver="highs")
            reg.fit(x_train, y)
            estimate = reg.predict(x_held)
            rating = np.floor(np.clip(estimate, 1, 10) + 0.5).astype(int)
            record("quantile_" + view, rating, estimate)
    return predictions


def run_episodes(dataset_root, output_root):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = output_root / f"episodes_{stamp}_{uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                "sklearn": sklearn.__version__, "inputs": {}, "features": CORE,
                "views": ["first eligible window", "coordinatewise median of all eligible windows"],
                "logistic_C": 0.1, "quantile": 0.5, "quantile_alpha": 0.1,
                "primary": {"v3": "logistic_median", "v4": "quantile_median"}}
    all_metrics, all_people, all_confusions = [], [], []
    for case in ("v3", "v4"):
        windows, labels, manifest["inputs"][case] = read_bundle(dataset_root, case)
        episodes, views = episode_views(dataset_root, case, windows)
        if case == "v3":
            folds = [("validation", episodes.split.eq("Train"), episodes.split.eq("validation"))]
        else:
            folds = [(f"LOPO_{i}", episodes.participant_id.ne(person), episodes.participant_id.eq(person))
                     for i, person in enumerate(sorted(episodes.participant_id.unique()), 1)]
        predictions = []
        memberships = []
        for fold, fit, held in folds:
            predictions.extend(evaluate_fold(case, episodes, views, fit, held, labels, fold))
            for role, mask in [("fit", fit), ("held", held)]:
                membership = episodes.loc[mask, ["participant_id", "actual"]].reset_index()
                memberships.append(membership.assign(fold=fold, role=role))
        pred = pd.concat(predictions, ignore_index=True)
        expected = episodes.index[folds[0][2]] if case == "v3" else episodes.index
        for model, subset in pred.groupby("model"):
            if subset.episode_id.duplicated().any() or set(subset.episode_id) != set(expected):
                raise ValueError("Prediksi episode ganda atau tidak lengkap.")
            info = {"case": case, "model": model}
            all_metrics.append({**info, **scores(subset, labels)})
            for person, group in subset.groupby("participant_id"):
                all_people.append({**info, "participant_id": person, **scores(group, labels)})
            matrix = confusion_matrix(subset.actual, subset.predicted, labels=labels)
            for label, counts in zip(labels, matrix):
                all_confusions.append({**info, "actual": label, "support": counts.sum(),
                                       **dict(zip([f"pred_{k}" for k in labels], counts))})
        pred.to_csv(out / f"{case}_episode_predictions.csv", index=False)
        pd.concat(memberships).to_csv(out / f"{case}_folds.csv", index=False)
    save_json(out / "manifest.json", manifest)
    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(all_people).to_csv(out / "participant_metrics.csv", index=False)
    pd.DataFrame(all_confusions).to_csv(out / "confusion.csv", index=False)
    archive = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    print(metrics.to_string(index=False))
    print("ZIP untuk dikembalikan:", archive)
    return Path(archive)
