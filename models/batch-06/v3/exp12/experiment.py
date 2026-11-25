"""Compare HRV durations at matched end times, keeping eye/head and pre-task references fixed."""


def pair_hrv(root, short, records):
    path = root / "v2_session_phase/window_45s_step_5s/all_data.csv"
    keys = ["participant_id", "session_id", "window_end"]
    long = pd.read_csv(path, usecols=keys + ["window_start", "window_seconds", "quality_tier",
                                             "hrv_rr_pairs", "window_phase_mix"] + HRV)
    if not long.window_seconds.eq(45).all():
        raise ValueError("Durasi sumber HRV berbeda dari 45 detik.")
    long = long[long.quality_tier.isin(["good", "acceptable"])]
    pairs = short[keys + ["episode_id"] + HRV].merge(long, on=keys, validate="one_to_one",
                                                     suffixes=("_15", "_45"))
    matcher = WindowIndex(records)
    keep = []
    for row in pairs.itertuples():
        episode, _ = matcher.match(row.session_id, epoch_ms(row.window_start), epoch_ms(row.window_end))
        active = all(p["study_phase"] == "active_task" for p in json.loads(row.window_phase_mix)
                     if p["duration_seconds"] > 0)
        if episode and episode["episode_id"] == row.episode_id and active:
            keep.append(row.Index)
    pairs = pairs.loc[keep].copy()
    short_path = root / "v2_session_phase/window_15s_step_5s/all_data.csv"
    counts = pd.read_csv(short_path, usecols=keys + ["hrv_rr_pairs"])
    return pairs.merge(counts, on=keys, validate="one_to_one", suffixes=("_45", "_15")), path


def run(root, output_root):
    x, refs, meta, records, train, validation, _, hashes = load_cohort(root)
    short = window_pool(root, records, meta)
    pairs, long_path = pair_hrv(root, short, records)
    matrices = {"exp11_15": x}
    for duration in (15, 45):
        features = [f"{col}_{duration}" for col in HRV]
        paired_medians = pairs.groupby("episode_id")[features].median()
        if not np.isfinite(paired_medians).all().all():
            raise ValueError("HRV berpasangan tidak lengkap.")
        updated = x.copy()
        updated.loc[paired_medians.index, HRV] = paired_medians.to_numpy()
        matrices[f"matched_{duration}"] = updated

    splits = [("Train_LOPO", p, train - {p}, {p}) for p in sorted(train)]
    splits.append(("validation", "validation", train, validation))
    results = []
    for protocol, fold, fitting, held in splits:
        labels = fold_labels(records, fitting).reindex(x.index)
        for name, matrix in matrices.items():
            result, _ = score_episodes(matrix, refs, meta, labels, fitting, held)
            if len(result):
                results.append(result.assign(recipe=name, protocol=protocol, fold=fold,
                                             predicted=np.where(result.score > 0, "High", "Non-High")).reset_index())
    pred = pd.concat(results, ignore_index=True)
    pred["paired_ends"] = pred.episode_id.map(pairs.groupby("episode_id").size()).fillna(0).astype(int)
    ratings_file = root / "v4_self_reported_rating/episode_ratings.csv"
    ratings = pd.read_csv(ratings_file, usecols=["episode_id", "participant_id", "difficulty_rating", "puzzle_set", "puzzle_id"])
    pred = pred.merge(ratings, how="left", on=["episode_id", "participant_id"], validate="many_to_one")
    tables = []
    for scope, frame in [("all", pred), ("has_45", pred[pred.paired_ends.gt(0)])]:
        metrics, _ = evaluate(frame)
        counts = []
        for (protocol, recipe), g in frame.groupby(["protocol", "recipe"]):
            tn, fp, fn, tp = confusion_matrix(g.actual, g.predicted, labels=TWO).ravel()
            counts.append(dict(protocol=protocol, recipe=recipe, people=g.participant_id.nunique(),
                               tn=tn, fp=fp, fn=fn, tp=tp))
        tables.append(metrics.merge(pd.DataFrame(counts), on=["protocol", "recipe"]).assign(scope=scope))
    summary = pd.concat(tables, ignore_index=True)
    alignment = compare(pred.rename(columns={"recipe": "model"}).drop(columns="difficulty_rating"), ratings)
    out = output_root / ("hrv_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    observed = pairs[pairs.episode_id.isin(pred.episode_id)]
    columns = ["episode_id", "window_end", "hrv_rr_pairs_15", "hrv_rr_pairs_45",
               *[f"{c}_{d}" for c in HRV for d in (15,45)]]
    for name, frame in [("episode_predictions", pred), ("metrics", summary), ("matched_windows", observed[columns]),
                        ("rating_alignment", alignment)]:
        frame.to_csv(out / (name + ".csv"), index=False)
    hashes[str(long_path.relative_to(root))] = hashlib.sha256(long_path.read_bytes()).hexdigest()
    hashes[str(ratings_file.relative_to(root))] = hashlib.sha256(ratings_file.read_bytes()).hexdigest()
    save_json(out / "manifest.json", {"inputs": hashes, "sklearn": sklearn.__version__, "numpy": np.__version__,
              "pandas": pd.__version__, "features": SENSORS, "threshold": 0, "reference_window_seconds": 15,
              "fallback": "original 15 s episode median if no paired end", "test_evaluated": False})
    print(summary[["scope", "protocol", "recipe", "episodes", "accuracy", "auc_pooled", "tn", "fp", "fn", "tp"]].to_string(index=False))
    archive = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    print("ZIP:", archive)
    return Path(archive)
