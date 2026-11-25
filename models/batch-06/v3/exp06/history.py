"""Notebook cells. read_bundle, CORE, scores and save_json come from the shared relabeling pipeline."""

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

HEAD = ["head_pitch_mean", "head_pitch_std", "head_yaw_mean", "head_yaw_std"]
SENSORS = CORE + HEAD


def history_streams(path):
    keys = ["participant_id", "session_id"]
    bank = pd.read_csv(path, usecols=keys + ["window_start", "window_end"] + SENSORS)
    for key in ["window_start", "window_end"]:
        bank[key] = pd.to_datetime(bank[key], utc=True, format="mixed", errors="raise")
    if bank[keys + ["window_start", "window_end"]].isna().any().any() or not bank.window_end.gt(bank.window_start).all():
        raise ValueError("Interval sensor historis tidak valid.")
    bank = bank[np.isfinite(bank[SENSORS]).all(axis=1)]
    return dict(tuple(bank.sort_values("window_end", ascending=False).groupby(keys)))


def prior_reference(stream, start):
    # Newest first. Work only with disjoint windows that finished before this task.
    chosen = []
    limit = start
    if stream is not None:
        for index, left, right in stream[["window_start", "window_end"]].itertuples():
            if right <= limit:
                chosen.append(index)
                limit = left
                if len(chosen) == 12:
                    break
    if len(chosen) < 3:
        return pd.Series(np.nan, index=SENSORS), len(chosen), None
    return stream.loc[chosen, SENSORS].median(), len(chosen), stream.at[chosen[0], "window_end"]


def cohort(root, case, streams):
    windows, labels, hashes = read_bundle(root, case)
    if case == "v3":
        windows = windows[windows.split.isin(["Train", "validation"])]
    package = root / ("v3_behavioral_proxy" if case == "v3" else "v4_self_reported_rating")
    source = package / "window_15s_step_5s/good_acceptable.csv"
    head = pd.read_csv(source, usecols=["window_id"] + HEAD)
    windows = windows.merge(head, on="window_id", how="left", validate="one_to_one")
    if not np.isfinite(windows[SENSORS]).all().all():
        raise ValueError("Perbandingan berpasangan memerlukan head pose dan core yang lengkap.")
    groups = windows.groupby("episode_id")
    x = groups[SENSORS].median()
    columns = ["participant_id", "session_id", "actual"] + (["split"] if case == "v3" else [])
    if groups[columns].nunique().ne(1).any().any():
        raise ValueError("Metadata episode tidak konsisten.")
    meta = groups[columns].first()
    if case == "v3":
        meta["weight"] = groups.sample_weight.sum(min_count=1)
    evidence = package / ("episode_labels.csv" if case == "v3" else "episode_ratings.csv")
    start_ms = pd.read_csv(evidence, usecols=["episode_id", "start_ms"]).set_index("episode_id").start_ms
    meta["start"] = pd.to_datetime(start_ms.loc[meta.index], unit="ms", utc=True, errors="raise")
    if meta.start.isna().any():
        raise ValueError("Awal episode tidak lengkap.")
    references = []
    for row in meta.itertuples():
        stream = streams.get((row.participant_id, row.session_id))
        ref, count, end = prior_reference(stream, row.start)
        references.append(ref)
        meta.at[row.Index, "history_windows"] = count
        meta.at[row.Index, "reference_end"] = end.isoformat() if end is not None else ""
    reference = pd.DataFrame(references, index=meta.index)
    return meta, x, reference, labels, hashes


def predict_fold(case, fold, meta, x, reference, labels, fit, held):
    train, test = meta.loc[fit], meta.loc[held]
    if set(train.participant_id) & set(test.participant_id):
        raise ValueError("Peserta fitting dan evaluasi tumpang tindih.")
    target = train.actual
    weights = train.weight / train.weight.mean() if case == "v3" else np.ones(len(train))
    # Cold start uses the same fold's fitting median, never other held episodes.
    centered = x - reference.fillna(x.loc[fit].median())
    predictions = []

    def record(name, values):
        frame = test[["participant_id", "actual", "history_windows"]].reset_index()
        predictions.append(frame.assign(model=name, predicted=values, fold=fold))

    record("dummy_mode", target.value_counts().reindex(labels, fill_value=0).idxmax())
    if case == "v4":
        record("dummy_median", target.quantile(0.5, interpolation="lower"))
    for name, view in [("raw_core", x[CORE]), ("relative_core", centered[CORE]),
                       ("raw_core_head", x), ("relative_core_head", centered)]:
        scaler = StandardScaler().fit(view.loc[fit], sample_weight=weights)
        x_fit, x_held = scaler.transform(view.loc[fit]), scaler.transform(view.loc[held])
        nominal = SVC(C=1, kernel="rbf", gamma="scale")
        nominal.fit(x_fit, target, sample_weight=weights)
        record("svc_" + name, nominal.predict(x_held))
        if case == "v4":
            ordinal = SVR(C=2, epsilon=0.5, kernel="rbf", gamma="scale")
            ordinal.fit(x_fit, target)
            record("svr_" + name, np.floor(ordinal.predict(x_held).clip(1, 10) + 0.5).astype(int))
    return predictions


def run_history(root, output_root):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = output_root / f"history_{stamp}_{uuid4().hex[:8]}"
    bank_path = root / "v2_session_phase/window_15s_step_5s/good_acceptable.csv"
    streams = history_streams(bank_path)
    all_path = bank_path.with_name("all_data.csv")
    source = pd.read_csv(all_path, usecols=["window_id", "participant_id", "session_id", "quality_tier",
                                           "acceptable_failed_criteria"] + CORE)
    out.mkdir(parents=True, exist_ok=False)
    inputs = {"v2_history_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
              "v2_all_sha256": hashlib.sha256(all_path.read_bytes()).hexdigest()}
    metrics, per_person = [], []
    for case in ["v3", "v4"]:
        meta, x, ref, labels, inputs[case] = cohort(root, case, streams)
        extras = recoverable(root, case, source, out)
        if case == "v3":
            folds = [("validation", meta.split.eq("Train"), meta.split.eq("validation"))]
        else:
            folds = [(f"LOPO_{i}", meta.participant_id.ne(p), meta.participant_id.eq(p))
                     for i, p in enumerate(sorted(meta.participant_id.unique()), 1)]
        predictions = []
        for fold, fit, held in folds:
            predictions.extend(predict_fold(case, fold, meta, x, ref, labels, fit, held))
            predictions.extend(recovery_fold(case, fold, meta, x, extras, fit, held))
        pred = pd.concat(predictions, ignore_index=True)
        expected = meta.index[meta.split.eq("validation")] if case == "v3" else meta.index
        for name, group in pred.groupby("model"):
            if group.episode_id.duplicated().any() or set(group.episode_id) != set(expected):
                raise ValueError("Cakupan prediksi salah.")
            info = {"case": case, "model": name}
            metrics.append({**info, **scores(group, labels), "correct": int(group.actual.eq(group.predicted).sum())})
            for person, rows in group.groupby("participant_id"):
                per_person.append({**info, "participant_id": person, **scores(rows, labels)})
        meta.to_csv(out / f"{case}_references.csv")
        pred.to_csv(out / f"{case}_predictions.csv", index=False)
    summary = pd.DataFrame(metrics)
    summary.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(per_person).to_csv(out / "participant_metrics.csv", index=False)
    save_json(out / "manifest.json", {"inputs": inputs, "features": SENSORS, "sklearn": sklearn.__version__,
              "pandas": pd.__version__, "numpy": np.__version__, "primary_exact": "svc_relative_core_head",
              "primary_distance": "svr_relative_core_head", "max_reference_windows": 12, "min_reference_windows": 3,
              "svc_C": 1, "svr_C": 2, "svr_epsilon": 0.5, "kernel": "rbf", "gamma": "scale"})
    print(summary.to_string(index=False))
    archive = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    print("ZIP:", archive)
    return Path(archive)
