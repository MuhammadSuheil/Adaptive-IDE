"""Paired evaluation of personal pre-episode IQR scaling for the v3 behavioral proxy."""

MIN_IQR_WINDOWS = 4


def historical_iqr(bank, meta, references):
    streams = history_streams(bank)
    spreads = pd.DataFrame(np.nan, index=meta.index, columns=SENSORS)
    audit = []
    for episode, row in meta.iterrows():
        start = pd.to_datetime(row.start_ms, unit="ms", utc=True)
        stream = streams.get((row.participant_id, row.session_id))
        selected = []
        limit = start
        if stream is not None:
            for idx, left, right in stream[["window_start", "window_end"]].itertuples():
                if right <= limit:
                    selected.append(idx)
                    limit = left
                    if len(selected) == 12:
                        break
        last = pd.NaT
        if selected:
            last = stream.at[selected[0], "window_end"]
        if len(selected) >= 3:
            history = stream.loc[selected, SENSORS]
            np.testing.assert_allclose(history.median(), references.loc[episode], rtol=0, atol=0)
            if len(history) >= MIN_IQR_WINDOWS:
                spreads.loc[episode] = history.quantile(0.75) - history.quantile(0.25)
        elif references.loc[episode].notna().any():
            raise ValueError("Referensi pusat tidak sesuai dengan riwayat.")
        audit.append(dict(episode_id=episode, history_windows=len(selected), last_history_end=last,
                          episode_start=start))
    return spreads, pd.DataFrame(audit).set_index("episode_id")


def evaluate_fold(x, refs, meta, labels, spread, fitting, held):
    if fitting & held:
        raise ValueError("Peserta fitting dan evaluasi tumpang tindih.")
    eligible = labels.ne("")
    fit = eligible & meta.participant_id.isin(fitting)
    test = eligible & meta.participant_id.isin(held)
    if not test.any():
        return [], []
    y = binary(labels[fit])
    people = meta.loc[fit, "participant_id"]
    weights = 1 / (people.groupby(y).transform("nunique") * y.groupby([y, people]).transform("size"))
    weights /= weights.mean()
    delta = x - refs.fillna(x.loc[fit].median())
    # Give each fitting participant one vote per feature for the typical historical IQR.
    by_person = spread.loc[fit].groupby(people).median()
    typical = by_person.median()
    # Average local and population IQR to moderate short-history estimates.
    # A zero population IQR or missing history leaves the exp11 feature unchanged.
    ratio = spread.div(typical.where(typical.gt(0)))
    multiplier = (2 / (1 + ratio)).fillna(1.0)
    results = []
    for name, features in [("exp11_center", delta), ("personal_iqr", delta * multiplier)]:
        scaler = StandardScaler().fit(features.loc[fit], sample_weight=weights)
        model = SVC(C=1, gamma="scale", kernel="rbf")
        model.fit(scaler.transform(features.loc[fit]), y, sample_weight=weights)
        if model.fit_status_ != 0:
            raise ValueError("SVM belum konvergen.")
        scores = high_score(model, scaler.transform(features.loc[test]))
        results.append(meta.loc[test, ["participant_id"]].assign(original=labels[test], actual=binary(labels[test]),
                       recipe=name, score=scores, predicted=np.where(scores > 0, "High", "Non-High"),
                       personal_scale_features=ratio.loc[test].count(axis=1)).reset_index())
    audit = pd.DataFrame({"typical_iqr": typical, "fit_people_with_iqr": by_person.count()}).rename_axis("feature")
    return results, audit.reset_index()


def run(root, output_root):
    x, refs, meta, records, train, val, bank, hashes = load_cohort(root)
    spread, history = historical_iqr(bank, meta, refs)
    splits = [("Train_LOPO", p, train - {p}, {p}) for p in sorted(train)]
    splits.append(("validation", "validation", train, val))
    results, audits = [], []
    for protocol, fold, fitting, held in splits:
        labels = fold_labels(records, fitting).reindex(x.index)
        rows, scales = evaluate_fold(x, refs, meta, labels, spread, fitting, held)
        if rows:
            results.extend(row.assign(protocol=protocol, fold=fold) for row in rows)
            audits.append(scales.assign(protocol=protocol, fold=fold))
    pred = pd.concat(results, ignore_index=True).join(history, on="episode_id", validate="many_to_one")
    metrics, pairs = evaluate(pred)
    counts = []
    for (protocol, recipe), g in pred.groupby(["protocol", "recipe"]):
        tn, fp, fn, tp = confusion_matrix(g.actual, g.predicted, labels=TWO).ravel()
        counts.append(dict(protocol=protocol, recipe=recipe, tn=tn, fp=fp, fn=fn, tp=tp,
                           people=g.participant_id.nunique(), episodes_with_scale=g.personal_scale_features.gt(0).sum()))
    metrics = metrics.merge(pd.DataFrame(counts), on=["protocol", "recipe"], validate="one_to_one")
    rating_file = root / "v4_self_reported_rating/episode_ratings.csv"
    ratings = pd.read_csv(rating_file, usecols=["episode_id", "participant_id", "difficulty_rating"])
    alignment = compare(pred.rename(columns={"recipe": "model"}), ratings)
    out = output_root / ("iqr_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    for name, frame in [("episode_predictions", pred), ("metrics", metrics), ("within_person", pairs),
                        ("fold_iqr", pd.concat(audits)), ("rating_alignment", alignment)]:
        frame.to_csv(out / (name + ".csv"), index=False)
    hashes["v4_episode_ratings"] = hashlib.sha256(rating_file.read_bytes()).hexdigest()
    save_json(out / "manifest.json", {"inputs": hashes, "features": SENSORS, "C": 1, "gamma": "scale",
              "threshold": 0, "minimum_iqr_windows": MIN_IQR_WINDOWS, "maximum_history_windows": 12,
              "multiplier": "2 / (1 + local_IQR / typical_fitting_IQR), fallback 1", "test_evaluated": False,
              "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__})
    print(metrics[["protocol", "recipe", "episodes", "accuracy", "auc_pooled", "tn", "fp", "fn", "tp",
                   "episodes_with_scale"]].to_string(index=False))
    print(alignment.to_string(index=False))
    archive = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    print("ZIP:", archive)
    return Path(archive)
