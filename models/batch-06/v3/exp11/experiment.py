"""Source cells for the controlled episode/window SVM comparison."""

from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def compare_fold(x, refs, meta, windows, records, fitting, held):
    if fitting & held:
        raise ValueError("Peserta fitting dan evaluasi bertumpang tindih.")
    labels = fold_labels(records, fitting).reindex(x.index)
    fit = labels.ne("") & meta.participant_id.isin(fitting)
    test = labels.ne("") & meta.participant_id.isin(held)
    if not test.any():
        return None
    y = binary(labels[fit])
    people = meta.loc[fit, "participant_id"]
    episode_weight = 1 / (people.groupby(y).transform("nunique") * y.groupby([y, people]).transform("size"))
    episode_weight /= episode_weight.mean()

    reference = refs.fillna(x.loc[fit].median())
    relative = x - reference
    scaler = StandardScaler().fit(relative.loc[fit], sample_weight=episode_weight)
    fit_a = scaler.transform(relative.loc[fit])
    # The numeric gamma matches exp09's gamma="scale" on A; recomputing it on windows would change the kernel.
    variance = fit_a.var()
    gamma = 1 / (fit_a.shape[1] * variance) if variance else 1.0
    model_a = SVC(kernel="rbf", C=1, gamma=gamma).fit(fit_a, y, sample_weight=episode_weight)
    a_score = high_score(model_a, scaler.transform(relative.loc[test]))

    tr = windows[windows.episode_id.isin(y.index)]
    te = windows[windows.episode_id.isin(labels[test].index)]
    window_weight = tr.episode_id.map(episode_weight) / tr.groupby("episode_id").window_id.transform("size")
    np.testing.assert_allclose(window_weight.groupby(tr.episode_id).sum().loc[y.index], episode_weight)

    def window_matrix(frame):
        centered = frame[SENSORS] - reference.loc[frame.episode_id, SENSORS].to_numpy()
        return scaler.transform(centered)

    model_b = SVC(kernel="rbf", C=1, gamma=gamma)
    model_b.fit(window_matrix(tr), tr.episode_id.map(y), sample_weight=window_weight)
    if model_a.fit_status_ or model_b.fit_status_:
        raise ValueError("SVC belum konvergen.")
    win_pred = te[["participant_id", "episode_id", "window_id"]].assign(score=high_score(model_b, window_matrix(te)))
    b_score = win_pred.groupby("episode_id").score.mean().reindex(labels[test].index)
    if not np.isfinite(b_score).all():
        raise ValueError("Skor window hilang.")
    endpoint = meta.loc[test, ["participant_id"]].assign(original=labels[test], actual=binary(labels[test]))
    rows = []
    for recipe, scores in [("A_episode", a_score), ("B_window", b_score)]:
        rows.append(endpoint.assign(recipe=recipe, score=scores,
                                    predicted=np.where(scores > 0, "High", "Non-High")).reset_index())
    audit = dict(gamma=gamma, fit_episodes=len(y), fit_windows=len(tr), held_episodes=len(endpoint),
                 held_windows=len(te), weight_A=episode_weight.sum(), weight_B=window_weight.sum())
    return pd.concat(rows), win_pred, audit


def run(dataset_root, output_root):
    ratings_file = dataset_root / "v4_self_reported_rating/episode_ratings.csv"
    if not ratings_file.is_file():
        raise FileNotFoundError(f"Salin CSV rating untuk evaluasi exp11: {ratings_file}")
    x, refs, meta, records, train, validation, _, hashes = load_cohort(dataset_root)
    windows = window_pool(dataset_root, records, meta)
    pd.testing.assert_frame_equal(windows.groupby("episode_id")[SENSORS].median().sort_index(), x.sort_index())
    results, scores, folds = [], [], []
    splits = [("Train_LOPO", p, train - {p}, {p}) for p in sorted(train)]
    splits.append(("validation", "validation", train, validation))
    for protocol, fold, fitting, held in splits:
        batch = compare_fold(x, refs, meta, windows, records, fitting, held)
        if batch is None:
            folds.append(dict(protocol=protocol, fold=fold, held_episodes=0))
            continue
        episodes, win_pred, audit = batch
        results.append(episodes.assign(protocol=protocol, fold=fold))
        scores.append(win_pred.assign(protocol=protocol, fold=fold))
        folds.append(dict(protocol=protocol, fold=fold, **audit))
    predictions = pd.concat(results, ignore_index=True)
    metrics, pairs = evaluate(predictions)
    ratings = pd.read_csv(ratings_file, usecols=["episode_id", "participant_id", "difficulty_rating"])
    alignment = compare(predictions.rename(columns={"recipe": "model"}), ratings)
    out = output_root / ("paired_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    for name, frame in [("episode_predictions", predictions), ("window_scores_B", pd.concat(scores)),
                        ("metrics", metrics), ("within_person", pairs), ("rating_alignment", alignment),
                        ("folds", pd.DataFrame(folds))]:
        frame.to_csv(out / (name + ".csv"), index=False)
    save_json(out / "manifest.json", {"inputs": hashes, "rating_sha256": hashlib.sha256(ratings_file.read_bytes()).hexdigest(),
              "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__, "features": SENSORS,
              "kernel": "rbf", "C": 1, "gamma": "scale on A, shared by both", "threshold": 0})
    print(metrics.to_string(index=False))
    print(alignment.to_string(index=False))
    archive = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    print("ZIP:", archive)
    return Path(archive)
