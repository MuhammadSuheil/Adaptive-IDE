"""Compare three estimators on identical held-out v3 episodes."""

from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

LABELS = ["Non-High", "High"]
MODEL_PARAMS = {
    "svm_rbf": {"C": 1, "kernel": "rbf", "gamma": "scale"},
    "logistic_l2": {"C": 1, "solver": "lbfgs", "max_iter": 2000},
    "extra_trees": {"n_estimators": 300, "max_depth": 3, "min_samples_leaf": 3, "random_state": 42},
}
ESTIMATORS = {"svm_rbf": SVC, "logistic_l2": LogisticRegression, "extra_trees": ExtraTreesClassifier}
THRESHOLDS = {"svm_rbf": 0.0, "logistic_l2": 0.5, "extra_trees": 0.5}


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


def load_cohort(root):
    official, _, hashes = read_bundle(root, "v3")
    package = root / "v3_behavioral_proxy"
    evidence = pd.read_csv(package / "episode_labels.csv", keep_default_na=False, float_precision="round_trip")
    evidence["intervals"] = evidence.intervals.map(json.loads)
    records = evidence.to_dict("records")
    people = pd.read_csv(package / "participant_splits.csv")
    train = frozenset(people.loc[people.split.eq("Train"), "participant_id"])
    val = frozenset(people.loc[people.split.eq("validation"), "participant_id"])
    if not evidence.episode_id.map(fold_labels(records, train)).eq(evidence.cognitive_load_label).all():
        raise ValueError("Label split resmi berbeda.")
    pool = root / "v2_session_phase/window_15s_step_5s/all_data.csv"
    hashes["v2_all"] = snapshot_hash(pool, "acbef9a60f4114de460ee3c57ca94ff5cae0d48b67229d8b14283d173fef1355")
    # The v2 source is shared by episode mapping and history. History applies its own quality-only filter.
    x, refs, meta = build_pool(pool, pool, records, official)
    return x, refs, meta, records, train, val, hashes


def fit_predict(fit_x, y, weights, held_x, endpoint):
    if set(y) != set(LABELS):
        raise ValueError("Fold fitting harus memuat kedua kelas.")
    rows, coefficients = [], []
    for name, estimator in ESTIMATORS.items():
        model = estimator(**MODEL_PARAMS[name]).fit(fit_x, y, sample_weight=weights)
        features = held_x.columns
        if name == "svm_rbf":
            if model.fit_status_ != 0:
                raise ValueError("SVM belum konvergen.")
            score = high_score(model, held_x)
        else:
            score = model.predict_proba(held_x)[:, list(model.classes_).index("High")]
        if name == "logistic_l2":
            sign = 1 if model.classes_[1] == "High" else -1
            coefficients.append(pd.DataFrame({"feature": features, "coefficient_high": model.coef_[0] * sign}))
        rows.append(endpoint.assign(recipe=name, score=score,
                    predicted=np.where(score > THRESHOLDS[name], "High", "Non-High")).reset_index())
    return pd.concat(rows, ignore_index=True), pd.concat(coefficients, ignore_index=True)


def summarize(pred):
    metrics = []
    for (protocol, recipe), g in pred.groupby(["protocol", "recipe"]):
        tn, fp, fn, tp = confusion_matrix(g.actual, g.predicted, labels=LABELS).ravel()
        if not ((tp + fn) and (tn + fp)):
            raise ValueError(f"Kedua kelas dibutuhkan untuk AUC {protocol}.")
        metrics.append(dict(protocol=protocol, recipe=recipe, participants=g.participant_id.nunique(),
                            episodes=len(g), accuracy=(tp + tn) / len(g),
                            balanced_accuracy=balanced_accuracy_score(g.actual, g.predicted),
                            macro_f1=f1_score(g.actual, g.predicted, labels=LABELS, average="macro", zero_division=0),
                            auc=roc_auc_score(g.actual.eq("High"), g.score), tn=tn, fp=fp, fn=fn, tp=tp))
    stats = pd.DataFrame(metrics)
    ranks = stats[stats.protocol.eq("Train_LOPO")].sort_values(
        ["auc", "balanced_accuracy", "recipe"], ascending=[False, False, True])
    return stats, ranks.reset_index(drop=True)


def run(root, output_root):
    x, refs, meta, records, train, val, hashes = load_cohort(root)
    folds = [("Train_LOPO", participant, train - {participant}, {participant})
             for participant in sorted(train)]
    folds.append(("validation", "validation", train, val))
    predictions, coefficients, audit = [], [], []
    for protocol, fold, fitting, held in folds:
        data = fold_data(x, refs, meta, records, fitting, held)
        if data is None:
            audit.append(dict(protocol=protocol, fold=fold, held_episodes=0))
            continue
        fit_x, y, weights, held_x, endpoint = data
        rows, coefs = fit_predict(fit_x, y, weights, held_x, endpoint)
        predictions.append(rows.assign(protocol=protocol, fold=fold))
        coefficients.append(coefs.assign(protocol=protocol, fold=fold))
        audit.append(dict(protocol=protocol, fold=fold, fit_episodes=len(fit_x), held_episodes=len(held_x),
                          held_with_history=int(refs.loc[held_x.index].notna().all(axis=1).sum()),
                          fit_people=meta.loc[fit_x.index, "participant_id"].nunique()))
    pred = pd.concat(predictions, ignore_index=True)
    endpoint_keys = ["protocol", "fold", "episode_id", "participant_id", "actual", "original"]
    endpoints = pred[pred.recipe.eq("svm_rbf")].set_index(endpoint_keys).sort_index()
    if endpoints.index.has_duplicates or pred.duplicated(endpoint_keys + ["recipe"]).any():
        raise ValueError("Prediksi episode berulang.")
    for name in ESTIMATORS:
        pd.testing.assert_index_equal(pred[pred.recipe.eq(name)].set_index(endpoint_keys).sort_index().index,
                                      endpoints.index)
    metrics, ranking = summarize(pred)
    out = output_root / ("models_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    for name, frame in [("episode_predictions", pred), ("metrics", metrics), ("lopo_ranking", ranking),
                        ("logistic_coefficients", pd.concat(coefficients)), ("folds", pd.DataFrame(audit))]:
        frame.to_csv(out / (name + ".csv"), index=False)
    save_json(out / "manifest.json", {"inputs": hashes, "models": MODEL_PARAMS, "thresholds": THRESHOLDS,
              "features": SENSORS, "history": "v2 all_data; good/acceptable, finite; prior disjoint windows, min 3 max 12",
              "test_evaluated": False, "candidate_order": ["LOPO AUC", "LOPO balanced accuracy", "recipe"],
              "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__})
    print(metrics.to_string(index=False))
    archive = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    print("ZIP:", archive)
    return Path(archive)
