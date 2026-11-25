"""Source cells following the exp07 data-loading functions in the standalone notebook."""

from functools import lru_cache

THRESHOLDS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)
MIN_RECALL = 0.75
BLOCKS = {"HRV": HRV, "Eye": EYE, "Head": HEAD}


def load_cohort(root):
    official, _, hashes = read_bundle(root, "v3")
    package = root / "v3_behavioral_proxy"
    ev = pd.read_csv(package / "episode_labels.csv", keep_default_na=False, float_precision="round_trip")
    ev["intervals"] = ev.intervals.map(json.loads)
    records = ev.to_dict("records")
    people = pd.read_csv(package / "participant_splits.csv")
    train = frozenset(people.loc[people.split.eq("Train"), "participant_id"])
    val = frozenset(people.loc[people.split.eq("validation"), "participant_id"])
    if not ev.episode_id.map(fold_labels(records, train)).eq(ev.cognitive_load_label).all():
        raise ValueError("Label split resmi berbeda.")
    wd = root / "v2_session_phase/window_15s_step_5s"
    bank, pool = wd / "good_acceptable.csv", wd / "all_data.csv"
    hashes["v2_history"] = snapshot_hash(bank, "39e72fd8504bb57d83e2edd91a371e904ba6203ad5a3df90fa560a3eaa715722")
    hashes["v2_all"] = snapshot_hash(pool, "acbef9a60f4114de460ee3c57ca94ff5cae0d48b67229d8b14283d173fef1355")
    x, refs, meta = build_pool(bank, pool, records, official)
    return x, refs, meta, records, train, val, bank, hashes


def score_episodes(x, refs, meta, labels, fitting, held, features=SENSORS):
    if fitting & held:
        raise ValueError("Overlap peserta.")
    fit_ids = labels.index[labels.ne("") & meta.participant_id.isin(fitting)]
    test_ids = labels.index[labels.ne("") & meta.participant_id.isin(held)]
    result = meta.loc[test_ids, ["participant_id"]].assign(original=labels.loc[test_ids])
    result["actual"] = binary(result.original)
    if result.empty:
        return result.assign(score=pd.Series(dtype=float)), None
    y = binary(labels.loc[fit_ids])
    if set(y) != set(TWO):
        raise ValueError("Kelas fitting tidak lengkap.")
    persons = meta.loc[fit_ids, "participant_id"]
    weights = 1 / (persons.groupby(y).transform("nunique") * y.groupby([y, persons]).transform("size"))
    weights /= weights.mean()
    delta = x[features] - refs[features].fillna(x.loc[fit_ids, features].median())
    scaler = StandardScaler().fit(delta.loc[fit_ids], sample_weight=weights)
    model = SVC(C=1, gamma="scale", kernel="rbf").fit(scaler.transform(delta.loc[fit_ids]), y,
                                                      sample_weight=weights)
    test = scaler.transform(delta.loc[test_ids])
    # sklearn's positive decision value corresponds to classes_[1].
    direction = 1 if model.classes_[1] == "High" else -1
    result["score"] = direction * model.decision_function(test)
    if not np.array_equal(np.where(result.score > 0, "High", "Non-High"), model.predict(test)):
        raise ValueError("Prediksi ambang nol berbeda dari SVC.")
    return result, (delta, scaler, y)


def alarm_metrics(actual, predicted):
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=TWO).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "accuracy": (tn + tp) / len(actual),
            "precision_high": tp / (tp + fp) if tp + fp else np.nan,
            "recall_high": tp / (tp + fn) if tp + fn else np.nan}


def choose_threshold(oof):
    if oof.empty or oof.index.has_duplicates:
        raise ValueError("OOF pemilihan ambang kosong atau duplikat.")
    evaluations = []
    for threshold in THRESHOLDS:
        pred = np.where(oof.score > threshold, "High", "Non-High")
        evaluations.append({"threshold": threshold, **alarm_metrics(oof.actual, pred)})
    table = pd.DataFrame(evaluations)
    feasible = table[table.recall_high.ge(MIN_RECALL)]
    selected = 0.0 if feasible.empty else float(feasible.sort_values(
        ["fp", "tp", "threshold"], ascending=[True, False, True]).iloc[0].threshold)
    return selected, table.assign(selected=table.threshold.eq(selected), fallback=feasible.empty)


def validation_audit(x, refs, meta, bank, decisions, fitted):
    delta, scaler, training_y = fitted
    streams = history_streams(bank)
    tn_ids = decisions.index[decisions.actual.eq("Non-High") & decisions.baseline.eq("Non-High")]
    scaled = pd.DataFrame(scaler.transform(delta), index=delta.index, columns=SENSORS)
    train_center = scaled.loc[training_y.index[training_y.eq("Non-High")]].median()
    tn_center = scaled.loc[tn_ids].median()
    detail, blocks = [], []
    for eid, row in decisions.iterrows():
        m = meta.loc[eid]
        start = pd.to_datetime(m.start_ms, unit="ms", utc=True)
        ref, count, end = prior_reference(streams.get((m.participant_id, m.session_id)), start)
        if (end is not None and end > start) or not np.allclose(ref, refs.loc[eid], equal_nan=True):
            raise ValueError("Audit referensi berbeda dari prediksi.")
        info = {"episode_id": eid, "participant_id": m.participant_id, "original": row.original,
                "actual": row.actual, "baseline": row.baseline, "tuned": row.tuned, "score": row.score,
                "history_windows": count, "reference_end": end, "episode_start": start}
        for block, columns in BLOCKS.items():
            z = scaled.loc[eid, columns]
            blocks.append({**info, "block": block,
                           "rms_vs_train_non_high": np.sqrt(((z - train_center[columns])**2).mean()),
                           "rms_vs_validation_tn": np.sqrt(((z - tn_center[columns])**2).mean())})
            for col in columns:
                detail.append({**info, "block": block, "feature": col, "raw": x.at[eid, col],
                               "session_reference": ref[col], "effective_reference": x.at[eid, col] - delta.at[eid, col],
                               "delta": delta.at[eid, col], "z": z[col],
                               "z_train_non_high_median": train_center[col], "z_validation_tn_median": tn_center[col]})
    return pd.DataFrame(detail), pd.DataFrame(blocks)


def run(dataset_root, output_root):
    x, refs, meta, records, train, validation, bank, hashes = load_cohort(dataset_root)

    @lru_cache(maxsize=None)
    def labels_for(fitting):
        return fold_labels(records, fitting).reindex(x.index)

    def fit(fitting, held):
        return score_episodes(x, refs, meta, labels_for(fitting), fitting, held)

    predictions, grids, tuning, memberships = [], [], [], []
    folds = [("Train_LOPO", p, train - {p}, frozenset([p])) for p in sorted(train)]
    folds.append(("validation", "validation", train, validation))
    for protocol, fold, fitting, held in folds:
        inner = []
        for person in sorted(fitting):
            inner_fit = fitting - {person}
            result, _ = fit(inner_fit, {person})
            inner.append(result)
            memberships.append({"protocol": protocol, "outer_fold": fold, "inner_held": person,
                                "inner_fitting": json.dumps(sorted(inner_fit)), "outer_held": json.dumps(sorted(held)),
                                "scored_episodes": len(result)})
        oof = pd.concat(inner)
        threshold, grid = choose_threshold(oof)
        grids.append(grid.assign(protocol=protocol, outer_fold=fold))
        tuning.append(oof.assign(protocol=protocol, outer_fold=fold).reset_index())
        outer, fitted = fit(fitting, held)
        outer = outer.assign(protocol=protocol, fold=fold, threshold=threshold,
                             baseline=np.where(outer.score > 0, "High", "Non-High"),
                             tuned=np.where(outer.score > threshold, "High", "Non-High"))
        predictions.append(outer.reset_index())
        if protocol == "validation":
            detail, blocks = validation_audit(x, refs, meta, bank, outer, fitted)
            # Exploratory refits on identical Train episodes to inspect the dependence on each sensor block.
            ablations = []
            for block, columns in BLOCKS.items():
                remaining = [name for name in SENSORS if name not in columns]
                reduced, _ = score_episodes(x, refs, meta, labels_for(fitting), fitting, held, remaining)
                ablations.append(reduced.assign(removed=block, predicted=np.where(reduced.score > 0, "High", "Non-High"),
                                                full_baseline=outer.baseline).reset_index())
    pred = pd.concat(predictions, ignore_index=True)
    summary = []
    for protocol, group in pred.groupby("protocol"):
        if group.episode_id.duplicated().any():
            raise ValueError("Duplikat endpoint.")
        for rule in ["baseline", "tuned"]:
            summary.append({"protocol": protocol, "rule": rule, "episodes": len(group),
                            "participants": group.participant_id.nunique(), **alarm_metrics(group.actual, group[rule])})
    out = output_root / ("alarms_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    tables = {"metrics": pd.DataFrame(summary), "episode_predictions": pred, "threshold_selection": pd.concat(grids),
              "inner_oof": pd.concat(tuning), "inner_folds": pd.DataFrame(memberships), "validation_features": detail,
              "validation_blocks": blocks, "validation_ablation": pd.concat(ablations)}
    for name, frame in tables.items():
        frame.to_csv(out / (name + ".csv"), index=False)
    save_json(out / "manifest.json", {"inputs": hashes, "features": SENSORS, "thresholds": THRESHOLDS,
              "min_recall_inner": MIN_RECALL, "sklearn": sklearn.__version__, "pandas": pd.__version__,
              "numpy": np.__version__, "positive_class": "High", "fallback_threshold": 0.0})
    print(tables["metrics"].to_string(index=False))
    print("Ambang untuk validation:", threshold)
    archive = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    print("ZIP:", archive)
    return Path(archive)
