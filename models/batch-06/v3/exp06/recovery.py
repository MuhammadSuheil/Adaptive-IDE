"""Paired single-modality controls using already recorded, labeled training episodes."""

HR_FAILURES = {"invalid_or_missing:rmssd_ms", "invalid_or_missing:sdnn_ms", "hrv_unusable",
               "hrv_coverage", "hrv_rr_pairs", "hrv_data_age_seconds", "rr_local_median_jump"}


def recoverable(root, case, source, out):
    package = root / ("v3_behavioral_proxy" if case == "v3" else "v4_self_reported_rating")
    target = "cognitive_load_label" if case == "v3" else "difficulty_rating"
    wd = package / "window_15s_step_5s"
    assignment = pd.read_csv(wd / "window_assignments.csv")
    accepted = pd.read_csv(wd / "good_acceptable.csv", usecols=["window_id"])
    covered = assignment.loc[assignment.window_id.isin(accepted.window_id), "episode_id"]
    candidates = assignment[assignment.quality_tier.eq("poor") & ~assignment.episode_id.isin(covered)]
    keys = ["window_id", "participant_id", "session_id", "quality_tier"]
    matched = candidates.merge(source, on=keys, validate="one_to_one")
    if len(matched) != len(candidates):
        raise ValueError("Sumber v2 tidak mencakup kandidat pemulihan.")
    evidence_file = "episode_labels.csv" if case == "v3" else "episode_ratings.csv"
    ids = ["episode_id", "participant_id", "session_id", target]
    ev = pd.read_csv(package / evidence_file, usecols=ids)
    if len(matched.merge(ev, on=ids, validate="many_to_one")) != len(matched):
        raise ValueError("Label episode kandidat berbeda dari bukti.")
    if case == "v3":
        splits = pd.read_csv(package / "participant_splits.csv")
        if len(matched.merge(splits, on=["participant_id", "split"], validate="many_to_one")) != len(matched):
            raise ValueError("Split peserta pada pemulihan berbeda dari split resmi.")
        matched = matched[matched.split.eq("Train")]
    flags = matched.acceptable_failed_criteria.fillna("").map(lambda s: set(s.split(";")))
    masks = {"eye": (EYE, flags.map(lambda f: f.issubset(HR_FAILURES))),
             "hrv": (HRV, flags.map(lambda f: f == {"min_iris_available_ratio"}))}
    extras = {}
    audit = []
    for name, (features, mask) in masks.items():
        usable = matched.loc[mask & np.isfinite(matched[features]).all(axis=1)].copy()
        usable = usable.rename(columns={target: "actual"})
        episode = usable.groupby("episode_id")
        extras[name] = episode[features].median().join(episode[["participant_id", "actual"]].first())
        audit.append(usable[["window_id", "episode_id", "participant_id", "actual", "acceptable_failed_criteria"]]
                     .assign(modality=name))
    pd.concat(audit).to_csv(out / f"{case}_recovered_windows.csv", index=False)
    return extras


def recovery_fold(case, fold, meta, x, extras, fit, held):
    predictions = []
    for name, features in [("eye", EYE), ("hrv", HRV)]:
        original = x.loc[fit, features].join(meta[["participant_id", "actual"]])
        additional = extras[name]
        # V3 additions already obey Train. For LOPO exclude the held person from all sources.
        additional = additional[~additional.participant_id.isin(meta.loc[held, "participant_id"])]
        for variant, train in [("original", original), ("expanded", pd.concat([original, additional]))]:
            if train.index.has_duplicates or train.participant_id.isin(meta.loc[held, "participant_id"]).any():
                raise ValueError("Episode duplikat atau peserta evaluasi muncul dalam fitting.")
            if case == "v3":
                participants = train.groupby("actual").participant_id.transform("nunique")
                episodes = train.groupby(["actual", "participant_id"]).actual.transform("size")
                weights = 1 / (participants * episodes)
                weights = weights / weights.mean()
                if variant == "original":
                    published = meta.loc[train.index, "weight"]
                    if not np.allclose(weights, published / published.mean()):
                        raise ValueError("Bobot kontrol tidak cocok dengan bobot episode resmi.")
            else:
                weights = np.ones(len(train))
            scaler = StandardScaler().fit(train[features], sample_weight=weights)
            model = SVC(C=1, kernel="rbf", gamma="scale")
            model.fit(scaler.transform(train[features]), train.actual, sample_weight=weights)
            result = meta.loc[held, ["participant_id", "actual", "history_windows"]].reset_index()
            result["predicted"] = model.predict(scaler.transform(x.loc[held, features]))
            predictions.append(result.assign(model=f"{name}_{variant}", fold=fold, fitting_episodes=len(train)))
    return predictions
