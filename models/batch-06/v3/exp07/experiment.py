"""Standalone notebook cells for participant-held-out evaluation of the v3 target."""

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

THREE = ["Low", "Medium", "High"]
TWO = ["Non-High", "High"]


def binary(y):
    return y.where(y.eq("High"), "Non-High")


def snapshot_hash(path, expected_lf_hash):
    data = path.read_bytes()
    if hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest() != expected_lf_hash:
        raise ValueError(f"Gunakan snapshot v2 15 detik dari repo untuk exp07: {path.name}")
    return hashlib.sha256(data).hexdigest()


def build_pool(bank, all_windows, records, official):
    # Use all quality-eligible sensor windows, including episodes unlabeled by the full-Train reference.
    streams = history_streams(bank)
    frame = pd.read_csv(all_windows, usecols=["window_id", "session_id", "participant_id", "window_start",
                                             "window_end", "window_phase_mix", "quality_tier"] + SENSORS)
    frame = frame[frame.quality_tier.isin(["good", "acceptable"])]
    matcher = WindowIndex(records)
    matches = []
    for row in frame.itertuples():
        episode, _ = matcher.match(row.session_id, epoch_ms(row.window_start), epoch_ms(row.window_end))
        active = all(p["study_phase"] == "active_task" for p in json.loads(row.window_phase_mix)
                     if p["duration_seconds"] > 0)
        if episode and active:
            if episode["participant_id"] != row.participant_id:
                raise ValueError("Pemetaan peserta berbeda.")
            matches.append((row.Index, episode["episode_id"]))
    mapped = frame.loc[[i for i, _ in matches]].copy()
    mapped["episode_id"] = [e for _, e in matches]
    known = pd.DataFrame(records).set_index("episode_id")
    published = mapped[mapped.episode_id.map(known.cognitive_load_label).ne("")]
    cols = ["window_id", "participant_id", "episode_id"]
    pd.testing.assert_frame_equal(published[cols].sort_values(cols).reset_index(drop=True),
                                  official[cols].sort_values(cols).reset_index(drop=True), check_dtype=False)
    # No Test feature vectors enter model development.
    dev = mapped[mapped.episode_id.map(known.split).isin(["Train", "validation"])]
    if not np.isfinite(dev[SENSORS]).all().all():
        raise ValueError("Fitur sensor tidak lengkap.")
    x = dev.groupby("episode_id")[SENSORS].median()
    meta = known.loc[x.index, ["participant_id", "session_id", "start_ms", "split"]]
    refs = []
    for row in meta.itertuples():
        stream = streams.get((row.participant_id, row.session_id))
        start = pd.to_datetime(row.start_ms, unit="ms", utc=True)
        ref, _, _ = prior_reference(stream, start)
        refs.append(ref)
    return x, pd.DataFrame(refs, index=x.index), meta


def fold_labels(records, fit_people):
    roles = {row["participant_id"]: ("Train" if row["participant_id"] in fit_people else "held")
             for row in records}
    reference = make_reference(records, roles)
    return pd.Series({r["episode_id"]: label_episode(r, reference)["cognitive_load_label"]
                      for r in records})


def classify(x, y, people, held_index):
    persons = people.loc[y.index]
    weight = 1 / (persons.groupby(y).transform("nunique") * y.groupby([y, persons]).transform("size"))
    weight = weight / weight.mean()
    scaler = StandardScaler().fit(x.loc[y.index], sample_weight=weight)
    model = SVC(C=1, kernel="rbf", gamma="scale")
    model.fit(scaler.transform(x.loc[y.index]), y, sample_weight=weight)
    return pd.Series(model.predict(scaler.transform(x.loc[held_index])), index=held_index)


def run_fold(x, refs, meta, records, protocol, fold, fit_people, held_people, coverage):
    if fit_people & held_people:
        raise ValueError("Peserta muncul pada kedua sisi.")
    labels = fold_labels(records, fit_people).reindex(x.index)
    fit = labels.ne("") & meta.participant_id.isin(fit_people)
    held = labels.ne("") & meta.participant_id.isin(held_people)
    y, actual = labels[fit], labels[held]
    coverage.append(dict(protocol=protocol, fold=fold, fit_people=len(fit_people),
                         held_people=len(held_people), fit_episodes=len(y), held_episodes=len(actual),
                         held_sensor_episodes=int(meta.participant_id.isin(held_people).sum()),
                         fit_classes="|".join(sorted(y.unique()))))
    if actual.empty:
        return []
    if set(y.unique()) != set(THREE):
        raise ValueError(f"{fold}: kelas fitting tidak lengkap.")
    relative = x - refs.fillna(x.loc[y.index].median())
    people = meta.participant_id
    result = []

    def record(name, task, pred):
        rows = actual.to_frame("original").assign(participant_id=people, predicted=pred)
        rows["actual"] = actual if task == "three" else binary(actual)
        result.append(rows.assign(protocol=protocol, fold=fold, task=task, model=name).reset_index())

    record("majority", "three", y.value_counts().reindex(THREE).idxmax())
    two = binary(y)
    record("majority", "binary", two.value_counts().reindex(TWO).idxmax())
    gates = {}
    for name, view in [("eye", x[EYE]), ("relative_core_head", relative)]:
        pred = classify(view, y, people, actual.index)
        record(name, "three", pred)
        record(name + "_mapped", "binary", binary(pred))
        gates[name] = classify(view, two, people, actual.index)
        record(name + "_direct", "binary", gates[name])
    # Fixed cascade: eye gate for High, relative sensors discriminate Low/Medium.
    lower = classify(relative, y[y.ne("High")], people, actual.index)
    record("eye_gate_relative_low_medium", "three", lower.where(gates["eye"].ne("High"), "High"))
    return result


def run(dataset_root, output_root):
    official, _, hashes = read_bundle(dataset_root, "v3")
    ev = pd.read_csv(dataset_root / "v3_behavioral_proxy/episode_labels.csv",
                     keep_default_na=False, float_precision="round_trip")
    ev["intervals"] = ev.intervals.map(json.loads)
    records = ev.to_dict("records")
    people = pd.read_csv(dataset_root / "v3_behavioral_proxy/participant_splits.csv")
    train = set(people.loc[people.split.eq("Train"), "participant_id"])
    validation = set(people.loc[people.split.eq("validation"), "participant_id"])
    relabeled = fold_labels(records, train)
    if not ev.episode_id.map(relabeled).eq(ev.cognitive_load_label).all():
        raise ValueError("Rekonstruksi label full Train berbeda dari paket resmi.")
    bank = dataset_root / "v2_session_phase/window_15s_step_5s/good_acceptable.csv"
    digest = snapshot_hash(bank, "39e72fd8504bb57d83e2edd91a371e904ba6203ad5a3df90fa560a3eaa715722")
    all_windows = bank.with_name("all_data.csv")
    all_digest = snapshot_hash(all_windows, "acbef9a60f4114de460ee3c57ca94ff5cae0d48b67229d8b14283d173fef1355")
    x, refs, meta = build_pool(bank, all_windows, records, official)
    coverage, parts = [], []
    for person in sorted(train):
        parts.extend(run_fold(x, refs, meta, records, "Train_LOPO", person,
                              train - {person}, {person}, coverage))
    parts.extend(run_fold(x, refs, meta, records, "validation", "validation", train, validation, coverage))
    predictions = pd.concat(parts, ignore_index=True)
    results, confusions = [], []
    for (protocol, task, name), group in predictions.groupby(["protocol", "task", "model"]):
        labels = THREE if task == "three" else TWO
        assert group.episode_id.is_unique
        cm = confusion_matrix(group.actual, group.predicted, labels=labels)
        info = dict(protocol=protocol, task=task, model=name)
        results.append({**info, "episodes": len(group), "participants": group.participant_id.nunique(),
                        "correct": int(cm.trace()), "accuracy": cm.trace()/len(group),
                        "macro_f1": f1_score(group.actual, group.predicted, labels=labels, average="macro", zero_division=0),
                        "high_tp": int(cm[-1,-1]), "high_fn": int(cm[-1,:-1].sum()),
                        "high_fp": int(cm[:-1,-1].sum())})
        for original, rows in group.groupby("original"):
            confusions.append({**info, "original": original, **rows.predicted.value_counts().to_dict()})
    out = output_root / ("v3_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(out / "episode_predictions.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "fold_coverage.csv", index=False)
    pd.DataFrame(confusions).fillna(0).to_csv(out / "confusion_by_original_label.csv", index=False)
    summary = pd.DataFrame(results)
    summary.to_csv(out / "metrics.csv", index=False)
    distribution = []
    for unit, frame in [("window", official), ("episode", official.drop_duplicates("episode_id"))]:
        distribution.append(frame.groupby(["split", "actual"]).size().rename("count").reset_index().assign(unit=unit))
    pd.concat(distribution).to_csv(out / "distribution.csv", index=False)
    save_json(out / "manifest.json", {"v3": hashes, "v2_history_sha256": digest, "v2_all_sha256": all_digest, "sklearn": sklearn.__version__,
              "pandas": pd.__version__, "numpy": np.__version__, "target": {"Low": "Non-High", "Medium": "Non-High", "High": "High"}})
    print(summary.to_string(index=False))
    archive = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    print("ZIP:", archive)
    return Path(archive)
