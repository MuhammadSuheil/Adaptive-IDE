def validate_frame(frame, window):
    required = set(
        CORE
        + [
            "window_id",
            "session_id",
            "participant_id",
            "window_start",
            "window_end",
            "window_seconds",
            "cognitive_load_label",
            "study_phase",
        ]
    )
    missing = required - set(frame)
    if missing or frame.empty:
        raise ValueError(
            f"Window {window}: kolom core/metadata hilang {sorted(missing)} atau data kosong."
        )
    d = frame.copy()
    if not d.cognitive_load_label.isin(LABELS).all():
        raise ValueError("Label tidak dikenal; jangan memetakan ulang difficulty.")
    if "quality_tier" in d and not d.quality_tier.isin(("good", "acceptable")).all():
        raise ValueError(
            "Gunakan good_acceptable terbaru, bukan all_data atau view kualitas lain."
        )
    if (
        "hrv_reason" in d
        and d.hrv_reason.fillna("")
        .str.split(";")
        .map(lambda tokens: "rr_local_median_jump" in tokens)
        .any()
    ):
        raise ValueError(
            "good_acceptable masih memuat window lonjakan RR yang seharusnya dikeluarkan."
        )
    for key in ("window_id", "session_id", "participant_id"):
        if d[key].isna().any() or d[key].astype(str).str.strip().eq("").any():
            raise ValueError(f"Identitas kosong: {key}")
        d[key] = d[key].astype(str)
    for key in ("window_start", "window_end"):
        if not d[key].map(lambda v: pd.Timestamp(v).tzinfo is not None).all():
            raise ValueError("Timestamp memerlukan zona waktu.")
    d["_start"] = (
        pd.to_datetime(d.window_start, utc=True, format="ISO8601")
        .dt.as_unit("ns")
        .astype("int64")
    )
    d["_end"] = (
        pd.to_datetime(d.window_end, utc=True, format="ISO8601")
        .dt.as_unit("ns")
        .astype("int64")
    )
    if (
        not np.allclose((d._end - d._start) / 1e9, window, rtol=0, atol=1e-6)
        or not d.window_seconds.eq(window).all()
    ):
        raise ValueError("Durasi window tidak sesuai.")
    if d.window_id.duplicated().any() or d.duplicated(["session_id", "_end"]).any():
        raise ValueError("ID/endpoint window duplikat.")
    d[CORE] = d[CORE].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(d[CORE].to_numpy(float)).all():
        raise ValueError("Fitur core missing/nonfinite; periksa versi dataset.")
    positive = [
        "rmssd_ms",
        "sdnn_ms",
        "blink_rate_per_minute",
        "ear_left_std",
        "ear_right_std",
        "iris_size_estimate_std",
    ]
    if d[positive].lt(0).any().any() or d.iris_size_estimate_mean.le(0).any():
        raise ValueError("Nilai core mentah di luar rentang fisik.")
    if (
        not d[["ear_left_mean", "ear_right_mean"]]
        .apply(lambda x: x.between(0, 1))
        .all()
        .all()
    ):
        raise ValueError("EAR di luar rentang.")
    d["_y"] = d.cognitive_load_label.map({v: i for i, v in enumerate(LABELS)}).astype(
        int
    )
    return d.sort_values(["session_id", "_end"]).reset_index(drop=True)


def purge_indices(frame, candidates, held, support=SUPPORT_SECONDS):
    candidates, held = np.asarray(candidates, int), np.asarray(held, int)
    keep = []
    for sid, rows in frame.iloc[candidates].groupby("session_id", sort=True):
        held_frame = frame.iloc[held]
        times = np.sort(
            held_frame.loc[held_frame.session_id.eq(sid), "_end"].to_numpy()
        )
        if not len(times):
            keep.extend(rows.index)
            continue
        ends = rows._end.to_numpy()
        pos = np.searchsorted(times, ends)
        left = times[np.maximum(pos - 1, 0)]
        right = times[np.minimum(pos, len(times) - 1)]
        good = (np.abs(ends - left) >= support * 10**9) & (
            np.abs(ends - right) >= support * 10**9
        )
        keep.extend(rows.index[good])
    return np.array(sorted(keep), int)


def has_classes(frame, ids):
    return set(frame.iloc[ids]._y) == {0, 1, 2}


def split_cost(frame, parts):
    total = np.bincount(frame._y, minlength=3) / len(frame)
    low = frame._y.eq(0)
    phase = np.where(
        frame.study_phase.eq("active_task"),
        "active",
        np.where(frame.study_phase.eq("mixed"), "mixed", "other"),
    )
    source = frame.session_id + "|" + frame._y.astype(str)
    full_source = source.value_counts(normalize=True)
    global_low = pd.Series(phase[low]).value_counts(normalize=True)
    cost = 0.0
    for part in ("train", "validation", "test"):
        ids = np.flatnonzero(parts == part)
        if not has_classes(frame, ids):
            return math.inf
        class_error = np.abs(
            np.bincount(frame.iloc[ids]._y, minlength=3) / len(ids) - total
        ).mean()
        low_ids = ids[low.to_numpy()[ids]]
        phase_error = (
            pd.Series(phase[low_ids])
            .value_counts(normalize=True)
            .reindex(global_low.index, fill_value=0)
            .sub(global_low)
            .abs()
            .mean()
        )
        source_error = (
            source.iloc[ids]
            .value_counts(normalize=True)
            .reindex(full_source.index, fill_value=0)
            .sub(full_source)
            .abs()
            .mean()
        )
        cost += 4 * class_error + 2 * phase_error + source_error
    return float(cost + 0.5 * np.mean(parts == "purged"))


def prepare_data(frames, candidates=128):
    if set(frames) != {15, 25, 45}:
        raise ValueError("Ketiga panjang window harus tersedia.")
    views = {w: validate_frame(d, w) for w, d in frames.items()}
    combined = pd.concat(
        [d.assign(_window=w) for w, d in views.items()], ignore_index=True
    )
    disagreement = combined.groupby(["session_id", "_end"])._y.nunique()
    if (
        disagreement.gt(1).any()
        or combined.groupby("session_id").participant_id.nunique().gt(1).any()
    ):
        raise ValueError("Identitas atau label endpoint antarwindow bertentangan.")
    union = (
        combined.sort_values(["session_id", "_end", "_window"])
        .drop_duplicates(["session_id", "_end"], keep="last")
        .reset_index(drop=True)
    )
    anchors = combined.groupby("session_id")._start.min()
    union["_block"] = pd.factorize(
        pd.MultiIndex.from_arrays(
            [
                union.session_id,
                (
                    (union._end - union.session_id.map(anchors))
                    // (BLOCK_SECONDS * 10**9)
                ).astype(int),
            ]
        ),
        sort=True,
    )[0]
    block_labels = union.groupby("_block", sort=True)._y.agg(
        lambda a: np.bincount(a, minlength=3).argmax()
    )
    best = None
    for seed in range(SEED, SEED + candidates):
        rng = np.random.default_rng(seed)
        assignment = {}
        for label in range(3):
            blocks = block_labels[block_labels.eq(label)].index.to_numpy().copy()
            rng.shuffle(blocks)
            n = max(1, round(0.15 * len(blocks)))
            for j, block in enumerate(blocks):
                assignment[int(block)] = (
                    "test" if j < n else "validation" if j < 2 * n else "train"
                )
        parts = union._block.map(assignment).to_numpy(dtype=object)
        test = np.flatnonzero(parts == "test")
        val = purge_indices(union, np.flatnonzero(parts == "validation"), test)
        train = purge_indices(union, np.flatnonzero(parts == "train"), np.r_[test, val])
        kept = np.full(len(union), "purged", dtype=object)
        kept[test], kept[val], kept[train] = "test", "validation", "train"
        cost = split_cost(union, kept)
        if best is None or cost < best[0]:
            best = (cost, seed, kept.copy())
    if best is None or not np.isfinite(best[0]):
        raise ValueError(
            "Tidak tersedia split dengan ketiga kelas; periksa jumlah blok/kelas."
        )
    union["_split"] = best[2]
    train = np.flatnonzero(union._split.eq("train"))
    assignments = {"_split": union._split.to_numpy()}
    for repeat, seed in enumerate(CV_SEEDS):
        source = union.iloc[train]
        splitter = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for fold, (fit, val) in enumerate(
            splitter.split(np.zeros(len(source)), source._y, source._block)
        ):
            fit, val = train[fit], train[val]
            fit = purge_indices(union, fit, val)
            col = np.full(len(union), "purged", dtype=object)
            col[fit], col[val] = "train", "validation"
            assignments[f"_cv_{repeat}_{fold}"] = col
    lookup = {
        (sid, int(end)): i
        for i, (sid, end) in enumerate(zip(union.session_id, union._end))
    }
    counts = []
    for window, frame in views.items():
        pos = np.array(
            [lookup[(sid, int(end))] for sid, end in zip(frame.session_id, frame._end)]
        )
        frame["_block"] = union._block.to_numpy()[pos]
        for name, values in assignments.items():
            frame[name] = values[pos]
            for part in (
                ("train", "validation")
                if name != "_split"
                else ("train", "validation", "test", "purged")
            ):
                rows = np.flatnonzero(frame[name].eq(part))
                if part != "purged" and not has_classes(frame, rows):
                    raise ValueError(
                        f"Window {window}/{name}/{part}: kelas tidak lengkap."
                    )
                c = frame.iloc[rows].cognitive_load_label.value_counts()
                counts.append(
                    {
                        "window": window,
                        "partition": name,
                        "split": part,
                        "rows": len(rows),
                        **{label: int(c.get(label, 0)) for label in LABELS},
                    }
                )
    manifest = {
        "seed": int(best[1]),
        "selection": "metadata_only",
        "block_seconds": BLOCK_SECONDS,
        "support_seconds": SUPPORT_SECONDS,
        "cv_seeds": list(CV_SEEDS),
        "rows": [
            {
                "session_id": sid,
                "end_ns": int(end),
                **{key: str(value[i]) for key, value in assignments.items()},
            }
            for i, (sid, end) in enumerate(
                union[["session_id", "_end"]].itertuples(index=False, name=None)
            )
        ],
    }
    return views, manifest, pd.DataFrame(counts)


def read_datasets(root):
    frames, hashes = {}, {}
    for window in (15, 25, 45):
        path = Path(root) / f"window_{window}s_step_5s/good_acceptable.csv"
        payload = path.read_bytes()
        frames[window] = pd.read_csv(io.BytesIO(payload))
        hashes[str(window)] = hashlib.sha256(payload).hexdigest()
    views, manifest, counts = prepare_data(frames)
    return views, manifest, counts, hashes


def history_indices(frame, seconds):
    result = [None] * len(frame)
    for _, group in frame.groupby("session_id", sort=False):
        ids = group.sort_values("_end").index.to_numpy()
        times = frame.loc[ids, "_end"].to_numpy()
        first = 0
        for j, i in enumerate(ids):
            if j and times[j] - times[j - 1] != 5 * 10**9:
                first = j
            lo = max(first, int(np.searchsorted(times, times[j] - seconds * 10**9)))
            result[i] = ids[lo : j + 1]
    return result


def monitor_indices(frame, ids):
    ids = np.asarray(ids, int)
    blocks = np.unique(frame.iloc[ids]._block)
    for seed in range(310, 374):
        rng = np.random.default_rng(seed)
        chosen = rng.choice(blocks, max(1, round(0.15 * len(blocks))), replace=False)
        monitor = ids[frame.iloc[ids]._block.isin(chosen).to_numpy()]
        fit = purge_indices(
            frame, ids[~frame.iloc[ids]._block.isin(chosen).to_numpy()], monitor
        )
        if has_classes(frame, fit) and has_classes(frame, monitor):
            return fit, monitor
    raise ValueError(
        "Monitor neural tidak memiliki tiga kelas; periksa dukungan blok training."
    )
