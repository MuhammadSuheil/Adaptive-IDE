import hashlib
import importlib.metadata
import io
import json
import os
import platform
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "batch05-stage1-v1"
LABELS = ("Low", "Medium", "High")
CORE = [
    "rmssd_ms",
    "sdnn_ms",
    "blink_rate_per_minute",
    "ear_left_mean",
    "ear_left_std",
    "ear_right_mean",
    "ear_right_std",
    "iris_size_estimate_mean",
    "iris_size_estimate_std",
    "iris_size_delta_mean",
]
HEAD = ["head_pitch_mean", "head_pitch_std", "head_yaw_mean", "head_yaw_std"]
BLOCK_SECONDS = 240
GUARD_SECONDS = 45
PHASES = {
    "active_task",
    "mixed",
    "feedback",
    "rating",
    "rest",
    "pre_task",
    "post_task",
    "interrupted",
    "practice",
}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def validate_frame(frame, window):
    required = CORE + [
        "window_id",
        "session_id",
        "participant_id",
        "window_start",
        "window_end",
        "window_seconds",
        "cognitive_load_label",
        "study_phase",
    ]
    missing = sorted(set(required) - set(frame))
    if missing or frame.empty:
        raise ValueError(f"Window {window}: data kosong atau kolom hilang: {missing}")
    d = frame.copy(deep=True)
    for key in ("window_id", "session_id", "participant_id", "study_phase"):
        if d[key].isna().any() or d[key].astype(str).str.strip().eq("").any():
            raise ValueError(f"Nilai kosong: {key}")
        d[key] = d[key].astype(str)
    if not d.cognitive_load_label.isin(LABELS).all():
        raise ValueError("Label harus Low, Medium, High pada cognitive_load_label.")
    if not d.study_phase.isin(PHASES).all():
        raise ValueError("Fase tidak dikenal; periksa protokol dataset.")
    if "quality_tier" in d and not d.quality_tier.isin(("good", "acceptable")).all():
        raise ValueError("Dataset harus view good_acceptable.")
    if (
        "hrv_reason" in d
        and d.hrv_reason.fillna("")
        .str.contains(r"(?:^|;)rr_local_median_jump(?:;|$)", regex=True)
        .any()
    ):
        raise ValueError("Window lonjakan RR masih masuk view kurasi.")
    for source, target in (("window_start", "_start"), ("window_end", "_end")):
        try:
            if (
                not d[source]
                .map(lambda value: pd.Timestamp(value).tzinfo is not None)
                .all()
            ):
                raise ValueError("Timestamp harus memiliki zona waktu.")
            parsed = pd.to_datetime(
                d[source], utc=True, format="ISO8601", errors="raise"
            )
            if parsed.isna().any():
                raise ValueError("Timestamp kosong.")
            d[target] = parsed.dt.as_unit("ns").astype("int64")
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"Timestamp tidak valid: {source}") from exc
    if (
        not np.allclose((d._end - d._start) / 1e9, window, atol=1e-6, rtol=0)
        or not pd.to_numeric(d.window_seconds, errors="raise").eq(window).all()
    ):
        raise ValueError("Durasi window tidak sesuai folder.")
    if d.window_id.duplicated().any() or d.duplicated(["session_id", "_end"]).any():
        raise ValueError("Window ID atau endpoint duplikat.")
    d[CORE] = d[CORE].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(d[CORE].to_numpy(float)).all():
        raise ValueError("Fitur core missing atau nonfinite; dataset tidak diimputasi.")
    positive = [c for c in CORE if c.endswith("_std")] + [
        "rmssd_ms",
        "sdnn_ms",
        "blink_rate_per_minute",
    ]
    if d[positive].lt(0).any().any() or d.iris_size_estimate_mean.le(0).any():
        raise ValueError("Nilai sensor mentah di luar rentang fisik.")
    if (
        not d[["ear_left_mean", "ear_right_mean"]]
        .apply(lambda column: column.between(0, 1))
        .all()
        .all()
    ):
        raise ValueError("EAR berada di luar rentang 0 sampai 1.")
    d["_y"] = d.cognitive_load_label.map(dict(zip(LABELS, range(3)))).astype(int)
    d["_key"] = d.session_id + "|" + d._end.astype(str)
    return d.sort_values(["session_id", "_end"]).reset_index(drop=True)


def build_union(frames):
    if set(frames) != {15, 25, 45}:
        raise ValueError("Ketiga dataset 15, 25, 45 detik diperlukan.")
    views = {w: validate_frame(frame, w) for w, frame in frames.items()}
    combined = pd.concat(
        [d.assign(_window=w) for w, d in views.items()], ignore_index=True
    )
    if (
        combined.groupby("_key")._y.nunique().gt(1).any()
        or combined.groupby("session_id").participant_id.nunique().gt(1).any()
    ):
        raise ValueError("Identitas atau label endpoint antarwindow bertentangan.")
    union = (
        combined.sort_values(["session_id", "_end", "_window"])
        .drop_duplicates("_key", keep="last")
        .reset_index(drop=True)
    )
    anchors = combined.groupby("session_id")._start.min()
    slots = (
        (union._end - union.session_id.map(anchors)) // (BLOCK_SECONDS * 10**9)
    ).astype(int)
    union["_block"] = pd.factorize(
        pd.MultiIndex.from_arrays([union.session_id, slots]), sort=True
    )[0]
    lookup = union.set_index("_key")._block
    for frame in views.values():
        frame["_block"] = frame._key.map(lookup).astype(int)
    return views, union


def purge(frame, candidates, held, guard=GUARD_SECONDS):
    candidates, held = np.asarray(candidates, dtype=int), np.asarray(held, dtype=int)
    if not len(held) or not len(candidates):
        return np.sort(candidates)
    result = []
    held_frame = frame.iloc[held]
    for sid, group in frame.iloc[candidates].groupby("session_id", sort=False):
        times = np.sort(
            held_frame.loc[held_frame.session_id.eq(sid), "_end"].to_numpy()
        )
        if not len(times):
            result.extend(group.index)
            continue
        ends = group._end.to_numpy()
        positions = np.searchsorted(times, ends)
        left = times[np.maximum(positions - 1, 0)]
        right = times[np.minimum(positions, len(times) - 1)]
        good = (np.abs(ends - left) >= guard * 10**9) & (
            np.abs(ends - right) >= guard * 10**9
        )
        result.extend(group.index[good])
    return np.array(sorted(result), dtype=int)


def has_classes(frame, ids):
    return set(frame.iloc[np.asarray(ids, int)]._y) == {0, 1, 2}


def metadata(frame):
    # Only identifiers, target, phase, and time enter split selection.
    return frame[["_key", "_block", "session_id", "_end", "_y", "study_phase"]].copy()


def source_deficit(frame, fitting, held):
    fit_keys = set(zip(frame.iloc[fitting].session_id, frame.iloc[fitting]._y))
    missing = [
        (sid, int(y)) not in fit_keys
        for sid, y in frame.iloc[held][["session_id", "_y"]].itertuples(
            index=False, name=None
        )
    ]
    return float(np.mean(missing)) if len(missing) else 1.0


def distribution_distance(frame, left, right, column, only_low=False):
    a, b = frame.iloc[left], frame.iloc[right]
    if only_low:
        a, b = a[a._y.eq(0)], b[b._y.eq(0)]
    if a.empty or b.empty:
        return 1.0
    keys = pd.Index(sorted(set(a[column]) | set(b[column])))
    return float(
        (
            a[column].value_counts(normalize=True).reindex(keys, fill_value=0)
            - b[column].value_counts(normalize=True).reindex(keys, fill_value=0)
        )
        .abs()
        .sum()
    )


def randomized_blocks(frame, ids, seed, fraction=None, folds=None):
    source = frame.iloc[ids]
    majority = source.groupby("_block", sort=True)._y.agg(
        lambda values: int(np.bincount(values, minlength=3).argmax())
    )
    rng = np.random.default_rng(seed)
    assigned = {}
    for label in range(3):
        blocks = majority[majority.eq(label)].index.to_numpy().copy()
        rng.shuffle(blocks)
        if folds is not None:
            offset = int(rng.integers(folds))
            assigned.update(
                {int(b): (i + offset) % folds for i, b in enumerate(blocks)}
            )
        else:
            count = max(1, round(len(blocks) * fraction)) if len(blocks) else 0
            assigned.update({int(b): int(i < count) for i, b in enumerate(blocks)})
    return source._block.map(assigned).to_numpy(int)


def development_split(views, union, reference, seeds=range(42, 170)):
    data = metadata(union)
    frozen = {
        str(r["session_id"]) + "|" + str(r["end_ns"])
        for r in reference["rows"]
        if r["_split"] == "test"
    }
    if frozen - set(data._key):
        raise ValueError(
            "Endpoint Test referensi hilang; jangan mengganti Test secara otomatis."
        )
    test = np.flatnonzero(data._key.isin(frozen))
    eligible = purge(data, np.flatnonzero(~data._key.isin(frozen)), test)
    view_positions = {
        w: data.index[data._key.isin(frame._key)].to_numpy()
        for w, frame in views.items()
    }
    best = None
    for seed in seeds:
        assignments = randomized_blocks(data, eligible, seed, fraction=0.2)
        validation = eligible[assignments == 1]
        train = purge(data, eligible[assignments == 0], validation)
        if not all(
            has_classes(data, np.intersect1d(ids, pos))
            for pos in view_positions.values()
            for ids in (train, validation)
        ):
            continue
        coverage = source_deficit(data, train, validation) + source_deficit(
            data, train, test
        )
        distance = distribution_distance(data, train, validation, "_y")
        distance += distribution_distance(
            data, train, validation, "study_phase", only_low=True
        )
        removed = len(eligible) - len(train) - len(validation)
        size_error = abs(len(validation) / (len(train) + len(validation)) - 0.2)
        cost = (coverage, distance, removed / max(1, len(eligible)) + size_error, seed)
        if best is None or cost < best[0]:
            best = (cost, seed, train, validation)
    if best is None:
        raise ValueError(
            "Split development tidak memiliki tiga kelas pada setiap window."
        )
    out = union.copy()
    out["_split"] = "purged"
    out.loc[test, "_split"] = "test"
    out.loc[best[2], "_split"] = "train"
    out.loc[best[3], "_split"] = "validation"
    mapping = out.set_index("_key")._split
    projected = {
        w: frame.assign(_split=frame._key.map(mapping)) for w, frame in views.items()
    }
    return {
        "union": out,
        "views": projected,
        "seed": int(best[1]),
        "rank": list(best[0]),
        "eligible_before_validation_purge": len(eligible),
    }


def development_cv(views, union, seeds=range(42, 74)):
    data = metadata(union)
    outer_train = np.flatnonzero(union._split.eq("train"))
    positions = {
        w: data.index[data._key.isin(frame._key)].to_numpy()
        for w, frame in views.items()
    }
    best = None
    for seed in seeds:
        assignment = randomized_blocks(data, outer_train, seed, folds=5)
        folds, deficits, distances, removed = [], [], [], 0
        for number in range(5):
            validation = outer_train[assignment == number]
            fit = purge(data, outer_train[assignment != number], validation)
            if not all(
                has_classes(data, np.intersect1d(ids, pos))
                for pos in positions.values()
                for ids in (fit, validation)
            ):
                break
            deficits.append(source_deficit(data, fit, validation))
            distances.append(
                distribution_distance(data, fit, validation, "_y")
                + distribution_distance(data, fit, validation, "study_phase", True)
            )
            removed += len(outer_train) - len(fit) - len(validation)
            folds.append(
                {"fold": number, "fit": fit.tolist(), "validation": validation.tolist()}
            )
        if len(folds) != 5:
            continue
        cost = (float(np.mean(deficits)), float(np.mean(distances)), removed, seed)
        if best is None or cost < best[0]:
            best = (cost, seed, folds)
    if best is None:
        return {
            "ready": False,
            "reason": "Lima fold dengan tiga kelas tidak tersedia.",
            "folds": [],
        }
    return {
        "ready": True,
        "seed": int(best[1]),
        "rank": list(best[0]),
        "folds": best[2],
    }


def choose_monitor(frame, ids, seeds=range(310, 438)):
    data = metadata(frame)
    ids = np.asarray(ids, dtype=int)
    best = None
    for seed in seeds:
        assignment = randomized_blocks(data, ids, seed, fraction=0.15)
        monitor = ids[assignment == 1]
        fitting = purge(data, ids[assignment == 0], monitor)
        counts = [
            np.bincount(data.iloc[part]._y, minlength=3) for part in (fitting, monitor)
        ]
        if any((values < 10).any() for values in counts):
            continue
        gap = float(
            np.max(np.abs(counts[0] / counts[0].sum() - counts[1] / counts[1].sum()))
        )
        if gap > 0.10 + 1e-12:
            continue
        phase = distribution_distance(
            data, fitting, monitor, "study_phase", only_low=True
        )
        source = distribution_distance(data, fitting, monitor, "session_id")
        cost = (phase + source, len(ids) - len(fitting) - len(monitor), seed)
        if best is None or cost < best[0]:
            best = (cost, seed, fitting, monitor, gap)
    if best is None:
        return {
            "ready": False,
            "reason": "Tidak ada monitor dengan >=10 baris/kelas dan gap <=10 poin persentase.",
            "fit": [],
            "monitor": [],
        }
    return {
        "ready": True,
        "seed": int(best[1]),
        "fit": best[2].tolist(),
        "monitor": best[3].tolist(),
        "max_class_gap": best[4],
        "rank": list(best[0]),
    }


class SensorPreprocessor:
    """Statistics from fitting rows only, including per-session centering."""

    def __init__(self, columns, mode="global"):
        if mode not in ("global", "session_centered", "session_robust"):
            raise ValueError("Normalisasi tidak dikenal.")
        if not columns or not set(columns).issubset(CORE + HEAD):
            raise ValueError("Gunakan daftar eksplisit fitur sensor core/head.")
        self.columns, self.mode = list(columns), mode

    def fit(self, frame, ids):
        ids = np.asarray(ids, dtype=int)
        if (
            not len(ids)
            or len(np.unique(ids)) != len(ids)
            or (ids < 0).any()
            or (ids >= len(frame)).any()
        ):
            raise ValueError("ID fitting kosong, duplikat, atau di luar rentang.")
        fitting = frame.iloc[ids]
        x = fitting[self.columns].to_numpy(float)
        if not np.isfinite(x).all():
            raise ValueError("Fitur fitting nonfinite.")
        self.center = np.median(x, axis=0)
        raw_scale = np.quantile(x, 0.75, axis=0) - np.quantile(x, 0.25, axis=0)
        self.scale = np.where(raw_scale > 1e-8, raw_scale, 1.0)
        self.session_stats = {}
        for sid in sorted(fitting.session_id.unique()):
            subset = x[fitting.session_id.eq(sid).to_numpy()]
            scale = np.quantile(subset, 0.75, axis=0) - np.quantile(
                subset, 0.25, axis=0
            )
            self.session_stats[str(sid)] = (
                np.median(subset, axis=0),
                np.where(scale > 1e-8, scale, self.scale),
            )
        return self

    def transform(self, frame):
        x = frame[self.columns].to_numpy(float).copy()
        if not np.isfinite(x).all():
            raise ValueError("Fitur transform nonfinite; tidak ada imputasi diam-diam.")
        for sid in frame.session_id.unique():
            mask = frame.session_id.eq(sid).to_numpy()
            center, scale = self.center, self.scale
            if self.mode != "global":
                center, local_scale = self.session_stats.get(
                    str(sid), (self.center, self.scale)
                )
                if self.mode == "session_robust":
                    scale = local_scale
            x[mask] = (x[mask] - center) / scale
        if not np.isfinite(x).all():
            raise ValueError("Hasil normalisasi nonfinite.")
        return x

    def state(self):
        return {
            "columns": self.columns,
            "mode": self.mode,
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "session_stats": {
                sid: [center.tolist(), scale.tolist()]
                for sid, (center, scale) in self.session_stats.items()
            },
        }


def select_candidate(candidates):
    eligible = []
    for candidate in candidates:
        recall = np.asarray(candidate["recall"], float)
        numeric = [
            candidate["macro_f1"],
            candidate["accuracy"],
            candidate["fold_std"],
            candidate["complexity"],
            *recall,
        ]
        if (
            recall.shape != (3,)
            or not np.isfinite(numeric).all()
            or (recall <= 0).any()
            or (recall > 1).any()
            or not 0 <= candidate["macro_f1"] <= 1
            or not 0 <= candidate["accuracy"] <= 1
            or candidate.get("finite_predictions", False) is not True
        ):
            continue
        eligible.append(candidate)
    return (
        min(
            eligible,
            key=lambda c: (
                -c["macro_f1"],
                -c["accuracy"],
                -min(c["recall"]),
                c["fold_std"],
                c["complexity"],
                c["name"],
            ),
        )
        if eligible
        else None
    )


def partition_counts(frame, assignments, context):
    records = []
    for part in sorted(set(assignments)):
        rows = frame.loc[np.asarray(assignments) == part]
        counts = rows.cognitive_load_label.value_counts()
        records.append(
            {
                "context": context,
                "partition": part,
                "rows": len(rows),
                "blocks": int(rows._block.nunique()) if "_block" in rows else 0,
                **{label: int(counts.get(label, 0)) for label in LABELS},
            }
        )
    return records


def support_table(frame):
    records = []
    for (sid, y), rows in frame.groupby(["session_id", "_y"], sort=True):
        entry = {
            "session_id": sid,
            "label": LABELS[y],
            "rows": len(rows),
            "blocks": int(rows._block.nunique()),
        }
        for part in ("train", "validation", "test", "purged"):
            subset = rows[rows._split.eq(part)]
            entry[part + "_rows"] = len(subset)
            entry[part + "_blocks"] = int(subset._block.nunique())
        entry["held_without_train"] = (
            entry["validation_rows"] + entry["test_rows"] > 0
            and entry["train_rows"] == 0
        )
        entry["short_cell"] = entry["blocks"] < 3
        records.append(entry)
    return pd.DataFrame(records)


def project_fold(frame, union, fold):
    fit_keys = set(union.iloc[fold["fit"]]._key)
    val_keys = set(union.iloc[fold["validation"]]._key)
    return (
        np.flatnonzero(frame._key.isin(fit_keys)),
        np.flatnonzero(frame._key.isin(val_keys)),
    )


def check_disjoint(frame, fitting, held):
    fitting, held = np.asarray(fitting, int), np.asarray(held, int)
    return not len(np.intersect1d(fitting, held)) and np.array_equal(
        purge(frame, fitting, held), np.sort(fitting)
    )


def preprocessing_check(frame, fitting, held, columns=CORE):
    if not len(fitting):
        return False
    for mode in ("global", "session_centered", "session_robust"):
        original = SensorPreprocessor(columns, mode).fit(frame, fitting)
        altered = frame.copy()
        altered.loc[held, columns] = 1e9
        changed = SensorPreprocessor(columns, mode).fit(altered, fitting)
        if original.state() != changed.state():
            return False
        if not np.isfinite(original.transform(frame.iloc[fitting])).all():
            return False
    return True


def preflight(
    frames,
    reference,
    split_seeds=range(42, 170),
    cv_seeds=range(42, 74),
    progress=print,
):
    views, union = build_union(frames)
    progress("Memilih split development dari metadata.")
    development = development_split(views, union, reference, split_seeds)
    views, union = development["views"], development["union"]
    progress("Memeriksa lima fold development.")
    cv = development_cv(views, union, cv_seeds)
    counts, monitors, readiness, details = [], {}, [], {}
    reference_lookup = {
        str(r["session_id"]) + "|" + str(r["end_ns"]): r["_split"]
        for r in reference["rows"]
    }
    for window, frame in sorted(views.items()):
        progress(f"Window {window}: monitor dan pemeriksaan preprocessing.")
        train = np.flatnonzero(frame._split.eq("train"))
        validation = np.flatnonzero(frame._split.eq("validation"))
        test = np.flatnonzero(frame._split.eq("test"))
        contexts = [("final", train)]
        for fold in cv["folds"]:
            fit, val = project_fold(frame, union, fold)
            assignments = np.full(len(frame), "outside_train", dtype=object)
            assignments[train] = "purged"
            assignments[fit], assignments[val] = "fit", "validation"
            counts.extend(
                dict(window=window, **r)
                for r in partition_counts(frame, assignments, f"cv_{fold['fold']}")
            )
            if not check_disjoint(frame, fit, val):
                raise ValueError("Pemisahan fold overlap.")
            contexts.append((f"cv_{fold['fold']}", fit))
        monitors[str(window)] = {}
        for name, ids in contexts:
            progress(f"  {window}s/{name}: pemilihan monitor.")
            monitor = choose_monitor(frame, ids)
            fit_ids, monitor_ids = monitor.pop("fit"), monitor.pop("monitor")
            if monitor["ready"]:
                if not check_disjoint(frame, fit_ids, monitor_ids):
                    raise ValueError("Monitor overlap dengan fitting.")
                if not set(fit_ids + monitor_ids).issubset(set(ids)):
                    raise ValueError("Monitor keluar dari fitting pool.")
            assignments = np.full(len(frame), "unused", dtype=object)
            assignments[ids] = "purged"
            assignments[fit_ids], assignments[monitor_ids] = "fit", "monitor"
            counts.extend(
                dict(window=window, **r)
                for r in partition_counts(frame, assignments, f"monitor_{name}")
            )
            monitor.update(
                fit_window_ids=frame.iloc[fit_ids].window_id.tolist(),
                monitor_window_ids=frame.iloc[monitor_ids].window_id.tolist(),
            )
            monitor["composition"] = (
                frame.loc[np.isin(np.arange(len(frame)), fit_ids + monitor_ids)]
                .assign(
                    role=assignments[
                        np.isin(np.arange(len(frame)), fit_ids + monitor_ids)
                    ]
                )
                .groupby(["role", "cognitive_load_label", "study_phase", "session_id"])
                .size()
                .reset_index(name="rows")
                .to_dict("records")
            )
            monitors[str(window)][name] = monitor
        no_overlap = check_disjoint(
            frame, train, np.r_[validation, test]
        ) and check_disjoint(frame, validation, test)
        reference_parts = frame._key.map(reference_lookup).fillna("absent")
        if not np.array_equal(reference_parts.eq("test"), frame._split.eq("test")):
            raise ValueError("Test berubah dari referensi.")
        transform_ok = preprocessing_check(frame, train, np.r_[validation, test])
        head_ready = set(HEAD).issubset(frame) and bool(
            np.isfinite(
                frame[HEAD].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            ).all()
        )
        if head_ready:
            head_ready = bool(
                frame[["head_pitch_std", "head_yaw_std"]].ge(0).all().all()
            )
        tabular = bool(
            cv["ready"]
            and no_overlap
            and transform_ok
            and all(has_classes(frame, part) for part in (train, validation, test))
        )
        neural = tabular and all(m["ready"] for m in monitors[str(window)].values())
        readiness.append(
            {
                "window": window,
                "tabular_ready": tabular,
                "neural_ready": neural,
                "core_head_available": head_ready,
                "no_overlap": no_overlap,
                "preprocessing_train_only": transform_ok,
                "test_unchanged": True,
            }
        )
        counts.extend(
            dict(window=window, **r)
            for r in partition_counts(frame, frame._split, "outer")
        )
        support = support_table(frame)
        details[str(window)] = {
            "source_support": support.to_dict("records"),
            "validation_without_train_fraction": source_deficit(
                frame, train, validation
            ),
            "test_without_train_fraction": source_deficit(frame, train, test),
            "changed_from_reference_rows": int(reference_parts.ne(frame._split).sum()),
            "head_profile_reason": "complete"
            if head_ready
            else "missing/nonfinite/negative head std",
            "constant_core": [c for c in CORE if frame[c].nunique() <= 1],
            "duplicate_core_vectors": int(frame.duplicated(CORE).sum()),
        }
    return {
        "views": views,
        "union": union,
        "development": development,
        "cv": cv,
        "counts": pd.DataFrame(counts),
        "monitors": monitors,
        "readiness": readiness,
        "details": details,
    }


def manifest_records(result):
    records = (
        result["union"][["_key", "session_id", "_end", "_block", "_split"]]
        .rename(
            columns={
                "_key": "endpoint_key",
                "_end": "end_ns",
                "_block": "block",
                "_split": "split",
            }
        )
        .to_dict("records")
    )
    cv = result["cv"].copy()
    cv["folds"] = [
        {
            "fold": f["fold"],
            "fit_endpoint_keys": result["union"].iloc[f["fit"]]._key.tolist(),
            "validation_endpoint_keys": result["union"]
            .iloc[f["validation"]]
            ._key.tolist(),
        }
        for f in cv["folds"]
    ]
    return {
        "block_seconds": BLOCK_SECONDS,
        "guard_seconds": GUARD_SECONDS,
        "history_seconds": 0,
        "seed": result["development"]["seed"],
        "selection": "metadata_only",
        "rank": result["development"]["rank"],
        "rows": records,
        "cv": cv,
        "window_ids": {
            str(w): frame[["window_id", "_key", "_split"]]
            .rename(columns={"_key": "endpoint_key", "_split": "split"})
            .to_dict("records")
            for w, frame in result["views"].items()
        },
    }


def report_text(result, hashes):
    lines = [
        "# Persiapan batch-05",
        "",
        "Tahap 1 memeriksa dataset dan rancangan evaluasi tanpa melatih classifier.",
        "",
        "## Kesiapan",
        "",
        "| Window | Tabular | Neural | Core + head |",
        "|---|---|---|---|",
    ]
    for r in result["readiness"]:
        lines.append(
            f"| {r['window']}s | {r['tabular_ready']} | {r['neural_ready']} | {r['core_head_available']} |"
        )
    lines += [
        "",
        "False berarti tahap terkait belum boleh dijalankan. Ambang monitor tidak dilonggarkan.",
        "",
        "## Pembagian utama",
        "",
        "| Window | Bagian | Baris | Low | Medium | High |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in result["counts"].query("context == 'outer'").to_dict("records"):
        lines.append(
            f"| {r['window']}s | {r['partition']} | {r['rows']} | {r['Low']} | {r['Medium']} | {r['High']} |"
        )
    lines += [
        "",
        "Split development memakai blok 240 detik dan guard 45 detik, tanpa history. Test mempertahankan ID batch-04.",
        f"Seed development: {result['development']['seed']}. Seed CV: {result['cv'].get('seed', 'tidak tersedia')}.",
        "",
        "## Dukungan sumber dan perubahan referensi",
        "",
        "| Window | Val tanpa sumber Train | Test tanpa sumber Train | Sel sesi-kelas <3 blok | Baris berganti bagian |",
        "|---|---:|---:|---:|---:|",
    ]
    for w, d in result["details"].items():
        lines.append(
            f"| {w}s | {d['validation_without_train_fraction']:.2%} | {d['test_without_train_fraction']:.2%} | "
            f"{sum(r['short_cell'] for r in d['source_support'])} | {d['changed_from_reference_rows']} |"
        )
    lines += [
        "",
        "Sel pendek tidak dipaksa muncul di semua bagian. source_support.csv memuat jumlah baris dan blok tiap sesi-kelas.",
        "Perubahan split, guard, dan normalisasi harus diuji terpisah pada tahap fitting. Test lama merupakan referensi eksploratif yang sudah pernah dianalisis.",
        "",
        "## Monitor neural",
        "",
        "| Window | Konteks | Siap | Gap kelas maksimum | Alasan |",
        "|---|---|---|---|---|",
    ]
    for w, contexts in result["monitors"].items():
        for name, m in contexts.items():
            gap = f"{m['max_class_gap']:.2%}" if m["ready"] else "-"
            lines.append(
                f"| {w}s | {name} | {m['ready']} | {gap} | {m.get('reason', '')} |"
            )
    lines += [
        "",
        "Monitor memerlukan minimal 10 baris tiap kelas pada fitting dan monitor, serta gap proporsi maksimum 10 poin persentase.",
        "Jumlah window overlap bukan jumlah observasi independen. Ambang tersebut memeriksa operasi pipeline, bukan kecukupan sampel penelitian.",
        "Komposisi fase, sumber, dan kelas serta ID monitor tersimpan pada monitors.json. Monitor hanya memakai fitting pool masing-masing.",
        "",
        "## Preprocessing dan seleksi",
        "",
        "Statistik global dan per sesi dihitung dari fitting saja. Sesi baru memakai statistik global. Nilai terpusat yang negatif tetap dipertahankan.",
        "Kelayakan kandidat memerlukan prediksi finite dan recall positif pada tiga kelas development. Urutan peringkat: Macro-F1, accuracy, recall minimum, kestabilan fold, kompleksitas.",
        "Target 70% belum diuji pada tahap ini.",
        "",
        "## SHA-256 dataset",
        "",
    ]
    lines.extend(f"- {w}s: `{value}`" for w, value in hashes.items())
    return "\n".join(lines) + "\n"


def run_preparation(
    dataset_root, output_root, reference, expected_hashes, implementation_sha256
):
    frames, hashes = {}, {}
    for window in (15, 25, 45):
        path = Path(dataset_root) / f"window_{window}s_step_5s/good_acceptable.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
        payload = path.read_bytes()
        hashes[str(window)] = hashlib.sha256(payload).hexdigest()
        frames[window] = pd.read_csv(io.BytesIO(payload))
    if hashes != expected_hashes:
        raise ValueError(
            "Hash dataset berbeda dari versi yang dibekukan. Audit ulang referensi sebelum membuat split baru."
        )
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "version": VERSION,
        "implementation_sha256": implementation_sha256,
        "stage": 1,
        "dataset_root": str(dataset_root),
        "dataset_sha256": hashes,
        "block_seconds": BLOCK_SECONDS,
        "guard_seconds": GUARD_SECONDS,
        "history_seconds": 0,
        "features": {"default": CORE, "optional": CORE + HEAD},
        "split_seeds": [42, 169],
        "cv_seeds": [42, 73],
        "monitor_seeds": [310, 437],
        "monitor_min_class_rows": 10,
        "monitor_max_class_gap": 0.1,
        "python": platform.python_version(),
        "environment": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "pandas")
        },
    }
    save_json(run_dir / "run.json", config)
    save_json(run_dir / "reference_manifest.json", reference)
    try:
        result = preflight(frames, reference)
        manifest = manifest_records(result)
        save_json(run_dir / "development_manifest.json", manifest)
        save_json(run_dir / "monitors.json", result["monitors"])
        result["counts"].to_csv(run_dir / "partition_counts.csv", index=False)
        support = pd.concat(
            [
                pd.DataFrame(d["source_support"]).assign(window=int(w))
                for w, d in result["details"].items()
            ],
            ignore_index=True,
        )
        support.to_csv(run_dir / "source_support.csv", index=False)
        gate = next(r for r in result["readiness"] if r["window"] == 25)
        status = {
            "stage": 1,
            "training_executed": False,
            "manifest_sha256": digest(manifest),
            "window_readiness": result["readiness"],
            "details": result["details"],
            "stage2_ready": bool(gate["tabular_ready"] and gate["neural_ready"]),
            "test_role": "previously_seen_exploratory_reference",
        }
        save_json(run_dir / "readiness.json", status)
        (run_dir / "PREPARATION.md").write_text(
            report_text(result, hashes), encoding="utf-8"
        )
        save_json(
            run_dir / "status.json",
            {"status": "complete", "stage": 1, "training_executed": False},
        )
    except Exception as exc:
        save_json(
            run_dir / "status.json",
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise
    print(f"Laporan persiapan: {run_dir}")
    print("Tahap 2 siap:", status["stage2_ready"])
    return result, run_dir
