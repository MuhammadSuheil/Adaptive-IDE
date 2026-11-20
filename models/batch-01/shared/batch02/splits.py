"""Grouping and aligned endpoints, never random-row validation."""

import numpy as np
import pandas as pd


def group_splits(frame, seed=42):
    from sklearn.model_selection import StratifiedGroupKFold

    y = frame.y.to_numpy()
    for count in (3, 2):
        if frame.person_id.nunique() < count:
            continue
        splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=seed)
        splits = list(splitter.split(np.zeros((len(frame), 1)), y, frame.person_id))
        if all(len(np.unique(y[train])) == 3 for train, _ in splits):
            return splits
    raise ValueError(
        "Tidak ada inner split grup 3/2-fold dengan tiga kelas pada setiap fitting partition."
    )


def make_manifests(frames, seed=42):
    people = sorted(set().union(*(set(f.person_id) for f in frames.values())))
    result = {"people": people, "seed": seed, "windows": {}}
    for seconds, frame in frames.items():
        folds = []
        for i, person in enumerate(people):
            train = np.flatnonzero(frame.person_id.to_numpy() != person)
            test = np.flatnonzero(frame.person_id.to_numpy() == person)
            if not len(test):
                folds.append(
                    {
                        "id": f"fold_{i:02d}",
                        "person": person,
                        "status": "no_test_support",
                    }
                )
                continue
            inner = [
                {"train": train[a].tolist(), "valid": train[b].tolist()}
                for a, b in group_splits(frame.iloc[train].reset_index(drop=True), seed)
            ]
            folds.append(
                {
                    "id": f"fold_{i:02d}",
                    "person": person,
                    "status": "ready",
                    "train": train.tolist(),
                    "test": test.tolist(),
                    "inner": inner,
                }
            )
        result["windows"][str(seconds)] = {
            "window_ids": frame.window_id.tolist(),
            "folds": folds,
        }
    return result


def paired_endpoints(frames):
    result = None
    for frame in frames.values():
        part = frame[["session_id", "window_end", "y"]].copy()
        part["window_end"] = pd.to_datetime(part.window_end, utc=True)
        result = (
            part
            if result is None
            else result.merge(
                part, on=["session_id", "window_end", "y"], validate="one_to_one"
            )
        )
    return result.sort_values(["session_id", "window_end"]).reset_index(drop=True)


def restrict_paired(frame, paired):
    keys = pd.MultiIndex.from_frame(paired[["session_id", "window_end"]])
    return (
        frame[pd.MultiIndex.from_frame(frame[["session_id", "window_end"]]).isin(keys)]
        .copy()
        .reset_index(drop=True)
    )
