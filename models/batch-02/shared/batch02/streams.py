"""Sensor-only causal transforms before supervised endpoint selection."""

import numpy as np
import pandas as pd


def checked_stream(frame):
    result = frame.copy().reset_index(drop=True)
    result["window_end"] = pd.to_datetime(result.window_end, utc=True)
    for _, group in result.groupby("session_id", sort=False):
        gaps = group.window_end.diff().dropna().dt.total_seconds()
        if not ((gaps > 0) & np.isclose(gaps / 5, np.round(gaps / 5))).all():
            raise ValueError(
                "Timestamp stream harus meningkat pada grid 5 detik, tanpa duplikasi."
            )
    if (
        "available_at" in result
        and (pd.to_datetime(result.available_at, utc=True) > result.window_end).any()
    ):
        raise ValueError(
            "Stream mengandung pengukuran yang baru tersedia setelah endpoint."
        )
    return result


def causal_history(frame, columns, seconds):
    result = checked_stream(frame)
    if seconds not in (15, 30):
        raise ValueError("Horizon history harus 15 atau 30 detik.")
    for name in columns:
        means = np.full(len(result), np.nan)
        stds = means.copy()
        deltas = means.copy()
        for _, group in result.groupby("session_id", sort=False):
            times = group.window_end
            blocks = times.diff().dt.total_seconds().gt(7.5).cumsum()
            for _, block in group.groupby(blocks, sort=False):
                series = pd.Series(
                    pd.to_numeric(block[name]).to_numpy(float), index=block.window_end
                )
                means[block.index] = (
                    series.rolling(f"{seconds}s", closed="right", min_periods=1)
                    .mean()
                    .to_numpy()
                )
                stds[block.index] = (
                    series.rolling(f"{seconds}s", closed="right", min_periods=2)
                    .std(ddof=1)
                    .to_numpy()
                )
                deltas[block.index] = series.diff().to_numpy()
        result[f"{name}__history_mean_{seconds}s"] = means
        result[f"{name}__history_std_{seconds}s"] = stds
        result[f"{name}__delta"] = deltas
    return result


def causal_center(frame, columns):
    result = checked_stream(frame)
    for name in columns:
        centered = np.full(len(result), np.nan)
        for _, group in result.groupby("session_id", sort=False):
            blocks = group.window_end.diff().dt.total_seconds().gt(7.5).cumsum()
            for _, block in group.groupby(blocks, sort=False):
                values = pd.Series(
                    pd.to_numeric(block[name]).to_numpy(float), index=block.window_end
                )
                prior = values.rolling("60s", closed="left", min_periods=3).median()
                centered[block.index] = (values - prior).to_numpy()
        result[f"{name}__relative_60s"] = centered
    return result


def nonoverlap_indices(frame):
    keep = []
    for _, group in frame.groupby("session_id", sort=False):
        previous = None
        for index, row in group.sort_values("window_end").iterrows():
            start, end = pd.Timestamp(row.window_start), pd.Timestamp(row.window_end)
            if previous is None or start >= previous:
                keep.append(index)
                previous = end
    return np.asarray(keep, int)


def sequence_indices(stream, endpoints):
    stream = checked_stream(stream)
    lookup = {(r.session_id, r.window_end): i for i, r in stream.iterrows()}
    previous_by_index = {}
    for _, group in stream.groupby("session_id", sort=False):
        indices = group.index.tolist()
        previous_by_index.update(zip(indices[1:], indices[:-1]))
    sequences = []
    for row in endpoints.itertuples():
        end = pd.Timestamp(row.window_end)
        index = lookup[(row.session_id, end)]
        indices = [index]
        while len(indices) < 7 and index in previous_by_index:
            previous_index = previous_by_index[index]
            previous = stream.iloc[previous_index]
            current = stream.iloc[index]
            if (
                previous.session_id != row.session_id
                or (current.window_end - previous.window_end).total_seconds() > 7.5
            ):
                break
            index = previous_index
            indices.append(index)
        sequences.append(list(reversed(indices)))
    return sequences
