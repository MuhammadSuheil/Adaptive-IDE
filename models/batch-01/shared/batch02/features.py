"""Feature matrices cannot depend on target masks or annotation resets."""

import hashlib

import numpy as np
import pandas as pd

from .artifacts import file_hash, read_json
from .registry import TEMPORAL, feature_group
from .streams import causal_center, causal_history, checked_stream, sequence_indices


def frame_hash(frame):
    h = hashlib.sha256()
    h.update(str(list(frame.columns)).encode())
    h.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return h.hexdigest()


class MissingStream(ValueError):
    pass


class FeatureStore:
    def __init__(self, features, stream_root=None, release_id=None):
        self.features = features
        self.stream_root = stream_root
        self.release_id = release_id
        self.loaded = {}
        self.transformed = {}

    def stream(self, seconds):
        if seconds not in self.loaded:
            if (
                self.stream_root is None
                or not (self.stream_root / "stream_manifest.json").is_file()
            ):
                raise MissingStream(
                    "skipped_missing_stream: cache seluruh grid sensor belum tersedia."
                )
            manifest = read_json(self.stream_root / "stream_manifest.json")
            if manifest["release_id"] != self.release_id:
                raise ValueError("Stream cache berasal dari release berbeda.")
            relative = f"window_{seconds}s_step_5s.csv"
            path = self.stream_root / relative
            if file_hash(path) != manifest["files"][relative]:
                raise ValueError("Hash stream cache berubah.")
            frame = pd.read_csv(path)
            if not set(self.features).issubset(frame):
                raise ValueError("Fitur stream cache tidak lengkap.")
            self.loaded[seconds] = checked_stream(frame)
        return self.loaded[seconds]

    def matrix(self, recipe, rows, seconds):
        names = feature_group(recipe.features, self.features, recipe.baseline_ratio)
        if not recipe.history and recipe.normalization != "N3":
            return rows[names].reset_index(drop=True).copy()
        stream = self.stream(seconds)
        key = (
            frame_hash(stream[["session_id", "window_end", *names]]),
            seconds,
            recipe.history,
            recipe.normalization,
            tuple(names),
        )
        if key not in self.transformed:
            transformed = stream.copy()
            if recipe.history:
                transformed = causal_history(
                    transformed, [n for n in names if n in TEMPORAL], recipe.history
                )
            if recipe.normalization == "N3":
                transformed = causal_center(
                    transformed,
                    [
                        n
                        for n in names
                        if n not in ("rmssd_ms", "sdnn_ms", "rmssd_baseline_ratio")
                        and n.endswith("_mean")
                    ],
                )
            derived = [
                n for n in transformed if "__" in n and n.split("__")[0] in names
            ]
            self.transformed[key] = transformed[
                ["session_id", "window_end", *names, *derived]
            ]
        merged = rows[["session_id", "window_end"]].merge(
            self.transformed[key],
            how="left",
            on=["session_id", "window_end"],
            validate="one_to_one",
            indicator=True,
        )
        if not merged._merge.eq("both").all():
            raise ValueError("Endpoint supervised tidak ditemukan pada stream lengkap.")
        return merged.drop(columns=["session_id", "window_end", "_merge"])

    def sequences(self, recipe, rows, seconds):
        stream = self.stream(seconds)
        names = feature_group(recipe.features, self.features, recipe.baseline_ratio)
        return [
            stream.iloc[indices][names].copy()
            for indices in sequence_indices(stream, rows)
        ]

    def verify_against(self, frames):
        report = {}
        for seconds, rows in frames.items():
            stream = self.stream(seconds)
            merged = rows.merge(
                stream,
                on=["session_id", "window_end"],
                suffixes=("_csv", "_stream"),
                validate="one_to_one",
            )
            if len(merged) != len(rows):
                raise ValueError("Cache tidak mencakup seluruh endpoint eligible.")
            for feature in self.features:
                if not np.allclose(
                    merged[f"{feature}_csv"],
                    merged[f"{feature}_stream"],
                    equal_nan=True,
                    rtol=1e-9,
                    atol=1e-8,
                ):
                    raise ValueError(
                        f"Nilai cache berbeda dari dataset: {seconds}/{feature}"
                    )
            report[seconds] = {"matched_rows": len(rows), "full_grid_rows": len(stream)}
        return report
