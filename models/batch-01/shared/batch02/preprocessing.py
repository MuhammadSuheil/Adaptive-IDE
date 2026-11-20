"""Training-only tabular transformations and balancing."""

import numpy as np
import pandas as pd


class CandidateInvalid(ValueError):
    pass


def representation(frame, normalization):
    result = frame.copy()
    if normalization == "N2":
        for name in ("rmssd_ms", "sdnn_ms"):
            if name in result:
                result[name] = np.log1p(result[name].where(result[name] >= 0))
        if "rmssd_baseline_ratio" in result:
            result["rmssd_baseline_ratio"] = np.log(
                result.rmssd_baseline_ratio.where(result.rmssd_baseline_ratio > 0)
            )
    elif normalization not in ("N0", "N1", "N3"):
        raise ValueError("Normalisasi tidak dikenal.")
    return result


class Preprocessor:
    def __init__(
        self,
        scaler="standard",
        missing="median",
        selector="none",
        indicator=False,
        winsor=False,
    ):
        self.scaler = scaler
        self.missing = missing
        self.selector = selector
        self.indicator = indicator
        self.winsor = winsor

    def fit(self, frame, y=None):
        self.columns_ = [
            c
            for c in frame
            if frame[c].notna().any() and frame[c].nunique(dropna=True) > 1
        ]
        if not self.columns_:
            raise CandidateInvalid(
                "Tidak ada fitur training yang tidak kosong/konstan."
            )
        values = frame[self.columns_].astype(float)
        self.bounds_ = values.quantile([0.01, 0.99]) if self.winsor else None
        if self.bounds_ is not None:
            values = values.clip(self.bounds_.loc[0.01], self.bounds_.loc[0.99], axis=1)
        self.medians_ = values.median()
        self.names_ = self.columns_ + (
            [f"{n}__missing" for n in self.columns_] if self.indicator else []
        )
        matrix = self._impute(values)
        if self.scaler == "standard":
            self.offset_ = np.nanmean(matrix, axis=0)
            self.scale_ = np.nanstd(matrix, axis=0)
        elif self.scaler == "robust":
            self.offset_ = np.nanmedian(matrix, axis=0)
            self.scale_ = np.nanpercentile(matrix, 75, axis=0) - np.nanpercentile(
                matrix, 25, axis=0
            )
        elif self.scaler == "none":
            self.offset_, self.scale_ = (
                np.zeros(matrix.shape[1]),
                np.ones(matrix.shape[1]),
            )
        else:
            raise ValueError("Scaler tidak dikenal.")
        self.scale_[~np.isfinite(self.scale_) | (self.scale_ == 0)] = 1.0
        matrix = (matrix - self.offset_) / self.scale_
        self.indices_ = np.arange(matrix.shape[1])
        self.pca_ = None
        if self.selector == "corr":
            correlation = pd.DataFrame(matrix).corr().abs().to_numpy()
            keep = []
            for i in range(len(self.names_)):
                if not any(correlation[i, j] > 0.95 for j in keep):
                    keep.append(i)
            self.indices_ = np.array(keep)
        elif self.selector.startswith("mi"):
            from sklearn.feature_selection import mutual_info_classif

            if y is None or not np.isfinite(matrix).all():
                raise ValueError("MI memerlukan target training dan input terimputasi.")
            scores = mutual_info_classif(matrix, y, n_neighbors=3, random_state=42)
            self.indices_ = np.argsort(-scores, kind="stable")[
                : min(int(self.selector[2:]), matrix.shape[1])
            ]
        elif self.selector == "pca":
            from sklearn.decomposition import PCA

            self.pca_ = PCA(n_components=0.95, whiten=False, svd_solver="full").fit(
                matrix
            )
        elif self.selector != "none":
            raise ValueError("Selector tidak dikenal.")
        self.feature_names_ = (
            [f"PC{i + 1}" for i in range(self.pca_.n_components_)]
            if self.pca_ is not None
            else [self.names_[i] for i in self.indices_]
        )
        return self

    def _impute(self, frame):
        if self.missing not in ("median", "native"):
            raise ValueError("Missing handler tidak dikenal.")
        values = frame.fillna(self.medians_) if self.missing == "median" else frame
        matrix = values.to_numpy(float)
        return (
            np.concatenate([matrix, frame.isna().to_numpy(float)], axis=1)
            if self.indicator
            else matrix
        )

    def transform(self, frame):
        values = frame[self.columns_].astype(float)
        if self.bounds_ is not None:
            values = values.clip(self.bounds_.loc[0.01], self.bounds_.loc[0.99], axis=1)
        matrix = (self._impute(values) - self.offset_) / self.scale_
        return (
            self.pca_.transform(matrix)
            if self.pca_ is not None
            else matrix[:, self.indices_]
        )


def sampling_strategy(y):
    counts = np.bincount(np.asarray(y, int), minlength=3)
    return {i: int(counts.max()) for i, n in enumerate(counts) if 0 < n < counts.max()}


def training_weights(y, method, sessions=None):
    y = np.asarray(y, int)
    if method == "weight":
        counts = np.bincount(y, minlength=3)
        if (counts == 0).any():
            raise ValueError("Class weight memerlukan ketiga kelas training.")
        values = len(y) / (3 * counts[y])
    elif method == "recording":
        _, inverse, counts = np.unique(
            sessions, return_inverse=True, return_counts=True
        )
        values = 1.0 / counts[inverse]
    elif method == "none":
        return None
    else:
        raise ValueError("Bobot tidak dikenal.")
    return values / values.mean()


def resample(matrix, y, method, seed=42):
    strategy = sampling_strategy(y)
    if method not in ("ros", "smote") or not strategy:
        return matrix, np.asarray(y), {"method": method, "added": 0}
    from imblearn.over_sampling import SMOTE, RandomOverSampler

    if method == "smote" and any(np.sum(np.asarray(y) == k) <= 3 for k in strategy):
        raise CandidateInvalid(
            "SMOTE k=3 memerlukan lebih dari tiga contoh setiap kelas yang disintesis."
        )
    sampler = (
        SMOTE(k_neighbors=3, sampling_strategy=strategy, random_state=seed)
        if method == "smote"
        else RandomOverSampler(sampling_strategy=strategy, random_state=seed)
    )
    x_new, y_new = sampler.fit_resample(matrix, y)
    return (
        x_new,
        y_new,
        {"method": method, "strategy": strategy, "added": len(y_new) - len(y)},
    )
