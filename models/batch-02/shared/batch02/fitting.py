"""Fit/predict boundaries with explicit score kind and training-only state."""

import warnings
from dataclasses import dataclass

import numpy as np

from .estimators import make_estimator
from .preprocessing import Preprocessor, representation, resample, training_weights


def prediction_values(model, matrix):
    if hasattr(model, "predict_proba"):
        raw, kind = np.asarray(model.predict_proba(matrix)), "probability"
    else:
        raw, kind = np.asarray(model.decision_function(matrix)), "decision"
    if raw.ndim != 2 or raw.shape[1] != 3 or sorted(model.classes_) != [0, 1, 2]:
        raise ValueError("Output model tidak memuat tiga kelas tetap.")
    return raw[:, [list(model.classes_).index(c) for c in range(3)]], kind


@dataclass
class FittedTabular:
    recipe: object
    preprocessing: object
    model: object
    resampling: dict
    epoch: int | None = None

    @property
    def feature_names(self):
        return self.preprocessing.feature_names_

    def predict(self, frame):
        matrix = self.preprocessing.transform(
            representation(frame, self.recipe.normalization)
        )
        values, kind = prediction_values(self.model, matrix)
        if tuple(self.recipe.decision) != (1.0, 1.0, 1.0):
            if kind != "probability":
                raise ValueError("Penyesuaian keputusan memerlukan probabilitas.")
            values = values * np.asarray(self.recipe.decision)
            values = values / values.sum(1, keepdims=True)
        return values, kind


def fit_tabular(recipe, matrix_frame, rows, seed=42, threads=2, factory=make_estimator):
    if recipe.missing == "native" and recipe.model not in (
        "xgboost",
        "hgb",
        "catboost",
    ):
        raise ValueError("Native NaN tidak didukung estimator ini.")
    if recipe.missing == "native" and recipe.balance in ("ros", "smote"):
        raise ValueError("Resampling awal tidak dikombinasikan dengan native NaN.")
    processor = Preprocessor(
        "standard" if recipe.balance == "smote" else recipe.scaler,
        recipe.missing,
        recipe.selector,
        recipe.indicator,
        recipe.winsor,
    )
    y = rows.y.to_numpy(int)
    matrix = processor.fit(
        representation(matrix_frame, recipe.normalization), y
    ).transform(representation(matrix_frame, recipe.normalization))
    matrix, sampled_y, sampling = resample(matrix, y, recipe.balance, seed)
    weight = training_weights(
        y,
        recipe.balance if recipe.balance in ("weight", "recording") else "none",
        rows.session_id.to_numpy(),
    )
    model = factory(recipe, seed, threads, 3)
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits

    with warnings.catch_warnings(), threadpool_limits(limits=threads):
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(
            matrix,
            sampled_y,
            **({"sample_weight": weight} if weight is not None else {}),
        )
    return FittedTabular(recipe, processor, model, sampling)
