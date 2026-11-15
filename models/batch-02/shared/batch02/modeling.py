"""Shared fit/predict interface for static, composite, and temporal models."""

from dataclasses import dataclass, replace

import numpy as np

from .artifacts import file_hash
from .fitting import fit_tabular
from .neural import Ineligible, fit_neural
from .preprocessing import Preprocessor


@dataclass
class Bundle:
    recipe: object
    fitted: object
    children: tuple = ()
    fallback: object = None
    fallback_count: int = 0

    @property
    def feature_names(self):
        if self.children:
            return sorted(set().union(*(set(c.feature_names) for c in self.children)))
        return self.fitted.feature_names

    @property
    def epoch(self):
        return getattr(self.fitted, "epoch", None)

    def predict(self, rows, store, seconds):
        if self.children:
            parts = [child.predict(rows, store, seconds) for child in self.children]
            if any(kind != "probability" for _, kind in parts):
                raise ValueError(
                    "Fusion/voting memerlukan probabilitas, bukan decision score."
                )
            alpha = self.recipe.params.get("alpha", 0.5)
            probabilities = alpha * parts[0][0] + (1 - alpha) * parts[1][0]
            if self.fallback is not None:
                unavailable = np.zeros(len(rows), bool)
                for child in self.children:
                    unavailable |= (
                        store.matrix(child.recipe, rows, seconds)
                        .isna()
                        .all(axis=1)
                        .to_numpy()
                    )
                if unavailable.any():
                    probabilities[unavailable] = self.fallback.predict(
                        rows.iloc[np.flatnonzero(unavailable)], store, seconds
                    )[0]
                self.fallback_count += int(unavailable.sum())
            return probabilities, "probability"
        sources = (
            store.sequences(self.recipe, rows, seconds)
            if self.recipe.model == "gru"
            else store.matrix(self.recipe, rows, seconds)
        )
        return self.fitted.predict(sources)


@dataclass
class TabPFNFit:
    preprocessing: object
    model: object
    epoch: int | None = None

    @property
    def feature_names(self):
        return self.preprocessing.feature_names_

    def predict(self, frame):
        x = self.preprocessing.transform(frame)
        # A single endpoint is deliberately the causal reference path.
        return np.concatenate(
            [self.model.predict_proba(row[None, :]) for row in x]
        ), "probability"


def fit_candidate(recipe, rows, store, seconds, config, epochs=None):
    if recipe.model in ("late_fusion", "vote"):
        if recipe.model == "late_fusion":
            base = recipe.params.get("branch", "logistic")
            children = tuple(
                replace(recipe, model=base, features=group, params={}, children=())
                for group in ("F5", "F6")
            )
        else:
            children = recipe.children
            if len(children) != 2 or children[0].model == children[1].model:
                raise ValueError(
                    "Voting memerlukan dua keluarga berbeda yang dipilih pada inner training."
                )
        fitted = tuple(fit_candidate(r, rows, store, seconds, config) for r in children)
        fallback = None
        if recipe.model == "late_fusion":
            from .registry import Recipe

            fallback = fit_candidate(
                Recipe("xgboost", scaler="none"), rows, store, seconds, config
            )
        return Bundle(recipe, None, fitted, fallback)
    sources = (
        store.sequences(recipe, rows, seconds)
        if recipe.model == "gru"
        else store.matrix(recipe, rows, seconds)
    )
    if recipe.model in ("mlp", "ft_transformer", "gru"):
        fitted = fit_neural(recipe, sources, rows, config.seed, config.threads, epochs)
    elif recipe.model == "tabpfn":
        import torch

        if not torch.cuda.is_available():
            raise Ineligible(
                "skipped_hardware: TabPFN memerlukan GPU pada protokol ini."
            )
        path = config.tabpfn_checkpoint
        if path is None or not path.is_file() or not config.tabpfn_license_confirmed:
            raise Ineligible(
                "skipped_checkpoint: checkpoint V2 lokal dan persetujuan lisensi diperlukan."
            )
        if (
            not config.tabpfn_checkpoint_sha256
            or file_hash(path) != config.tabpfn_checkpoint_sha256
        ):
            raise ValueError("Hash checkpoint TabPFN tidak sesuai.")
        if len(rows) > 10000 or sources.shape[1] > 500:
            raise Ineligible("skipped_size: batas input V2 terlampaui.")
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        processor = Preprocessor(scaler="none", missing="native").fit(sources)
        model = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2,
            model_path=str(path),
            n_estimators=8,
            random_state=config.seed,
            device="cuda",
        )
        model.fit(processor.transform(sources), rows.y.to_numpy())
        fitted = TabPFNFit(processor, model)
    else:
        fitted = fit_tabular(recipe, sources, rows, config.seed, config.threads)
    return Bundle(recipe, fitted)
