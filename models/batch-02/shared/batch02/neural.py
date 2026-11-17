"""Group-aware neural stopping. Called only by explicit training cells."""

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .preprocessing import Preprocessor
from .streams import nonoverlap_indices


class Ineligible(ValueError):
    pass


def support_ok(rows):
    if rows.person_id.nunique() < 8:
        return False
    subset = rows.reset_index(drop=True).iloc[
        nonoverlap_indices(rows.reset_index(drop=True))
    ]
    return all(np.sum(subset.y == k) >= 20 for k in range(3))


def stopping_split(rows, seed):
    from sklearn.model_selection import StratifiedGroupKFold

    for count in (3, 4):
        if rows.person_id.nunique() < count:
            continue
        splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=seed)
        for fit, stop in splitter.split(
            np.zeros((len(rows), 1)), rows.y, rows.person_id
        ):
            if support_ok(rows.iloc[fit]) and rows.iloc[fit].y.nunique() == 3:
                return fit, stop
    raise Ineligible(
        "skipped_insufficient_support: tidak ada group stopping dengan >=8 grup dan >=20 endpoint nonoverlap per kelas fitting."
    )


def subset_sources(sources, indices):
    return (
        sources.iloc[indices]
        if isinstance(sources, pd.DataFrame)
        else [sources[i] for i in indices]
    )


def flattened(sources):
    if isinstance(sources, pd.DataFrame):
        return sources
    joined = pd.concat(sources)
    return joined.loc[~joined.index.duplicated()]


def prepared(sources, processor):
    if isinstance(sources, pd.DataFrame):
        return processor.transform(sources).astype("float32"), None
    matrices = [processor.transform(frame).astype("float32") for frame in sources]
    padded = np.zeros((len(matrices), 7, matrices[0].shape[1]), dtype="float32")
    lengths = np.array([len(m) for m in matrices], dtype="int64")
    for i, matrix in enumerate(matrices):
        padded[i, : len(matrix)] = matrix
    return padded, lengths


def network(name, width):
    from .networks import FeatureTransformer, SensorGRU, TabularMLP

    return {"mlp": TabularMLP, "ft_transformer": FeatureTransformer, "gru": SensorGRU}[
        name
    ](width)


@dataclass
class NeuralFit:
    recipe: object
    preprocessing: object
    model: object
    epoch: int

    @property
    def feature_names(self):
        return self.preprocessing.feature_names_

    def predict(self, sources):
        import torch

        matrix, lengths = prepared(sources, self.preprocessing)
        self.model.eval()
        results = []
        with torch.no_grad():
            for start in range(0, len(matrix), 256):
                x = torch.from_numpy(matrix[start : start + 256])
                logits = (
                    self.model(x, torch.from_numpy(lengths[start : start + 256]))
                    if lengths is not None
                    else self.model(x)
                )
                results.append(torch.softmax(logits, dim=-1).numpy())
        return np.concatenate(results), "probability"


def _train(recipe, sources, rows, seed, threads, epochs, stopping=None):
    import torch

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    processor = Preprocessor().fit(flattened(sources))
    matrix, lengths = prepared(sources, processor)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    model = network(recipe.model, matrix.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=recipe.params.get(
            "lr", 0.0003 if recipe.model == "ft_transformer" else 0.001
        ),
        weight_decay=0.001,
    )
    target = torch.tensor(rows.y.to_numpy(), dtype=torch.long, device=device)
    loss_fn = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    best_loss, best_epoch, stale = float("inf"), 1, 0
    if stopping is not None:
        stop_sources, stop_rows = stopping
        stop_matrix, stop_lengths = prepared(stop_sources, processor)
    for epoch in range(1, epochs + 1):
        model.train()
        indices = rng.permutation(len(rows))
        for start in range(0, len(rows), 64):
            batch = indices[start : start + 64]
            x = torch.from_numpy(matrix[batch]).to(device)
            optimizer.zero_grad()
            logits = (
                model(x, torch.from_numpy(lengths[batch]))
                if lengths is not None
                else model(x)
            )
            loss_fn(logits, target[batch]).backward()
            optimizer.step()
        if stopping is not None:
            model.eval()
            with torch.no_grad():
                x = torch.from_numpy(stop_matrix).to(device)
                logits = (
                    model(x, torch.from_numpy(stop_lengths))
                    if stop_lengths is not None
                    else model(x)
                )
                loss = float(
                    loss_fn(
                        logits,
                        torch.tensor(
                            stop_rows.y.to_numpy(), dtype=torch.long, device=device
                        ),
                    )
                )
            if loss < best_loss - 1e-8:
                best_loss, best_epoch, stale = loss, epoch, 0
            else:
                stale += 1
            if stale >= 20:
                break
    return NeuralFit(
        recipe, processor, model.cpu(), best_epoch if stopping is not None else epochs
    )


def fit_neural(recipe, sources, rows, seed=42, threads=2, epochs=None):
    if not support_ok(rows):
        raise Ineligible(
            "skipped_insufficient_support: neural support gate tidak terpenuhi."
        )
    if epochs is None:
        fit, stop = stopping_split(rows.reset_index(drop=True), seed)
        selected = _train(
            recipe,
            subset_sources(sources, fit),
            rows.iloc[fit],
            seed,
            threads,
            200,
            (subset_sources(sources, stop), rows.iloc[stop]),
        )
        epochs = selected.epoch
    return _train(recipe, sources, rows, seed, threads, int(epochs))
