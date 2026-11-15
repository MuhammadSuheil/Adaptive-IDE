"""Fixed-three-class metrics; absent true classes remain explicit."""

import numpy as np

LABELS = ("Low", "Medium", "High")


def classification_metrics(y, predicted, probabilities=None):
    y, predicted = np.asarray(y, int), np.asarray(predicted, int)
    if (
        len(y) == 0
        or y.shape != predicted.shape
        or not np.isin(y, range(3)).all()
        or not np.isin(predicted, range(3)).all()
    ):
        raise ValueError(
            "Target/prediksi harus berupa array kelas 0/1/2 yang tidak kosong."
        )
    matrix = np.zeros((3, 3), dtype=int)
    np.add.at(matrix, (y, predicted), 1)
    support, counts, correct = matrix.sum(1), matrix.sum(0), matrix.diagonal()
    precision = np.divide(correct, counts, out=np.zeros(3, float), where=counts > 0)
    recall = np.divide(correct, support, out=np.zeros(3, float), where=support > 0)
    f1 = np.divide(
        2 * correct,
        support + counts,
        out=np.zeros(3, float),
        where=(support + counts) > 0,
    )
    result = {
        "macro_f1": float(f1.mean()),
        "accuracy": float(correct.sum() / len(y)),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "confusion_matrix": matrix.tolist(),
        "classes": {
            name: {
                "precision": float(precision[i]) if support[i] else None,
                "recall": float(recall[i]) if support[i] else None,
                "f1": float(f1[i]) if support[i] else None,
                "support": int(support[i]),
            }
            for i, name in enumerate(LABELS)
        },
    }
    if probabilities is not None:
        p = np.asarray(probabilities, float)
        if (
            p.shape != (len(y), 3)
            or not np.isfinite(p).all()
            or (p < 0).any()
            or not np.allclose(p.sum(1), 1, atol=1e-6)
        ):
            raise ValueError(
                "Probabilitas harus finite, nonnegatif, dan berjumlah satu."
            )
        result["log_loss"] = float(
            -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1)).mean()
        )
    return result


def paired_bootstrap(frame, first, second, repetitions=2000, seed=42):
    groups = sorted(frame.person_id.unique())
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        selected = rng.choice(groups, len(groups), replace=True)
        indices = np.concatenate(
            [np.flatnonzero(frame.person_id.to_numpy() == g) for g in selected]
        )
        a = classification_metrics(
            frame.y_true.to_numpy()[indices], frame[first].to_numpy()[indices]
        )["macro_f1"]
        b = classification_metrics(
            frame.y_true.to_numpy()[indices], frame[second].to_numpy()[indices]
        )["macro_f1"]
        differences.append(a - b)
    return {
        "difference_95ci": np.quantile(differences, [0.025, 0.975]).tolist(),
        "unit": "person",
        "seed": seed,
    }
