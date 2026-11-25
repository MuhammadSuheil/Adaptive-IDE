"""Source cells for the two standalone relabeling notebooks."""

# %% Input CSV
import hashlib
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import RobustScaler

HRV = ["rmssd_ms", "sdnn_ms"]
EYE = ["blink_rate_per_minute", "ear_left_mean", "ear_left_std", "ear_right_mean",
       "ear_right_std", "iris_size_estimate_mean", "iris_size_estimate_std", "iris_size_delta_mean"]
CORE = HRV + EYE
ID = ["participant_id", "episode_id"]
SEED = 42


def read_bundle(root, case):
    proxy = case == "v3"
    package = root / ("v3_behavioral_proxy" if proxy else "v4_self_reported_rating")
    target = "cognitive_load_label" if proxy else "difficulty_rating"
    labels = ["Low", "Medium", "High"] if proxy else list(range(1, 11))
    window_dir = Path("window_15s_step_5s")
    hashes = {}

    def read(relative, columns):
        path = package / relative
        hashes[str(Path(relative).as_posix())] = hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path, usecols=columns)

    def attach(left, right, keys, validation):
        merged = left.merge(right, on=keys, how="left", validate=validation, indicator=True)
        if not merged.pop("_merge").eq("both").all():
            raise ValueError(f"CSV pendamping tidak cocok pada {keys}.")
        return merged

    keys = ["window_id", "participant_id", "session_id", target]
    data = read(window_dir / "good_acceptable.csv", keys + CORE)
    assignments = read(window_dir / "window_assignments.csv",
                       keys + ["episode_id"] + (["split"] if proxy else []))
    for table in (data, assignments):
        if table.window_id.duplicated().any() or table[keys].isna().any().any():
            raise ValueError("Kunci window tidak lengkap atau window_id berulang.")
    data = attach(data, assignments, keys, "one_to_one")
    evidence = read("episode_labels.csv" if proxy else "episode_ratings.csv",
                    ID + ["session_id", target])
    if evidence.episode_id.isna().any() or evidence.episode_id.duplicated().any():
        raise ValueError("episode_id pada bukti harus unik dan lengkap.")
    data = attach(data, evidence, ID + ["session_id", target], "many_to_one")
    if set(data[target]) != set(labels) or not np.isfinite(data[CORE].to_numpy(float)).all():
        raise ValueError("Periksa kelas target dan kelengkapan sepuluh fitur core.")
    if proxy:
        splits = read("participant_splits.csv", ["participant_id", "split"])
        if splits.participant_id.duplicated().any() or set(splits.split) != {"Train", "validation", "Test"}:
            raise ValueError("Periksa pembagian peserta v3.")
        data = attach(data, splits, ["participant_id", "split"], "many_to_one")
        weights = read(window_dir / "train_sample_weights.csv", ["window_id", *ID, target, "sample_weight"])
        train = data.split.eq("Train")
        if set(weights.window_id) != set(data.loc[train, "window_id"]) or weights.window_id.duplicated().any():
            raise ValueError("Daftar bobot berbeda dari window Train.")
        checked = attach(data.loc[train], weights, ["window_id", *ID, target], "one_to_one")
        if not np.isfinite(checked.sample_weight).all() or not checked.sample_weight.gt(0).all():
            raise ValueError("Bobot Train harus positif dan finite.")
        data["sample_weight"] = data.window_id.map(checked.set_index("window_id").sample_weight)
    return data.rename(columns={target: "actual"}), labels, hashes


# %% Model
def fit_probabilities(train, held, features, family, labels, weights):
    x, x_held = train[features], held[features]
    y = train.actual.to_numpy()
    if family in ("logistic", "ordinal"):
        scaler = RobustScaler().fit(x)
        x, x_held = scaler.transform(x), scaler.transform(x_held)
    if family == "ordinal":
        # P(Y > k) for k=1,...,9. A missing side in a fold has constant probability.
        tails = []
        for k in labels[:-1]:
            above = y > k
            if above.all() or not above.any():
                tails.append(np.full(len(held), float(above[0])))
            else:
                model = LogisticRegression(C=1, max_iter=3000)
                model.fit(x, above, sample_weight=weights)
                tails.append(model.predict_proba(x_held)[:, 1])
        # Enforce non-increasing tails before differencing into ten probabilities.
        tails = np.minimum.accumulate(np.column_stack(tails), axis=1)
        return -np.diff(np.column_stack([np.ones(len(held)), tails, np.zeros(len(held))]), axis=1)
    if family == "trees":
        model = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=5, max_features=0.8,
                                     random_state=SEED, n_jobs=2)
    elif family == "logistic":
        model = LogisticRegression(C=1, max_iter=3000)
    elif family == "prior":
        model = DummyClassifier(strategy="prior")
    else:
        raise ValueError(family)
    model.fit(x, y, sample_weight=weights)
    probs = np.zeros((len(held), len(labels)))
    probs[:, [labels.index(c) for c in model.classes_]] = model.predict_proba(x_held)
    return probs


def predict(train, held, labels, features, family, weights, name, split, fold):
    if set(train.participant_id) & set(held.participant_id):
        raise ValueError("Peserta fitting juga muncul pada evaluasi.")
    probs = fit_probabilities(train, held, features, family, labels, weights)
    if not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(axis=1), 1):
        raise ValueError("Distribusi prediksi tidak valid.")
    result = held[["window_id", *ID, "actual"]].copy()
    result["model"], result["split"], result["fold"] = name, split, fold
    for i, label in enumerate(labels):
        result[f"p_{label}"] = probs[:, i]
    return result


# %% Metrik dan ekspor
def predictions_by_unit(windows, labels):
    prob = [f"p_{c}" for c in labels]
    keys = ["split", "fold", "model", *ID, "actual"]
    episodes = windows.groupby(keys, as_index=False)[prob].mean()
    for table in (windows, episodes):
        p = table[prob].to_numpy()
        indices = p.argmax(axis=1)
        median = table.model.isin(["ordinal_logistic", "logistic_median", "dummy_median"])
        indices[median] = (p[median].cumsum(axis=1) >= 0.5).argmax(axis=1)
        table["predicted"] = np.asarray(labels)[indices]
    return {"window": windows, "episode": episodes}


def scores(table, labels):
    actual, predicted = table.actual, table.predicted
    row = dict(n=len(table), participants=table.participant_id.nunique(),
               accuracy=accuracy_score(actual, predicted),
               macro_f1=f1_score(actual, predicted, labels=labels, average="macro", zero_division=0))
    if len(labels) == 10:
        distance = (actual - predicted).abs()
        row.update(mae=distance.mean(), within_one=distance.le(1).mean())
    return row


def write_results(out, predictions, labels):
    metrics, by_person, confusion = [], [], []
    for unit, table in predictions_by_unit(pd.concat(predictions, ignore_index=True), labels).items():
        table.to_csv(out / f"{unit}_predictions.csv", index=False)
        for (split, model), group in table.groupby(["split", "model"]):
            info = dict(unit=unit, split=split, model=model)
            metrics.append({**info, **scores(group, labels)})
            for person, subset in group.groupby("participant_id"):
                by_person.append({**info, "participant_id": person, **scores(subset, labels)})
            matrix = confusion_matrix(group.actual, group.predicted, labels=labels)
            for label, counts in zip(labels, matrix):
                support = counts.sum()
                confusion.append({**info, "actual": label, "support": support,
                                  "recall": counts[labels.index(label)] / support if support else np.nan,
                                  **dict(zip([f"pred_{c}" for c in labels], counts))})
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(by_person).to_csv(out / "participant_metrics.csv", index=False)
    pd.DataFrame(confusion).to_csv(out / "confusion.csv", index=False)
    return metrics


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def prepare_run(output_root, case, data, labels, hashes, recipes):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = output_root / f"{case}_{stamp}_{uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "manifest.json", dict(case=case, labels=labels, seed=SEED, csv_sha256=hashes,
              recipes=recipes, python=platform.python_version(), sklearn=sklearn.__version__,
              numpy=np.__version__, pandas=pd.__version__))
    keys = ["split" if case == "v3" else "participant_id", "actual"]
    support = data.groupby(keys).agg(windows=("window_id", "size"), episodes=("episode_id", "nunique"),
                                     participants=("participant_id", "nunique"))
    support.to_csv(out / "input_support.csv")
    return out


def finish(out, predictions, labels):
    metrics = write_results(out, predictions, labels)
    archive = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    print(metrics[metrics.unit.eq("episode")].to_string(index=False))
    print("Arsip hasil:", archive)
    return Path(archive)


# %% V3
def run_v3(dataset_root, output_root):
    data, labels, hashes = read_bundle(dataset_root, "v3")
    recipes = {"dummy_mode": ("prior", CORE), "logistic_core": ("logistic", CORE),
               "trees_core": ("trees", CORE), "trees_hrv": ("trees", HRV), "trees_eye": ("trees", EYE)}
    out = prepare_run(output_root, "v3", data, labels, hashes, recipes)
    data[["participant_id", "split"]].drop_duplicates().to_csv(out / "participant_splits.csv", index=False)
    train = data[data.split.eq("Train")]
    validation = data[data.split.eq("validation")]
    predictions = []
    ranking = []
    for name, (family, features) in recipes.items():
        # The prevalence control describes the observed, unweighted fitting windows.
        weights = None if family == "prior" else train.sample_weight
        pred = predict(train, validation, labels, features, family, weights, name, "validation", "validation")
        predictions.append(pred)
        if family != "prior":
            episodes = predictions_by_unit(pred, labels)["episode"]
            rank = scores(episodes, labels)
            ranking.append((rank["macro_f1"], rank["accuracy"], name))
    # Stable ties retain the predeclared recipe order.
    best = max(ranking, key=lambda row: row[:2])[2]
    save_json(out / "selection.json", dict(model=best, selection_unit="episode", split="validation",
              criterion=["max macro_f1", "max accuracy", "recipe order"], refit="same Train and seed"))
    test = data[data.split.eq("Test")]
    for name in (best, "dummy_mode"):
        family, features = recipes[name]
        weights = None if family == "prior" else train.sample_weight
        predictions.append(predict(train, test, labels, features, family, weights, name, "Test", "Test"))
    return finish(out, predictions, labels)


# %% V4
def run_v4(dataset_root, output_root):
    data, labels, hashes = read_bundle(dataset_root, "v4")
    recipes = {"dummy_mode": ("prior", CORE), "logistic_core": ("logistic", CORE), "trees_core": ("trees", CORE),
               "ordinal_logistic": ("ordinal", CORE)}
    out = prepare_run(output_root, "v4", data, labels, hashes, recipes)
    people = sorted(data.participant_id.unique())
    folds = pd.DataFrame({"fold": range(1, len(people) + 1), "held_participant": people})
    folds.to_csv(out / "folds.csv", index=False)
    predictions = []
    for fold, person in folds.itertuples(index=False, name=None):
        held = data.participant_id.eq(person)
        train = data.loc[~held]
        weights = 1 / train.groupby("episode_id").window_id.transform("size")
        weights = weights / weights.mean()
        for name, (family, features) in recipes.items():
            pred = predict(train, data.loc[held], labels, features, family, weights, name, "LOPO", fold)
            predictions.append(pred)
            median_control = {"dummy_mode": "dummy_median", "logistic_core": "logistic_median"}
            if name in median_control:
                predictions.append(pred.assign(model=median_control[name]))
    return finish(out, predictions, labels)
