TABULAR_FAMILIES = (
    "logistic",
    "lda",
    "knn",
    "svm",
    "forest",
    "extra",
    "hist",
    "xgboost",
    "catboost",
)
NEURAL_FAMILIES = ("mlp", "residual", "tabm", "transformer", "gru", "tcn")


def base_recipe(family, preset=0):
    params = {
        "dummy": {},
        "logistic": {"C": 1.0 if preset == 0 else 0.1},
        "lda": {"shrinkage": "auto"},
        "knn": {"n_neighbors": 15 if preset == 0 else 35, "weights": "distance"},
        "svm": {"C": 1.0 if preset == 0 else 10.0, "kernel": "rbf", "gamma": "scale"},
        "forest": {
            "n_estimators": 400,
            "min_samples_leaf": 5 if preset == 0 else 15,
            "max_features": 0.8,
        },
        "extra": {
            "n_estimators": 400,
            "min_samples_leaf": 5 if preset == 0 else 15,
            "max_features": 0.8,
        },
        "hist": {
            "max_iter": 300,
            "max_leaf_nodes": 15 if preset == 0 else 7,
            "min_samples_leaf": 25,
            "l2_regularization": 5.0,
            "learning_rate": 0.05,
        },
        "xgboost": {
            "n_estimators": 400,
            "max_depth": 3 if preset == 0 else 5,
            "min_child_weight": 10,
            "reg_lambda": 10.0,
            "learning_rate": 0.05,
        },
        "catboost": {
            "iterations": 400,
            "depth": 4 if preset == 0 else 6,
            "l2_leaf_reg": 10.0,
            "learning_rate": 0.05,
        },
    }
    return {
        "family": family,
        "group": "core",
        "normalization": "standard",
        "representation": "current",
        "history": 0,
        "derived": False,
        "selector": "all",
        "balance": "none",
        "objective": "multiclass",
        "params": copy.deepcopy(params.get(family, {})),
        "network": {
            "width": 64,
            "depth": 2,
            "dropout": 0.2,
            "lr": 0.001,
            "weight_decay": 0.001,
            "batch": 128,
            "smoothing": 0.0,
            "max_epochs": 200,
            "patience": 20,
        },
    }


def make_estimator(recipe, classes, seed=42):
    family, p = recipe["family"], dict(recipe["params"])
    if family == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if family == "logistic":
        return LogisticRegression(max_iter=4000, random_state=seed, **p)
    if family == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", **p)
    if family == "knn":
        return KNeighborsClassifier(**p)
    if family == "svm":
        return SVC(random_state=seed, **p)
    if family in ("forest", "extra"):
        cls = RandomForestClassifier if family == "forest" else ExtraTreesClassifier
        return cls(random_state=seed, n_jobs=2, **p)
    if family == "hist":
        return HistGradientBoostingClassifier(
            early_stopping=False, random_state=seed, **p
        )
    if family == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="multi:softprob" if classes > 2 else "binary:logistic",
            eval_metric="mlogloss" if classes > 2 else "logloss",
            n_jobs=2,
            random_state=seed,
            **p,
        )
    if family == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            loss_function="MultiClass" if classes > 2 else "Logloss",
            random_seed=seed,
            thread_count=2,
            allow_writing_files=False,
            verbose=False,
            **p,
        )
    raise ValueError(f"Keluarga tidak dikenal: {family}")


def fit_classifier(recipe, x, y, seed):
    if set(np.unique(y)) != set(range(len(np.unique(y)))) or len(np.unique(y)) < 2:
        raise ValueError("Label fitting harus lengkap dan berurutan.")
    if recipe["balance"] in ("ros", "smote"):
        if (
            recipe["group"] != "core"
            or recipe["derived"]
            or recipe["representation"] != "current"
        ):
            raise ValueError("Resampling hanya untuk core tanpa fitur turunan/riwayat.")
        from imblearn.over_sampling import SMOTE, RandomOverSampler

        minimum = int(np.bincount(y).min())
        sampler = (
            RandomOverSampler(random_state=seed)
            if recipe["balance"] == "ros"
            else SMOTE(random_state=seed, k_neighbors=min(5, minimum - 1))
        )
        if minimum < 2 and recipe["balance"] == "smote":
            raise ValueError("Contoh kelas terlalu sedikit untuk SMOTE.")
        x, y = sampler.fit_resample(x, y)
    estimator = make_estimator(recipe, len(np.unique(y)), seed)
    kwargs = {}
    if recipe["balance"] == "weight":
        if "sample_weight" not in inspect.signature(estimator.fit).parameters:
            raise ValueError("Keluarga ini tidak mendukung sample_weight.")
        kwargs["sample_weight"] = compute_sample_weight("balanced", y)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        estimator.fit(x, y, **kwargs)
    estimator.runtime_diagnostics_ = {
        "fit_status": int(estimator.fit_status_)
        if hasattr(estimator, "fit_status_")
        else None,
        "n_iter": np.asarray(estimator.n_iter_).astype(int).reshape(-1).tolist()
        if hasattr(estimator, "n_iter_")
        else [],
        "warnings": [f"{w.category.__name__}: {w.message}" for w in captured],
    }
    for warning in captured:
        warnings.warn(str(warning.message), warning.category, stacklevel=2)
    return estimator


def fit_model(recipe, frame, train_ids, seed=42, epochs=None, trial=None):
    if recipe["family"] in NEURAL_FAMILIES:
        return fit_neural(recipe, frame, train_ids, seed, epochs=epochs, trial=trial)
    ids = np.asarray(train_ids, int)
    started = time.perf_counter()
    pre = SensorTransform(recipe).fit(frame, ids)
    x, y = pre.transform(frame, ids), frame.iloc[ids]._y.to_numpy()
    preprocess_seconds = time.perf_counter() - started
    if set(y) != {0, 1, 2}:
        raise ValueError("Train tidak memuat tiga kelas.")
    started = time.perf_counter()
    if recipe["objective"] == "ordinal":
        estimators = [
            fit_classifier(recipe, x, (y > boundary).astype(int), seed)
            for boundary in (0, 1)
        ]
    elif recipe["objective"] == "hierarchy":
        estimators = [
            fit_classifier(recipe, x, (y > 0).astype(int), seed),
            fit_classifier(recipe, x[y > 0], (y[y > 0] == 2).astype(int), seed),
        ]
    else:
        estimators = [fit_classifier(recipe, x, y, seed)]
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    if recipe["objective"] == "multiclass":
        training_labels = estimators[0].predict(x).astype(int).reshape(-1)
    else:
        training_labels = None
    return {
        "kind": "tabular",
        "recipe": copy.deepcopy(recipe),
        "pre": pre,
        "estimators": estimators,
        "training_accuracy": float(accuracy_score(y, training_labels))
        if training_labels is not None
        else None,
        "runtime": {
            "preprocess_seconds": preprocess_seconds,
            "fit_seconds": fit_seconds,
            "training_predict_seconds": time.perf_counter() - started,
        },
        "convergence": [estimator.runtime_diagnostics_ for estimator in estimators],
    }


def aligned_scores(estimator, x, classes):
    if hasattr(estimator, "predict_proba"):
        values = np.asarray(estimator.predict_proba(x))
        result = np.zeros((len(x), classes))
        result[:, np.asarray(estimator.classes_, int)] = values
        return result, "probability"
    score = np.asarray(estimator.decision_function(x))
    if score.ndim == 1:
        score = np.column_stack([-score, score])
    result = np.zeros((len(x), classes))
    result[:, np.asarray(estimator.classes_, int)] = score
    return result, "decision_score"


def predict_model(model, frame, ids):
    ids = np.asarray(ids, int)
    if model["kind"] == "neural":
        return predict_neural(model, frame, ids)
    if model["kind"] in ("vote", "average", "stack"):
        outputs = [predict_model(base, frame, ids) for base in model["models"]]
        if model["kind"] == "stack":
            z = np.concatenate([out["scores"] for out in outputs], axis=1)
            scores = model["meta"].predict_proba(z)
            return {
                "labels": scores.argmax(axis=1),
                "scores": scores,
                "score_type": "probability",
            }
        if model["kind"] == "average":
            if not all(out["score_type"] == "probability" for out in outputs):
                raise ValueError(
                    "Decision score tidak boleh dirata-ratakan sebagai probabilitas."
                )
            scores = np.mean([out["scores"] for out in outputs], axis=0)
        else:
            scores = np.mean([np.eye(3)[out["labels"]] for out in outputs], axis=0)
        return {
            "labels": scores.argmax(axis=1),
            "scores": scores,
            "score_type": "probability"
            if model["kind"] == "average"
            else "vote_fraction",
        }
    x = model["pre"].transform(frame, ids)
    objective = model["recipe"]["objective"]
    if objective == "multiclass":
        estimator = model["estimators"][0]
        scores, kind = aligned_scores(estimator, x, 3)
        return {
            "labels": estimator.predict(x).astype(int).reshape(-1),
            "scores": scores,
            "score_type": kind,
        }
    scores = [aligned_scores(estimator, x, 2) for estimator in model["estimators"]]
    if any(kind != "probability" for _, kind in scores):
        raise ValueError("Ordinal/hierarchy memerlukan classifier probabilistik.")
    a, b = (value[:, 1] for value, _ in scores)
    if objective == "ordinal":
        crossed = a < b
        middle = (a + b) / 2
        a, b = np.where(crossed, middle, a), np.where(crossed, middle, b)
        p = np.column_stack([1 - a, a - b, b])
    else:
        p = np.column_stack([1 - a, a * (1 - b), a * b])
    return {"labels": p.argmax(axis=1), "scores": p, "score_type": "probability"}


def pack_model(model):
    def encode(value):
        module = type(value).__module__
        if module.startswith(("xgboost", "catboost")):
            native = "xgboost" if module.startswith("xgboost") else "catboost"
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / (
                    "model.json" if native == "xgboost" else "model.cbm"
                )
                value.save_model(str(path))
                return {"__native__": native, "payload": path.read_bytes()}
        if isinstance(value, dict):
            if value.get("kind") == "neural":
                import torch

                buffer = io.BytesIO()
                torch.save(
                    {
                        k: v.detach().cpu()
                        for k, v in value["network"].state_dict().items()
                    },
                    buffer,
                )
                return {
                    **{k: encode(v) for k, v in value.items() if k != "network"},
                    "__network_state__": buffer.getvalue(),
                }
            return {k: encode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [encode(v) for v in value]
        return value

    return cloudpickle.dumps(encode(model))


def unpack_model(payload):
    def decode(value):
        if isinstance(value, dict) and "__native__" in value:
            native = value["__native__"]
            if native == "xgboost":
                from xgboost import XGBClassifier

                model = XGBClassifier()
            else:
                from catboost import CatBoostClassifier

                model = CatBoostClassifier()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / (
                    "model.json" if native == "xgboost" else "model.cbm"
                )
                path.write_bytes(value["payload"])
                model.load_model(str(path))
            return model
        if isinstance(value, dict):
            result = {
                k: decode(v) for k, v in value.items() if k != "__network_state__"
            }
            if "__network_state__" in value:
                import torch

                result["network"] = build_network(result["recipe"], result["input_dim"])
                result["network"].load_state_dict(
                    torch.load(
                        io.BytesIO(value["__network_state__"]),
                        map_location="cpu",
                        weights_only=True,
                    )
                )
                result["network"].eval()
            return result
        if isinstance(value, list):
            return [decode(v) for v in value]
        return value

    return decode(cloudpickle.loads(payload))
