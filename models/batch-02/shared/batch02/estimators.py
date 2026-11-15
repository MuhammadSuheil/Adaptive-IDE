"""Lazy estimator construction. Constructors do not fit or download weights."""

import numpy as np


def make_estimator(recipe, seed=42, threads=2, classes=3):
    p = recipe.params.copy()
    if recipe.model == "dummy":
        from sklearn.dummy import DummyClassifier

        return DummyClassifier(strategy="most_frequent")
    if recipe.model == "logistic":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            C=p.get("C", 1.0),
            solver="lbfgs",
            max_iter=5000,
            tol=1e-4,
            random_state=seed,
        )
    if recipe.model == "svm":
        from sklearn.svm import SVC

        return SVC(
            C=p.get("C", 1.0),
            gamma=p.get("gamma", "scale"),
            kernel="rbf",
            probability=False,
            decision_function_shape="ovr",
            random_state=seed,
        )
    if recipe.model in ("random_forest", "extra_trees"):
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

        cls = (
            RandomForestClassifier
            if recipe.model == "random_forest"
            else ExtraTreesClassifier
        )
        return cls(
            n_estimators=300,
            min_samples_leaf=5,
            max_features=0.7,
            max_depth=p.get("max_depth", 5),
            class_weight=None,
            random_state=seed,
            n_jobs=threads,
        )
    if recipe.model == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=300,
            min_samples_leaf=20,
            l2_regularization=1,
            max_leaf_nodes=p.get("max_leaf_nodes", 7),
            learning_rate=p.get("learning_rate", 0.05),
            early_stopping=False,
            random_state=seed,
        )
    if recipe.model == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300,
            max_depth=p.get("max_depth", 3),
            learning_rate=p.get("learning_rate", 0.05),
            min_child_weight=p.get("min_child_weight", 5),
            reg_lambda=p.get("reg_lambda", 5),
            reg_alpha=0,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=threads,
            random_state=seed,
            objective="multi:softprob" if classes == 3 else "binary:logistic",
            **({"num_class": 3} if classes == 3 else {}),
        )
    if recipe.model == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=p.get("depth", 4),
            l2_leaf_reg=p.get("l2_leaf_reg", 5),
            loss_function="MultiClass" if classes == 3 else "Logloss",
            use_best_model=False,
            allow_writing_files=False,
            verbose=False,
            task_type="CPU",
            random_seed=seed,
            thread_count=threads,
        )
    if recipe.model in ("ordinal", "hierarchy"):
        return BinaryStructure(recipe, seed, threads)
    raise ValueError(
        f"Estimator memerlukan adapter khusus atau tidak dikenal: {recipe.model}"
    )


class BinaryStructure:
    def __init__(self, recipe, seed, threads):
        self.recipe, self.seed, self.threads = recipe, seed, threads
        self.classes_ = np.arange(3)

    def fit(self, x, y, sample_weight=None):
        from dataclasses import replace

        branch = (
            "logistic"
            if self.recipe.model == "ordinal"
            else self.recipe.params.get("branch", "logistic")
        )
        child = replace(
            self.recipe, model=branch, params={"C": self.recipe.params.get("C", 1.0)}
        )
        self.models_ = [
            make_estimator(child, self.seed, self.threads, 2) for _ in range(2)
        ]
        if self.recipe.model == "ordinal":
            tasks = [
                (np.ones(len(y), bool), (y > 0).astype(int)),
                (np.ones(len(y), bool), (y > 1).astype(int)),
            ]
        else:
            tasks = [
                (np.ones(len(y), bool), (y > 0).astype(int)),
                (y > 0, (y[y > 0] == 2).astype(int)),
            ]
        for model, (mask, target) in zip(self.models_, tasks):
            model.fit(
                x[mask],
                target,
                **(
                    {"sample_weight": sample_weight[mask]}
                    if sample_weight is not None
                    else {}
                ),
            )
        return self

    def predict_proba(self, x):
        q1, q2 = [m.predict_proba(x)[:, 1] for m in self.models_]
        if self.recipe.model == "ordinal":
            crossed = q2 > q1
            middle = (q1 + q2) / 2
            q1, q2 = np.where(crossed, middle, q1), np.where(crossed, middle, q2)
            return np.column_stack([1 - q1, q1 - q2, q2])
        return np.column_stack([1 - q1, q1 * (1 - q2), q1 * q2])
