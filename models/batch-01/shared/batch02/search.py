"""Inner-only screening and bounded recipe search; no outer predictions here."""

from dataclasses import asdict, replace

import numpy as np

from .artifacts import digest, read_json, write_json
from .features import MissingStream
from .metrics import classification_metrics
from .modeling import fit_candidate
from .neural import Ineligible
from .persistence import checkpoint
from .preprocessing import CandidateInvalid
from .registry import Recipe, controls, model_bank, recipes_for, unique
from .runtime import capabilities


def context_seconds(recipe):
    return max(
        recipe.history,
        60 if recipe.normalization == "N3" else 0,
        30 if recipe.model == "gru" else 0,
    )


def estimator_count(recipe):
    if recipe.children:
        return sum(estimator_count(child) for child in recipe.children)
    if recipe.model == "late_fusion":
        return 900 if recipe.params.get("branch") == "xgboost" else 302
    if recipe.model in ("ordinal", "hierarchy"):
        return 600 if recipe.params.get("branch") == "xgboost" else 2
    if recipe.model == "tabpfn":
        return 8
    return (
        300
        if recipe.model
        in ("xgboost", "random_forest", "extra_trees", "catboost", "hgb")
        else 1
    )


def select_record(records):
    valid = [
        r for r in records if r.get("status", "ok") == "ok" and np.isfinite(r["score"])
    ]
    if not valid:
        raise Ineligible("Tidak ada kandidat eligible pada fitting partition ini.")
    best = max(r["score"] for r in valid)
    tied = [r for r in valid if best - r["score"] <= 0.005 + 1e-12]
    return min(
        tied,
        key=lambda r: (
            r["recipe"].baseline_ratio,
            r["n_features"],
            context_seconds(r["recipe"]),
            estimator_count(r["recipe"]),
            r["recipe"].key,
            r.get("window", 0),
        ),
    )


def ranked(records):
    remaining = [r for r in records if r.get("status") == "ok"]
    result = []
    while remaining:
        chosen = select_record(remaining)
        result.append(chosen)
        remaining = [r for r in remaining if r["recipe"].key != chosen["recipe"].key]
    return result


class Evaluator:
    def __init__(self, frame, store, seconds, config, directory, fit=fit_candidate):
        self.frame, self.store, self.seconds, self.config = (
            frame,
            store,
            seconds,
            config,
        )
        self.directory, self.fit = directory, fit
        self.results = {}

    def score(self, recipe, inner):
        key = digest(
            {"recipe": asdict(recipe), "inner": inner, "seed": self.config.seed}
        )
        if key in self.results:
            return self.results[key]
        path = self.directory / f"{key}.json"
        if path.exists():
            result = read_json(path)
            result["recipe"] = recipe
            self.results[key] = result
            return result
        predictions, truth, kinds, values, epochs, n_features = [], [], [], [], [], []
        try:
            if recipe.model in ("mlp", "ft_transformer", "gru", "tabpfn"):
                capability = capabilities(self.config)[recipe.model]
                if capability.startswith("skipped"):
                    raise Ineligible(capability)
            for fold in inner:
                train = self.frame.iloc[fold["train"]].copy()
                valid = self.frame.iloc[fold["valid"]].copy()
                if recipe.thin == 10:
                    train = thin_training(train)
                if (
                    train.person_id.isin(valid.person_id).any()
                    or train.y.nunique() != 3
                ):
                    raise ValueError(
                        "Inner fitting/validation groep atau kelas tidak sesuai kontrak."
                    )
                fitted = self.fit(recipe, train, self.store, self.seconds, self.config)
                output, kind = fitted.predict(valid, self.store, self.seconds)
                predictions.extend(output.argmax(1).tolist())
                truth.extend(valid.y.tolist())
                values.extend(output.tolist())
                kinds.append(kind)
                n_features.append(len(fitted.feature_names))
                if fitted.epoch is not None:
                    epochs.append(fitted.epoch)
            if len(set(kinds)) != 1:
                raise ValueError("Jenis skor berubah antar-inner fold.")
            metrics = classification_metrics(
                truth,
                predictions,
                np.array(values) if kinds[0] == "probability" else None,
            )
            result = {
                "status": "ok",
                "score": metrics["macro_f1"],
                "metrics": metrics,
                "n_features": float(np.median(n_features)),
                "epoch": int(np.median(epochs)) if epochs else None,
                "predicted": predictions,
                "y_true": truth,
                "values": values,
                "kind": kinds[0],
            }
        except (Ineligible, MissingStream) as error:
            result = {"status": "skipped", "reason": str(error)}
        except (CandidateInvalid, Warning) as error:
            result = {"status": "invalid", "reason": f"{type(error).__name__}: {error}"}
        write_json(path, {**result, "recipe": asdict(recipe)})
        result["recipe"] = recipe
        self.results[key] = result
        checkpoint()
        return result


def thin_training(frame):
    if "grid_origin" not in frame:
        raise MissingStream(
            "skipped_missing_stream: step-10 memerlukan grid_origin, bukan indeks row filtered."
        )
    ticks = (frame.window_end - frame.grid_origin).dt.total_seconds()
    result = frame[np.isclose(ticks % 10, 0)].copy()
    if result.y.nunique() != 3:
        raise Ineligible(
            "skipped_insufficient_support: thinning menghilangkan kelas fitting."
        )
    return result


def vote_from_records(records):
    first = []
    for record in ranked(records):
        recipe = record["recipe"]
        if recipe.model in (
            "logistic",
            "xgboost",
            "random_forest",
            "extra_trees",
            "catboost",
            "hgb",
        ) and all(r.model != recipe.model for r in first):
            first.append(recipe)
        if len(first) == 2:
            return Recipe("vote", children=tuple(first))
    raise Ineligible("Voting tidak mempunyai dua keluarga probabilistik yang eligible.")


def adaptive_search(evaluator, inner):
    records, seen, truncations = [], set(), []

    def add(recipes, optional=False):
        output = []
        for recipe in unique(recipes):
            if recipe.key in seen:
                output.extend(r for r in records if r["recipe"].key == recipe.key)
                continue
            if len(seen) >= 120:
                if not optional:
                    raise ValueError(
                        "Budget registry inti melebihi 120; perbaiki spec sebelum full."
                    )
                truncations.append(recipe.key)
                continue
            seen.add(recipe.key)
            result = evaluator.score(recipe, inner)
            records.append(result)
            output.append(result)
        return output

    screening = ranked(add(recipes_for("exp02")))
    representations = []
    for record in screening:
        signature = (record["recipe"].features, record["recipe"].baseline_ratio)
        if signature not in representations:
            representations.append(signature)
        if len(representations) == 2:
            break
    bank = add(
        [
            replace(r, features=f, baseline_ratio=b)
            for f, b in representations
            for r in model_bank()
        ]
    )
    static = ranked(bank + add(controls()))[:2]
    for record in static:
        base = record["recipe"]
        variants = [replace(base, history=h) for h in (15, 30)]
        variants += [replace(base, selector=s) for s in ("corr", "mi8", "mi12")]
        variants += [replace(base, balance=b) for b in ("weight", "ros", "smote")]
        refined = add(variants)
        history = [
            r for r in refined if r.get("status") == "ok" and r["recipe"].history
        ]
        selectors = [
            r
            for r in refined
            if r.get("status") == "ok" and r["recipe"].selector != "none"
        ]
        balances = [
            r
            for r in refined
            if r.get("status") == "ok" and r["recipe"].balance != "none"
        ]
        if history:
            h = select_record([record, *history])["recipe"].history
            combinations = []
            if selectors:
                combinations.append(
                    replace(
                        base,
                        history=h,
                        selector=select_record([record, *selectors])["recipe"].selector,
                    )
                )
            if balances:
                combinations.append(
                    replace(
                        base,
                        history=h,
                        balance=select_record([record, *balances])["recipe"].balance,
                    )
                )
            add(combinations)
    complex_recipes = [
        r
        for r in recipes_for("exp09") + recipes_for("exp10")
        if r.model not in ("vote", "logistic", "xgboost")
    ]
    try:
        complex_recipes.append(vote_from_records(bank))
    except Ineligible:
        pass
    add(complex_recipes)
    # Reserve advanced candidates before optional representation refinements.
    add(
        recipes_for("exp11") + recipes_for("exp12") + [Recipe("gru", features="F4")],
        optional=True,
    )
    for record in static:
        base = record["recipe"]
        extras = [
            replace(base, scaler="robust"),
            replace(base, winsor=True),
            replace(base, indicator=True),
            replace(base, selector="pca"),
        ]
        if base.features != "F6":
            extras.append(replace(base, normalization="N2"))
        if base.features not in ("F2", "F5"):
            extras.append(replace(base, normalization="N3"))
        if base.model in ("xgboost", "hgb", "catboost"):
            extras.append(replace(base, missing="native"))
        add(extras, optional=True)
    categories = {
        "static": [],
        "history": [],
        "normal": [],
        "complex": [],
        "advanced": [],
    }
    for record in ranked(records):
        r = record["recipe"]
        category = (
            "advanced"
            if r.model in ("mlp", "ft_transformer", "tabpfn", "gru")
            else "complex"
            if r.model in ("late_fusion", "vote", "ordinal", "hierarchy")
            else "normal"
            if r.normalization in ("N2", "N3")
            else "history"
            if r.history
            else "static"
        )
        categories[category].append(record)
    quotas = {"static": 6, "history": 4, "normal": 2, "complex": 4, "advanced": 8}
    pool = [r for name, group in categories.items() for r in group[: quotas[name]]]
    mandatory = [evaluator.score(recipe, inner) for recipe in controls()]
    mandatory = [r for r in mandatory if r.get("status") == "ok"]
    keys = {r["recipe"].key for r in mandatory}
    static_pool = (
        mandatory
        + [r for r in categories["static"] if r["recipe"].key not in keys][
            : 6 - len(mandatory)
        ]
    )
    pool = static_pool + [r for r in pool if r not in categories["static"]]
    winner = select_record(pool)
    return (
        winner,
        records,
        {
            "effective_recipes": len(seen),
            "truncated": truncations,
            "final_pool": [r["recipe"].key for r in pool],
            "limit": 120,
        },
    )
