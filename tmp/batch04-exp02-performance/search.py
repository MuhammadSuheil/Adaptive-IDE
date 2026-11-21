import optuna


def evaluate_recipe(store, frame, window, recipe, repeats=(0,), trial=None):
    rid = digest(recipe)[:16]
    key = f"cv_w{window}_{rid}_n{len(repeats)}"
    cached = store.read(key)
    if cached is not None:
        return cached
    folds, predictions, best_epochs, repeat_metrics = [], [], [], []
    for repeat in repeats:
        repeat_rows = []
        for fold in range(5):
            fold_started = time.perf_counter()
            io_before = dict(store.io_seconds)
            column = f"_cv_{repeat}_{fold}"
            fitting = np.flatnonzero(frame[column].eq("train"))
            scoring = np.flatnonzero(frame[column].eq("validation"))
            fold_key = f"fold_w{window}_{rid}_r{repeat}f{fold}"
            record = store.read(fold_key)
            progress(
                store,
                stage="cross_validation",
                window=window,
                family=recipe["family"],
                repeat=repeat + 1,
                fold=fold + 1,
                folds=5,
                recipe=rid,
                trial=trial.number + 1 if trial is not None else None,
                started_at=datetime.now(UTC).isoformat(),
                status="cached" if record is not None else "running",
            )
            if record is None:
                if trial is not None:
                    trial.set_user_attr("fold_offset", (repeat * 5 + fold) * 202)
                label = (
                    f"window={window}s {recipe['family']} "
                    f"kernel={recipe['params'].get('kernel', '-')} C={recipe['params'].get('C', '-')} "
                    f"recipe={rid} repeat={repeat + 1} fold={fold + 1}/5 "
                    f"trial={trial.number + 1 if trial is not None else '-'} fit_rows={len(fitting)}"
                )
                with FitHeartbeat(label):
                    model = fit_model(recipe, frame, fitting, seed=42, trial=trial)
                started = time.perf_counter()
                output = predict_model(model, frame, scoring)
                prediction_seconds = time.perf_counter() - started
                score = metrics(frame.iloc[scoring]._y, output["labels"])
                training_accuracy = model.get("training_accuracy")
                if training_accuracy is None:
                    training = predict_model(model, frame, fitting)
                    training_accuracy = metrics(
                        frame.iloc[fitting]._y, training["labels"]
                    )["accuracy"]
                record = {
                    "window": window,
                    "recipe": recipe,
                    "fold": fold,
                    "repeat": repeat,
                    "validation": score,
                    "train_accuracy": training_accuracy,
                    "best_epoch": model.get("best_epoch"),
                    "score_type": output["score_type"],
                    "runtime": {
                        **model.get("runtime", {}),
                        "validation_predict_seconds": prediction_seconds,
                        "checkpoint_read_seconds": store.io_seconds["read_seconds"]
                        - io_before["read_seconds"],
                        "compute_seconds": time.perf_counter() - fold_started,
                    },
                    "convergence": model.get("convergence", []),
                }
                rows = [
                    {
                        "row": int(i),
                        "y": int(frame.iloc[i]._y),
                        "prediction": int(label),
                        "scores": [float(v) for v in scores],
                        "repeat": repeat,
                    }
                    for i, label, scores in zip(
                        scoring, output["labels"], output["scores"]
                    )
                ]
                artifacts = {"predictions.json": canonical(rows).encode()}
                if model.get("history"):
                    artifacts["history.csv"] = csv_bytes(pd.DataFrame(model["history"]))
                started = time.perf_counter()
                store.commit(fold_key, record, artifacts)
                checkpoint_seconds = time.perf_counter() - started
                store.commit(
                    f"timing_{fold_key}",
                    {
                        "checkpoint": fold_key,
                        "write_seconds": checkpoint_seconds,
                        "total_seconds": time.perf_counter() - fold_started,
                    },
                )
                print(
                    f"Checkpoint tersimpan {fold_key} | write={checkpoint_seconds:.2f}s",
                    flush=True,
                )
                del model
            else:
                print(f"Checkpoint dipakai ulang {fold_key}", flush=True)
            rows = json.loads(store.artifact(fold_key, "predictions.json"))
            progress(
                store,
                stage="cross_validation",
                status="completed",
                window=window,
                family=recipe["family"],
                repeat=repeat + 1,
                fold=fold + 1,
                folds=5,
                recipe=rid,
                elapsed_seconds=time.perf_counter() - fold_started,
            )
            repeat_rows.extend(rows)
            folds.append(record)
            if record["best_epoch"] is not None:
                best_epochs.append(record["best_epoch"])
            if trial is not None:
                step = (
                    (repeat * 5 + fold + 1) * 202 - 1
                    if recipe["family"] in NEURAL_FAMILIES
                    else fold
                )
                trial.report(
                    float(np.mean([r["validation"]["accuracy"] for r in folds])),
                    step=step,
                )
                if fold >= 2 and trial.should_prune():
                    raise optuna.TrialPruned()
        expected = set(np.flatnonzero(frame._split.eq("train")))
        if {r["row"] for r in repeat_rows} != expected or len(repeat_rows) != len(
            expected
        ):
            raise ValueError(
                "OOF tidak mencakup tepat satu prediksi setiap baris Train."
            )
        repeat_metrics.append(
            metrics(
                [r["y"] for r in repeat_rows], [r["prediction"] for r in repeat_rows]
            )
        )
        predictions.extend(repeat_rows)
    record = {
        "id": rid,
        "window": window,
        "recipe": recipe,
        "validation": {
            name: float(np.mean([r[name] for r in repeat_metrics]))
            for name in repeat_metrics[0]
        },
        "accuracy_std": float(np.std([r["validation"]["accuracy"] for r in folds])),
        "training_accuracy": float(np.mean([r["train_accuracy"] for r in folds])),
        "epochs": int(np.median(best_epochs)) if best_epochs else None,
        "repeats": list(repeats),
        "complexity": len(GROUPS[recipe["group"]]),
    }
    store.commit(
        key,
        record,
        {
            "oof.json": canonical(predictions).encode(),
            "folds.csv": csv_bytes(
                pd.DataFrame(
                    [
                        {
                            "repeat": r["repeat"],
                            "fold": r["fold"],
                            **r["validation"],
                            "train_accuracy": r["train_accuracy"],
                            "best_epoch": r["best_epoch"],
                        }
                        for r in folds
                    ]
                )
            ),
        },
    )
    return record


def suggest_recipe(trial, family, experiment, groups=None):
    recipe = base_recipe(family)
    recipe["group"] = trial.suggest_categorical(
        "group", groups or ["core", "core_head"]
    )
    if experiment == "exp02":
        normalization = trial.suggest_categorical(
            "normalization", ["standard", "session_center", "session_robust", "causal"]
        )
        recipe.update(
            normalization=normalization,
            representation=trial.suggest_categorical(
                "representation", ["current", "history"]
            ),
            history=trial.suggest_categorical("history", [15, 30, 60]),
        )
    else:
        recipe["normalization"] = trial.suggest_categorical(
            "normalization", ["standard", "robust", "log_robust"]
        )
    if family not in NEURAL_FAMILIES:
        recipe["derived"] = trial.suggest_categorical("derived", [False, True])
        recipe["selector"] = trial.suggest_categorical(
            "selector", ["all", "mi8", "pca95"]
        )
        p = recipe["params"]
        if family == "logistic":
            p["C"] = trial.suggest_float("C", 1e-4, 1e3, log=True)
        elif family == "lda":
            p["shrinkage"] = trial.suggest_float("shrinkage", 0.01, 1.0)
        elif family == "knn":
            p.update(
                n_neighbors=trial.suggest_int("neighbors", 5, 81, step=2),
                weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
                p=trial.suggest_categorical("distance", [1, 2]),
            )
        elif family == "svm":
            p.update(
                C=trial.suggest_float("C", 1e-3, 1e3, log=True),
                kernel=trial.suggest_categorical("kernel", ["rbf", "linear"]),
            )
            if p["kernel"] == "rbf":
                p["gamma"] = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
        elif family in ("forest", "extra"):
            p.update(
                n_estimators=trial.suggest_categorical("trees", [300, 600]),
                min_samples_leaf=trial.suggest_int("leaf", 1, 32),
                max_depth=trial.suggest_categorical("depth", [4, 8, 16, None]),
                max_features=trial.suggest_categorical("features", [0.5, 0.8, 1.0]),
            )
        elif family == "hist":
            p.update(
                max_iter=trial.suggest_int("iterations", 200, 800, step=100),
                max_leaf_nodes=trial.suggest_categorical("leaves", [7, 15, 31]),
                min_samples_leaf=trial.suggest_int("leaf", 10, 80),
                learning_rate=trial.suggest_float("lr", 0.01, 0.2, log=True),
                l2_regularization=trial.suggest_float("l2", 0.001, 100.0, log=True),
            )
        elif family == "xgboost":
            p.update(
                n_estimators=trial.suggest_int("iterations", 200, 1000, step=100),
                max_depth=trial.suggest_int("depth", 2, 7),
                min_child_weight=trial.suggest_float("child", 2, 40, log=True),
                learning_rate=trial.suggest_float("lr", 0.01, 0.15, log=True),
                reg_lambda=trial.suggest_float("l2", 0.1, 100, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample", 0.6, 1.0),
            )
        elif family == "catboost":
            p.update(
                iterations=trial.suggest_int("iterations", 200, 1000, step=100),
                depth=trial.suggest_int("depth", 3, 7),
                learning_rate=trial.suggest_float("lr", 0.01, 0.15, log=True),
                l2_leaf_reg=trial.suggest_float("l2", 1.0, 100.0, log=True),
            )
    else:
        recipe["balance"] = trial.suggest_categorical("balance", ["none", "weight"])
        cfg = recipe["network"]
        cfg.update(
            width=trial.suggest_categorical(
                "width",
                [32, 64, 128, 256] if family not in ("gru", "tcn") else [16, 32, 64],
            ),
            depth=trial.suggest_int(
                "depth", 1, 3 if family not in ("gru", "tcn") else 2
            ),
            dropout=trial.suggest_float("dropout", 0.0, 0.6),
            lr=trial.suggest_float("lr", 1e-4, 0.003, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 0.1, log=True),
            batch=trial.suggest_categorical("batch", [64, 128, 256]),
            smoothing=trial.suggest_categorical("smoothing", [0.0, 0.05, 0.1]),
        )
        if family in ("gru", "tcn"):
            recipe["history"] = trial.suggest_categorical("history", [15, 30, 60])
    if (
        recipe["representation"] == "current"
        and recipe["normalization"] != "causal"
        and family not in ("gru", "tcn")
    ):
        recipe["history"] = 0
    return recipe


def optimize(store, frame, window, name, families, count, experiment):
    sid = (
        f"w{window}_"
        + digest({"name": name, "families": list(families), "experiment": experiment})[
            :12
        ]
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
    )
    records = []
    for number in range(count):
        key = f"study_{sid}_t{number}"
        done = store.read(key)
        if done is not None:
            study = cloudpickle.loads(store.artifact(key, "study.pkl"))
            if done["candidate"] is not None:
                records.append(done["candidate"])
            continue
        pending = f"pending_{sid}_t{number}"
        saved = store.read(pending)
        if saved is None:
            trial = study.ask()
            family = (
                families[0]
                if len(families) == 1
                else trial.suggest_categorical("family", list(families))
            )
            recipe = suggest_recipe(trial, family, experiment, available_groups(frame))
            store.commit(
                pending,
                {"trial_number": trial.number, "recipe": recipe},
                {"study.pkl": cloudpickle.dumps(study)},
            )
        else:
            study = cloudpickle.loads(store.artifact(pending, "study.pkl"))
            frozen = study.get_trials(deepcopy=False)[saved["trial_number"]]
            trial = optuna.trial.FixedTrial(frozen.params, number=frozen.number)
            recipe = saved["recipe"]
        progress(
            store,
            stage="tuning",
            window=window,
            study=name,
            trial=number + 1,
            total_trials=count,
        )
        try:
            record = evaluate_recipe(store, frame, window, recipe, trial=trial)
            study.tell(
                saved["trial_number"] if saved else trial.number,
                record["validation"]["accuracy"],
            )
            records.append(record)
            state = "complete"
        except optuna.TrialPruned:
            study.tell(
                saved["trial_number"] if saved else trial.number,
                state=optuna.trial.TrialState.PRUNED,
            )
            record, state = None, "pruned"
        store.commit(
            key,
            {"candidate": record, "state": state},
            {"study.pkl": cloudpickle.dumps(study)},
        )
        store.publish(f"trials_{sid}.csv", csv_bytes(study.trials_dataframe()))
        print(f"Trial {number + 1}/{count} window={window}s status={state}", flush=True)
    return records


def unique_records(records):
    return sorted({r["id"]: r for r in records}.values(), key=rank_key)


def family_leaders(records, count):
    selected, seen = [], set()
    for record in unique_records(records):
        family = record["recipe"]["family"]
        if family not in seen:
            selected.append(record)
            seen.add(family)
        if len(selected) == count:
            break
    return selected


def leaderboard(records):
    return pd.DataFrame(
        [
            {
                "id": r["id"],
                "window": r["window"],
                "family": r["recipe"]["family"],
                "group": r["recipe"]["group"],
                "normalization": r["recipe"]["normalization"],
                "representation": r["recipe"]["representation"],
                "history": r["recipe"]["history"],
                "objective": r["recipe"]["objective"],
                "balance": r["recipe"]["balance"],
                "selector": r["recipe"]["selector"],
                "derived": r["recipe"]["derived"],
                "training_accuracy": r["training_accuracy"],
                "accuracy_std": r["accuracy_std"],
                "repeats": len(r["repeats"]),
                **r["validation"],
            }
            for r in records
        ]
    )


def run_search(store, views, experiment):
    all_records, shortlists = [], {}
    for window, frame in views.items():
        records = [evaluate_recipe(store, frame, window, base_recipe("dummy"))]
        if experiment == "exp01":
            for family in TABULAR_FAMILIES:
                for group in available_groups(frame):
                    for preset in range(2):
                        recipe = base_recipe(family, preset)
                        recipe["group"] = group
                        records.append(evaluate_recipe(store, frame, window, recipe))
            families = list(
                dict.fromkeys(
                    r["recipe"]["family"]
                    for r in unique_records(records)
                    if r["recipe"]["family"] != "dummy"
                )
            )[:4]
            for family in families:
                records += optimize(
                    store,
                    frame,
                    window,
                    family,
                    [family],
                    BUDGET["tabular_trials"],
                    experiment,
                )
            leaders = family_leaders(
                [
                    r
                    for r in records
                    if r["recipe"]["family"] not in ("dummy", "knn", "lda")
                ],
                2,
            )
            for leader in leaders:
                for selector in ("all", "mi8", "pca95"):
                    for derived in (False, True):
                        recipe = copy.deepcopy(leader["recipe"])
                        recipe.update(selector=selector, derived=derived)
                        records.append(evaluate_recipe(store, frame, window, recipe))
                for group in ("eye", "hrv"):
                    recipe = copy.deepcopy(leader["recipe"])
                    recipe.update(group=group, derived=False, selector="all")
                    records.append(evaluate_recipe(store, frame, window, recipe))
                for balance in ("none", "weight", "ros", "smote"):
                    recipe = copy.deepcopy(leader["recipe"])
                    recipe.update(
                        group="core",
                        derived=False,
                        representation="current",
                        history=0,
                        selector="all",
                        balance=balance,
                    )
                    records.append(evaluate_recipe(store, frame, window, recipe))
            probabilistic = family_leaders(
                [
                    r
                    for r in records
                    if r["recipe"]["family"]
                    in ("logistic", "hist", "xgboost", "catboost", "forest", "extra")
                ],
                2,
            )
            for leader in probabilistic:
                for objective in ("ordinal", "hierarchy"):
                    recipe = copy.deepcopy(leader["recipe"])
                    recipe["objective"] = objective
                    records.append(evaluate_recipe(store, frame, window, recipe))
        elif experiment == "exp02":
            for family in ("logistic", "svm", "hist"):
                for normalization in ("standard", "session_center", "session_robust"):
                    recipe = base_recipe(family)
                    recipe["normalization"] = normalization
                    records.append(evaluate_recipe(store, frame, window, recipe))
            records += optimize(
                store,
                frame,
                window,
                "reference",
                ["logistic", "svm", "hist"],
                BUDGET["reference_trials"],
                experiment,
            )
            winner = unique_records(
                [r for r in records if r["recipe"]["family"] != "dummy"]
            )[0]["recipe"]
            for history in (15, 30, 60):
                for representation in ("current", "history"):
                    recipe = copy.deepcopy(winner)
                    recipe.update(
                        history=history,
                        representation=representation,
                        normalization="causal",
                    )
                    records.append(evaluate_recipe(store, frame, window, recipe))
        else:
            families = (
                ("mlp", "residual", "tabm", "transformer")
                if experiment == "exp03"
                else ("gru", "tcn")
            )
            for family in families:
                recipe = base_recipe(family)
                if family in ("gru", "tcn"):
                    recipe["history"] = 30
                records.append(evaluate_recipe(store, frame, window, recipe))
                records += optimize(
                    store,
                    frame,
                    window,
                    family,
                    [family],
                    BUDGET["neural_trials"],
                    experiment,
                )
            if experiment == "exp04":
                for family in ("mlp", "hist"):
                    for history in (0, 15, 30, 60):
                        recipe = base_recipe(family)
                        recipe.update(
                            history=history,
                            representation="history" if history else "current",
                        )
                        records.append(evaluate_recipe(store, frame, window, recipe))
        records = unique_records(records)
        by_family = {}
        for record in records:
            group = by_family.setdefault(record["recipe"]["family"], [])
            if len(group) < 2:
                group.append(record)
        verified = [
            evaluate_recipe(store, frame, window, record["recipe"], repeats=(0, 1))
            for group in by_family.values()
            for record in group
            if record["recipe"]["family"] != "dummy"
        ]
        controls = []
        if experiment in ("exp02", "exp04"):
            current = [
                r
                for r in records
                if r["recipe"]["history"] == 0 and r["recipe"]["family"] != "dummy"
            ]
            history = [
                r
                for r in records
                if r["recipe"]["representation"] == "history"
                or r["recipe"]["family"] in ("gru", "tcn")
            ]
            for group in (current, history):
                if group:
                    controls.append(
                        evaluate_recipe(
                            store,
                            frame,
                            window,
                            min(group, key=rank_key)["recipe"],
                            repeats=(0, 1),
                        )
                    )
            verified += controls
        records = unique_records(records + verified)
        shortlist = unique_records(unique_records(verified)[:8] + controls)
        if not shortlist:
            raise ValueError(
                "Tidak ada kandidat selesai; eksperimen tidak dapat dinyatakan complete."
            )
        shortlists[window] = shortlist
        all_records.extend(records)
        store.publish("validation_leaderboard.csv", csv_bytes(leaderboard(all_records)))
        store.runtime_summary()
        print("\nWindow", window, ": kandidat teratas dari CV internal")
        display(leaderboard(shortlist).head(8).round(4))
    return shortlists
