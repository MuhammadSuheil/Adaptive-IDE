def model_schema(model):
    if model["kind"] in ("vote", "average", "stack"):
        return {
            "kind": model["kind"],
            "components": [model_schema(m) for m in model["models"]],
        }
    return {
        "kind": model["kind"],
        "recipe": model["recipe"],
        "features": GROUPS[model["recipe"]["group"]],
        "target": "cognitive_load_label",
        "classes": LABELS,
        "context_rule": "same_session_contiguous_5s_no_future",
    }


def fitted_candidate(store, frame, window, candidate, seed):
    key = f"fit_w{window}_{candidate['id']}_s{seed}"
    record = store.read(key)
    if record is None:
        fit_started = time.perf_counter()
        io_before = dict(store.io_seconds)
        train = np.flatnonzero(frame._split.eq("train"))
        val = np.flatnonzero(frame._split.eq("validation"))
        progress(
            store,
            stage="fit_full_train",
            window=window,
            recipe=candidate["id"],
            seed=seed,
            family=candidate["recipe"]["family"],
            started_at=datetime.now(UTC).isoformat(),
            status="running",
        )
        with FitHeartbeat(
            f"full_train window={window}s {candidate['recipe']['family']} recipe={candidate['id']} seed={seed} fit_rows={len(train)}"
        ):
            model = fit_model(
                candidate["recipe"], frame, train, seed=seed, epochs=candidate["epochs"]
            )
        started = time.perf_counter()
        output = predict_model(model, frame, val)
        prediction_seconds = time.perf_counter() - started
        payload = pack_model(model)
        restored = unpack_model(payload)
        np.testing.assert_array_equal(
            output["labels"], predict_model(restored, frame, val)["labels"]
        )
        training_accuracy = model.get("training_accuracy")
        if training_accuracy is None:
            training_accuracy = metrics(
                frame.iloc[train]._y, predict_model(model, frame, train)["labels"]
            )["accuracy"]
        record = {
            "id": key,
            "window": window,
            "recipe": candidate["recipe"],
            "validation": metrics(frame.iloc[val]._y, output["labels"]),
            "accuracy_std": candidate["accuracy_std"],
            "training_accuracy": training_accuracy,
            "seed": seed,
            "complexity": candidate["complexity"],
            "epochs": candidate["epochs"],
            "runtime": {
                **model.get("runtime", {}),
                "validation_predict_seconds": prediction_seconds,
                "checkpoint_read_seconds": store.io_seconds["read_seconds"]
                - io_before["read_seconds"],
                "compute_seconds": time.perf_counter() - fit_started,
            },
            "convergence": model.get("convergence", []),
        }
        checkpoint_started = time.perf_counter()
        store.commit(
            key,
            record,
            {
                "model.pkl": payload,
                "schema.json": canonical(model_schema(model)).encode(),
            },
        )
        store.commit(
            f"timing_{key}",
            {
                "checkpoint": key,
                "write_seconds": time.perf_counter() - checkpoint_started,
                "total_seconds": time.perf_counter() - fit_started,
            },
        )
    return record, unpack_model(store.artifact(key, "model.pkl"))


def save_selection_candidate(store, frame, window, key, model, base_record):
    cached = store.read(key)
    if cached is not None:
        return cached
    validation = np.flatnonzero(frame._split.eq("validation"))
    output = predict_model(model, frame, validation)
    recipe = copy.deepcopy(base_record["recipe"])
    record = {
        "id": key,
        "window": window,
        "recipe": recipe,
        "kind": model["kind"],
        "validation": metrics(frame.iloc[validation]._y, output["labels"]),
        "accuracy_std": base_record.get("accuracy_std", 0),
        "complexity": base_record.get("complexity", 0),
    }
    rows = frame.iloc[validation][
        ["window_id", "session_id", "cognitive_load_label"]
    ].reset_index(drop=True)
    rows["prediction"] = np.asarray(LABELS)[output["labels"]]
    store.commit(
        key,
        record,
        {
            "model.pkl": pack_model(model),
            "predictions.csv": csv_bytes(rows),
            "schema.json": canonical(model_schema(model)).encode(),
        },
    )
    return record


def select_models(store, views, shortlists, experiment):
    previous = store.read("selection")
    if previous is not None:
        return previous
    selected, selection_records = {}, []
    seed_records = []
    for window, candidates in shortlists.items():
        frame = views[window]
        final_records, bases = [], []
        for candidate in candidates:
            first, model = fitted_candidate(store, frame, window, candidate, 42)
            seed_records.append(first)
            key = f"selection_w{window}_{candidate['id']}_single"
            final_records.append(
                save_selection_candidate(store, frame, window, key, model, first)
            )
            bases.append((candidate, model))
            if candidate["recipe"]["family"] in NEURAL_FAMILIES:
                models = [model]
                for seed in FINAL_SEEDS[1:]:
                    seed_record, other = fitted_candidate(
                        store, frame, window, candidate, seed
                    )
                    seed_records.append(seed_record)
                    models.append(other)
                ensemble = {"kind": "average", "models": models}
                key = f"selection_w{window}_{candidate['id']}_seeds"
                final_records.append(
                    save_selection_candidate(store, frame, window, key, ensemble, first)
                )
        diverse, seen = [], set()
        for candidate, model in bases:
            family = candidate["recipe"]["family"]
            if family not in seen:
                diverse.append((candidate, model))
                seen.add(family)
            if len(diverse) == 3:
                break
        if len(diverse) >= 2:
            models = [model for _, model in diverse]
            for kind in ("vote", "average"):
                validation = np.flatnonzero(frame._split.eq("validation"))
                if kind == "average" and not all(
                    predict_model(m, frame, validation)["score_type"] == "probability"
                    for m in models
                ):
                    continue
                ensemble = {"kind": kind, "models": models}
                key = f"selection_w{window}_{kind}"
                final_records.append(
                    save_selection_candidate(
                        store, frame, window, key, ensemble, diverse[0][0]
                    )
                )
            arrays, indices = [], None
            for candidate, _ in diverse:
                key = f"cv_w{window}_{candidate['id']}_n2"
                rows = [
                    r
                    for r in json.loads(store.artifact(key, "oof.json"))
                    if r["repeat"] == 0
                ]
                rows = sorted(rows, key=lambda r: r["row"])
                current = np.array([r["row"] for r in rows])
                if indices is not None:
                    np.testing.assert_array_equal(indices, current)
                indices = current
                arrays.append(np.array([r["scores"] for r in rows]))
            meta = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(C=0.1, max_iter=2000)),
                ]
            )
            meta.fit(np.concatenate(arrays, axis=1), frame.iloc[indices]._y)
            stack = {"kind": "stack", "models": models, "meta": meta}
            final_records.append(
                save_selection_candidate(
                    store,
                    frame,
                    window,
                    f"selection_w{window}_stack",
                    stack,
                    diverse[0][0],
                )
            )
        final_records = sorted(final_records, key=rank_key)
        winner = final_records[0]
        roles = {"winner": winner["id"]}
        dummy = evaluate_recipe(store, frame, window, base_recipe("dummy"))
        dummy_record, dummy_model = fitted_candidate(store, frame, window, dummy, 42)
        dummy_key = f"selection_w{window}_dummy"
        save_selection_candidate(
            store, frame, window, dummy_key, dummy_model, dummy_record
        )
        roles["dummy_control"] = dummy_key
        if experiment in ("exp02", "exp04"):
            for role, predicate in (
                (
                    "current_control",
                    lambda r: (
                        r.get("kind") in ("tabular", "neural")
                        and r["recipe"]["representation"] == "current"
                        and r["recipe"]["history"] == 0
                    ),
                ),
                (
                    "history_control",
                    lambda r: (
                        r.get("kind") in ("tabular", "neural")
                        and (
                            r["recipe"]["representation"] == "history"
                            or r["recipe"]["family"] in ("gru", "tcn")
                        )
                    ),
                ),
            ):
                matches = [r for r in final_records if predicate(r)]
                if matches:
                    roles[role] = matches[0]["id"]
        if experiment == "exp04":
            sequences = [
                r
                for r in final_records
                if r.get("kind") == "neural" and r["recipe"]["family"] in ("gru", "tcn")
            ]
            if sequences:
                roles["sequence_control"] = sequences[0]["id"]
        selected[str(window)] = {"winner": winner, "roles": roles}
        selection_records.extend(final_records)
    primary = min((d["winner"] for d in selected.values()), key=rank_key)
    lock = {
        "windows": selected,
        "primary": primary["id"],
        "selection_rule": "validation_accuracy_macro_f1_min_recall_stability_complexity",
    }
    store.publish(
        "seed_stability.csv",
        csv_bytes(
            pd.DataFrame(
                [
                    {
                        "id": r["id"],
                        "window": r["window"],
                        "seed": r["seed"],
                        **r["validation"],
                    }
                    for r in seed_records
                ]
            )
        ),
    )
    store.publish(
        "selection_leaderboard.csv",
        csv_bytes(
            pd.DataFrame(
                [
                    {
                        "id": r["id"],
                        "window": r["window"],
                        "kind": r.get("kind", ""),
                        "family": r["recipe"]["family"],
                        **r["validation"],
                    }
                    for r in selection_records
                ]
            )
        ),
    )
    store.commit("selection", lock)
    return lock


def evaluate_test(store, views, lock):
    rows = []
    for window, frame in views.items():
        chosen = lock["windows"][str(window)]
        test = np.flatnonzero(frame._split.eq("test"))
        for identifier in dict.fromkeys(chosen["roles"].values()):
            key = "test_" + identifier
            record = store.read(key)
            if record is None:
                selection = store.read(identifier)
                payload = store.artifact(identifier, "model.pkl")
                model = unpack_model(payload)
                output = predict_model(model, frame, test)
                y = frame.iloc[test]._y.to_numpy()
                score = metrics(y, output["labels"])
                record = {
                    "id": identifier,
                    "window": window,
                    "test_rows": len(test),
                    "primary": identifier == lock["primary"],
                    "window_winner": identifier == chosen["winner"]["id"],
                    "validation_accuracy": selection["validation"]["accuracy"],
                    **score,
                    "model_sha256": hashlib.sha256(payload).hexdigest(),
                }
                predictions = frame.iloc[test][
                    [
                        "window_id",
                        "participant_id",
                        "session_id",
                        "window_start",
                        "window_end",
                        "study_phase",
                        "cognitive_load_label",
                    ]
                ].reset_index(drop=True)
                predictions["prediction"] = np.asarray(LABELS)[output["labels"]]
                for i, name in enumerate(LABELS):
                    predictions[f"{output['score_type']}_{name}"] = output["scores"][
                        :, i
                    ]
                report = pd.DataFrame(
                    classification_report(
                        y,
                        output["labels"],
                        labels=[0, 1, 2],
                        target_names=LABELS,
                        zero_division=0,
                        output_dict=True,
                    )
                ).T.reset_index(names="class")
                matrix = confusion_matrix(y, output["labels"], labels=[0, 1, 2])
                fig, axis = plt.subplots(figsize=(5, 4))
                axis.imshow(matrix, cmap="Blues")
                axis.set(
                    xticks=range(3),
                    yticks=range(3),
                    xticklabels=LABELS,
                    yticklabels=LABELS,
                    xlabel="Prediksi",
                    ylabel="Label dataset",
                    title=f"Window {window}s",
                )
                for i in range(3):
                    for j in range(3):
                        axis.text(
                            j,
                            i,
                            str(matrix[i, j]),
                            ha="center",
                            va="center",
                            color="white"
                            if matrix[i, j] > matrix.max() / 2
                            else "black",
                        )
                artifacts = {
                    "predictions.csv": csv_bytes(predictions),
                    "classification_report.csv": csv_bytes(report),
                    "confusion_matrix.csv": csv_bytes(
                        pd.DataFrame(matrix, index=LABELS, columns=LABELS).reset_index()
                    ),
                    "confusion_matrix.png": figure_bytes(fig),
                }
                history_model = model
                while history_model.get("models"):
                    history_model = history_model["models"][0]
                if history_model.get("history"):
                    history = pd.DataFrame(history_model["history"])
                    fig, axis = plt.subplots(figsize=(6, 3))
                    axis.plot(history.epoch, history.train_loss, label="Train loss")
                    if "monitor_loss" in history:
                        axis.plot(
                            history.epoch, history.monitor_loss, label="Monitor loss"
                        )
                    axis.legend()
                    axis.set(xlabel="Epoch", ylabel="Loss")
                    artifacts["learning_curve.png"] = figure_bytes(fig)
                # Kinerja menurut fase dan kecukupan riwayat; tidak memilih model kembali.
                groups = []
                for phase, part in predictions.groupby("study_phase"):
                    yy = part.cognitive_load_label.map(dict(zip(LABELS, range(3))))
                    pp = part.prediction.map(dict(zip(LABELS, range(3))))
                    groups.append(
                        {
                            "group": "phase:" + phase,
                            "rows": len(part),
                            "accuracy": float(accuracy_score(yy, pp)),
                        }
                    )
                for lookback in (15, 30, 60):
                    context = history_indices(frame, lookback)
                    enough = np.array(
                        [len(context[i]) == lookback // 5 + 1 for i in test]
                    )
                    for ready in (False, True):
                        ids = np.flatnonzero(enough == ready)
                        if len(ids):
                            groups.append(
                                {
                                    "group": f"history_{lookback}s_complete={ready}",
                                    "rows": len(ids),
                                    "accuracy": float(
                                        accuracy_score(y[ids], output["labels"][ids])
                                    ),
                                }
                            )
                artifacts["test_groups.csv"] = csv_bytes(pd.DataFrame(groups))
                store.commit(key, record, artifacts)
            rows.append(record)
            print("\nTest:", identifier)
            display(
                pd.read_csv(
                    io.BytesIO(store.artifact(key, "classification_report.csv"))
                ).round(4)
            )
            try:
                from IPython.display import Image
            except ImportError:
                Image = None
            if Image is not None:
                display(Image(data=store.artifact(key, "confusion_matrix.png")))
                if (store.local / key / "learning_curve.png").exists():
                    display(Image(data=store.artifact(key, "learning_curve.png")))
    table = pd.DataFrame(rows)
    store.publish("test_summary.csv", csv_bytes(table))
    winner = table.loc[table.primary].iloc[0]
    summary_lines = ["| Window | Model | Accuracy | Macro-F1 |", "|---|---|---:|---:|"]
    summary_lines += [
        f"| {r.window}s | {r.id} | {r.accuracy:.2%} | {r.macro_f1:.4f} |"
        for r in table.itertuples()
    ]
    text = [
        "# Hasil Eksperimen",
        "",
        f"Model utama pilihan validation: `{winner.id}`.",
        "",
        f"Accuracy test: {winner.accuracy:.2%}. Macro-F1: {winner.macro_f1:.4f}.",
        "",
        "\n".join(summary_lines),
        "",
        "Pemenang dikunci dari validation. Test tidak dipakai untuk mengganti konfigurasi, seed, atau split.",
        "",
    ]
    store.publish("RESULTS.md", "\n".join(text).encode())
    store.runtime_summary()
    store.audit_remote()
    store.commit(
        "complete",
        {
            "test_models": len(table),
            "primary": lock["primary"],
            "finished_at": datetime.now(UTC).isoformat(),
        },
    )
    progress(store, stage="complete", primary=lock["primary"])
    display(table.round(4))
    return table
