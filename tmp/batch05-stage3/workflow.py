def stage3_monitor(frame, monitors, context, pool, held):
    record = monitors[context]
    if not record["ready"]:
        raise ValueError("Monitor persiapan belum siap.")
    lookup = {name: i for i, name in enumerate(frame.window_id)}
    try:
        fitting = np.array([lookup[v] for v in record["fit_window_ids"]], int)
        monitor = np.array([lookup[v] for v in record["monitor_window_ids"]], int)
    except KeyError as exc:
        raise ValueError("ID monitor persiapan tidak tersedia.") from exc
    if not set(np.r_[fitting, monitor]).issubset(set(pool)):
        raise ValueError("Monitor keluar dari fitting pool.")
    validate_neural_roles(frame, fitting, monitor, held)
    return fitting, monitor


def select_stage3_candidate(candidates):
    eligible = [
        c for c in candidates if c["finite_predictions"] and min(c["recall"]) > 0
    ]
    source_order = {"full": 0, "matched": 1, "neural": 2}
    return (
        min(
            eligible,
            key=lambda c: (
                -c["macro_f1"],
                -c["accuracy"],
                -min(c["recall"]),
                c["fold_std"],
                c["estimators"],
                NORMALIZATIONS.index(c["normalization"]),
                source_order[c["source"]],
                CHECKPOINT_POLICIES.index(c["policy"])
                if c["source"] == "neural"
                else 0,
            ),
        )
        if eligible
        else None
    )


def assemble_stage3_candidate(
    frame, train, fold_tables, name, source, normalization, policy=None
):
    records, fold_scores = [], []
    for fold, table in enumerate(fold_tables):
        score = classification_metrics(
            table.target.to_numpy(), table.prediction.to_numpy()
        )
        fold_scores.append(score["macro_f1"])
        records.append(table.assign(fold=fold))
    table = pd.concat(records, ignore_index=True)
    if table.window_id.duplicated().any() or set(table.window_id) != set(
        frame.iloc[train].window_id
    ):
        raise ValueError("Cakupan OOF kandidat tidak lengkap/duplikat.")
    table = table.set_index("window_id").loc[frame.iloc[train].window_id].reset_index()
    verify_prediction_table(frame, train, table)
    table_probabilities(table)
    metrics = classification_metrics(
        table.target.to_numpy(), table.prediction.to_numpy()
    )
    candidate = {
        "name": name,
        "source": source,
        "normalization": normalization,
        "policy": policy,
        "estimators": 3 if source == "neural" else 1,
        "finite_predictions": True,
        "fold_std": float(np.std(fold_scores)),
        "fold_macro_f1": fold_scores,
        **metrics,
    }
    candidate["eligible"] = min(metrics["recall"]) > 0
    return candidate, table.assign(candidate=name)


def stage3_graphs(
    store, leaderboard, epoch_choices, histories, winner, validation, test
):
    fig, ax = plt.subplots(figsize=(11, 6))
    ordered = leaderboard.sort_values("macro_f1")
    ax.barh(
        ordered.name,
        ordered.macro_f1,
        color=["#4768b5" if s == "neural" else "#8a6534" for s in ordered.source],
    )
    ax.set(
        xlim=(0, 1), xlabel="OOF Macro-F1", title="Neural dan kontrol tabular, 25 detik"
    )
    save_figure(store, "stage3_oof.png", fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, normalization in zip(axes, NORMALIZATIONS):
        subset = histories[histories.normalization.eq(normalization)]
        for column in ("fit_loss", "monitor_loss"):
            grouped = subset.groupby("epoch")[column]
            mean, std = grouped.mean(), grouped.std().fillna(0)
            ax.plot(mean.index, mean, label=column)
            ax.fill_between(mean.index, mean - std, mean + std, alpha=0.15)
        ax.set(title=normalization, xlabel="Epoch", ylabel="Cross-entropy")
        ax.legend()
    save_figure(store, "training_monitor_curves.png", fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for policy in CHECKPOINT_POLICIES:
        subset = epoch_choices[epoch_choices.policy.eq(policy)]
        ax.scatter(np.arange(len(subset)), subset.epoch, label=policy, alpha=0.6, s=20)
    ax.set(
        xlabel="Urutan kombinasi fold/seed/normalisasi",
        ylabel="Epoch terpilih",
        title="Pemilihan checkpoint",
    )
    ax.legend()
    save_figure(store, "checkpoint_epochs.png", fig)
    for name, metrics in (("validation", validation), ("test_reference", test)):
        fig, ax = plt.subplots(figsize=(6, 4))
        matrix = np.asarray(metrics["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    str(matrix[i, j]),
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black",
                )
        ax.set(
            xticks=range(3),
            yticks=range(3),
            xticklabels=LABELS,
            yticklabels=LABELS,
            xlabel="Prediksi",
            ylabel="Target",
            title=f"{name}: {winner['name']}",
        )
        save_figure(store, f"confusion_{name}.png", fig)


def stage3_results(store, leaderboard, winner, validation, test, epoch_choices):
    lines = [
        "# Hasil Tahap 3",
        "",
        "Window 25 detik, 10 fitur core, split dan monitor tahap 1.",
        "MLP menggunakan tiga seed dan tiga normalisasi; checkpoint loss, Macro-F1, dan epoch 120 berasal dari lintasan training yang sama.",
        "",
        "## Perbandingan OOF",
        "",
        "| Kandidat | Accuracy | Macro-F1 | Recall Low/Medium/High | Layak |",
        "|---|---:|---:|---|---|",
    ]
    for r in leaderboard.sort_values("macro_f1", ascending=False).to_dict("records"):
        recalls = "/".join(f"{r['recall_' + label]:.2%}" for label in LABELS)
        lines.append(
            f"| {r['name']} | {r['accuracy']:.2%} | {r['macro_f1']:.2%} | {recalls} | {r['eligible']} |"
        )
    lines += [
        "",
        "## Model terkunci",
        "",
        f"Kandidat terpilih melalui OOF: **{winner['name']}**.",
        f"Validation accuracy {validation['accuracy']:.2%}, Macro-F1 {validation['macro_f1']:.2%}.",
        f"Test referensi accuracy {test['accuracy']:.2%}, Macro-F1 {test['macro_f1']:.2%}.",
        "Test sudah pernah dianalisis pada rangkaian sebelumnya. Hasilnya merupakan referensi eksploratif, bukan pengujian independen baru.",
        "Model, preprocessing, dan bobot dikunci sebelum Test dibuka. Tidak ada penggantian pemenang berdasarkan validation/Test.",
        "",
        "| Kelas | Recall validation | Recall Test referensi | F1 Test referensi |",
        "|---|---:|---:|---:|",
    ]
    for i, label in enumerate(LABELS):
        lines.append(
            f"| {label} | {validation['recall'][i]:.2%} | {test['recall'][i]:.2%} | {test['f1'][i]:.2%} |"
        )
    lines += [
        "",
        "## Dukungan fitting dan checkpoint",
        "",
        "Matched-data Extra Trees memakai baris optimizer-fitting yang sama dengan MLP. Baseline penuh memakai seluruh fitting pool CV, seperti tahap 2.",
        "Fitting akhir neural/matched memakai 2.121 window; monitor 501 window. Baseline penuh memakai 2.724 window.",
        "Neural tidak di-refit dengan median epoch CV. Checkpoint akhir dipilih dari monitor akhir dengan aturan yang telah ditetapkan.",
        "",
        "| Aturan | Median epoch CV | Epoch minimum | Epoch maksimum |",
        "|---|---:|---:|---:|",
    ]
    for policy, rows in epoch_choices.groupby("policy", sort=False):
        lines.append(
            f"| {policy} | {rows.epoch.median():.1f} | {rows.epoch.min()} | {rows.epoch.max()} |"
        )
    lines += [
        "",
        "Epoch di atas merupakan deskripsi checkpoint, bukan durasi refit model akhir.",
        "",
        "## Artefak",
        "",
        "cv_leaderboard.csv, seed_fold_metrics.csv, checkpoint_choices.csv, history.csv, validation_metrics.csv, test_reference.json, dan model_lock.json memuat bukti pelaksanaan.",
        "Grafik tersedia pada figures. Prediksi/probabilitas dan checkpoint per pekerjaan berada pada jobs.",
        f"Pekerjaan fitting unik selesai: {len(store.completed)} dari batas 72. Tidak ada pencarian atau perluasan batch otomatis.",
    ]
    return "\n".join(lines) + "\n"


def run_stage3(
    dataset_root,
    output_root,
    development,
    reference,
    expected_hashes,
    manifest_hash,
    monitors,
    monitor_hash,
    code_hash,
    resume_dir=None,
):
    if digest(monitors) != monitor_hash:
        raise ValueError("Snapshot monitor berubah.")
    data = locked_data(
        dataset_root, development, reference, expected_hashes, manifest_hash
    )
    frame, train, validation = data["frame"], data["train"], data["validation"]
    roles = {
        f"cv_{i}": stage3_monitor(frame, monitors, f"cv_{i}", fitting, held)
        for i, (fitting, held) in enumerate(data["folds"])
    }
    roles["final"] = stage3_monitor(frame, monitors, "final", train, validation)
    device = stage3_device()
    signature = {
        "stage": 3,
        "version": STAGE3_VERSION,
        "code_sha256": code_hash,
        "dataset_sha256": expected_hashes,
        "manifest_sha256": manifest_hash,
        "monitor_sha256": monitor_hash,
        "neural": NEURAL_CONFIG,
        "seeds": list(NEURAL_SEEDS),
        "policies": list(CHECKPOINT_POLICIES),
        "normalizations": list(NORMALIZATIONS),
        "tabular": MODEL_PARAMETERS["extra_trees"],
        "maximum_neural_jobs": 48,
        "maximum_tabular_jobs": 24,
        "maximum_total_jobs": 72,
        "device": str(device),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
        "python": platform.python_version(),
        "environment": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pandas",
                "scikit-learn",
                "joblib",
                "matplotlib",
                "threadpoolctl",
            )
        },
    }
    store = Stage3Store(output_root, signature, resume_dir)
    print("Run tahap 3:", store.remote, flush=True)
    save_json(store.remote / "development_manifest.json", development)
    save_json(store.remote / "monitors.json", monitors)
    store.progress("starting")
    seed_rows, epochs, history_tables, fold_rows = [], [], [], []
    candidates, oof_tables, candidate_tables = [], [], {}
    try:
        for normalization in NORMALIZATIONS:
            for number, (_, held) in enumerate(data["folds"]):
                fitting, monitor = roles[f"cv_{number}"]
                for seed in NEURAL_SEEDS:
                    key = f"nn_{normalization}_f{number}_s{seed}"
                    record = neural_job(
                        frame, fitting, monitor, held, normalization, seed, store, key
                    )
                    history_tables.append(
                        pd.read_csv(store.get(key) / "history.csv").assign(
                            normalization=normalization, fold=number, seed=seed
                        )
                    )
                    for policy, chosen in record["policies"].items():
                        epochs.append(
                            {
                                "normalization": normalization,
                                "fold": number,
                                "seed": seed,
                                "policy": policy,
                                "epoch": chosen["epoch"],
                                "monitor_loss": chosen["monitor_loss"],
                                "monitor_macro_f1": chosen["monitor_metrics"][
                                    "macro_f1"
                                ],
                                "monitor_min_recall": min(
                                    chosen["monitor_metrics"]["recall"]
                                ),
                            }
                        )
                        seed_rows.append(
                            metric_row(
                                chosen["held_metrics"],
                                "held_fold",
                                normalization=normalization,
                                policy=policy,
                                seed=seed,
                                fold=number,
                                selected_epoch=chosen["epoch"],
                            )
                        )
                stage3_tree_job(
                    frame,
                    fitting,
                    held,
                    normalization,
                    store,
                    f"tree_matched_{normalization}_f{number}",
                )
            for policy in CHECKPOINT_POLICIES:
                tables = []
                for number, (_, held) in enumerate(data["folds"]):
                    arrays = []
                    for seed in NEURAL_SEEDS:
                        table = pd.read_csv(
                            store.get(f"nn_{normalization}_f{number}_s{seed}")
                            / f"predictions_{policy}.csv"
                        )
                        verify_prediction_table(frame, held, table)
                        arrays.append(table_probabilities(table))
                    tables.append(
                        probability_table(frame, held, average_probabilities(arrays))
                    )
                candidate, table = assemble_stage3_candidate(
                    frame,
                    train,
                    tables,
                    f"mlp__{normalization}__{policy}",
                    "neural",
                    normalization,
                    policy,
                )
                candidates.append(candidate)
                oof_tables.append(table)
                candidate_tables[candidate["name"]] = tables
            tables = [
                pd.read_csv(
                    store.get(f"tree_matched_{normalization}_f{i}") / "predictions.csv"
                )
                for i in range(5)
            ]
            candidate, table = assemble_stage3_candidate(
                frame,
                train,
                tables,
                "extra_trees_matched__" + normalization,
                "matched",
                normalization,
            )
            candidates.append(candidate)
            oof_tables.append(table)
            candidate_tables[candidate["name"]] = tables
        for number, (fitting, held) in enumerate(data["folds"]):
            stage3_tree_job(
                frame, fitting, held, "session_robust", store, f"tree_full_f{number}"
            )
        tables = [
            pd.read_csv(store.get(f"tree_full_f{i}") / "predictions.csv")
            for i in range(5)
        ]
        candidate, table = assemble_stage3_candidate(
            frame,
            train,
            tables,
            "extra_trees_full__session_robust",
            "full",
            "session_robust",
        )
        candidates.append(candidate)
        oof_tables.append(table)
        candidate_tables[candidate["name"]] = tables
        winner = select_stage3_candidate(candidates)
        best_neural = select_stage3_candidate(
            [c for c in candidates if c["source"] == "neural"]
        )
        save_json(
            store.remote / "selection.json",
            {
                "winner": winner,
                "best_neural": best_neural,
                "candidates": candidates,
                "selection_uses_validation": False,
                "selection_uses_test": False,
            },
        )
        leaderboard = pd.DataFrame(
            [
                metric_row(
                    c,
                    "oof",
                    name=c["name"],
                    source=c["source"],
                    normalization=c["normalization"],
                    policy=c["policy"],
                    fold_std=c["fold_std"],
                    eligible=c["eligible"],
                )
                for c in candidates
            ]
        )
        for candidate in candidates:
            for number, table in enumerate(candidate_tables[candidate["name"]]):
                fold_rows.append(
                    metric_row(
                        classification_metrics(
                            table.target.to_numpy(), table.prediction.to_numpy()
                        ),
                        "held_fold",
                        name=candidate["name"],
                        fold=number,
                    )
                )
        epoch_table, history_table = (
            pd.DataFrame(epochs),
            pd.concat(history_tables, ignore_index=True),
        )
        write_frame(store, "cv_leaderboard.csv", leaderboard)
        write_frame(store, "fold_metrics.csv", pd.DataFrame(fold_rows))
        write_frame(store, "seed_fold_metrics.csv", pd.DataFrame(seed_rows))
        write_frame(store, "checkpoint_choices.csv", epoch_table)
        write_frame(store, "history.csv", history_table)
        write_frame(
            store, "oof_predictions.csv", pd.concat(oof_tables, ignore_index=True)
        )
        if winner is None:
            save_json(
                store.remote / "status.json",
                {
                    "status": "complete",
                    "reason": "no_eligible_candidate",
                    "test_evaluated": False,
                },
            )
            return {"run_dir": store.remote, "leaderboard": leaderboard, "winner": None}
        final_descriptors = {}
        fitting, monitor = roles["final"]
        if best_neural is not None:
            normalization = best_neural["normalization"]
            keys = [f"final_nn_{normalization}_s{seed}" for seed in NEURAL_SEEDS]
            for seed, key in zip(NEURAL_SEEDS, keys):
                neural_job(
                    frame, fitting, monitor, validation, normalization, seed, store, key
                )
            for policy in CHECKPOINT_POLICIES:
                name = f"mlp__{normalization}__{policy}"
                final_descriptors[name] = {
                    "kind": "neural",
                    "jobs": keys,
                    "name": name,
                    "normalization": normalization,
                    "policy": policy,
                }
        for normalization in NORMALIZATIONS:
            key = "final_tree_matched_" + normalization
            stage3_tree_job(
                frame, fitting, validation, normalization, store, key, save_model=True
            )
            name = "extra_trees_matched__" + normalization
            final_descriptors[name] = {
                "kind": "tree",
                "jobs": [key],
                "name": name,
                "normalization": normalization,
            }
        stage3_tree_job(
            frame,
            train,
            validation,
            "session_robust",
            store,
            "final_tree_full",
            save_model=True,
        )
        name = "extra_trees_full__session_robust"
        final_descriptors[name] = {
            "kind": "tree",
            "jobs": ["final_tree_full"],
            "name": name,
            "normalization": "session_robust",
        }
        validation_rows, validation_scores = [], {}
        for name, descriptor in final_descriptors.items():
            p = predict_stage3_descriptor(frame, validation, store, descriptor)
            table = probability_table(frame, validation, p)
            validation_scores[name] = classification_metrics(
                table.target.to_numpy(), table.prediction.to_numpy()
            )
            validation_rows.append(
                metric_row(
                    validation_scores[name],
                    "validation",
                    name=name,
                    selected=name == winner["name"],
                )
            )
            if name == winner["name"]:
                write_frame(store, "validation_predictions.csv", table)
        write_frame(store, "validation_metrics.csv", pd.DataFrame(validation_rows))
        descriptor = final_descriptors[winner["name"]]
        seal_stage3_model(store, descriptor)
        test = evaluate_stage3_test(frame, store)
        final_validation = validation_scores[winner["name"]]
        breakdown = []
        fit_ids = train if winner["source"] == "full" else fitting
        supported = set(zip(frame.iloc[fit_ids].session_id, frame.iloc[fit_ids]._y))
        for partition, filename in (
            ("validation", "validation_predictions.csv"),
            ("test_reference", "test_reference_predictions.csv"),
        ):
            table = pd.read_csv(store.remote / filename)
            table["correct"] = table.target.eq(table.prediction)
            table["source_supported"] = [
                (s, y) in supported for s, y in zip(table.session_id, table.target)
            ]
            for grouping in ("cognitive_load_label", "study_phase", "source_supported"):
                for value, rows in table.groupby(grouping):
                    breakdown.append(
                        {
                            "partition": partition,
                            "group": grouping,
                            "value": str(value),
                            "rows": len(rows),
                            "correct": int(rows.correct.sum()),
                            "accuracy": float(rows.correct.mean()),
                        }
                    )
        write_frame(store, "error_breakdown.csv", pd.DataFrame(breakdown))
        stage3_graphs(
            store,
            leaderboard,
            epoch_table,
            history_table,
            winner,
            final_validation,
            test["metrics"],
        )
        (store.remote / "RESULTS.md").write_text(
            stage3_results(
                store,
                leaderboard,
                winner,
                final_validation,
                test["metrics"],
                epoch_table,
            ),
            encoding="utf-8",
        )
        verified = store.audit()
        adequate = (
            min(final_validation["recall"]) > 0 and min(test["metrics"]["recall"]) > 0
        )
        save_json(
            store.remote / "status.json",
            {
                "status": "complete",
                "stage": 3,
                "fit_jobs": len(store.completed),
                "neural_jobs": sum("nn_" in k for k in store.completed),
                "tabular_jobs": sum("tree_" in k for k in store.completed),
                "verified_artifacts": verified,
                "winner": winner["name"],
                "recognizes_three_classes_on_validation_and_test": adequate,
                "test_evaluated": True,
                "test_role": "previously_seen_exploratory_reference",
            },
        )
        store.progress("complete", "complete", test_evaluated=True)
        print("Hasil tahap 3:", store.remote / "RESULTS.md", flush=True)
        return {
            "run_dir": store.remote,
            "leaderboard": leaderboard,
            "winner": winner,
            "validation": final_validation,
            "test": test,
            "best_neural": best_neural,
        }
    except Exception as exc:
        save_json(
            store.remote / "status.json",
            {"status": "failed", "stage": 3, "error": f"{type(exc).__name__}: {exc}"},
        )
        raise
