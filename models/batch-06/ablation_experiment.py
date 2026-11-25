"""Exp02 definitions, embedded after the unchanged Exp01 helper definitions.

Four fixed representations isolate the transforms on one frozen cohort. This
module uses the shared functions embedded in the preceding notebook cell; it is
not an additional Colab runtime dependency.
"""

import warnings


ABLATION_VERSION = "batch06-exp02-ablation-v1"
ABLATION_MODES = ("raw", "rmssd_only", "eye_only", "combined")
ABLATION_CONTRASTS = (
    ("rmssd_only", "raw"), ("eye_only", "raw"), ("combined", "raw"),
    ("combined", "rmssd_only"), ("combined", "eye_only"),
)


def ablation_representation(frame, references, mode):
    if mode == "raw":
        return representation(frame, references, "hrv_eye", "raw")
    if mode == "rmssd_only":
        return representation(frame, references, "hrv_rmssd", "baseline_relative")
    if mode in ("eye_only", "combined"):
        x = representation(frame, references, "hrv_eye", "baseline_relative")
        if mode == "eye_only":
            x[:, 0] = frame.rmssd_ms.to_numpy(float)
        return x
    raise ValueError(f"Varian ablasi tidak dikenal: {mode}")


def ablation_metrics(frame, prediction):
    scored = metrics(frame._y.to_numpy(), np.asarray(prediction, dtype=int))
    support = np.bincount(frame._y.to_numpy(), minlength=3)
    scored["three_class_evaluable"] = bool((support > 0).all())
    for i, label in enumerate(LABELS):
        scored["support_" + label.lower()] = int(support[i])
        # The fixed-label Macro-F1 above remains comparable to Exp01. Recall
        # itself is unavailable, rather than zero, when there is no support.
        scored["recall_" + label.lower()] = scored["recall"][i] if support[i] else None
    per_person = []
    for person in sorted(frame.participant_id.unique()):
        mask = frame.participant_id.eq(person).to_numpy()
        per_person.append(metrics(frame.loc[mask, "_y"].to_numpy(), np.asarray(prediction)[mask]))
    scored["participants"] = int(len(per_person))
    scored["mean_participant_accuracy"] = float(np.mean([r["accuracy"] for r in per_person]))
    scored["mean_participant_macro_f1"] = float(np.mean([r["macro_f1"] for r in per_person]))
    return scored


def ablation_metric_row(frame, prediction, protocol, mode, family="extra_trees"):
    scored = ablation_metrics(frame, prediction)
    return {
        "protocol": protocol, "representation": mode, "family": family,
        **{key: value for key, value in scored.items() if key not in ("recall", "confusion_matrix")},
    }


def prepare_ablation(dataset_root, snapshot):
    views, hashes, paths = load_views(dataset_root)
    if hashes != snapshot["dataset_sha256"]:
        raise ValueError(
            "Hash CSV Drive berbeda dari run Exp01. Exp02 memerlukan data yang sama "
            "untuk kontrol baris/fold identik; siapkan manifest eksperimen baru "
            "jika dataset memang diperbarui, jangan mengganti split otomatis."
        )
    frame = views[25]
    train_ids, validation_ids = snapshot["train_window_ids"], snapshot["validation_window_ids"]
    if len(set(train_ids)) != len(train_ids) or len(set(validation_ids)) != len(validation_ids):
        raise ValueError("ID snapshot berulang.")
    if set(train_ids) & set(validation_ids):
        raise ValueError("Train dan validation snapshot bertabrakan.")
    reserved = set(snapshot["test_window_ids"])
    if reserved & (set(train_ids) | set(validation_ids)):
        raise ValueError("ID Test masuk pembanding.")
    indexed = frame.set_index("window_id", drop=False)
    train = indexed.loc[train_ids].reset_index(drop=True)
    validation = indexed.loc[validation_ids].reset_index(drop=True)
    refs = snapshot["baseline_references"]
    evidence = snapshot["source_evidence"]
    if set(train.participant_id) != set(snapshot["participants"]):
        raise ValueError("Peserta Train berubah dari snapshot.")
    if set(train._y) != {0, 1, 2}:
        raise ValueError("Train tidak memuat tiga kelas.")
    # Audit the same references against the values in the Drive CSV. No reference
    # can be selected anew after viewing ablation scores.
    for sid, ref in refs.items():
        source = evidence[sid]
        if int(source["baseline_end_ns"]) > int(source["first_task_ns"]):
            raise ValueError("Baseline RMSSD bertabrakan dengan tugas.")
        rows = frame[frame.session_id.eq(sid)]
        if not bool_values(rows.baseline_usable).all() or not rows.baseline_coverage.ge(0.8 - 1e-9).all():
            raise ValueError("Baseline RMSSD tidak layak pada CSV.")
        if not np.allclose(rows.baseline_rmssd_ms, ref["rmssd"], rtol=1e-9, atol=1e-8):
            raise ValueError("Baseline RMSSD berubah.")
        initial = indexed.loc[ref["eye_window_id"]]
        if initial.study_phase != "pre_task" or int(initial._end) >= int(source["first_task_ns"]):
            raise ValueError("Acuan eye bukan window sebelum tugas pertama.")
        if int(initial._start) < int(source["baseline_end_ns"]):
            raise ValueError("Acuan eye mendahului penyelesaian baseline recorder.")
        if int(initial._end) != int(ref["eye_available_ns"]):
            raise ValueError("Waktu akhir acuan eye berubah.")
        for col in CORE:
            if not np.isclose(initial[col], ref["eye_values"][col], rtol=1e-9, atol=1e-8):
                raise ValueError(f"Nilai acuan eye berubah: {col}")
        for part in (train, validation):
            subset = part[part.session_id.eq(sid)]
            if not subset.participant_id.eq(source["participant_id"]).all():
                raise ValueError("Grup peserta berbeda dari bukti sumber.")
            available = max(int(ref["hrv_available_ns"]), int(ref["eye_available_ns"]))
            if not subset._start.ge(available + 10**9).all():
                raise ValueError("Window supervised bertabrakan dengan acuan awal.")
    if set(pd.concat([train, validation]).session_id) != set(refs):
        raise ValueError("Sumber pembanding tidak cocok dengan baseline.")
    position = dict(zip(train.window_id, range(len(train))))
    folds = {}
    for protocol, cases in snapshot["folds"].items():
        folds[protocol] = []
        covered = []
        for case in cases:
            if len(set(case["fit_window_ids"])) != len(case["fit_window_ids"]) or len(set(case["held_window_ids"])) != len(case["held_window_ids"]):
                raise ValueError("ID fold berulang.")
            fit = np.array([position[key] for key in case["fit_window_ids"]], dtype=int)
            held = np.array([position[key] for key in case["held_window_ids"]], dtype=int)
            if set(fit) & set(held) or not len(held) or set(train.iloc[fit]._y) != {0, 1, 2}:
                raise ValueError("Fitting/held-out fold tidak sah.")
            if protocol == "participant_held_out":
                if set(train.iloc[fit].participant_id) & set(train.iloc[held].participant_id):
                    raise ValueError("Peserta bocor ke fitting.")
                if train.iloc[held].participant_id.nunique() != 1:
                    raise ValueError("Fold utama harus menahan tepat satu orang.")
            elif protocol == "historical_block_cv":
                for sid in set(train.iloc[fit].session_id) & set(train.iloc[held].session_id):
                    a = train.iloc[fit].loc[lambda f: f.session_id.eq(sid), "_end"].to_numpy()
                    b = train.iloc[held].loc[lambda f: f.session_id.eq(sid), "_end"].to_numpy()
                    if np.abs(a[:, None] - b).min() < 45 * 10**9:
                        raise ValueError("Guard waktu kurang dari 45 detik.")
            else:
                raise ValueError("Protokol snapshot tidak dikenal.")
            covered.extend(held.tolist())
            folds[protocol].append((int(case["fold"]), fit, held))
        if sorted(covered) != list(range(len(train))):
            raise ValueError("OOF tidak mencakup Train tepat sekali.")
    if set(folds) != {"participant_held_out", "historical_block_cv"}:
        raise ValueError("Dua protokol tetap harus tersedia.")
    return {"train": train, "validation": validation, "references": refs, "folds": folds, "hashes": hashes, "paths": paths}


def ablation_predictions(frame, prediction, protocol, mode, fold, family="extra_trees"):
    result = frame[["window_id", "participant_id", "session_id", "study_phase", "cognitive_load_label"]].copy()
    result["predicted_label"] = [LABELS[int(value)] for value in prediction]
    result["protocol"], result["representation"], result["fold"] = protocol, mode, fold
    result["family"] = family
    return result


def ablation_contrasts(summary):
    records = []
    for protocol, rows in summary[summary.family.eq("extra_trees")].groupby("protocol"):
        by_mode = rows.set_index("representation")
        for treatment, control in ABLATION_CONTRASTS:
            a, b = by_mode.loc[treatment], by_mode.loc[control]
            result = {"protocol": protocol, "treatment": treatment, "control": control, "rows": int(a.rows)}
            if a.rows != b.rows or a.participants != b.participants:
                raise ValueError("Kontras tidak memakai endpoint yang sama.")
            for metric in ("accuracy", "macro_f1", "mean_participant_accuracy", "mean_participant_macro_f1"):
                result[metric + "_delta_pp"] = float(100 * (a[metric] - b[metric]))
            records.append(result)
    return pd.DataFrame(records)


def ablation_participant_metrics(predictions):
    records = []
    label_map = dict(zip(LABELS, range(3)))
    for keys, group in predictions.groupby(["protocol", "family", "representation", "participant_id"], sort=True):
        frame = group.assign(_y=group.cognitive_load_label.map(label_map)).reset_index(drop=True)
        prediction = group.predicted_label.map(label_map).to_numpy()
        record = dict(zip(("protocol", "family", "representation", "participant_id"), keys))
        record.update(ablation_metric_row(frame, prediction, keys[0], keys[2], keys[1]))
        records.append(record)
    return pd.DataFrame(records)


def ablation_control_check(summary, snapshot):
    records = []
    for old in snapshot["exp01_controls"]:
        current = summary[summary.protocol.eq(old["protocol"]) & summary.representation.eq(old["representation"]) & summary.family.eq("extra_trees")].iloc[0]
        records.append({
            "protocol": old["protocol"], "representation": old["representation"],
            "exp01_accuracy": old["accuracy"], "exp02_accuracy": float(current.accuracy),
            "exp01_macro_f1": old["macro_f1"], "exp02_macro_f1": float(current.macro_f1),
            "same_metrics": bool(np.isclose(current.accuracy, old["accuracy"], rtol=0, atol=1e-12) and np.isclose(current.macro_f1, old["macro_f1"], rtol=0, atol=1e-12)),
        })
    return pd.DataFrame(records)


def ablation_report(run, summary, validation, contrasts, controls, plan):
    def percent(value):
        return "N/A" if pd.isna(value) else f"{value:.2%}"
    lines = [
        "# Hasil ablasi baseline Batch 06 Exp02", "",
        "Empat representasi memakai peserta, window, fold, dan Extra Trees yang sama dengan subset eye Exp01. "
        "Seluruh fitur tetap ada; yang berubah adalah normalisasi RMSSD dan/atau eye. SDNN tetap mentah. "
        "Test tidak dievaluasi; semua hasil merupakan eksplorasi development pada subset kecil.", "",
        "## OOF peserta yang ditahan dari fitting", "",
        "| Varian | Accuracy window | Macro-F1 tiga kelas | Mean accuracy orang | Mean Macro-F1 orang | Support Low/Medium/High |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary[summary.protocol.eq("participant_held_out")].itertuples():
        lines.append(f"| {row.representation} | {row.accuracy:.2%} | {row.macro_f1:.2%} | {row.mean_participant_accuracy:.2%} | {row.mean_participant_macro_f1:.2%} | {row.support_low}/{row.support_medium}/{row.support_high} |")
    lines.extend(["", "Macro-F1 per orang memakai tiga kelas tetap, termasuk kelas yang tidak hadir pada fold tertentu. "
                  "Rata-rata orang memberi setiap peserta bobot yang sama; accuracy window memberi setiap baris bobot yang sama.", "",
                  "## Kontras yang dibakukan", "",
                  "| Perlakuan dibanding kontrol | Delta accuracy | Delta Macro-F1 | Delta mean accuracy orang |", "|---|---:|---:|---:|"])
    for row in contrasts[contrasts.protocol.eq("participant_held_out")].itertuples():
        lines.append(f"| {row.treatment} vs {row.control} | {row.accuracy_delta_pp:+.2f} pp | {row.macro_f1_delta_pp:+.2f} pp | {row.mean_participant_accuracy_delta_pp:+.2f} pp |")
    lines.extend(["", "`eye_only vs raw` menilai transformasi eye saat RMSSD tetap asli. "
                  "`combined vs rmssd_only` menilai penambahan transformasi eye saat RMSSD sudah relatif. "
                  "Perbedaan keduanya menunjukkan ketergantungan pada representasi lain, bukan efek fisiologis kausal.", "",
                  "## Validation historis", "",
                  "Validation sudah diperiksa pada Exp01 dan tidak merupakan hold-out baru. "
                  "Semua empat varian dibakukan sebelum fitting; tidak ada pemilihan ulang dari validation.", "",
                  "| Varian | Window | Accuracy | Macro-F1 tiga kelas | Recall Low | Recall Medium | Recall High | Support Low/Medium/High |", "|---|---:|---:|---:|---:|---:|---:|---|"])
    for row in validation.itertuples():
        lines.append(f"| {row.representation} | {row.rows} | {row.accuracy:.2%} | {row.macro_f1:.2%} | {percent(row.recall_low)} | {percent(row.recall_medium)} | {percent(row.recall_high)} | {row.support_low}/{row.support_medium}/{row.support_high} |")
    if not validation.three_class_evaluable.all():
        lines.extend(["", "**Validation tidak mencakup ketiga kelas. Recall kelas dengan support nol adalah N/A. "
                      "Macro-F1 tetap dihitung dengan tiga label untuk konsistensi dan memasukkan F1 nol pada kelas tanpa support. "
                      "Accuracy validation tidak boleh dipromosikan sebagai pengujian tiga kelas lengkap atau pengguna baru.**"])
    lines.extend(["", "## Pemeriksaan kontrol Exp01", "",
                  "Raw dan combined dijalankan ulang dalam Exp02 sebagai kontrol internal. "
                  "Kontras selalu dihitung dari hasil Exp02 yang baru, bukan mencampur skor lintas run.", "",
                  "| Protokol | Varian | Metrik sama dengan Exp01 |", "|---|---|---|"])
    for row in controls.itertuples():
        lines.append(f"| {row.protocol} | {row.representation} | {row.same_metrics} |")
    lines.extend(["", "Perbedaan kontrol perlu dibaca bersama versi paket dan numerik runtime. "
                  "Tidak ada seed, parameter, atau fold yang diubah untuk memaksakan kecocokan.", "",
                  "## Batas kesimpulan", "",
                  "Acuan eye adalah pre_task, bukan pengukuran pupil atau baseline istirahat tervalidasi. "
                  "Label tetap difficulty/fase; belum ada rating beban mental independen. Window overlap bukan observasi independen. "
                  "Empat peserta tidak cukup untuk klaim populasi luas; perubahan deskriptif tidak dianggap signifikansi statistik.", "",
                  f"Budget: {plan['cv_fit_budget']} fitting CV dan {plan['validation_fit_budget']} fitting Train penuh. "
                  "Model validation disimpan sebagai empat pembanding, bukan pemenang final menurut validation.", ""])
    (run / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def ablation_figures(run, summary, contrasts, validation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = run / "figures"
    destination.mkdir(exist_ok=True)
    data = summary[(summary.protocol.eq("participant_held_out")) & summary.family.eq("extra_trees")].set_index("representation").loc[list(ABLATION_MODES)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["#68859b", "#b98d40", "#637ac8", "#159b88"]
    for ax, metric, title in ((axes[0], "macro_f1", "OOF window Macro-F1, three fixed labels"), (axes[1], "mean_participant_accuracy", "OOF mean accuracy, equal participant weights")):
        ax.bar(data.index, data[metric] * 100, color=colors)
        ax.set(ylim=(0, 100), ylabel="Percent", title=title)
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(destination / "ablation_oof.png", dpi=170)
    plt.close(fig)
    data = contrasts[contrasts.protocol.eq("participant_held_out")]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(data.treatment + " vs " + data.control, data.macro_f1_delta_pp, color="#159b88")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Macro-F1 change (percentage points)", title="Paired transforms on identical windows and folds")
    fig.tight_layout()
    fig.savefig(destination / "ablation_contrasts.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    row = validation.iloc[0]
    ax.bar(LABELS, [row.support_low, row.support_medium, row.support_high], color=["#68859b", "#c95c58", "#159b88"])
    ax.set(ylabel="Windows", title="Historical validation class support")
    fig.tight_layout()
    fig.savefig(destination / "validation_support.png", dpi=170)
    plt.close(fig)


def run_ablation(dataset_root, output_root, snapshot, implementation_hashes):
    data = prepare_ablation(dataset_root, snapshot)
    train, validation, refs, folds = data["train"], data["validation"], data["references"], data["folds"]
    run = Path(output_root) / f"ablation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    run.mkdir(parents=True, exist_ok=False)
    plan = {
        "version": ABLATION_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_exp01_run": snapshot["source_run"], "snapshot_sha256": object_hash(snapshot),
        "dataset_sha256": data["hashes"], "implementation_sha256": implementation_hashes,
        "features": CORE, "variants": list(ABLATION_MODES), "model": MODELS["extra_trees"],
        "dummy": MODELS["dummy"], "contrasts": list(ABLATION_CONTRASTS),
        "cv_fit_budget": sum(len(cases) for cases in folds.values()) * 5,
        "validation_fit_budget": 4, "test_evaluated": False,
        "search": False, "validation_model_selection": False,
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib", "matplotlib")},
        "python": platform.python_version(),
    }
    save_json(run / "plan.json", plan)
    save_json(run / "cohort_snapshot.json", snapshot)
    save_json(run / "status.json", {"stage": "cv", "complete": False, "test_evaluated": False})
    print("Run:", run, flush=True)
    print("Train:", len(train), "window,", train.participant_id.nunique(), "orang", flush=True)
    val_support = np.bincount(validation._y, minlength=3)
    print("Support validation Low/Medium/High:", val_support.tolist(), flush=True)
    if (val_support == 0).any():
        print("PERHATIAN: validation historis tidak mencakup tiga kelas; recall tanpa support = N/A.", flush=True)
    summaries, fold_rows, tables, fit_count = [], [], [], 0
    for protocol, cases in folds.items():
        for mode in (*ABLATION_MODES, "dummy"):
            family = "dummy" if mode == "dummy" else "extra_trees"
            x = ablation_representation(train, refs, "raw" if mode == "dummy" else mode)
            oof = np.full(len(train), -1, dtype=int)
            for number, fit, held in cases:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    prediction, fitting, _ = fit_predict(x, train._y.to_numpy(), fit, held, family)
                if (oof[held] >= 0).any():
                    raise ValueError("OOF ditulis lebih dari sekali.")
                oof[held] = prediction
                row = ablation_metric_row(train.iloc[held], prediction, protocol, mode, family)
                row.update({"fold": number, "fit_rows": len(fit), "train_accuracy": float(accuracy_score(train.iloc[fit]._y, fitting)), "warnings": [str(w.message) for w in caught]})
                fold_rows.append(row)
                tables.append(ablation_predictions(train.iloc[held], prediction, protocol, mode, number, family))
                save_json(run / "jobs" / f"{protocol}_{mode}_{number}.json", row)
                fit_count += 1
                print(f"CV {fit_count}/{plan['cv_fit_budget']}: {protocol}, {mode}, fold {number}", flush=True)
            if (oof < 0).any():
                raise ValueError("OOF tidak lengkap.")
            summaries.append(ablation_metric_row(train, oof, protocol, mode, family))
    summary = pd.DataFrame(summaries)
    summary.to_csv(run / "cv_leaderboard.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run / "fold_metrics.csv", index=False)
    predictions = pd.concat(tables, ignore_index=True)
    predictions.to_csv(run / "oof_predictions.csv", index=False)
    # All four fixed variants, including weaker controls, are locked for the
    # historical validation before that endpoint is evaluated.
    save_json(run / "comparison_lock.json", {
        "variants": list(ABLATION_MODES), "plan_sha256": object_hash(plan),
        "oof_sha256": sha256_file(run / "oof_predictions.csv"),
        "validation_evaluated": False, "test_evaluated": False,
        "selection": "all_four_predeclared_controls_no_validation_selection",
    })
    val_rows = []
    joined = pd.concat([train, validation], ignore_index=True)
    fit = np.arange(len(train))
    held = np.arange(len(train), len(joined))
    for mode in ABLATION_MODES:
        x = ablation_representation(joined, refs, mode)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prediction, fitting, bundle = fit_predict(x, joined._y.to_numpy(), fit, held, "extra_trees")
        bundle.update({"version": ABLATION_VERSION, "representation": mode, "features": CORE, "labels": LABELS,
                       "baseline_contract": "Provide audited initial references for a new session; no task-session statistics are learned as references."})
        path = run / "models" / f"extra_trees_{mode}.joblib"
        path.parent.mkdir(exist_ok=True)
        joblib.dump(bundle, path)
        row = ablation_metric_row(validation, prediction, "validation", mode)
        row.update({"train_accuracy": float(accuracy_score(train._y, fitting)), "model_sha256": sha256_file(path), "warnings": [str(w.message) for w in caught]})
        val_rows.append(row)
        tables.append(ablation_predictions(validation, prediction, "validation", mode, -1))
        print("Validation:", mode, row["accuracy"], flush=True)
    val_table = pd.DataFrame(val_rows)
    val_table.to_csv(run / "validation_metrics.csv", index=False)
    all_predictions = pd.concat(tables, ignore_index=True)
    all_predictions[all_predictions.protocol.eq("validation")].to_csv(run / "validation_predictions.csv", index=False)
    ablation_participant_metrics(all_predictions).to_csv(run / "participant_metrics.csv", index=False)
    contrasts = ablation_contrasts(pd.concat([summary, val_table], ignore_index=True))
    contrasts.to_csv(run / "paired_contrasts.csv", index=False)
    controls = ablation_control_check(pd.concat([summary, val_table], ignore_index=True), snapshot)
    controls.to_csv(run / "exp01_control_checks.csv", index=False)
    if not controls.same_metrics.all():
        print("PERHATIAN: sebagian kontrol berbeda dari Exp01. Baca exp01_control_checks.csv dan versi paket; semua kontras memakai hasil Exp02 saja.", flush=True)
    for key, path in data["paths"].items():
        if sha256_file(path) != data["hashes"][key]:
            raise ValueError("CSV sumber berubah selama run.")
    ablation_figures(run, summary, contrasts, val_table)
    ablation_report(run, summary, val_table, contrasts, controls, plan)
    save_json(run / "status.json", {
        "complete": True, "stage": "complete", "cv_fits": fit_count,
        "validation_fits": len(val_rows), "test_evaluated": False,
        "dataset_unchanged": True, "plan_sha256": object_hash(plan),
        "validation_three_class_evaluable": bool(val_table.three_class_evaluable.all()),
        "exp01_control_metrics_match": bool(controls.same_metrics.all()),
    })
    print("Laporan:", run / "RESULTS.md", flush=True)
    return {"run_dir": run, "leaderboard": summary, "validation": val_table, "contrasts": contrasts, "controls": controls}
