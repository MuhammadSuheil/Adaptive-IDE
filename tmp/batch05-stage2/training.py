import re
import shutil
import tempfile
import time
import warnings

import joblib
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

STAGE2_VERSION = "batch05-stage2-v1"
NORMALIZATIONS = ("global", "session_centered", "session_robust")
FAMILIES = ("logistic", "extra_trees", "hist_gradient")
MODEL_PARAMETERS = {
    "logistic": {"C": 1.0, "solver": "lbfgs", "max_iter": 5000, "random_state": 42},
    "extra_trees": {
        "n_estimators": 200,
        "min_samples_leaf": 5,
        "max_features": 0.8,
        "n_jobs": 2,
        "random_state": 42,
    },
    "hist_gradient": {
        "max_iter": 200,
        "learning_rate": 0.05,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
        "early_stopping": False,
        "random_state": 42,
    },
    "dummy": {"strategy": "most_frequent"},
}


def classification_metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if (
        y.ndim != 1
        or not len(y)
        or prediction.shape != y.shape
        or not np.isfinite(prediction).all()
        or not np.isin(prediction, [0, 1, 2]).all()
        or not np.isin(y, [0, 1, 2]).all()
    ):
        raise ValueError(
            "Target/prediksi harus berupa kelas 0, 1, 2 yang finite dan sejajar."
        )
    matrix = np.zeros((3, 3), dtype=int)
    np.add.at(matrix, (y.astype(int), prediction.astype(int)), 1)
    true_count, predicted_count = matrix.sum(1), matrix.sum(0)
    precision = np.divide(
        np.diag(matrix), predicted_count, out=np.zeros(3), where=predicted_count > 0
    )
    recall = np.divide(
        np.diag(matrix), true_count, out=np.zeros(3), where=true_count > 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(3),
        where=precision + recall > 0,
    )
    return {
        "rows": len(y),
        "accuracy": float(np.trace(matrix) / matrix.sum()),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "confusion_matrix": matrix.tolist(),
        "true_counts": true_count.tolist(),
        "predicted_counts": predicted_count.tolist(),
    }


def validate_training_indices(frame, fitting, held, guard=45):
    fitting, held = np.asarray(fitting, int), np.asarray(held, int)
    for ids in (fitting, held):
        if (
            not len(ids)
            or len(np.unique(ids)) != len(ids)
            or (ids < 0).any()
            or (ids >= len(frame)).any()
        ):
            raise ValueError(
                "Indeks fitting/evaluasi kosong, duplikat, atau tidak valid."
            )
        if frame.iloc[ids]._split.eq("test").any():
            raise ValueError("Test tidak boleh masuk fitting atau evaluasi tahap 2.")
    if not check_disjoint(frame, fitting, held) or not np.array_equal(
        purge(frame, fitting, held, guard), np.sort(fitting)
    ):
        raise ValueError("Fitting dan held-out overlap atau tidak memenuhi guard.")
    if not has_classes(frame, fitting) or not has_classes(frame, held):
        raise ValueError("Fitting/evaluasi memerlukan tiga kelas.")


def model_for(family):
    constructors = {
        "logistic": LogisticRegression,
        "extra_trees": ExtraTreesClassifier,
        "hist_gradient": HistGradientBoostingClassifier,
        "dummy": DummyClassifier,
    }
    return constructors[family](**MODEL_PARAMETERS[family])


def verify_prediction_table(frame, ids, table):
    expected = frame.iloc[np.asarray(ids, int)]
    if (
        table.window_id.tolist() != expected.window_id.tolist()
        or table.target.tolist() != expected._y.tolist()
    ):
        raise ValueError("Prediksi tidak sejajar dengan ID window/target.")
    classification_metrics(table.target.to_numpy(), table.prediction.to_numpy())


def prediction_table(frame, ids, prediction):
    columns = [
        "window_id",
        "session_id",
        "study_phase",
        "cognitive_load_label",
        "_block",
        "_y",
    ]
    table = (
        frame.iloc[ids][columns]
        .rename(columns={"_y": "target", "_block": "block"})
        .copy()
    )
    table["prediction"] = np.asarray(prediction)
    table["prediction_label"] = [LABELS[int(p)] for p in prediction]
    verify_prediction_table(frame, ids, table)
    return table.reset_index(drop=True)


class Stage2Store:
    """Local-first, checksummed checkpoints; the remote marker is written last."""

    def __init__(self, output_root, signature, resume_dir=None):
        self.signature = signature
        self.fingerprint = digest(signature)
        self.local = Path(tempfile.mkdtemp(prefix="codearrange-b05-stage2-"))
        self.cached = {}
        self.remote = (
            Path(resume_dir)
            if resume_dir is not None
            else Path(output_root)
            / (
                "stage2_"
                + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
                + "_"
                + self.fingerprint[:10]
            )
        )
        if resume_dir is not None:
            path = self.remote / "run.json"
            if (
                not path.is_file()
                or json.loads(path.read_text(encoding="utf-8")) != signature
            ):
                raise ValueError(
                    "Resume tidak kompatibel dengan data, manifest, kode, konfigurasi, atau lingkungan."
                )
        else:
            self.remote.mkdir(parents=True, exist_ok=False)
            save_json(self.remote / "run.json", signature)
        self.completed = {
            p.parent.name for p in (self.remote / "jobs").glob("*/done.json")
        }

    def job_dir(self, key):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", key):
            raise ValueError("Nama checkpoint tidak valid.")
        path = self.local / "jobs" / key
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get(self, key):
        if key in self.cached:
            return self.cached[key]
        local = self.job_dir(key)
        remote = self.remote / "jobs" / key
        marker = remote / "done.json"
        if not marker.is_file():
            return None
        record = json.loads(marker.read_text(encoding="utf-8"))
        if record["run_fingerprint"] != self.fingerprint:
            raise ValueError("Fingerprint checkpoint berbeda.")
        for name, checksum in record["files"].items():
            if Path(name).name != name:
                raise ValueError("Nama artefak checkpoint tidak valid.")
            source = remote / name
            if not source.is_file():
                raise ValueError(f"Artefak checkpoint hilang: {name}")
            shutil.copy2(source, local / name)
            if hashlib.sha256((local / name).read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Checksum checkpoint tidak cocok: {name}")
        self.cached[key] = local
        return local

    def complete(self, key):
        local = self.job_dir(key)
        files = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in local.iterdir()
            if p.is_file() and p.name != "done.json"
        }
        if "result.json" not in files:
            raise ValueError("Checkpoint belum memiliki hasil lengkap.")
        remote = self.remote / "jobs" / key
        remote.mkdir(parents=True, exist_ok=True)
        for name in files:
            shutil.copy2(local / name, remote / name)
        save_json(
            remote / "done.json", {"run_fingerprint": self.fingerprint, "files": files}
        )
        self.cached[key] = local
        self.completed.add(key)

    def progress(self, key, state="running", **extra):
        save_json(
            self.remote / "progress.json",
            {
                "stage": 2,
                "job": key,
                "state": state,
                "completed_fit_jobs": len(self.completed),
                "maximum_fit_jobs": 71,
                "updated_utc": datetime.now(UTC).isoformat(),
                **extra,
            },
        )

    def audit(self):
        verified = 0
        for marker in (self.remote / "jobs").glob("*/done.json"):
            data = json.loads(marker.read_text(encoding="utf-8"))
            for name, expected in data["files"].items():
                path = marker.parent / name
                if (
                    not path.is_file()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != expected
                ):
                    raise ValueError(f"Audit arsip gagal: {path}")
                verified += 1
        return verified


def restore_transform(state):
    transform = SensorPreprocessor(state["columns"], state["mode"])
    transform.center, transform.scale = (
        np.array(state["center"]),
        np.array(state["scale"]),
    )
    transform.session_stats = {
        s: (np.array(c), np.array(v)) for s, (c, v) in state["session_stats"].items()
    }
    if (
        transform.center.shape != (len(transform.columns),)
        or transform.scale.shape != transform.center.shape
        or not np.isfinite(transform.center).all()
        or not np.isfinite(transform.scale).all()
        or (transform.scale <= 0).any()
    ):
        raise ValueError("Statistik preprocessing tidak valid.")
    return transform


def load_saved_model(path):
    # Only load this workflow's own, previously checksummed estimator artifact.
    return joblib.load(Path(path) / "estimator.joblib"), restore_transform(
        json.loads((Path(path) / "preprocessing.json").read_text(encoding="utf-8"))
    )


def fit_job(
    frame, fitting, held, family, normalization, store, key, save_model=False, guard=45
):
    fitting, held = np.asarray(fitting, int), np.asarray(held, int)
    validate_training_indices(frame, fitting, held, guard)
    job_signature = digest(
        {
            "fit": frame.iloc[fitting].window_id.tolist(),
            "held": frame.iloc[held].window_id.tolist(),
            "family": family,
            "normalization": normalization,
            "save_model": save_model,
            "parameters": MODEL_PARAMETERS[family],
            "guard": guard,
        }
    )
    cached = store.get(key)
    if cached is not None:
        result = json.loads((cached / "result.json").read_text(encoding="utf-8"))
        if result["job_signature"] != job_signature:
            raise ValueError(
                "Checkpoint pekerjaan memiliki ID atau konfigurasi berbeda."
            )
        verify_prediction_table(
            frame, fitting, pd.read_csv(cached / "fit_predictions.csv")
        )
        verify_prediction_table(
            frame, held, pd.read_csv(cached / "held_predictions.csv")
        )
        return result
    if len(store.completed) >= 71:
        raise ValueError("Batas 71 fitting tercapai.")
    store.progress(key)
    print(
        f"[{len(store.completed) + 1}/71] {key}: fit={len(fitting)}, evaluasi={len(held)}",
        flush=True,
    )
    start = time.perf_counter()
    local = store.job_dir(key)
    try:
        transform = SensorPreprocessor(CORE, normalization).fit(frame, fitting)
        x_fit, x_held = (
            transform.transform(frame.iloc[fitting]),
            transform.transform(frame.iloc[held]),
        )
        estimator = model_for(family)
        with (
            warnings.catch_warnings(record=True) as caught,
            threadpool_limits(limits=2),
        ):
            warnings.simplefilter("always")
            estimator.fit(x_fit, frame.iloc[fitting]._y.to_numpy())
            p_fit, p_held = estimator.predict(x_fit), estimator.predict(x_held)
        fit_predictions = prediction_table(frame, fitting, p_fit)
        held_predictions = prediction_table(frame, held, p_held)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        result = {
            "job_signature": job_signature,
            "family": family,
            "normalization": normalization,
            "converged": converged,
            "finite_predictions": True,
            "fit_metrics": classification_metrics(
                frame.iloc[fitting]._y.to_numpy(), p_fit
            ),
            "held_metrics": classification_metrics(
                frame.iloc[held]._y.to_numpy(), p_held
            ),
            "fit_rows": len(fitting),
            "fit_blocks": int(frame.iloc[fitting]._block.nunique()),
            "fit_class_counts": np.bincount(
                frame.iloc[fitting]._y, minlength=3
            ).tolist(),
            "elapsed_seconds": time.perf_counter() - start,
            "warnings": [
                {"category": w.category.__name__, "message": str(w.message)}
                for w in caught
            ],
        }
        fit_predictions.to_csv(local / "fit_predictions.csv", index=False)
        held_predictions.to_csv(local / "held_predictions.csv", index=False)
        if save_model:
            joblib.dump(estimator, local / "estimator.joblib")
            save_json(local / "preprocessing.json", transform.state())
            loaded, loaded_transform = load_saved_model(local)
            with threadpool_limits(limits=2):
                restored = loaded.predict(loaded_transform.transform(frame.iloc[held]))
            if not np.array_equal(p_held, restored):
                raise ValueError("Prediksi model setelah dimuat ulang berbeda.")
            result["reload_predictions_identical"] = True
        save_json(local / "result.json", result)
        store.complete(key)
        store.progress(key, "completed", elapsed_seconds=result["elapsed_seconds"])
        print(
            f"  selesai {result['elapsed_seconds']:.2f}s; Macro-F1={result['held_metrics']['macro_f1']:.4f}",
            flush=True,
        )
        return result
    except Exception as exc:
        store.progress(key, "failed", error=f"{type(exc).__name__}: {exc}")
        raise


def learning_subsets(frame, fitting):
    fitting = np.asarray(fitting, int)
    majority = (
        frame.iloc[fitting]
        .groupby("_block", sort=True)
        ._y.agg(lambda v: int(np.bincount(v, minlength=3).argmax()))
    )
    rng = np.random.default_rng(42)
    ordered = []
    for label in range(3):
        blocks = majority[majority.eq(label)].index.to_numpy().copy()
        rng.shuffle(blocks)
        ordered.append(blocks)
    return {
        fraction: fitting[
            frame.iloc[fitting]
            ._block.isin(
                np.concatenate(
                    [
                        blocks[: max(1, int(np.ceil(len(blocks) * fraction)))]
                        for blocks in ordered
                    ]
                )
            )
            .to_numpy()
        ]
        for fraction in (0.25, 0.5, 0.75, 1.0)
    }


def locked_data(
    dataset_root, development, reference, expected_hashes, expected_manifest_hash
):
    if digest(development) != expected_manifest_hash:
        raise ValueError("Manifest development berubah dari tahap 1.")
    raw, hashes = {}, {}
    for w in (15, 25, 45):
        payload = (
            Path(dataset_root) / f"window_{w}s_step_5s/good_acceptable.csv"
        ).read_bytes()
        hashes[str(w)] = hashlib.sha256(payload).hexdigest()
        raw[w] = pd.read_csv(io.BytesIO(payload))
    if hashes != expected_hashes:
        raise ValueError("Hash dataset berbeda dari versi persiapan.")
    views, union = build_union(raw)
    manifest = pd.DataFrame(development["rows"]).set_index("endpoint_key")
    if set(manifest.index) != set(union._key) or manifest.index.duplicated().any():
        raise ValueError("Cakupan endpoint development berubah.")
    if development["guard_seconds"] != 45 or development["history_seconds"] != 0:
        raise ValueError("Dukungan fitur/guard tidak sesuai tahap 2.")
    for frame in views.values():
        frame["_split"] = frame._key.map(manifest.split)
        if not frame._key.map(manifest.block).eq(frame._block).all():
            raise ValueError("Blok endpoint berubah.")
    frame = views[25]
    reference_lookup = {
        r["session_id"] + "|" + str(r["end_ns"]): r["_split"] for r in reference["rows"]
    }
    old = frame._key.map(reference_lookup)
    if old.isna().any() or not old.eq("test").eq(frame._split.eq("test")).all():
        raise ValueError("Test referensi berubah.")
    train = np.flatnonzero(frame._split.eq("train"))
    validation = np.flatnonzero(frame._split.eq("validation"))
    test = np.flatnonzero(frame._split.eq("test"))
    validate_training_indices(frame, train, validation)
    if not np.array_equal(
        purge(frame, np.r_[train, validation], test), np.sort(np.r_[train, validation])
    ):
        raise ValueError("Development overlap dengan Test.")
    folds, seen = [], []
    for fold in development["cv"]["folds"]:
        fitting = np.flatnonzero(frame._key.isin(fold["fit_endpoint_keys"]))
        held = np.flatnonzero(frame._key.isin(fold["validation_endpoint_keys"]))
        if not set(np.r_[fitting, held]).issubset(set(train)):
            raise ValueError("Fold keluar dari Train.")
        validate_training_indices(frame, fitting, held)
        folds.append((fitting, held))
        seen.extend(held)
    if len(folds) != 5 or sorted(seen) != train.tolist():
        raise ValueError("OOF harus mencakup setiap baris Train tepat sekali.")
    old_train, old_validation = (
        np.flatnonzero(old.eq("train")),
        np.flatnonzero(old.eq("validation")),
    )
    validate_training_indices(
        frame, old_train, old_validation, guard=reference["support_seconds"]
    )
    common = np.intersect1d(validation, old_validation)
    if len(common) != 217 or np.bincount(
        frame.iloc[common]._y, minlength=3
    ).tolist() != [73, 60, 84]:
        raise ValueError("217 window validation bersama berubah.")
    return {
        "frame": frame,
        "folds": folds,
        "train": train,
        "validation": validation,
        "old_train": old_train,
        "old_validation": old_validation,
        "common": common,
        "hashes": hashes,
    }


def aggregate_candidate(frame, train, store, family, normalization):
    records, tables = [], []
    for number in range(5):
        key = f"cv_{family}_{normalization}_f{number}"
        path = store.get(key)
        result = json.loads((path / "result.json").read_text(encoding="utf-8"))
        records.append(result)
        tables.append(pd.read_csv(path / "held_predictions.csv"))
    table = (
        pd.concat(tables, ignore_index=True)
        .set_index("window_id")
        .loc[frame.iloc[train].window_id]
        .reset_index()
    )
    if table.window_id.duplicated().any():
        raise ValueError("OOF duplikat.")
    verify_prediction_table(frame, train, table)
    scores = classification_metrics(
        table.target.to_numpy(), table.prediction.to_numpy()
    )
    complexity = {"logistic": 1, "hist_gradient": 2, "extra_trees": 3, "dummy": 0}[
        family
    ] * 10 + NORMALIZATIONS.index(normalization)
    candidate = {
        "name": family + "__" + normalization,
        "family": family,
        "normalization": normalization,
        **scores,
        "fold_std": float(np.std([r["held_metrics"]["macro_f1"] for r in records])),
        "complexity": complexity,
        "finite_predictions": True,
        "converged": all(r["converged"] for r in records),
        "fold_macro_f1": [r["held_metrics"]["macro_f1"] for r in records],
    }
    candidate["eligible"] = bool(
        family != "dummy" and candidate["converged"] and min(scores["recall"]) > 0
    )
    return candidate, table


def metric_row(result, context, **extra):
    return {
        "context": context,
        **extra,
        "accuracy": result["accuracy"],
        "balanced_accuracy": result["balanced_accuracy"],
        "macro_f1": result["macro_f1"],
        **{
            f"{metric}_{label}": result[metric][i]
            for metric in ("precision", "recall", "f1")
            for i, label in enumerate(LABELS)
        },
    }


def write_frame(store, name, frame):
    path = store.local / name
    frame.to_csv(path, index=False)
    shutil.copy2(path, store.remote / name)


def save_figure(store, name, figure):
    path = store.local / "figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    destination = store.remote / "figures" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    plt.close(figure)


def export_graphs(store, leaderboard, winner, validation_result, curve):
    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = leaderboard.sort_values("macro_f1")
    ax.barh(
        ordered.name,
        ordered.macro_f1,
        color=["#999999" if n.startswith("dummy") else "#4768b5" for n in ordered.name],
    )
    ax.set(xlim=(0, 1), xlabel="OOF Macro-F1", title="Kontrol development 25 detik")
    save_figure(store, "cv_macro_f1.png", fig)
    if winner is None:
        return
    for partition, result in (("oof", winner), ("validation", validation_result)):
        fig, ax = plt.subplots(figsize=(5, 4))
        matrix = np.asarray(result["confusion_matrix"])
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
            title=f"{partition}: {winner['name']}",
        )
        save_figure(store, f"confusion_{partition}.png", fig)
    if not curve.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, metric in zip(axes, ("macro_f1", "accuracy")):
            for partition in ("fitting", "validation"):
                groups = curve[curve.partition.eq(partition)].groupby("fraction")[
                    metric
                ]
                mean, std = groups.mean(), groups.std().fillna(0)
                ax.errorbar(
                    mean.index * 100, mean, yerr=std, marker="o", label=partition
                )
            ax.set(ylim=(0, 1), xlabel="Persentase blok fitting", ylabel=metric)
            ax.legend()
        fig.suptitle("Learning curve development")
        save_figure(store, "learning_curve.png", fig)


def stage2_report(store, leaderboard, winner, finals, controls, curve, gate):
    lines = [
        "# Hasil Tahap 2",
        "",
        "Window 25 detik, 10 fitur core; lima fold blok waktu yang dibekukan pada tahap 1.",
        "Test tidak dievaluasi. Skor berikut merupakan evaluasi development, bukan skor pengujian independen.",
        "",
        "## Perbandingan CV",
        "",
        "| Model | Normalisasi | OOF accuracy | OOF Macro-F1 | Recall Low/Medium/High | Layak |",
        "|---|---|---:|---:|---|---|",
    ]
    for r in leaderboard.sort_values("macro_f1", ascending=False).to_dict("records"):
        recalls = "/".join(f"{r[f'recall_{label}']:.2%}" for label in LABELS)
        lines.append(
            f"| {r['family']} | {r['normalization']} | {r['accuracy']:.2%} | {r['macro_f1']:.2%} | {recalls} | {r['eligible']} |"
        )
    lines += ["", "## Kandidat terpilih", ""]
    if winner is None:
        lines.append(
            "Tidak ada kandidat yang memenuhi kelayakan tiga kelas. Tidak ada pemenang yang dipaksakan."
        )
    else:
        value = finals[winner["name"]]["held_metrics"]
        lines += [
            f"Kandidat berdasarkan OOF: **{winner['name']}**.",
            f"Validation: accuracy {value['accuracy']:.2%}; Macro-F1 {value['macro_f1']:.2%}.",
            "Pemilihan kandidat selesai sebelum validation dibaca; hasil validation tidak mengganti pemenang.",
            "",
            "| Kelas | Precision validation | Recall validation | F1 validation |",
            "|---|---:|---:|---:|",
        ]
        for i, label in enumerate(LABELS):
            lines.append(
                f"| {label} | {value['precision'][i]:.2%} | {value['recall'][i]:.2%} | {value['f1'][i]:.2%} |"
            )
    lines += [
        "",
        "## Kontrol pembagian",
        "",
        "217 window validation yang sama: 73 Low, 60 Medium, 84 High.",
        "| Model | Pembagian fitting | Baris fitting | Accuracy bersama | Macro-F1 bersama |",
        "|---|---|---:|---:|---:|",
    ]
    for r in controls.to_dict("records"):
        lines.append(
            f"| {r['family']} | {r['split']} | {r['fit_rows']} | {r['accuracy']:.2%} | {r['macro_f1']:.2%} |"
        )
    lines += [
        "",
        "Pembagian lama dan baru berbeda pada komposisi fitting, guard, serta retensi baris. Selisih ini bukan efek normalisasi saja.",
        "",
        "## Learning curve",
        "",
    ]
    if curve.empty:
        lines.append("Learning curve tidak tersedia karena tidak ada kandidat layak.")
    else:
        lines += [
            "| Blok fitting | Accuracy fitting | Accuracy validation | Macro-F1 fitting | Macro-F1 validation |",
            "|---|---:|---:|---:|---:|",
        ]
        for fraction in sorted(curve.fraction.unique()):
            subset = curve[curve.fraction.eq(fraction)]
            fit = subset[subset.partition.eq("fitting")]
            val = subset[subset.partition.eq("validation")]
            lines.append(
                f"| {fraction:.0%} | {fit.accuracy.mean():.2%} | {val.accuracy.mean():.2%} | {fit.macro_f1.mean():.2%} | {val.macro_f1.mean():.2%} |"
            )
        lines.append(
            "\nNilai merupakan rata-rata fold; CSV menyertakan jumlah baris/blok aktual, dukungan kelas, dan titik yang dilewati."
        )
    lines += [
        "",
        "## Keputusan berikutnya",
        "",
        f"Gate perencanaan neural terarah: **{gate['ready_for_stage3_planning']}**.",
        "Gate memerlukan pengenalan tiga kelas pada OOF dan validation, serta mengungguli Dummy pada minimal tiga fold.",
        "Status tersebut bukan bukti model optimal atau jaminan 70%. Tahap 3 tidak berjalan otomatis.",
        "",
        f"Fitting penelitian selesai: {len(store.completed)} dari batas 71 pekerjaan unik.",
        "Grafik tersedia pada folder figures. Prediksi dan warning per pekerjaan tersedia pada jobs.",
    ]
    return "\n".join(lines) + "\n"


def run_stage2(
    dataset_root,
    output_root,
    development,
    reference,
    expected_hashes,
    manifest_hash,
    code_hash,
    resume_dir=None,
):
    data = locked_data(
        dataset_root, development, reference, expected_hashes, manifest_hash
    )
    signature = {
        "stage": 2,
        "version": STAGE2_VERSION,
        "code_sha256": code_hash,
        "dataset_sha256": expected_hashes,
        "manifest_sha256": manifest_hash,
        "reference_sha256": reference["source_sha256"],
        "features": CORE,
        "window_seconds": 25,
        "models": MODEL_PARAMETERS,
        "normalizations": list(NORMALIZATIONS),
        "maximum_fits": 71,
        "python": platform.python_version(),
        "environment": {
            package: importlib.metadata.version(package)
            for package in (
                "numpy",
                "pandas",
                "scikit-learn",
                "matplotlib",
                "joblib",
                "threadpoolctl",
            )
        },
    }
    store = Stage2Store(output_root, signature, resume_dir)
    print("Run tahap 2:", store.remote, flush=True)
    save_json(store.remote / "development_manifest.json", development)
    save_json(store.remote / "reference_manifest.json", reference)
    frame, train, validation = data["frame"], data["train"], data["validation"]
    fold_rows, candidates, all_oof = [], [], []
    configurations = [
        (family, mode) for family in FAMILIES for mode in NORMALIZATIONS
    ] + [("dummy", "global")]
    try:
        for family, normalization in configurations:
            for number, (fitting, held) in enumerate(data["folds"]):
                result = fit_job(
                    frame,
                    fitting,
                    held,
                    family,
                    normalization,
                    store,
                    f"cv_{family}_{normalization}_f{number}",
                )
                row = {
                    "family": family,
                    "normalization": normalization,
                    "fold": number,
                    "converged": result["converged"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "fit_rows": len(fitting),
                    "held_rows": len(held),
                    "warnings": json.dumps(result["warnings"]),
                    "fit_accuracy_minus_validation": result["fit_metrics"]["accuracy"]
                    - result["held_metrics"]["accuracy"],
                    "fit_macro_f1_minus_validation": result["fit_metrics"]["macro_f1"]
                    - result["held_metrics"]["macro_f1"],
                }
                for name, metric in (
                    ("fitting", "fit_metrics"),
                    ("validation", "held_metrics"),
                ):
                    fold_rows.append(metric_row(result[metric], name, **row))
            candidate, oof = aggregate_candidate(
                frame, train, store, family, normalization
            )
            candidates.append(candidate)
            all_oof.append(oof.assign(candidate=candidate["name"]))
        winner = select_candidate([c for c in candidates if c["eligible"]])
        save_json(
            store.remote / "selection.json",
            {
                "winner": winner,
                "candidates": candidates,
                "selected_using": "out_of_fold_only",
                "validation_read_for_selection": False,
            },
        )
        leaderboard = pd.DataFrame(
            [
                metric_row(
                    c,
                    "oof",
                    name=c["name"],
                    family=c["family"],
                    normalization=c["normalization"],
                    fold_std=c["fold_std"],
                    converged=c["converged"],
                    eligible=c["eligible"],
                )
                for c in candidates
            ]
        )
        write_frame(store, "cv_leaderboard.csv", leaderboard)
        fold_table = pd.DataFrame(fold_rows)
        write_frame(store, "fold_metrics.csv", fold_table)
        metrics = ["accuracy", "balanced_accuracy", "macro_f1"] + [
            f"{metric}_{label}"
            for metric in ("precision", "recall", "f1")
            for label in LABELS
        ]
        summary = fold_table.groupby(["family", "normalization", "context"])[
            metrics
        ].agg(["mean", "std"])
        summary.columns = ["_".join(c) for c in summary.columns]
        write_frame(store, "cv_fold_summary.csv", summary.reset_index())
        write_frame(store, "oof_predictions.csv", pd.concat(all_oof, ignore_index=True))
        final_configurations = {
            ("logistic", "global"),
            ("extra_trees", "global"),
            ("dummy", "global"),
        }
        if winner is not None:
            final_configurations.add((winner["family"], winner["normalization"]))
        finals, final_rows = {}, []
        for family, mode in sorted(final_configurations):
            name = family + "__" + mode
            result = fit_job(
                frame,
                train,
                validation,
                family,
                mode,
                store,
                f"final_{family}_{mode}",
                save_model=True,
            )
            finals[name] = result
            final_rows.append(
                metric_row(
                    result["held_metrics"],
                    "validation",
                    name=name,
                    converged=result["converged"],
                    selected=winner is not None and name == winner["name"],
                )
            )
        write_frame(store, "validation_metrics.csv", pd.DataFrame(final_rows))
        save_json(
            store.remote / "selected_model.json",
            None
            if winner is None
            else {
                "candidate": winner["name"],
                "selection": "out_of_fold_only",
                "estimator": f"jobs/final_{winner['family']}_{winner['normalization']}/estimator.joblib",
                "preprocessing": f"jobs/final_{winner['family']}_{winner['normalization']}/preprocessing.json",
                "fit_split": "train",
                "validation_added_to_fitting": False,
            },
        )
        controls, common_predictions = [], []
        for family in ("logistic", "extra_trees"):
            previous = fit_job(
                frame,
                data["old_train"],
                data["old_validation"],
                family,
                "global",
                store,
                f"reference_{family}_global",
                save_model=True,
                guard=reference["support_seconds"],
            )
            for split, record, key in (
                ("reference", previous, f"reference_{family}_global"),
                ("development", finals[family + "__global"], f"final_{family}_global"),
            ):
                predictions = (
                    pd.read_csv(store.get(key) / "held_predictions.csv")
                    .set_index("window_id")
                    .loc[frame.iloc[data["common"]].window_id]
                    .reset_index()
                )
                verify_prediction_table(frame, data["common"], predictions)
                score = classification_metrics(
                    predictions.target.to_numpy(), predictions.prediction.to_numpy()
                )
                controls.append(
                    metric_row(
                        score,
                        "common_validation",
                        family=family,
                        split=split,
                        fit_rows=record["fit_rows"],
                        fit_blocks=record["fit_blocks"],
                        **{
                            f"fit_{label}": record["fit_class_counts"][i]
                            for i, label in enumerate(LABELS)
                        },
                    )
                )
                common_predictions.append(
                    predictions.assign(family=family, split=split)
                )
        controls = pd.DataFrame(controls)
        write_frame(store, "split_controls.csv", controls)
        write_frame(
            store,
            "common_validation_predictions.csv",
            pd.concat(common_predictions, ignore_index=True),
        )
        curve_rows, skipped = [], []
        if winner is not None:
            family, mode = winner["family"], winner["normalization"]
            for number, (fitting, held) in enumerate(data["folds"]):
                for fraction, subset in learning_subsets(frame, fitting).items():
                    counts = np.bincount(frame.iloc[subset]._y, minlength=3)
                    if (counts < 10).any():
                        skipped.append(
                            {
                                "fold": number,
                                "fraction": fraction,
                                "reason": "fewer_than_10_rows_per_class",
                            }
                        )
                        continue
                    key = (
                        f"cv_{family}_{mode}_f{number}"
                        if fraction == 1
                        else f"curve_{int(fraction * 100)}_f{number}"
                    )
                    result = fit_job(frame, subset, held, family, mode, store, key)
                    for partition, metric in (
                        ("fitting", "fit_metrics"),
                        ("validation", "held_metrics"),
                    ):
                        curve_rows.append(
                            metric_row(
                                result[metric],
                                "learning_curve",
                                partition=partition,
                                fold=number,
                                fraction=fraction,
                                fit_rows=result["fit_rows"],
                                fit_blocks=result["fit_blocks"],
                                converged=result["converged"],
                                fit_Low=int(counts[0]),
                                fit_Medium=int(counts[1]),
                                fit_High=int(counts[2]),
                            )
                        )
        curve = pd.DataFrame(curve_rows)
        write_frame(store, "learning_curve.csv", curve)
        save_json(store.remote / "learning_curve_skipped.json", skipped)
        dummy = next(c for c in candidates if c["family"] == "dummy")
        wins = (
            0
            if winner is None
            else sum(
                a > b for a, b in zip(winner["fold_macro_f1"], dummy["fold_macro_f1"])
            )
        )
        gate = {
            "ready_for_stage3_planning": bool(
                winner is not None
                and wins >= 3
                and finals[winner["name"]]["converged"]
                and min(finals[winner["name"]]["held_metrics"]["recall"]) > 0
            ),
            "folds_better_than_dummy": wins,
            "test_evaluated": False,
            "stage3_executed": False,
        }
        if winner is not None:
            predictions = pd.read_csv(
                store.get(f"final_{winner['family']}_{winner['normalization']}")
                / "held_predictions.csv"
            )
            fit_sources = set(zip(frame.iloc[train].session_id, frame.iloc[train]._y))
            predictions["source_supported_in_train"] = [
                (s, t) in fit_sources
                for s, t in zip(predictions.session_id, predictions.target)
            ]
            predictions["correct"] = predictions.target.eq(predictions.prediction)
            breakdown = []
            for group_name, group in (
                ("class", "cognitive_load_label"),
                ("phase", "study_phase"),
                ("source_support", "source_supported_in_train"),
            ):
                for value, rows in predictions.groupby(group):
                    breakdown.append(
                        {
                            "group": group_name,
                            "value": str(value),
                            "rows": len(rows),
                            "correct": int(rows.correct.sum()),
                            "accuracy": float(rows.correct.mean()),
                        }
                    )
            for phase, rows in predictions[predictions.target.eq(0)].groupby(
                "study_phase"
            ):
                breakdown.append(
                    {
                        "group": "Low_phase",
                        "value": phase,
                        "rows": len(rows),
                        "correct": int(rows.correct.sum()),
                        "accuracy": float(rows.correct.mean()),
                    }
                )
            write_frame(store, "validation_breakdown.csv", pd.DataFrame(breakdown))
        save_json(store.remote / "gate.json", gate)
        export_graphs(
            store,
            leaderboard,
            winner,
            None if winner is None else finals[winner["name"]]["held_metrics"],
            curve,
        )
        (store.remote / "RESULTS.md").write_text(
            stage2_report(store, leaderboard, winner, finals, controls, curve, gate),
            encoding="utf-8",
        )
        verified = store.audit()
        save_json(
            store.remote / "status.json",
            {
                "status": "complete",
                "stage": 2,
                "fit_jobs": len(store.completed),
                "verified_job_artifacts": verified,
                "test_evaluated": False,
                "winner": None if winner is None else winner["name"],
            },
        )
        store.progress("complete", "complete", test_evaluated=False)
    except Exception as exc:
        save_json(
            store.remote / "status.json",
            {"status": "failed", "stage": 2, "error": f"{type(exc).__name__}: {exc}"},
        )
        raise
    print("Hasil tahap 2:", store.remote / "RESULTS.md", flush=True)
    return {
        "run_dir": store.remote,
        "leaderboard": leaderboard,
        "winner": winner,
        "finals": finals,
        "controls": controls,
        "learning_curve": curve,
        "gate": gate,
    }
