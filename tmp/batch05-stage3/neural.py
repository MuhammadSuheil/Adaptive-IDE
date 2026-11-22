import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn

NEURAL_CONFIG = {
    "epochs": 120,
    "batch_size": 128,
    "learning_rate": 0.001,
    "weight_decay": 0.001,
    "dropout": 0.2,
    "gradient_clip": 1.0,
}
NEURAL_SEEDS = (42, 43, 44)
CHECKPOINT_POLICIES = ("loss", "macro", "last")
STAGE3_VERSION = "batch05-stage3-v1"


class Stage3Store(Stage2Store):
    def __init__(self, output_root, signature, resume_dir=None):
        if resume_dir is None:
            resume_dir = Path(output_root) / (
                "stage3_"
                + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
                + "_"
                + digest(signature)[:10]
            )
            resume_dir.mkdir(parents=True, exist_ok=False)
            save_json(resume_dir / "run.json", signature)
        super().__init__(output_root, signature, resume_dir)

    def progress(self, key, state="running", **extra):
        save_json(
            self.remote / "progress.json",
            {
                "stage": 3,
                "job": key,
                "state": state,
                "completed_fit_jobs": len(self.completed),
                "maximum_fit_jobs": 72,
                "updated_utc": datetime.now(UTC).isoformat(),
                **extra,
            },
        )

    def partial(self, key):
        self.job_dir(key)
        root = self.remote / "partial" / key
        marker = root / "resume.json"
        if not marker.is_file():
            return None
        value = json.loads(marker.read_text(encoding="utf-8"))
        if value["run_fingerprint"] != self.fingerprint or value["slot"] not in (0, 1):
            raise ValueError("Checkpoint epoch tidak kompatibel.")
        source = root / f"slot{value['slot']}.pt"
        if (
            not source.is_file()
            or hashlib.sha256(source.read_bytes()).hexdigest() != value["sha256"]
        ):
            raise ValueError("Checkpoint epoch rusak atau tidak lengkap.")
        # Restore only the workflow's own checksummed tensors and primitive state.
        return torch.load(source, map_location="cpu", weights_only=True)

    def save_partial(self, key, payload):
        root = self.remote / "partial" / key
        root.mkdir(parents=True, exist_ok=True)
        marker = root / "resume.json"
        old = (
            json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else None
        )
        slot = 0 if old is None else 1 - old["slot"]
        temporary = self.local / f"resume_{key}.pt"
        torch.save(payload, temporary)
        checksum = hashlib.sha256(temporary.read_bytes()).hexdigest()
        shutil.copy2(temporary, root / f"slot{slot}.pt")
        save_json(
            marker,
            {
                "slot": slot,
                "epoch": payload["epoch"],
                "sha256": checksum,
                "run_fingerprint": self.fingerprint,
            },
        )


def build_stage3_mlp():
    return nn.Sequential(
        nn.Linear(10, 64),
        nn.LayerNorm(64),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.LayerNorm(32),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(32, 3),
    )


def copy_weights(model):
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def checkpoint_rank(policy, metrics, loss, epoch):
    if policy == "loss":
        return (float(loss), int(epoch))
    if policy == "macro":
        return (-metrics["macro_f1"], -min(metrics["recall"]), float(loss), int(epoch))
    raise ValueError("Aturan checkpoint tidak dikenal.")


def check_probabilities(probabilities):
    p = np.asarray(probabilities, float)
    if (
        p.ndim != 2
        or p.shape[1] != 3
        or not len(p)
        or not np.isfinite(p).all()
        or (p < 0).any()
        or (p > 1 + 1e-6).any()
        or not np.allclose(p.sum(1), 1, atol=1e-6, rtol=1e-6)
    ):
        raise ValueError("Probabilitas kelas tidak valid.")
    return p


def average_probabilities(arrays):
    if not arrays:
        raise ValueError("Prediksi seed kosong.")
    values = [check_probabilities(p) for p in arrays]
    if any(p.shape != values[0].shape for p in values):
        raise ValueError("Prediksi seed tidak sejajar.")
    return check_probabilities(np.mean(values, axis=0))


def probability_table(frame, ids, probabilities):
    p = check_probabilities(probabilities)
    table = prediction_table(frame, ids, p.argmax(1))
    for i, label in enumerate(LABELS):
        table["probability_" + label] = p[:, i]
    return table


def table_probabilities(table):
    p = check_probabilities(table[["probability_" + c for c in LABELS]].to_numpy())
    if not np.array_equal(p.argmax(1), table.prediction.to_numpy()):
        raise ValueError("Argmax probabilitas tidak cocok dengan prediksi.")
    return p


def validate_neural_roles(frame, fitting, monitor, held):
    for a, b in ((fitting, monitor), (fitting, held), (monitor, held)):
        validate_training_indices(frame, a, b)
    counts = [
        np.bincount(frame.iloc[ids]._y, minlength=3) for ids in (fitting, monitor)
    ]
    if any((v < 10).any() for v in counts):
        raise ValueError("Fitting/monitor memerlukan minimal 10 baris per kelas.")
    if (
        np.max(np.abs(counts[0] / sum(counts[0]) - counts[1] / sum(counts[1])))
        > 0.1 + 1e-12
    ):
        raise ValueError(
            "Proporsi kelas fitting dan monitor berbeda lebih dari 10 poin."
        )


def stage3_device():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def neural_probabilities_stage3(network, x, device):
    network.eval()
    with torch.no_grad():
        p = (
            network(torch.as_tensor(x, dtype=torch.float32, device=device))
            .softmax(-1)
            .cpu()
            .numpy()
        )
    return check_probabilities(p)


def logits_evaluation(logits, y):
    if logits.ndim != 2 or logits.shape[1] != 3 or not torch.isfinite(logits).all():
        raise ValueError("Logit neural tidak valid.")
    loss = nn.functional.cross_entropy(
        logits, torch.tensor(y, dtype=torch.long, device=logits.device)
    )
    if not torch.isfinite(loss):
        raise ValueError("Cross-entropy evaluasi nonfinite.")
    return check_probabilities(logits.softmax(-1).detach().cpu().numpy()), float(
        loss.detach()
    )


def neural_evaluation(network, x, y, device):
    network.eval()
    with torch.no_grad():
        return logits_evaluation(
            network(torch.as_tensor(x, dtype=torch.float32, device=device)), y
        )


def neural_job(
    frame,
    fitting,
    monitor,
    held,
    normalization,
    seed,
    store,
    key,
    config=None,
    on_epoch=None,
):
    cfg = dict(NEURAL_CONFIG if config is None else config)
    if cfg["dropout"] != 0.2 or cfg["epochs"] < 1:
        raise ValueError("Arsitektur/durasi tidak sesuai.")
    fitting, monitor, held = [np.asarray(ids, int) for ids in (fitting, monitor, held)]
    validate_neural_roles(frame, fitting, monitor, held)
    job_signature = digest(
        {
            "fitting": frame.iloc[fitting].window_id.tolist(),
            "monitor": frame.iloc[monitor].window_id.tolist(),
            "held": frame.iloc[held].window_id.tolist(),
            "normalization": normalization,
            "seed": seed,
            "config": cfg,
        }
    )
    cached = store.get(key)
    if cached is not None:
        result = json.loads((cached / "result.json").read_text(encoding="utf-8"))
        if result["job_signature"] != job_signature:
            raise ValueError("Checkpoint neural memiliki konfigurasi/ID berbeda.")
        for policy in CHECKPOINT_POLICIES:
            table = pd.read_csv(cached / f"predictions_{policy}.csv")
            verify_prediction_table(frame, held, table)
            table_probabilities(table)
        return result
    if len(store.completed) >= 72:
        raise ValueError("Batas fitting tahap 3 tercapai.")
    device = stage3_device()
    torch.manual_seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    pre = SensorPreprocessor(CORE, normalization).fit(frame, fitting)
    x = pre.transform(frame.iloc[fitting]).astype(np.float32)
    xm = pre.transform(frame.iloc[monitor]).astype(np.float32)
    y, ym = frame.iloc[fitting]._y.to_numpy(), frame.iloc[monitor]._y.to_numpy()
    network = build_stage3_mlp().to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    loss_fn = nn.CrossEntropyLoss()
    history, best, first_epoch, elapsed = [], {}, 1, 0.0
    partial = store.partial(key)
    if partial is not None:
        if partial["job_signature"] != job_signature:
            raise ValueError("Checkpoint epoch berasal dari pekerjaan berbeda.")
        network.load_state_dict(partial["model"])
        optimizer.load_state_dict(partial["optimizer"])
        history, best = partial["history"], partial["best"]
        first_epoch, elapsed = partial["epoch"] + 1, partial["elapsed_seconds"]
        rng.bit_generator.state = partial["numpy_rng"]
        random.setstate(partial["python_rng"])
        torch.set_rng_state(partial["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(partial["cuda_rng"])
    print(
        f"[{len(store.completed) + 1}/72] {key}: optimizer={len(fitting)}, monitor={len(monitor)}, device={device}",
        flush=True,
    )
    local = store.job_dir(key)
    try:
        for epoch in range(first_epoch, cfg["epochs"] + 1):
            tick = time.perf_counter()
            network.train()
            order = rng.permutation(len(fitting))
            optimizer_loss = 0.0
            for start in range(0, len(order), cfg["batch_size"]):
                batch = order[start : start + cfg["batch_size"]]
                logits = network(torch.as_tensor(x[batch], device=device))
                loss = loss_fn(
                    logits, torch.as_tensor(y[batch], dtype=torch.long, device=device)
                )
                if not torch.isfinite(loss):
                    raise ValueError("Loss neural nonfinite.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    network.parameters(), cfg["gradient_clip"], error_if_nonfinite=True
                )
                optimizer.step()
                optimizer_loss += float(loss.detach()) * len(batch)
            pf, fit_loss = neural_evaluation(network, x, y, device)
            pm, monitor_loss = neural_evaluation(network, xm, ym, device)
            fm, mm = (
                classification_metrics(y, pf.argmax(1)),
                classification_metrics(ym, pm.argmax(1)),
            )
            row = {
                "epoch": epoch,
                "optimizer_loss": optimizer_loss / len(fitting),
                "fit_loss": fit_loss,
                "monitor_loss": monitor_loss,
                "fit_accuracy": fm["accuracy"],
                "fit_macro_f1": fm["macro_f1"],
                "monitor_accuracy": mm["accuracy"],
                "monitor_macro_f1": mm["macro_f1"],
                "monitor_min_recall": min(mm["recall"]),
                "learning_rate": cfg["learning_rate"],
            }
            for i, label in enumerate(LABELS):
                row["monitor_recall_" + label] = mm["recall"][i]
                row["monitor_predicted_" + label] = mm["predicted_counts"][i]
            history.append(row)
            for policy in ("loss", "macro"):
                rank = checkpoint_rank(policy, mm, monitor_loss, epoch)
                if policy not in best or rank < tuple(best[policy]["rank"]):
                    best[policy] = {
                        "epoch": epoch,
                        "rank": list(rank),
                        "state": copy_weights(network),
                    }
            elapsed += time.perf_counter() - tick
            if epoch % 5 == 0 or epoch == cfg["epochs"]:
                store.save_partial(
                    key,
                    {
                        "job_signature": job_signature,
                        "epoch": epoch,
                        "model": copy_weights(network),
                        "optimizer": optimizer.state_dict(),
                        "history": history,
                        "best": best,
                        "elapsed_seconds": elapsed,
                        "numpy_rng": rng.bit_generator.state,
                        "python_rng": random.getstate(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all()
                        if device.type == "cuda"
                        else [],
                    },
                )
                store.progress(
                    key,
                    epoch=epoch,
                    maximum_epochs=cfg["epochs"],
                    monitor_loss=monitor_loss,
                )
            if epoch == 1 or epoch % 20 == 0 or epoch == cfg["epochs"]:
                print(
                    f"  epoch {epoch}/{cfg['epochs']}; fitting loss={row['fit_loss']:.4f}; monitor loss={monitor_loss:.4f}; monitor F1={mm['macro_f1']:.4f}",
                    flush=True,
                )
            if on_epoch is not None:
                on_epoch(epoch)
        best["last"] = {"epoch": cfg["epochs"], "state": copy_weights(network)}
        result = {
            "kind": "neural",
            "job_signature": job_signature,
            "normalization": normalization,
            "seed": seed,
            "config": cfg,
            "elapsed_seconds": elapsed,
            "policies": {},
        }
        predictions = {"fitting": {}, "monitor": {}}
        for policy in CHECKPOINT_POLICIES:
            state = best[policy]["state"]
            torch.save(state, local / f"{policy}.pt")
            network.load_state_dict(state)
            pf, fit_loss = neural_evaluation(network, x, y, device)
            pm, monitor_loss = neural_evaluation(network, xm, ym, device)
            ph = neural_probabilities_stage3(
                network, pre.transform(frame.iloc[held]).astype(np.float32), device
            )
            reloaded = build_stage3_mlp().to(device)
            reloaded.load_state_dict(
                torch.load(
                    local / f"{policy}.pt", map_location=device, weights_only=True
                )
            )
            loaded_p = neural_probabilities_stage3(
                reloaded, pre.transform(frame.iloc[held]).astype(np.float32), device
            )
            if not np.array_equal(ph.argmax(1), loaded_p.argmax(1)) or not np.allclose(
                ph, loaded_p, atol=1e-6
            ):
                raise ValueError("Prediksi checkpoint neural setelah reload berbeda.")
            probability_table(frame, held, ph).to_csv(
                local / f"predictions_{policy}.csv", index=False
            )
            predictions["fitting"][policy], predictions["monitor"][policy] = (
                pf.argmax(1).tolist(),
                pm.argmax(1).tolist(),
            )
            result["policies"][policy] = {
                "epoch": best[policy]["epoch"],
                "fit_metrics": classification_metrics(y, pf.argmax(1)),
                "monitor_metrics": classification_metrics(ym, pm.argmax(1)),
                "held_metrics": classification_metrics(
                    frame.iloc[held]._y.to_numpy(), ph.argmax(1)
                ),
                "fit_loss": fit_loss,
                "monitor_loss": monitor_loss,
                "reload_predictions_identical": True,
            }
        save_json(local / "preprocessing.json", pre.state())
        save_json(
            local / "indices.json",
            {
                role: frame.iloc[ids].window_id.tolist()
                for role, ids in (
                    ("fitting", fitting),
                    ("monitor", monitor),
                    ("held", held),
                )
            },
        )
        save_json(local / "internal_predictions.json", predictions)
        pd.DataFrame(history).to_csv(local / "history.csv", index=False)
        save_json(local / "result.json", result)
        store.complete(key)
        store.progress(key, "completed", elapsed_seconds=elapsed)
        return result
    except Exception as exc:
        store.progress(key, "failed", error=f"{type(exc).__name__}: {exc}")
        raise


def stage3_tree_job(frame, fitting, held, normalization, store, key, save_model=False):
    fitting, held = np.asarray(fitting, int), np.asarray(held, int)
    validate_training_indices(frame, fitting, held)
    job_signature = digest(
        {
            "fitting": frame.iloc[fitting].window_id.tolist(),
            "held": frame.iloc[held].window_id.tolist(),
            "normalization": normalization,
            "save_model": save_model,
            "model": MODEL_PARAMETERS["extra_trees"],
        }
    )
    cached = store.get(key)
    if cached is not None:
        value = json.loads((cached / "result.json").read_text(encoding="utf-8"))
        if value["job_signature"] != job_signature:
            raise ValueError("Checkpoint kontrol tabular berbeda.")
        table = pd.read_csv(cached / "predictions.csv")
        verify_prediction_table(frame, held, table)
        table_probabilities(table)
        return value
    if len(store.completed) >= 72:
        raise ValueError("Batas fitting tahap 3 tercapai.")
    print(f"[{len(store.completed) + 1}/72] {key}: fit={len(fitting)}", flush=True)
    store.progress(key)
    tick = time.perf_counter()
    pre = SensorPreprocessor(CORE, normalization).fit(frame, fitting)
    model = model_for("extra_trees")
    with threadpool_limits(limits=2):
        model.fit(pre.transform(frame.iloc[fitting]), frame.iloc[fitting]._y.to_numpy())
        ph = model.predict_proba(pre.transform(frame.iloc[held]))
        pf = model.predict(pre.transform(frame.iloc[fitting]))
    local = store.job_dir(key)
    probability_table(frame, held, ph).to_csv(local / "predictions.csv", index=False)
    result = {
        "kind": "tree",
        "normalization": normalization,
        "job_signature": job_signature,
        "fit_metrics": classification_metrics(frame.iloc[fitting]._y.to_numpy(), pf),
        "held_metrics": classification_metrics(
            frame.iloc[held]._y.to_numpy(), ph.argmax(1)
        ),
        "fit_rows": len(fitting),
        "elapsed_seconds": time.perf_counter() - tick,
    }
    save_json(
        local / "indices.json",
        {
            "fitting": frame.iloc[fitting].window_id.tolist(),
            "held": frame.iloc[held].window_id.tolist(),
        },
    )
    save_json(local / "internal_predictions.json", {"fitting": pf.tolist()})
    if save_model:
        joblib.dump(model, local / "estimator.joblib")
        save_json(local / "preprocessing.json", pre.state())
        restored, restored_pre = load_saved_model(local)
        with threadpool_limits(limits=2):
            if not np.allclose(
                restored.predict_proba(restored_pre.transform(frame.iloc[held])),
                ph,
                atol=1e-12,
            ):
                raise ValueError("Prediksi kontrol setelah reload berbeda.")
    save_json(local / "result.json", result)
    store.complete(key)
    store.progress(key, "completed")
    return result


def predict_stage3_descriptor(frame, ids, store, descriptor):
    arrays = []
    for key in descriptor["jobs"]:
        path = store.get(key)
        if path is None:
            raise ValueError("Model akhir belum selesai.")
        pre = restore_transform(
            json.loads((path / "preprocessing.json").read_text(encoding="utf-8"))
        )
        if descriptor["kind"] == "tree":
            model, _ = load_saved_model(path)
            with threadpool_limits(limits=2):
                p = model.predict_proba(pre.transform(frame.iloc[ids]))
        elif descriptor["kind"] == "neural":
            model = build_stage3_mlp()
            model.load_state_dict(
                torch.load(
                    path / (descriptor["policy"] + ".pt"),
                    map_location="cpu",
                    weights_only=True,
                )
            )
            p = neural_probabilities_stage3(
                model,
                pre.transform(frame.iloc[ids]).astype(np.float32),
                torch.device("cpu"),
            )
        else:
            raise ValueError("Jenis model akhir tidak dikenal.")
        arrays.append(p)
    return average_probabilities(arrays)


def seal_stage3_model(store, descriptor):
    if descriptor["kind"] not in ("neural", "tree") or not descriptor["jobs"]:
        raise ValueError("Deskripsi model akhir tidak valid.")
    if (
        descriptor["kind"] == "neural"
        and descriptor.get("policy") not in CHECKPOINT_POLICIES
    ):
        raise ValueError("Checkpoint model akhir tidak dikenal.")
    files = {}
    for key in descriptor["jobs"]:
        path = store.get(key)
        if path is None:
            raise ValueError("Model belum selesai.")
        for filename in (
            "preprocessing.json",
            descriptor["policy"] + ".pt"
            if descriptor["kind"] == "neural"
            else "estimator.joblib",
        ):
            source = path / filename
            files[f"jobs/{key}/{filename}"] = hashlib.sha256(
                source.read_bytes()
            ).hexdigest()
    lock = {
        "descriptor": descriptor,
        "files": files,
        "run_fingerprint": store.fingerprint,
        "selected_using": "out_of_fold_only",
        "test_role": "previously_seen_exploratory_reference",
    }
    target = store.remote / "model_lock.json"
    if target.is_file() and json.loads(target.read_text(encoding="utf-8")) != lock:
        raise ValueError("Model terkunci tidak boleh diganti setelah evaluasi.")
    save_json(target, lock)
    return lock


def evaluate_stage3_test(frame, store):
    lock_path = store.remote / "model_lock.json"
    if not lock_path.is_file():
        raise ValueError("Test hanya boleh dibuka setelah model terkunci.")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["run_fingerprint"] != store.fingerprint:
        raise ValueError("Model terkunci berasal dari run lain.")
    for name, checksum in lock["files"].items():
        path = store.remote / name
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != checksum
        ):
            raise ValueError("Artefak model terkunci berubah.")
    ids = np.flatnonzero(frame._split.eq("test"))
    target = store.remote / "test_reference.json"
    if target.is_file():
        result = json.loads(target.read_text(encoding="utf-8"))
        csv_path = store.remote / "test_reference_predictions.csv"
        if (
            result["lock_sha256"] != digest(lock)
            or hashlib.sha256(csv_path.read_bytes()).hexdigest()
            != result["predictions_sha256"]
        ):
            raise ValueError("Hasil Test referensi berubah.")
        table = pd.read_csv(csv_path)
        verify_prediction_table(frame, ids, table)
        table_probabilities(table)
        return result
    p = predict_stage3_descriptor(frame, ids, store, lock["descriptor"])
    table = probability_table(frame, ids, p)
    write_frame(store, "test_reference_predictions.csv", table)
    result = {
        "metrics": classification_metrics(frame.iloc[ids]._y.to_numpy(), p.argmax(1)),
        "lock_sha256": digest(lock),
        "test_evaluations": 1,
        "test_role": "previously_seen_exploratory_reference",
        "predictions_sha256": hashlib.sha256(
            (store.remote / "test_reference_predictions.csv").read_bytes()
        ).hexdigest(),
    }
    save_json(target, result)
    return result
