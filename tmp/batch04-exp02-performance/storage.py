class ArtifactStore:
    def __init__(self, local, remote, signature):
        self.local, self.remote = Path(local), Path(remote)
        self._synced = set()
        self._published = {}
        self._runtime_rows = None
        self.io_seconds = {"read_seconds": 0.0, "write_seconds": 0.0}
        self.local.mkdir(parents=True, exist_ok=True)
        self.remote.mkdir(parents=True, exist_ok=True)
        identity = self.remote / "run.json"
        if (
            identity.exists()
            and json.loads(identity.read_text(encoding="utf-8")) != signature
        ):
            raise ValueError(
                "Resume ditolak: identitas data, split, implementasi, atau environment berubah."
            )
        self._copy_verified(identity, canonical(signature).encode())

    def _copy_verified(self, target, payload):
        started = time.perf_counter()
        for attempt in range(5):
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                if (
                    hashlib.sha256(target.read_bytes()).digest()
                    != hashlib.sha256(payload).digest()
                ):
                    raise OSError(f"Checksum salinan tidak cocok: {target}")
                self.io_seconds["write_seconds"] += time.perf_counter() - started
                return
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(0.5 * 2**attempt)

    def _folder(self, root, key):
        if not key or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in key
        ):
            raise ValueError("ID checkpoint tidak valid.")
        return root / key

    def _verified(self, root, key):
        folder = self._folder(root, key)
        if not (folder / "done.json").exists():
            return None
        marker = json.loads((folder / "done.json").read_text())
        payloads = {}
        for name, checksum in marker["checksums"].items():
            if Path(name).name != name or name in ("", ".", "..", "done.json"):
                raise ValueError("Nama artefak tidak valid.")
            payload = (folder / name).read_bytes()
            if hashlib.sha256(payload).hexdigest() != checksum:
                raise ValueError(f"Checkpoint rusak: {key}/{name}")
            payloads[name] = payload
        if "record.json" not in payloads:
            raise ValueError("Checkpoint tidak mempunyai record.")
        return payloads, marker

    def commit(self, key, record, artifacts=None):
        self._synced.discard(key)
        payloads = dict(artifacts or {})
        payloads["record.json"] = canonical(record).encode()
        for name in payloads:
            if Path(name).name != name or name in ("", ".", "..", "done.json"):
                raise ValueError("Nama artefak tidak valid.")
        marker = canonical(
            {
                "checksums": {
                    n: hashlib.sha256(v).hexdigest() for n, v in payloads.items()
                }
            }
        ).encode()
        for root in (self.local, self.remote):
            folder = self._folder(root, key)
            for name, payload in payloads.items():
                self._copy_verified(folder / name, payload)
            self._copy_verified(folder / "done.json", marker)
        self._synced.add(key)

    def read(self, key):
        started = time.perf_counter()
        try:
            return self._read(key)
        finally:
            self.io_seconds["read_seconds"] += time.perf_counter() - started

    def _read(self, key):
        if key in self._synced:
            complete = self._verified(self.local, key)
            if complete is not None:
                return json.loads(complete[0]["record.json"])
            self._synced.discard(key)
        complete = None
        failure = None
        for attempt in range(5):
            try:
                complete = self._verified(self.remote, key)
                failure = None
                break
            except OSError as exc:
                failure = exc
                if attempt < 4:
                    time.sleep(0.5 * 2**attempt)
        if complete is None:
            complete = self._verified(self.local, key)
            if complete is None:
                if failure is not None:
                    raise failure
                return None
            payloads, marker = complete
            for name, payload in payloads.items():
                self._copy_verified(self._folder(self.remote, key) / name, payload)
            self._copy_verified(
                self._folder(self.remote, key) / "done.json", canonical(marker).encode()
            )
        payloads, _ = complete
        for name, payload in payloads.items():
            self._copy_verified(self._folder(self.local, key) / name, payload)
        self._copy_verified(
            self._folder(self.local, key) / "done.json", canonical(complete[1]).encode()
        )
        self._synced.add(key)
        return json.loads(payloads["record.json"])

    def audit_remote(self):
        for key in sorted(self._synced):
            local = self._verified(self.local, key)
            remote = self._verified(self.remote, key)
            if local is None or remote is None or local[1] != remote[1]:
                raise ValueError(f"Checkpoint tidak lengkap atau berubah: {key}")
        for name, checksum in self._published.items():
            if (
                hashlib.sha256((self.remote / name).read_bytes()).hexdigest()
                != checksum
            ):
                raise ValueError(f"Output berubah atau rusak: {name}")

    def runtime_summary(self):
        if self._runtime_rows is None:
            self._runtime_rows = {}
            previous = self.remote / "runtime_summary.csv"
            if previous.exists():
                for row in pd.read_csv(io.BytesIO(previous.read_bytes())).to_dict(
                    "records"
                ):
                    self._runtime_rows[row["checkpoint"]] = row
        rows = []
        for key in sorted(self._synced):
            record = self.read(key)
            if key.startswith(("fold_", "fit_")) and record and "runtime" in record:
                timing_key = f"timing_{key}"
                timing = self.read(timing_key) or {}
                rows.append(
                    {
                        "checkpoint": key,
                        "window": record["window"],
                        "family": record["recipe"]["family"],
                        **record["runtime"],
                        **{
                            name: value
                            for name, value in timing.items()
                            if name != "checkpoint"
                        },
                        "convergence": canonical(record.get("convergence", [])),
                    }
                )
        for row in rows:
            self._runtime_rows[row["checkpoint"]] = row
        self.publish(
            "runtime_summary.csv",
            csv_bytes(
                pd.DataFrame(
                    [self._runtime_rows[k] for k in sorted(self._runtime_rows)],
                    columns=None
                    if self._runtime_rows
                    else ["checkpoint", "window", "family"],
                )
            ),
        )

    def artifact(self, key, name):
        if Path(name).name != name or self.read(key) is None:
            raise ValueError("Artefak belum lengkap atau nama tidak valid.")
        return (self.local / key / name).read_bytes()

    def publish(self, name, payload):
        if Path(name).name != name:
            raise ValueError("Nama output tidak valid.")
        self._copy_verified(self.remote / name, payload)
        self._published[name] = hashlib.sha256(payload).hexdigest()


def progress(store, **values):
    store.publish(
        "progress.json",
        canonical({"updated_at": datetime.now(UTC).isoformat(), **values}).encode(),
    )


class FitHeartbeat:
    def __init__(self, label, interval=30.0):
        self.label, self.interval = label, interval
        self.stop = threading.Event()

    def __enter__(self):
        self.started = time.perf_counter()
        print(f"Mulai {self.label}", flush=True)

        def report():
            while not self.stop.wait(self.interval):
                print(
                    f"Berjalan {self.label} | elapsed={time.perf_counter() - self.started:.1f}s",
                    flush=True,
                )

        self.thread = threading.Thread(target=report, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set()
        self.thread.join()
        state = "Selesai" if exc_type is None else "Terhenti"
        print(
            f"{state} {self.label} | elapsed={time.perf_counter() - self.started:.1f}s",
            flush=True,
        )


def start_run(project, experiment, hashes, manifest, resume=None):
    packages = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "cloudpickle",
        "optuna",
        "matplotlib",
    ]
    if experiment == "exp01":
        packages += ["xgboost", "catboost", "imbalanced-learn"]
    if experiment in ("exp03", "exp04"):
        packages += ["torch"] + (
            ["tabm", "rtdl_num_embeddings"] if experiment == "exp03" else []
        )
    signature = {
        "version": CODE_VERSION,
        "implementation": IMPLEMENTATION_HASH,
        "experiment": experiment,
        "dataset_sha256": hashes,
        "split_sha256": digest(manifest),
        "environment": {p: metadata.version(p) for p in packages},
        "python": platform.python_version(),
        "budget": BUDGET,
        "final_seeds": list(FINAL_SEEDS),
    }
    parent = Path(project) / "models/batch-04/outputs" / experiment
    remote = (
        Path(resume)
        if resume
        else parent
        / (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            + "_"
            + digest(signature)[:10]
        )
    )
    if resume and (
        remote.parent.resolve() != parent.resolve()
        or not (remote / "run.json").is_file()
    ):
        raise ValueError("RESUME_DIR harus menunjuk folder run eksperimen yang sesuai.")
    store = ArtifactStore(
        Path(tempfile.mkdtemp(prefix="cl-batch04-")), remote, signature
    )
    store.publish("split.json", canonical(manifest).encode())
    print("Run:", remote)
    print(
        "Evaluasi internal Train–validation–test; CV lima bagian di dalam Train, bukan per peserta."
    )
    return store
