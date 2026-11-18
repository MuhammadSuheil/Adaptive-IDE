"""Verified incremental checkpoints; training writes only to the Colab workspace."""

import errno
import shutil
import tarfile
import tempfile
import time
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path, PurePosixPath
from uuid import uuid4

from .artifacts import digest, file_hash, read_json, write_json

_ACTIVE = ContextVar("batch02_persistence", default=None)


class PersistenceError(RuntimeError):
    """The local workspace is intact, but durable publication did not finish."""


def safe_relative(value):
    text = str(value)
    path = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or ":" in text
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in text.split("/"))
    ):
        raise ValueError(f"Path artefak tidak aman: {text}")
    return path


def checkpoint(*, force=False, run=None):
    storage = _ACTIVE.get()
    if storage is not None:
        if run is not None:
            storage.bind(run)
        storage.publish(force=force)


class RunPersistence:
    def __init__(self, work_batch, drive_batch, experiment, interval=300):
        if experiment not in {f"exp{i:02d}" for i in range(3, 15)}:
            raise ValueError("Workspace Colab tersedia untuk exp03 sampai exp14.")
        self.work_batch = Path(work_batch).resolve()
        self.drive_batch = Path(drive_batch).resolve()
        if self.work_batch == self.drive_batch or self.work_batch.is_relative_to(
            self.drive_batch
        ):
            raise ValueError("Workspace harus di luar mount Drive.")
        self.experiment, self.interval = experiment, interval
        self.run = self.remote = None
        self.previous = {"sequence": 0, "files": {}, "parts": {}}
        self.last_saved = 0.0
        self.final_pending = False
        self.last_commit = None
        self.sequence = 0
        registry = self.work_batch / "registry/dataset_release.json"
        self.registry_sha256 = file_hash(registry) if registry.is_file() else None

    def _drive_available(self):
        if not self.drive_batch.is_dir():
            raise FileNotFoundError(
                errno.ENOENT,
                "Folder proyek Drive tidak tersedia; mount/pintasan mungkin terputus.",
            )
        if self.registry_sha256 is not None:
            registry = self.drive_batch / "registry/dataset_release.json"
            if file_hash(registry) != self.registry_sha256:
                raise ValueError(
                    "Registry Drive berubah dari snapshot workspace. Jangan mengganti release saat training berjalan."
                )

    @staticmethod
    def _retry(operation):
        for attempt in range(5):
            try:
                return operation()
            except OSError as error:
                if error.errno in (
                    errno.EACCES,
                    errno.EPERM,
                    errno.ENOSPC,
                    errno.EROFS,
                    getattr(errno, "EDQUOT", -2),
                ):
                    raise
                if attempt == 4:
                    raise
                time.sleep(2**attempt)

    def _upload(self, source, target, expected):
        def copy():
            self._drive_available()
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if file_hash(target) != expected:
                raise OSError(errno.EIO, f"Verifikasi upload gagal: {target}")

        return self._retry(copy)

    def preflight(self):
        root = self.drive_batch / "outputs" / self.experiment
        probe = root / f".probe-{uuid4().hex}"
        try:

            def check():
                self._drive_available()
                root.mkdir(parents=True, exist_ok=True)
                probe.write_bytes(b"batch02-storage-check")
                if probe.read_bytes() != b"batch02-storage-check":
                    raise OSError(errno.EIO, "Probe Drive tidak cocok")
                probe.unlink()

            self._retry(check)
        except OSError as error:
            raise PersistenceError(
                f"Output Drive tidak dapat ditulis/dibaca: {root}. Periksa izin Editor, mount, dan kuota."
            ) from error
        finally:
            try:
                probe.unlink(missing_ok=True)
            except OSError:
                pass

    def bind(self, run):
        run = Path(run).resolve()
        parent = self.work_batch / "outputs" / self.experiment
        if run.parent != parent or run.is_symlink():
            raise ValueError("Run tidak berada di workspace eksperimen ini.")
        if self.run is not None and self.run != run:
            raise ValueError("Satu persistence hanya boleh menangani satu run.")
        identity = read_json(run / "identity.json")
        if not identity.get("hash"):
            raise ValueError("Identitas run tidak lengkap.")
        self.run = run
        self.remote = self.drive_batch / "outputs" / self.experiment / run.name
        if (self.remote / "identity.json").exists() and read_json(
            self.remote / "identity.json"
        )["hash"] != identity["hash"]:
            raise ValueError("Identitas folder output Drive berbeda.")

    @contextmanager
    def session(self):
        self.preflight()
        token = _ACTIVE.set(self)
        try:
            yield self
        except BaseException as error:
            if self.run is not None and not isinstance(error, PersistenceError):
                try:
                    self.publish(force=True)
                except Exception as storage_error:  # noqa: BLE001 - preserve the original exception.
                    error.add_note(f"Checkpoint juga gagal: {storage_error}")
            raise
        else:
            self.finish()
        finally:
            _ACTIVE.reset(token)

    def _inventory(self):
        files = {}
        for path in sorted(self.run.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Symlink tidak boleh masuk checkpoint: {path}")
            if (
                path.is_file()
                and path.name != "writer.lock"
                and not path.name.endswith(".tmp")
            ):
                name = path.relative_to(self.run).as_posix()
                safe_relative(name)
                files[name] = {"sha256": file_hash(path), "size": path.stat().st_size}
        return files

    @staticmethod
    def _is_report(name):
        path = PurePosixPath(name)
        return not any(
            p.startswith("fold_") or p in ("inner", "inner_refit") for p in path.parts
        ) and path.suffix in (".json", ".csv", ".md", ".png")

    def publish(self, *, force=False, final=False):
        if self.run is None:
            return None
        if not force and time.monotonic() - self.last_saved < self.interval:
            return self.last_commit
        self.final_pending = self.final_pending or final
        state = self.work_batch / "_persistence"
        state.mkdir(parents=True, exist_ok=True)
        try:
            inventory = self._inventory()
            self.sequence = max(self.sequence, self.previous["sequence"]) + 1
            sequence = self.sequence
            changed = [
                name
                for name, item in inventory.items()
                if item["sha256"] != self.previous["files"].get(name, {}).get("sha256")
            ]
            entries = {
                name: dict(self.previous["files"][name])
                for name in inventory
                if name not in changed
            }
            parts = dict(self.previous["parts"])
            with tempfile.TemporaryDirectory(prefix="publish-", dir=state) as directory:
                temporary = Path(directory)
                if changed:
                    part = f"{sequence:010d}_{uuid4().hex}.tar.gz"
                    archive_path = temporary / part
                    with tarfile.open(archive_path, "w:gz", compresslevel=1) as archive:
                        for name in changed:
                            archive.add(self.run / name, arcname=name, recursive=False)
                            entries[name] = {**inventory[name], "part": part}
                    checksum = file_hash(archive_path)
                    self._upload(
                        archive_path,
                        self.remote / "checkpoints" / "parts" / part,
                        checksum,
                    )
                    parts[part] = checksum
                used = {entry["part"] for entry in entries.values()}
                parts = {name: sha for name, sha in parts.items() if name in used}
                reports = {}
                if self.final_pending:
                    for name, item in inventory.items():
                        if name != "status.json" and self._is_report(name):
                            self._upload(
                                self.run / name, self.remote / name, item["sha256"]
                            )
                            reports[name] = item["sha256"]
                else:
                    for name in (
                        "identity.json",
                        "progress.json",
                        "workload.json",
                        "capabilities.json",
                    ):
                        if name in inventory:
                            self._upload(
                                self.run / name,
                                self.remote / name,
                                inventory[name]["sha256"],
                            )
                manifest = {
                    "storage_version": 1,
                    "sequence": sequence,
                    "run_id": self.run.name,
                    "experiment": self.experiment,
                    "identity_hash": read_json(self.run / "identity.json")["hash"],
                    "files": entries,
                    "parts": parts,
                    "reports": reports,
                    "final": self.final_pending,
                }
                manifest_name = f"commit_{sequence:010d}_{digest(manifest)}.json"
                manifest_path = temporary / manifest_name
                write_json(manifest_path, manifest)
                self._upload(
                    manifest_path,
                    self.remote / "checkpoints" / manifest_name,
                    file_hash(manifest_path),
                )
                local_status = read_json(self.run / "status.json")
                terminal = local_status.get("status", "").startswith("complete")
                durable_status = {
                    **local_status,
                    "status": "storage_pending"
                    if terminal and not self.final_pending
                    else local_status.get("status", "running"),
                    "compute_status": local_status.get("status"),
                    "storage_status": "synced",
                    "checkpoint": manifest_name,
                    "last_checkpoint_sequence": sequence,
                }
                status_path = temporary / "status.json"
                write_json(status_path, durable_status)
                self._upload(
                    status_path, self.remote / "status.json", file_hash(status_path)
                )
                self.previous, self.last_commit = manifest, manifest_name
                self.last_saved = time.monotonic()
                print(
                    f"Checkpoint Drive tersimpan: {self.remote} / {sequence}",
                    flush=True,
                )
                return manifest_name
        except OSError as error:
            raise PersistenceError(
                f"Penyimpanan Drive belum selesai. Workspace tetap ada di {self.run}. "
                f"Checkpoint terakhir: {self.last_commit or 'belum ada'}. "
                "Perbaiki akses Drive lalu jalankan persistence.retry_publish(); jangan hapus runtime."
            ) from error

    def finish(self):
        return self.publish(force=True, final=True)

    def retry_publish(self):
        return self.publish(force=True, final=self.final_pending)

    def _validated_manifest(self, path, remote, downloads):
        manifest = read_json(path)
        if path.name.split("_")[-1] != digest(manifest) + ".json":
            raise ValueError("Hash commit tidak sesuai.")
        if (
            manifest.get("storage_version") != 1
            or manifest["experiment"] != self.experiment
            or manifest["run_id"] != remote.name
        ):
            raise ValueError("Identitas checkpoint tidak sesuai.")
        for name, expected in manifest["parts"].items():
            if len(safe_relative(name).parts) != 1:
                raise ValueError("Nama arsip tidak aman.")
            local = downloads / name
            if not local.exists() or file_hash(local) != expected:
                self._retry(
                    lambda name=name, local=local: shutil.copyfile(
                        remote / "checkpoints/parts" / name, local
                    )
                )
            if file_hash(local) != expected:
                raise ValueError("Arsip checkpoint hilang atau korup.")
        for name, entry in manifest["files"].items():
            safe_relative(name)
            if entry["part"] not in manifest["parts"]:
                raise ValueError("Referensi arsip tidak lengkap.")
        return manifest

    def restore(self, remote):
        remote = Path(remote).resolve()
        if remote.parent != self.drive_batch / "outputs" / self.experiment:
            raise ValueError("Resume harus menunjuk run Drive eksperimen yang sama.")
        destination = self.work_batch / "outputs" / self.experiment / remote.name
        cache = self.work_batch / "_persistence"
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="restore-", dir=cache) as directory:
            return self._restore(remote, destination, Path(directory))

    def _restore(self, remote, destination, temporary):
        downloads = temporary / "parts"
        downloads.mkdir()
        candidates = sorted(
            (remote / "checkpoints").glob("commit_*.json"), reverse=True
        )
        errors = []
        for path in candidates:
            try:
                manifest = self._validated_manifest(path, remote, downloads)
                break
            except (OSError, ValueError, KeyError, TypeError) as error:
                errors.append(f"{path.name}: {error}")
        else:
            raise ValueError(f"Tidak ada checkpoint valid untuk resume. {errors[:3]}")
        if errors:
            warnings.warn(
                "Checkpoint terbaru tidak valid; memakai commit valid sebelumnya.",
                RuntimeWarning,
            )
        self.previous, self.last_commit = manifest, path.name
        self.sequence = manifest["sequence"]
        if destination.exists():
            if (
                read_json(destination / "identity.json")["hash"]
                != manifest["identity_hash"]
            ):
                raise ValueError("Workspace lokal memiliki identitas berbeda.")
            print(
                "Melanjutkan workspace lokal tanpa menimpa progres:",
                destination,
                flush=True,
            )
            return destination
        staging = temporary / "run"
        staging.mkdir()
        by_part = {}
        for name, entry in manifest["files"].items():
            by_part.setdefault(entry["part"], []).append(name)
        for part, names in by_part.items():
            with tarfile.open(downloads / part, "r:gz") as archive:
                for name in names:
                    member = archive.getmember(name)
                    if (
                        not member.isfile()
                        or member.size != manifest["files"][name]["size"]
                    ):
                        raise ValueError(
                            "Anggota arsip tidak aman atau ukurannya berbeda."
                        )
                    target = staging / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with (
                        archive.extractfile(member) as source,
                        target.open("wb") as output,
                    ):
                        shutil.copyfileobj(source, output)
                    if file_hash(target) != manifest["files"][name]["sha256"]:
                        raise ValueError("Hash file hasil restore berbeda.")
        if read_json(staging / "identity.json")["hash"] != manifest["identity_hash"]:
            raise ValueError("Identitas dalam arsip berbeda dari manifest commit.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(destination)
        self.final_pending = False
        print(f"Resume dari checkpoint {path.name}", flush=True)
        return destination
