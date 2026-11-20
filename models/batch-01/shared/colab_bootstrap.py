"""Standard-library-only bootstrap, run before importing the training package."""

import errno
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path, PurePosixPath


def _hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _relative(value):
    text = str(value)
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or "\\" in text
        or ":" in text
        or any(p in ("", ".", "..") for p in text.split("/"))
    ):
        raise ValueError(f"Path snapshot tidak aman: {value}")
    return text


def _copy(source, target, expected):
    for attempt in range(5):
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if _hash(target) != expected:
                raise ValueError(f"Hash input/kode berubah: {source}")
            return
        except OSError as error:
            if (
                error.errno
                in (
                    errno.EACCES,
                    errno.EPERM,
                    errno.ENOSPC,
                    errno.EROFS,
                    getattr(errno, "EDQUOT", -2),
                )
                or attempt == 4
            ):
                raise
            time.sleep(2**attempt)


def prepare_workspace(drive_batch, work_parent=Path("/content/batch02-work")):
    """Copy one immutable release and code snapshot, never write to Drive."""
    drive_batch, work_parent = Path(drive_batch).resolve(), Path(work_parent).resolve()
    if work_parent == drive_batch or work_parent.is_relative_to(drive_batch):
        raise ValueError("Workspace tidak boleh berada di folder Drive sumber.")
    registry = drive_batch / "registry/dataset_release.json"
    if not registry.is_file():
        raise FileNotFoundError(
            "Release belum tersedia. Jalankan exp00 sampai selesai pada folder Drive bersama terlebih dahulu."
        )
    release = _json(registry)
    release_id = _relative(release["release_id"])
    if len(PurePosixPath(release_id).parts) != 1 or any(
        release.get("gates", {}).get(g) != "approved"
        for g in ("integrity", "additional_data", "proportions", "frozen")
    ):
        raise ValueError("Release belum disahkan atau ID release tidak valid.")
    source = drive_batch / "inputs" / release_id
    if _json(source / "release.json") != release:
        raise ValueError("Registry dan snapshot release tidak sama.")
    stream_manifest = _json(source / "streams/stream_manifest.json")
    if stream_manifest.get("release_id") != release_id or stream_manifest.get(
        "dataset_manifest_sha256"
    ) != release["files"].get("manifest.json"):
        raise ValueError(
            "Paket sensor tidak cocok dengan release. Selesaikan exp00 terlebih dahulu."
        )
    if set(stream_manifest["files"]) != {
        f"window_{s}s_step_5s.csv" for s in (15, 25, 45)
    }:
        raise ValueError("Paket sensor release tidak lengkap.")
    files = {
        f"inputs/{release_id}/{_relative(name)}": sha
        for name, sha in release["files"].items()
    }
    for name, sha in stream_manifest["files"].items():
        files[f"inputs/{release_id}/streams/{_relative(name)}"] = sha
    required = [
        registry,
        source / "release.json",
        source / "streams/stream_manifest.json",
        source / "streams/verification.json",
        drive_batch / "requirements.txt",
        drive_batch / "shared/colab_bootstrap.py",
    ]
    required.extend(sorted((drive_batch / "shared/batch02").glob("*.py")))
    if not (drive_batch / "shared/batch02/__init__.py").is_file():
        raise FileNotFoundError("Folder shared/batch02 belum lengkap.")
    for path in required:
        files[path.relative_to(drive_batch).as_posix()] = _hash(path)
    for relative, expected in files.items():
        if _hash(drive_batch / relative) != expected:
            raise ValueError(f"Checksum sumber tidak sesuai: {relative}")
    destination = work_parent / release_id
    if destination.exists():
        if _json(destination / "workspace.json").get("files") != files:
            raise ValueError(
                "Kode/input workspace berbeda. Jangan menimpa runtime yang sudah dipakai; mulai runtime baru setelah upload paket terbaru."
            )
        for relative, expected in files.items():
            if _hash(destination / relative) != expected:
                raise ValueError(
                    f"File workspace berubah: {relative}. Gunakan runtime baru."
                )
        return destination
    work_parent.mkdir(parents=True, exist_ok=True)
    required_bytes = (
        sum((drive_batch / name).stat().st_size for name in files) * 2 + 512 * 1024**2
    )
    if shutil.disk_usage(work_parent).free < required_bytes:
        raise OSError(
            errno.ENOSPC, "Ruang disk Colab tidak cukup untuk menyiapkan workspace."
        )
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=work_parent) as directory:
        staging = Path(directory) / "batch"
        staging.mkdir()
        for relative, expected in files.items():
            _copy(drive_batch / relative, staging / relative, expected)
        if _hash(registry) != files["registry/dataset_release.json"]:
            raise ValueError(
                "Registry berubah saat persiapan. Jangan menjalankan ulang exp00 selama eksperimen aktif."
            )
        (staging / "workspace.json").write_text(
            json.dumps({"release_id": release_id, "files": files}, indent=2),
            encoding="utf-8",
        )
        staging.rename(destination)
    print("Workspace Colab:", destination, flush=True)
    return destination


def prepare_checkpoint(source, expected, work_batch):
    """Stage explicitly licensed optional weights once; never download a model."""
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(
            "Isi SHA-256 checkpoint yang benar sebelum mengaktifkan TabPFN."
        )
    source, work_batch = Path(source), Path(work_batch)
    if _hash(source) != expected:
        raise ValueError("SHA-256 checkpoint TabPFN tidak sesuai.")
    target = work_batch / "checkpoints" / f"{expected}.ckpt"
    if target.exists():
        if _hash(target) != expected:
            raise ValueError("Cache checkpoint lokal berubah. Gunakan runtime baru.")
    else:
        _copy(source, target, expected)
    return target
