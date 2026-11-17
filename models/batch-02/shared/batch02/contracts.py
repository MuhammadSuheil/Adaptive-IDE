"""Dataset release validation; no implicit approval or participant guessing."""

import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import digest, file_hash, read_json, write_json
from .metrics import LABELS

GATES = ("integrity", "additional_data", "proportions", "frozen")
WINDOWS = (15, 25, 45)


@dataclass(frozen=True)
class Config:
    project_root: Path
    dataset_root: Path | None = None
    batch_root: Path | None = None
    seed: int = 42
    threads: int = 4
    tabpfn_checkpoint: Path | None = None
    tabpfn_checkpoint_sha256: str | None = None
    tabpfn_license_confirmed: bool = False

    @property
    def batch(self):
        return Path(
            self.batch_root or self.project_root / "models" / "batch-02"
        ).resolve()

    @property
    def dataset(self):
        if self.dataset_root is not None:
            return Path(self.dataset_root).resolve()
        local = (
            self.project_root
            / "analysis_sessions"
            / "dataset_fusion"
            / "output"
            / "v2_session_phase"
        )
        return (
            local
            if local.exists()
            else self.project_root / "Dataset" / "v2_session_phase"
        ).resolve()


def require_approved_release(release):
    if any(release.get("gates", {}).get(g) != "approved" for g in GATES):
        raise ValueError("Training diblokir: semua gate dataset harus approved.")
    if not str(release.get("approved_by", "")).strip() or not release.get(
        "additional_sessions"
    ):
        raise ValueError(
            "Release memerlukan penanggung jawab dan daftar sesi tambahan aktual."
        )
    if not release.get("release_id") or not release.get("files"):
        raise ValueError("Snapshot release belum dibekukan.")


def validate_frame(frame, seconds, features):
    required = [
        "window_id",
        "session_id",
        "participant_id",
        "window_start",
        "window_end",
        "window_seconds",
        "quality_tier",
        "cognitive_load_label",
        *features,
    ]
    if set(required) - set(frame.columns):
        raise ValueError(
            f"Kolom wajib tidak lengkap: {sorted(set(required) - set(frame.columns))}"
        )
    if frame.empty or frame.window_id.duplicated().any():
        raise ValueError("Dataset kosong atau window_id berulang.")
    if not frame.cognitive_load_label.isin(LABELS).all():
        raise ValueError(
            "Target harus Low/Medium/High, bukan pemetaan difficulty lama."
        )
    if not frame.quality_tier.isin(["good", "acceptable"]).all():
        raise ValueError("Input utama hanya good_acceptable.")
    for name in ["session_id", "participant_id", "window_id"]:
        if frame[name].isna().any() or frame[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"Identitas kosong: {name}")
    for name in ["window_start", "window_end"]:
        if not frame[name].map(lambda x: pd.Timestamp(x).tzinfo is not None).all():
            raise ValueError("Timestamp memerlukan timezone.")
        frame[name] = pd.to_datetime(frame[name], utc=True)
    if (
        not np.allclose(
            (frame.window_end - frame.window_start).dt.total_seconds(), seconds
        )
        or not frame.window_seconds.eq(seconds).all()
    ):
        raise ValueError("Durasi window tidak sesuai.")
    if frame.duplicated(["session_id", "window_end"]).any():
        raise ValueError("Endpoint stream berulang.")
    frame[features] = frame[features].apply(pd.to_numeric, errors="raise")
    if np.isinf(frame[features].to_numpy(float)).any():
        raise ValueError("Fitur sensor mengandung infinity.")
    frame["y"] = frame.cognitive_load_label.map(dict(zip(LABELS, range(3)))).astype(int)
    return frame.sort_values(["session_id", "window_end"]).reset_index(drop=True)


def inspect_dataset(config):
    frames, features, files = {}, None, {}
    manifest = read_json(config.dataset / "manifest.json")
    policy = manifest.get(
        "admission_policy",
        manifest.get("quality_configuration", {}).get("admission_policy"),
    )
    if (
        policy
        not in (
            "window_quality_only",
            "conservative_quality_replacement_v1",
            "conservative_quality_replacement_v2",
        )
        or manifest.get("source_integrity_verified") is not True
    ):
        raise ValueError("Kebijakan/integritas dataset tidak didukung.")
    for seconds in WINDOWS:
        subdir = f"window_{seconds}s_step_5s"
        definition = read_json(config.dataset / subdir / "feature_columns.json")
        current = definition["features"]
        if (
            definition["target"] != "cognitive_load_label"
            or len(current) != 26
            or len(set(current)) != 26
        ):
            raise ValueError("Allowlist/target tidak sesuai kontrak 26 fitur.")
        if features is not None and current != features:
            raise ValueError("Allowlist antarwindow berbeda.")
        features = current
        frames[seconds] = validate_frame(
            pd.read_csv(config.dataset / subdir / "good_acceptable.csv"),
            seconds,
            features,
        )
        if policy != "window_quality_only":
            selected = frames[seconds].get("curation_selected")
            if (
                selected is None
                or not selected.astype(str).str.lower().isin(("true", "1")).all()
            ):
                raise ValueError("CSV curated memuat window yang tidak dipilih.")
        for name in ["good_acceptable.csv", "feature_columns.json"]:
            key = f"{subdir}/{name}"
            files[key] = file_hash(config.dataset / key)
            if manifest.get("output_sha256", {}).get(key) != files[key]:
                raise ValueError(f"Checksum dataset tidak sesuai manifest: {key}")
    if policy != "window_quality_only":
        for name in ("replacement_selection.json", "replacement_pairs.csv"):
            files[name] = file_hash(config.dataset / name)
            if files[name] != manifest.get("output_sha256", {}).get(name):
                raise ValueError(f"Checksum seleksi dataset tidak sesuai: {name}")
    files["manifest.json"] = file_hash(config.dataset / "manifest.json")
    summary = {
        seconds: {
            "rows": len(frame),
            "classes": frame.cognitive_load_label.value_counts().to_dict(),
            "groups_metadata": frame.participant_id.nunique(),
        }
        for seconds, frame in frames.items()
    }
    return {
        "frames": frames,
        "features": features,
        "files": files,
        "summary": summary,
        "manifest": manifest,
    }


def attach_registry(frames, registry):
    records = registry.get("sessions", [])
    if not records or any(
        r.get("confirmed") is not True or not str(r.get("person_id", "")).strip()
        for r in records
    ):
        raise ValueError("Konfirmasi person_id lintas sesi diperlukan pada registry.")
    mapping = {r["session_id"]: r["person_id"] for r in records}
    if len(mapping) != len(records):
        raise ValueError("Session ID berulang pada registry.")
    results = {}
    for seconds, frame in frames.items():
        frame = frame.copy()
        frame["person_id"] = frame.session_id.map(mapping)
        for name in ("collection_wave", "device_id"):
            frame[name] = frame.session_id.map(
                {r["session_id"]: r.get(name, "") for r in records}
            )
        if frame.person_id.isna().any():
            raise ValueError("Registry belum mencakup seluruh sesi.")
        results[seconds] = frame
    return results


def freeze_release(
    config,
    registry,
    additional_sessions,
    approved_by,
    proportions_approved=False,
    *,
    evaluation_scope="current_snapshot",
    collection_status="ongoing",
    expected_manifest_sha256=None,
):
    if not proportions_approved or not approved_by.strip() or not additional_sessions:
        raise ValueError(
            "Persetujuan proporsi, penanggung jawab, dan sesi tambahan wajib diisi."
        )
    inspected = inspect_dataset(config)
    if (
        expected_manifest_sha256 is not None
        and inspected["files"]["manifest.json"] != expected_manifest_sha256
    ):
        raise ValueError(
            "Hash manifest berbeda dari snapshot yang menerima persetujuan."
        )
    frames = attach_registry(inspected["frames"], registry)
    source_sessions = {s for f in frames.values() for s in f.session_id}
    if not set(additional_sessions).issubset(source_sessions):
        raise ValueError("Sesi tambahan tidak ditemukan pada dataset final.")
    from .splits import make_manifests

    manifests = make_manifests(frames, config.seed)
    identity = {
        "files": inspected["files"],
        "registry": registry,
        "splits": manifests,
        "evaluation_scope": evaluation_scope,
        "collection_status": collection_status,
    }
    release_id = digest(identity)[:20]
    destination = config.batch / "inputs" / release_id
    if destination.exists():
        raise ValueError(
            "Snapshot sudah ada; gunakan release yang tersimpan atau periksa perubahan input."
        )
    destination.mkdir(parents=True)
    for relative in inspected["files"]:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config.dataset / relative, target)
        if file_hash(target) != inspected["files"][relative]:
            raise ValueError(f"Sumber berubah selama pembekuan snapshot: {relative}")
    write_json(destination / "session_registry.json", registry)
    write_json(destination / "splits.json", manifests)
    release = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "release_id": release_id,
        "gates": dict.fromkeys(GATES, "approved"),
        "approved_by": approved_by,
        "evaluation_scope": evaluation_scope,
        "collection_status": collection_status,
        "additional_sessions": additional_sessions,
        "files": {
            **inspected["files"],
            "session_registry.json": file_hash(destination / "session_registry.json"),
            "splits.json": file_hash(destination / "splits.json"),
        },
        "summary": inspected["summary"],
    }
    write_json(destination / "release.json", release)
    write_json(config.batch / "registry" / "dataset_release.json", release)
    return release


def load_release(config):
    release = read_json(config.batch / "registry" / "dataset_release.json")
    require_approved_release(release)
    source = config.batch / "inputs" / release["release_id"]
    for relative, expected in release["files"].items():
        if file_hash(source / relative) != expected:
            raise ValueError(f"Hash snapshot berubah: {relative}")
    snapshot = Config(
        config.project_root, source, config.batch, config.seed, config.threads
    )
    inspected = inspect_dataset(snapshot)
    frames = attach_registry(
        inspected["frames"], read_json(source / "session_registry.json")
    )
    return release, frames, inspected["features"], read_json(source / "splits.json")
