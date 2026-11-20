"""Prepare portable sensor inputs once; Colab needs no raw recordings."""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import digest, file_hash, read_json, write_json
from .contracts import inspect_dataset, load_release
from .features import FeatureStore
from .registry import HRV


def build_portable_streams(config, source_root, destination=None):
    inspected = inspect_dataset(config)
    frames, features, manifest = (
        inspected["frames"],
        inspected["features"],
        inspected["manifest"],
    )
    source_root = source_root.resolve()
    for relative, expected in manifest["source_sha256"].items():
        if file_hash(source_root / relative) != expected:
            raise ValueError(f"Sumber mentah berubah: {relative}")
    sys.path.insert(0, str(config.project_root))
    from analysis_sessions.dataset_fusion import build_dataset as shared
    from analysis_sessions.dataset_fusion import build_dataset_v2 as canonical
    from analysis_sessions.dataset_fusion import v2_timestamps as timing

    configuration = manifest["quality_configuration"]
    collections = {seconds: [] for seconds in frames}
    context_path = config.dataset / "replacement_selection.json"
    context = read_json(context_path) if context_path.exists() else {}
    for folder in manifest["selected_folders"]:
        sid = str(
            pd.read_csv(
                source_root / folder / "eye_tracking.csv",
                nrows=1,
                usecols=["session_id"],
            ).session_id.iloc[0]
        )
        timestamp_policy = context.get("timestamp_policies", {}).get(sid)
        eye, replay, _events, _ready, metadata, _audit = canonical.load_session(
            source_root / folder, configuration, timestamp_policy
        )
        for seconds, windows in replay.items():
            for window in windows:
                start, end = (
                    shared.epoch_ms(window[n]) for n in ("window_start", "window_end")
                )
                rmssd, sdnn = (
                    shared.finite(window["rmssd_ms"]),
                    shared.finite(window["sdnn_ms"]),
                )
                usable = window["status"] == "ready" and rmssd >= 0 and sdnn >= 0
                row = {
                    "session_id": metadata["session_id"],
                    "window_start": window["window_start"],
                    "window_end": window["window_end"],
                    "available_at": window["window_end"],
                    "window_seconds": seconds,
                    "hrv_usable": usable,
                    "hrv_status": window["status"],
                    "hrv_coverage": window["coverage"],
                    "rmssd_ms": rmssd if usable else np.nan,
                    "sdnn_ms": sdnn if usable else np.nan,
                    "rmssd_baseline_ratio": rmssd / metadata["baseline_rmssd_ms"]
                    if usable and metadata["baseline_usable"]
                    else np.nan,
                    **shared.aggregate_eye(eye, start, end, configuration),
                }
                row.update(shared.quality_tier(row, configuration))
                row["eye_timestamp_ambiguous"] = timing.touches(
                    start, end, timestamp_policy
                )
                if row["eye_timestamp_ambiguous"]:
                    for feature in features:
                        if feature not in HRV:
                            row[feature] = np.nan
                    row["quality_tier"] = "poor"
                collections[seconds].append(row)
        print(f"Sensor inputs: {folder}", flush=True)
    destination = Path(destination or config.batch / "sensor_streams")
    if destination.exists():
        raise ValueError(
            "Cache sudah ada; jangan menimpa input run. Periksa manifest cache yang tersimpan."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + "_staging")
    staging.mkdir(exist_ok=False)
    files = {}
    for seconds, rows in collections.items():
        name = f"window_{seconds}s_step_5s.csv"
        pd.DataFrame(rows).sort_values(["session_id", "window_end"]).to_csv(
            staging / name, index=False
        )
        files[name] = file_hash(staging / name)
    dependencies = [
        *config.project_root.joinpath("analysis_sessions/dataset_fusion").glob("*.py"),
        *config.project_root.joinpath("hrv_monitor_prototype").glob("*.py"),
        config.project_root / "fuse_and_analyze.py",
        Path(__file__),
    ]
    code = {str(p.relative_to(config.project_root)): file_hash(p) for p in dependencies}
    write_json(
        staging / "stream_manifest.json",
        {
            "release_id": "portable",
            "dataset_manifest_sha256": inspected["files"]["manifest.json"],
            "training_performed": False,
            "timestamp_policy": "full_grid_mask_ambiguous_eye_features_keep_independent_hrv",
            "files": files,
            "source_sha256": manifest["source_sha256"],
            "builder_sha256": code,
            "cache_id": digest([files, code, inspected["files"]["manifest.json"]]),
        },
    )
    store = FeatureStore(features, staging, "portable")
    checked = store.verify_against(frames)
    write_json(staging / "verification.json", checked)
    for relative, expected in manifest["source_sha256"].items():
        if file_hash(source_root / relative) != expected:
            raise ValueError(f"Sumber berubah selama persiapan: {relative}")
    if (
        file_hash(config.dataset / "manifest.json")
        != inspected["files"]["manifest.json"]
    ):
        raise ValueError("Dataset berubah selama persiapan paket sensor.")
    staging.rename(destination)
    return checked


def attach_portable_streams(config, source=None):
    release, frames, features, _ = load_release(config)
    source = Path(source or config.batch / "sensor_streams")
    manifest = read_json(source / "stream_manifest.json")
    if manifest.get("dataset_manifest_sha256") != release["files"]["manifest.json"]:
        raise ValueError(
            "Paket sensor berasal dari dataset berbeda; salin paket dan dataset terbaru."
        )
    expected = {f"window_{s}s_step_5s.csv" for s in (15, 25, 45)}
    if set(manifest.get("files", {})) != expected:
        raise ValueError("Paket sensor tidak lengkap.")
    for name, checksum in manifest["files"].items():
        if file_hash(source / name) != checksum:
            raise ValueError(f"Hash paket sensor berubah: {name}")
    snapshot = config.batch / "inputs" / release["release_id"]
    destination = snapshot / "streams"
    if destination.exists():
        if (
            read_json(destination / "stream_manifest.json").get("cache_id")
            != manifest["cache_id"]
        ):
            raise ValueError("Cache release berbeda; jangan menimpa input eksperimen.")
        return FeatureStore(
            features, destination, release["release_id"]
        ).verify_against(frames)
    with tempfile.TemporaryDirectory(prefix=".streams-", dir=snapshot) as temporary:
        staging = Path(temporary) / "streams"
        staging.mkdir()
        for name, checksum in manifest["files"].items():
            shutil.copy2(source / name, staging / name)
            if file_hash(staging / name) != checksum:
                raise ValueError("Paket sensor berubah saat disalin.")
        write_json(
            staging / "stream_manifest.json",
            {**manifest, "release_id": release["release_id"]},
        )
        checked = FeatureStore(features, staging, release["release_id"]).verify_against(
            frames
        )
        write_json(staging / "verification.json", checked)
        staging.rename(destination)
    return checked
