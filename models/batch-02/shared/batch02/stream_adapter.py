"""Read-only bridge to the canonical v2 sensor calculations, loaded only locally."""

import sys

import numpy as np
import pandas as pd

from .artifacts import digest, file_hash, read_json, write_json
from .contracts import load_release
from .features import FeatureStore


def prepare_stream_cache(config, source_root):
    release, frames, features, _ = load_release(config)
    snapshot = config.batch / "inputs" / release["release_id"]
    manifest = read_json(snapshot / "manifest.json")
    source_root = source_root.resolve()
    for relative, expected in manifest["source_sha256"].items():
        if file_hash(source_root / relative) != expected:
            raise ValueError(f"Sumber mentah berubah: {relative}")
    sys.path.insert(0, str(config.project_root))
    from analysis_sessions.dataset_fusion import build_dataset as shared
    from analysis_sessions.dataset_fusion import build_dataset_v2 as canonical

    configuration = manifest["quality_configuration"]
    collections = {seconds: [] for seconds in frames}
    for folder in manifest["selected_folders"]:
        eye, replay, _events, _ready, metadata, _audit = canonical.load_session(
            source_root / folder, configuration
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
                collections[seconds].append(row)
    destination = snapshot / "streams"
    if destination.exists():
        raise ValueError(
            "Cache sudah ada; jangan menimpa input run. Periksa manifest cache yang tersimpan."
        )
    staging = snapshot / "streams_staging"
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
    ]
    code = {str(p.relative_to(config.project_root)): file_hash(p) for p in dependencies}
    write_json(
        staging / "stream_manifest.json",
        {
            "release_id": release["release_id"],
            "files": files,
            "source_sha256": manifest["source_sha256"],
            "builder_sha256": code,
            "cache_id": digest([files, code]),
        },
    )
    store = FeatureStore(features, staging, release["release_id"])
    checked = store.verify_against(frames)
    write_json(staging / "verification.json", checked)
    staging.rename(destination)
    return checked
