"""Identify archive copies without counting one acquisition twice.

Only exact CSV prefixes with extra post_task rows and identical event/RR/
calibration records are treated as equivalent. Source files are never changed.
"""
from collections import defaultdict
import csv
import hashlib
from pathlib import Path
import re

import pandas as pd


def participant_group(name):
    name = re.sub(r" - Copy(?: \(\d+\))?$", "", name, flags=re.IGNORECASE)
    return "Ramadhan" if name == "Ramadhan-2" else name


def _hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _equivalent_recording(primary, other):
    differences = []
    for filename in ["eye_tracking.csv", "hrv.csv"]:
        a = pd.read_csv(primary / filename, keep_default_na=False, low_memory=False)
        b = pd.read_csv(other / filename, keep_default_na=False, low_memory=False)
        if not a.columns.equals(b.columns):
            raise ValueError(f"ID sama tetapi kolom berbeda: {primary.name}, {other.name}, {filename}")
        short, long = (a, b) if len(a) <= len(b) else (b, a)
        if not short.reset_index(drop=True).equals(long.iloc[:len(short)].reset_index(drop=True)):
            raise ValueError(f"ID sama tetapi pengukuran berbeda: {primary.name}, {other.name}. Pilih satu dengan --subjects.")
        extra = long.iloc[len(short):]
        if len(extra) and ("study_phase" not in extra or not extra.study_phase.eq("post_task").all()):
            raise ValueError(f"Varian {other.name} memiliki tambahan selain post_task; pilih satu rekaman secara eksplisit.")
        differences.append(f"{filename}: {len(a)} vs {len(b)} baris; tambahan hanya post_task")
    for filename in ["task_events.csv", "rr_raw.csv", "eye_summary_calibration.json"]:
        a, b = primary / filename, other / filename
        if a.exists() != b.exists() or (a.exists() and _hash(a) != _hash(b)):
            raise ValueError(f"ID sama tetapi {filename} berbeda: {primary.name}, {other.name}; pilih satu dengan --subjects.")
    return "; ".join(differences)


def select_session_directories(root, subjects=None):
    root = Path(root)
    directories = [root] if (root / "eye_tracking.csv").exists() else [
        p for p in sorted(root.iterdir()) if p.is_dir() and (p / "eye_tracking.csv").exists()]
    if not directories:
        raise ValueError(f"Tidak ada eye_tracking.csv pada {root}")
    available = {p.name: p for p in directories}
    if subjects and set(subjects) - set(available):
        raise ValueError(f"Nama peserta tidak ditemukan: {sorted(set(subjects)-set(available))}. Pilihan: {list(available)}")
    grouped = defaultdict(list)
    for directory in directories:
        with (directory / "eye_tracking.csv").open(encoding="utf-8-sig", newline="") as stream:
            first = next(csv.DictReader(stream), None)
        session_id = first.get("session_id", "") if first else ""
        if not session_id:
            raise ValueError(f"ID sesi mata kosong: {directory}")
        grouped[session_id].append(directory)
    inventory, selected = [], []
    for session_id, group in grouped.items():
        group.sort(key=lambda p: (bool(re.search(r" - Copy(?: \(\d+\))?$", p.name, re.I)), p.name.casefold()))
        canonical = group[0]
        requested = [p for p in group if p.name in subjects] if subjects else [canonical]
        if subjects and len(requested) > 1:
            raise ValueError(f"Beberapa folder memakai ID {session_id}; pilih satu varian: {[p.name for p in requested]}")
        for directory in group:
            reason = "rekaman utama"
            if directory != canonical:
                # Explicit single-variant inspection is permitted even if variants differ.
                reason = (_equivalent_recording(canonical, directory) if not subjects
                          else "varian ID sesi sama; pemilihan eksplisit")
            chosen = directory in requested
            inventory.append({"folder": directory.name, "session_id": session_id,
                              "participant_group": participant_group(canonical.name),
                              "canonical_folder": canonical.name, "is_variant": directory != canonical,
                              "selected": chosen, "reason": reason})
            if chosen:
                selected.append(directory)
    order = {name: i for i, name in enumerate(subjects)} if subjects else None
    selected.sort(key=lambda p: order[p.name] if order else p.name.casefold())
    return selected, sorted(inventory, key=lambda row: row["folder"].casefold())
