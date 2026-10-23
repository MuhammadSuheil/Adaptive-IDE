#!/usr/bin/env python3
"""fuse_and_analyze.py - Multimodal Data Fusion & Cognitive Load Analysis

Menggabungkan data time-series dari Eye Tracking (per-frame ~30 FPS) dan HRV Monitor
(sliding window 25s dengan step 5s) ke dalam satu timeline terpadu, mengekstraksi
fitur kognitif neuroergonomis, dan mengklasifikasikan Cognitive Load State programmer.

Dapat dijalankan secara mandiri pada folder sesi:
    python fuse_and_analyze.py --session-dir sessions/SESSION_NAME
"""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import statistics
import sys


TIMELINE_COLUMNS = [
    'session_id', 'study_session_status', 'study_phase', 'difficulty_level',
    'puzzle_set', 'puzzle_id', 'window_phase_mix',
    'window_idx', 'window_start', 'window_end', 'timestamp_relative', 'timestamp_relative_ms',
    'window_seconds', 'hrv_status', 'hrv_coverage',
    'rmssd_ms', 'sdnn_ms', 'baseline_rmssd_ms',
    'rmssd_ratio', 'eye_frames_count', 'on_screen_ratio',
    'face_detected_ratio', 'mean_dwell_ms', 'max_dwell_ms',
    'section_transitions', 'dominant_section', 'mean_iris_delta',
    'mean_blink_rate_bpm', 'head_shifted_ratio', 'cognitive_state',
    'cognitive_load_score', 'cognitive_state_desc',
]

INDONESIA_WESTERN_TIME = timezone(timedelta(hours=7))


def format_wib(iso_str: str) -> str:
    """Konversi ISO-8601 aware ke waktu Indonesia Barat (UTC+7)."""
    return datetime.fromisoformat(iso_str).astimezone(INDONESIA_WESTERN_TIME).isoformat()


def parse_iso_to_ms(iso_str: str) -> float:
    """Konversi string ISO 8601 ke epoch milliseconds UTC."""
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.timestamp() * 1000.0
    except Exception:
        return 0.0


def joint_origin_ms(session_dir: Path, task_windows: list) -> float | None:
    """Shared zero is the UTC time when both sensor calibrations are complete."""
    status_path = session_dir / "hrv_calib_status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        value = status.get("both_calibrated_at_utc")
        if value:
            return parse_iso_to_ms(value)
    except (OSError, json.JSONDecodeError):
        pass
    if task_windows:
        first = task_windows[0]
        end_ms = parse_iso_to_ms(first.get("window_end", ""))
        return end_ms - safe_float(first.get("window_seconds")) * 1000.0
    return None


def safe_float(val, default=0.0):
    try:
        if val is None or val == "" or val == "None":
            return default
        f = float(val)
        return f if math.isfinite(f) else default
    except (ValueError, TypeError):
        return default


def load_study_events(session_dir: Path):
    """Read CodeArrange events; a missing/invalid ledger stays explicitly unlabelled."""
    event_path = Path(session_dir) / "task_events.csv"
    if not event_path.exists():
        return [], "unlabeled"
    events = []
    try:
        with event_path.open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("session_id") != Path(session_dir).name:
                    continue
                timestamp = safe_float(row.get("event_time_ms"), None)
                if timestamp is None or timestamp <= 0:
                    continue
                events.append({
                    "event_time_ms": timestamp,
                    "event_type": (row.get("event_type") or "").strip(),
                    "puzzle_set": (row.get("puzzle_set") or "").strip(),
                    "difficulty_level": (row.get("difficulty_level") or "").strip(),
                    "puzzle_id": (row.get("puzzle_id") or "").strip(),
                    "puzzle_title": (row.get("puzzle_title") or "").strip(),
                    "outcome": (row.get("outcome") or "").strip(),
                })
    except (OSError, csv.Error, UnicodeError) as error:
        print(f"[Fuse] Tidak dapat membaca event penelitian: {error}", file=sys.stderr)
        return [], "unlabeled"
    events.sort(key=lambda event: event["event_time_ms"])
    if not events:
        return [], "unlabeled"
    complete = any(event["event_type"] == "study_ended" for event in events)
    return events, "completed" if complete else "interrupted"


def calibration_ready_ms(session_dir: Path):
    try:
        with (Path(session_dir) / "hrv_calib_status.json").open(encoding="utf-8") as f:
            status = json.load(f)
        value = status.get("both_calibrated_at_utc")
        if value:
            return parse_iso_to_ms(value)
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return None


def _empty_context(phase):
    return {"study_phase": phase, "difficulty_level": "", "puzzle_set": "", "puzzle_id": "", "puzzle_title": ""}


def context_at(timestamp_ms: float, events: list, ready_ms=None):
    """Resolve a sample's phase and current set/question from the event ledger."""
    if not events:
        return _empty_context("unlabeled")
    if ready_ms is not None and timestamp_ms < ready_ms:
        return _empty_context("baseline")

    before = [event for event in events if event["event_time_ms"] <= timestamp_ms]
    latest_set_start = None
    latest_set_end = None
    latest_study_end = None
    for event in before:
        kind = event["event_type"]
        if kind == "set_started":
            latest_set_start = event
            latest_set_end = None
            latest_study_end = None
        elif kind == "set_ended":
            latest_set_end = event
        elif kind == "study_ended":
            latest_study_end = event

    if latest_study_end is not None:
        post_start = (latest_set_end["event_time_ms"] if latest_set_end is not None
                      and (latest_set_start is None or latest_set_end["event_time_ms"] > latest_set_start["event_time_ms"])
                      else latest_study_end["event_time_ms"])
        if timestamp_ms >= post_start:
            return _empty_context("post_task")

    if latest_set_start is None:
        return _empty_context("pre_task")

    if latest_set_end is not None and latest_set_end["event_time_ms"] > latest_set_start["event_time_ms"]:
        later_rating = [event for event in before if event["event_time_ms"] > latest_set_end["event_time_ms"]
                        and event["event_type"] in ("rating_started", "rating_ended")]
        if later_rating and later_rating[-1]["event_type"] == "rating_started":
            return _empty_context("rating")
        future_starts = [event for event in events
                         if event["event_type"] == "set_started"
                         and event["event_time_ms"] > latest_set_end["event_time_ms"]]
        future_study_ends = [event for event in events
                             if event["event_type"] == "study_ended"
                             and event["event_time_ms"] > latest_set_end["event_time_ms"]]
        next_start_ms = future_starts[0]["event_time_ms"] if future_starts else None
        end_ms = future_study_ends[0]["event_time_ms"] if future_study_ends else None
        if end_ms is not None and (next_start_ms is None or end_ms < next_start_ms):
            return _empty_context("post_task")
        if next_start_ms is not None:
            return _empty_context("rest")
        return _empty_context("interrupted")

    puzzle_starts = [event for event in before
                     if event["event_type"] == "puzzle_started"
                     and event["event_time_ms"] >= latest_set_start["event_time_ms"]]
    current_puzzle = puzzle_starts[-1] if puzzle_starts else {}
    phase = "active_task" if current_puzzle else "feedback"
    for event in before:
        if event["event_time_ms"] < latest_set_start["event_time_ms"]:
            continue
        if event["event_type"] == "puzzle_started":
            phase = "active_task"
        elif event["event_type"] == "attempt_submitted":
            phase = "feedback"
        elif event["event_type"] == "rating_started":
            phase = "rating"
        elif event["event_type"] == "rating_ended":
            phase = "feedback"
    return {
        "study_phase": phase,
        "difficulty_level": latest_set_start.get("difficulty_level", ""),
        "puzzle_set": latest_set_start.get("puzzle_set", ""),
        "puzzle_id": current_puzzle.get("puzzle_id", ""),
        "puzzle_title": current_puzzle.get("puzzle_title", ""),
    }


def interval_segments(start_ms: float, end_ms: float, events: list, ready_ms=None):
    """Return time-proportional phase/context segments for an HRV window."""
    if not events or end_ms <= start_ms:
        return []
    boundaries = {start_ms, end_ms}
    if ready_ms is not None and start_ms < ready_ms < end_ms:
        boundaries.add(ready_ms)
    boundaries.update(event["event_time_ms"] for event in events
                      if start_ms < event["event_time_ms"] < end_ms)
    ordered = sorted(boundaries)
    segments = []
    for left, right in zip(ordered, ordered[1:]):
        if right <= left:
            continue
        segment = {
            **context_at((left + right) / 2.0, events, ready_ms),
            "duration_seconds": round((right - left) / 1000.0, 3),
            "proportion": round((right - left) / (end_ms - start_ms), 4),
        }
        context_keys = ("study_phase", "difficulty_level", "puzzle_set", "puzzle_id", "puzzle_title")
        if segments and all(segments[-1].get(key) == segment.get(key) for key in context_keys):
            segments[-1]["duration_seconds"] = round(segments[-1]["duration_seconds"] + segment["duration_seconds"], 3)
            segments[-1]["proportion"] = round(segments[-1]["proportion"] + segment["proportion"], 4)
        else:
            segments.append(segment)
    return segments


def summarize_segments(segments):
    if not segments:
        return {"study_phase": "unlabeled", "difficulty_level": "", "puzzle_set": "", "puzzle_id": "", "window_phase_mix": ""}

    def unique_or_mixed(key):
        values = [segment.get(key, "") for segment in segments]
        nonempty = list(dict.fromkeys(value for value in values if value))
        if not nonempty:
            return ""
        if len(nonempty) == 1 and all(value == nonempty[0] for value in values):
            return nonempty[0]
        return "mixed"

    phases = list(dict.fromkeys(segment["study_phase"] for segment in segments))
    return {
        "study_phase": phases[0] if len(phases) == 1 else "mixed",
        "difficulty_level": unique_or_mixed("difficulty_level"),
        "puzzle_set": unique_or_mixed("puzzle_set"),
        "puzzle_id": unique_or_mixed("puzzle_id"),
        "window_phase_mix": json.dumps(segments, ensure_ascii=False, separators=(",", ":")),
    }


def annotate_sensor_csv(path: Path, session_id: str, events: list, ready_ms=None, kind="eye"):
    """Add/recompute labels in one finished sensor CSV without changing measurements."""
    if not path.exists() or not events:
        return 0
    temp_path = path.with_name(path.name + ".labeling.tmp")
    changed = 0
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            fieldnames = list(reader.fieldnames or [])
            for column in ("session_id", "study_phase", "difficulty_level", "puzzle_set", "puzzle_id"):
                if column not in fieldnames:
                    fieldnames.append(column)
            if kind == "hrv" and "window_phase_mix" not in fieldnames:
                fieldnames.append("window_phase_mix")
            with temp_path.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for row in reader:
                    row["session_id"] = session_id
                    if kind == "hrv" and row.get("phase") == "baseline":
                        row["study_phase"] = "baseline"
                        row["difficulty_level"] = row["puzzle_set"] = row["puzzle_id"] = ""
                        row["window_phase_mix"] = ""
                    elif kind == "hrv":
                        start_ms = parse_iso_to_ms(row.get("window_start", ""))
                        end_ms = parse_iso_to_ms(row.get("window_end", ""))
                        row.update(summarize_segments(interval_segments(start_ms, end_ms, events, ready_ms)))
                    else:
                        timestamp_ms = (safe_float(row.get("timestamp_ms"), None) if kind == "eye"
                                        else parse_iso_to_ms(row.get("received_at", "")))
                        context = (context_at(timestamp_ms, events, ready_ms)
                                   if timestamp_ms is not None and timestamp_ms > 0 else _empty_context("unlabeled"))
                        row.update({key: context[key] for key in ("study_phase", "difficulty_level", "puzzle_set", "puzzle_id")})
                    writer.writerow(row)
                    changed += 1
        os.replace(temp_path, path)
    except (OSError, csv.Error, UnicodeError) as error:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        print(f"[Fuse] Gagal memberi label {path.name}: {error}", file=sys.stderr)
        return 0
    return changed


def load_hrv_data(hrv_csv_path: Path):
    """Membaca baseline dan baris window tugas dari hrv.csv."""
    if not hrv_csv_path.exists():
        raise FileNotFoundError(f"File HRV tidak ditemukan: {hrv_csv_path}")

    baseline_info = {
        "rmssd_ms": None,
        "sdnn_ms": None,
        "status": "unavailable"
    }
    task_windows = []

    with hrv_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            phase = row.get("phase", "").strip()
            if phase == "baseline":
                rmssd = safe_float(row.get("rmssd_ms"), None)
                sdnn = safe_float(row.get("sdnn_ms"), None)
                status = row.get("status", "")
                if rmssd is not None:
                    baseline_info["rmssd_ms"] = rmssd
                if sdnn is not None:
                    baseline_info["sdnn_ms"] = sdnn
                baseline_info["status"] = status
            elif phase == "task":
                task_windows.append(row)

    return baseline_info, task_windows


def load_eye_frames(eye_csv_path: Path):
    """Membaca semua frame dari eye_tracking.csv (atau session_*.csv)."""
    if not eye_csv_path.exists():
        raise FileNotFoundError(f"File Eye Tracking tidak ditemukan: {eye_csv_path}")

    frames = []
    with eye_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_ms = safe_float(row.get("timestamp_ms"))
            if ts_ms > 0:
                frames.append({
                    "timestamp_ms": ts_ms,
                    "frame_index": int(safe_float(row.get("frame_index"))),
                    "gaze_x": safe_float(row.get("gaze_x_smooth")),
                    "gaze_y": safe_float(row.get("gaze_y_smooth")),
                    "grid_row": int(safe_float(row.get("grid_row"), -1)),
                    "grid_col": int(safe_float(row.get("grid_col"), -1)),
                    "section": row.get("section", "unknown"),
                    "confidence": safe_float(row.get("confidence")),
                    "dwell_time_ms": safe_float(row.get("dwell_time_ms")),
                    "nrevisit_count": int(safe_float(row.get("nrevisit_count"))),
                    "transition_rate": safe_float(row.get("transition_rate")),
                    "iris_size_delta": safe_float(row.get("iris_size_delta")),
                    "iris_valid": (row.get("iris_valid", "true").lower() in ("true", "1")
                                   and math.isfinite(safe_float(row.get("iris_size_delta"), float('nan')))),
                    "face_detected": row.get("face_detected", "").lower() in ("true", "1"),
                    "gaze_status": row.get("gaze_status", "unknown"),
                    "head_pose_shifted": row.get("head_pose_shifted", "").lower() in ("true", "1"),
                    "is_blinking": row.get("is_blinking", "").lower() in ("true", "1"),
                    "blink_rate_bpm": safe_float(row.get("blink_rate_bpm"))
                })
    return frames


def aggregate_eye_features(window_frames: list):
    """Menghitung ringkasan fitur eye tracking dalam satu window waktu."""
    if not window_frames:
        return {
            "eye_frame_count": 0,
            "on_screen_ratio": 0.0,
            "face_detected_ratio": 0.0,
            "mean_dwell_ms": 0.0,
            "max_dwell_ms": 0.0,
            "total_transitions": 0,
            "dominant_section": "none",
            "mean_iris_delta": 0.0,
            "mean_blink_rate_bpm": 0.0,
            "head_shifted_ratio": 0.0
        }

    n_frames = len(window_frames)
    on_screen_count = sum(1 for f in window_frames if f["gaze_status"] == "on_screen")
    face_count = sum(1 for f in window_frames if f["face_detected"])
    head_shift_count = sum(1 for f in window_frames if f["head_pose_shifted"])

    dwells = [f["dwell_time_ms"] for f in window_frames if f["dwell_time_ms"] > 0]
    iris_deltas = [f["iris_size_delta"] for f in window_frames
                   if f["face_detected"] and f.get("iris_valid", True)
                   and f['gaze_status'] in ('on_screen', 'gaze_outside_screen')
                   and not f.get('is_blinking', False)]
    blink_rates = [f["blink_rate_bpm"] for f in window_frames if f["blink_rate_bpm"] > 0]

    # Hitung transisi antar section
    transitions = 0
    prev_section = None
    section_counts = {}
    for f in window_frames:
        sec = f["section"]
        section_counts[sec] = section_counts.get(sec, 0) + 1
        if prev_section is not None and sec != prev_section and sec not in ("unknown", "off_screen"):
            transitions += 1
        prev_section = sec

    dominant_section = max(section_counts, key=section_counts.get) if section_counts else "none"

    return {
        "eye_frame_count": n_frames,
        "on_screen_ratio": round(on_screen_count / n_frames, 3),
        "face_detected_ratio": round(face_count / n_frames, 3),
        "mean_dwell_ms": round(statistics.mean(dwells), 1) if dwells else 0.0,
        "max_dwell_ms": round(max(dwells), 1) if dwells else 0.0,
        "total_transitions": transitions,
        "dominant_section": dominant_section,
        "mean_iris_delta": round(statistics.mean(iris_deltas), 4) if iris_deltas else 0.0,
        "mean_blink_rate_bpm": round(statistics.mean(blink_rates), 1) if blink_rates else 0.0,
        "head_shifted_ratio": round(head_shift_count / n_frames, 3)
    }


def classify_cognitive_state(rmssd_ratio: float, on_screen_ratio: float,
                             transitions: int, mean_dwell_ms: float,
                             blink_rate: float) -> tuple:
    """Mengklasifikasikan status kognitif programmer berdasarkan kaidah neuroergonomics.

    Returns:
        (state_label: str, cognitive_load_score: float [0-100], explanation: str)
    """
    # 1. Deteksi Distracted / Off-screen
    if on_screen_ratio < 0.45:
        return "DISTRACTED", 20.0, "Pandangan dominan keluar layar / tidak menatap editor"

    # Baseline ratio: Jika tidak ada baseline, asumsikan 1.0 (normal)
    hrv_factor = rmssd_ratio if (rmssd_ratio is not None and rmssd_ratio > 0) else 1.0

    # Komponen beban kognitif (0 - 100)
    # Penurunan HRV mengindikasikan beban simpatis tinggi (stres/mental strain)
    hrv_strain = max(0.0, min(100.0, (1.2 - hrv_factor) * 80.0))

    # Transisi tinggi (mata bolak-balik) menandakan pencarian intensif / confusion
    transition_strain = min(50.0, transitions * 3.5)

    # Durasi tatap lama (stuck pada baris kode tertentu)
    dwell_strain = min(30.0, (mean_dwell_ms / 1500.0) * 20.0)

    # Indeks beban kognitif terpadu (skala 0 - 100)
    score = round(min(100.0, max(0.0, (0.45 * hrv_strain) + (0.35 * transition_strain) + (0.20 * dwell_strain))), 1)

    # Klasifikasi state berbasis profil paper
    if (hrv_factor < 0.60) or (hrv_factor < 0.70 and score >= 40.0) or (blink_rate > 35.0 and score >= 50.0):
        state = "OVERLOADED"
        desc = "Penurunan HRV tajam (<70% baseline) disertai beban kognitif tinggi / kelelahan mental"
    elif transitions >= 8 or (mean_dwell_ms > 900.0 and transitions >= 4) or (dwell_strain > 15.0 and score >= 35.0):
        state = "CONFUSED"
        desc = "Transisi area tinggi / mata bolak-balik dan fiksasi lama (indikasi debugging intensif)"
    elif transitions >= 5 and mean_dwell_ms < 600.0:
        state = "SCANNING"
        desc = "Mata bergerak cepat memindai struktur kode umum tanpa fiksasi dalam"
    else:
        state = "FOCUSED"
        desc = "Variabilitas jantung stabil, tatapan teratur pada editor kode utama"


    return state, score, desc


def fuse_session(session_dir: Path, output_csv: Path = None, output_json: Path = None):
    """Proses fusi utama: menyelaraskan HRV dan Eye Tracking ke dalam timeline terpadu."""
    session_dir = Path(session_dir)
    hrv_csv = session_dir / "hrv.csv"
    
    # Cari file eye tracking (bisa eye_tracking.csv atau session_*.csv)
    eye_csv = session_dir / "eye_tracking.csv"
    if not eye_csv.exists():
        candidates = list(session_dir.glob("session_*_*.csv"))
        if candidates:
            eye_csv = candidates[0]

    baseline_info, task_windows = load_hrv_data(hrv_csv)
    session_id = session_dir.name
    study_events, study_session_status = load_study_events(session_dir)
    ready_ms = calibration_ready_ms(session_dir)
    if study_events:
        for filename, kind in (
            ("eye_tracking.csv", "eye"),
            ("eye_tracking_compact.csv", "eye"),
            ("hrv.csv", "hrv"),
            ("rr_raw.csv", "rr"),
        ):
            count = annotate_sensor_csv(session_dir / filename, session_id, study_events, ready_ms, kind)
            if count:
                print(f"[Fuse] Label fase ditambahkan ke {filename}: {count} baris.")
    else:
        print("[Fuse] task_events.csv tidak tersedia/valid; data sesi ini tetap tanpa label fase.", file=sys.stderr)
    eye_frames = load_eye_frames(eye_csv)

    baseline_rmssd = baseline_info.get("rmssd_ms")
    shared_origin_ms = joint_origin_ms(session_dir, task_windows)
    print(f"[Fuse] Memproses sesi: {session_dir.name}")
    print(f"[Fuse] Baseline RMSSD: {baseline_rmssd} ms (status: {baseline_info.get('status')})")
    print(f"[Fuse] Ditemukan {len(task_windows)} window HRV dan {len(eye_frames)} frame mata.")

    if not task_windows:
        print("[Fuse] Peringatan: Tidak ada window tugas HRV yang ditemukan.")

    # Sort frame berdasarkan timestamp_ms
    eye_frames.sort(key=lambda x: x["timestamp_ms"])

    timeline = []
    state_distribution = {"FOCUSED": 0, "CONFUSED": 0, "SCANNING": 0, "OVERLOADED": 0, "DISTRACTED": 0}

    for idx, w in enumerate(task_windows):
        w_start_str = w.get("window_start", "")
        w_end_str = w.get("window_end", "")
        w_start_ms = parse_iso_to_ms(w_start_str)
        w_end_ms = parse_iso_to_ms(w_end_str)
        study_labels = summarize_segments(interval_segments(w_start_ms, w_end_ms, study_events, ready_ms))

        # Filter frame mata dalam rentang window
        window_frames = [f for f in eye_frames if w_start_ms <= f["timestamp_ms"] <= w_end_ms]
        eye_stats = aggregate_eye_features(window_frames)

        # Metrik HRV
        curr_rmssd = safe_float(w.get("rmssd_ms"), None)
        curr_sdnn = safe_float(w.get("sdnn_ms"), None)
        hrv_coverage = safe_float(w.get("coverage"), 0.0)
        hrv_status = w.get("status", "unknown")

        rmssd_ratio = (curr_rmssd / baseline_rmssd) if (curr_rmssd and baseline_rmssd and baseline_rmssd > 0) else None

        # Tentukan status kognitif
        state, score, desc = classify_cognitive_state(
            rmssd_ratio=rmssd_ratio,
            on_screen_ratio=eye_stats["on_screen_ratio"],
            transitions=eye_stats["total_transitions"],
            mean_dwell_ms=eye_stats["mean_dwell_ms"],
            blink_rate=eye_stats["mean_blink_rate_bpm"]
        )

        state_distribution[state] = state_distribution.get(state, 0) + 1

        row = {
            "session_id": session_id,
            "study_session_status": study_session_status,
            **study_labels,
            "window_idx": idx,
            "window_start": format_wib(w_start_str) if w_start_str else "",
            "window_end": format_wib(w_end_str) if w_end_str else "",
            "timestamp_relative": ((w_end_ms - shared_origin_ms) / 1000.0
                                    if shared_origin_ms is not None else ""),
            "timestamp_relative_ms": (round(w_end_ms - shared_origin_ms, 3)
                                      if shared_origin_ms is not None else ""),
            "window_seconds": safe_float(w.get("window_seconds")),
            # Sinyal HRV
            "hrv_status": hrv_status,
            "hrv_coverage": hrv_coverage,
            "rmssd_ms": curr_rmssd,
            "sdnn_ms": curr_sdnn,
            "baseline_rmssd_ms": baseline_rmssd,
            "rmssd_ratio": round(rmssd_ratio, 3) if rmssd_ratio is not None else "",
            # Sinyal Eye Tracking
            "eye_frames_count": eye_stats["eye_frame_count"],
            "on_screen_ratio": eye_stats["on_screen_ratio"],
            "face_detected_ratio": eye_stats["face_detected_ratio"],
            "mean_dwell_ms": eye_stats["mean_dwell_ms"],
            "max_dwell_ms": eye_stats["max_dwell_ms"],
            "section_transitions": eye_stats["total_transitions"],
            "dominant_section": eye_stats["dominant_section"],
            "mean_iris_delta": eye_stats["mean_iris_delta"],
            "mean_blink_rate_bpm": eye_stats["mean_blink_rate_bpm"],
            "head_shifted_ratio": eye_stats["head_shifted_ratio"],
            # Inferensi Kognitif
            "cognitive_state": state,
            "cognitive_load_score": score,
            "cognitive_state_desc": desc
        }
        timeline.append(row)

    # Simpan output CSV
    if output_csv is None:
        output_csv = session_dir / "multimodal_timeline.csv"
    
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TIMELINE_COLUMNS)
        writer.writeheader()
        writer.writerows(timeline)
    print(f"[Fuse] Timeline multimodal berhasil disimpan: {output_csv}")

    # Buat ringkasan sesi JSON
    total_windows = len(timeline)
    task_windows_by_set = {}
    eligible_task_rows = []
    excluded_windows = {
        "mixed": 0, "feedback": 0, "rating": 0, "rest": 0, "pre_task": 0,
        "post_task": 0, "interrupted": 0, "unlabeled": 0, "baseline": 0,
        "other_phase": 0, "sensor_quality": 0,
    }
    for row in timeline:
        phase = row["study_phase"]
        if phase != "active_task" or row["puzzle_set"] in ("", "mixed"):
            reason = phase if phase in excluded_windows else "other_phase"
            excluded_windows[reason] += 1
            continue
        if row["hrv_status"] != "ready" or not row["eye_frames_count"]:
            excluded_windows["sensor_quality"] += 1
            continue
        key = row["puzzle_set"]
        eligible_task_rows.append(row)
        task_windows_by_set.setdefault(key, {"difficulty_level": row["difficulty_level"], "scores": []})["scores"].append(row["cognitive_load_score"])
    task_summary_by_set = {
        key: {"difficulty_level": value["difficulty_level"], "window_count": len(value["scores"]),
              "average_cognitive_load_score": round(statistics.mean(value["scores"]), 1)}
        for key, value in task_windows_by_set.items()
    }
    summary_report = {
        "session_name": session_dir.name,
        "study_session_status": study_session_status,
        "study_event_count": len(study_events),
        "study_events_file": "task_events.csv" if study_events else None,
        "analyzed_at": datetime.now(INDONESIA_WESTERN_TIME).isoformat(),
        "baseline": baseline_info,
        "total_windows": total_windows,
        "task_summary_by_set": task_summary_by_set,
        "excluded_window_count": excluded_windows,
        "task_state_breakdown_count": {
            state: sum(row["cognitive_state"] == state for row in eligible_task_rows)
            for state in state_distribution
        },
        "state_breakdown_count": state_distribution,
        "state_breakdown_percent": {
            k: round((v / total_windows * 100), 1) if total_windows > 0 else 0.0
            for k, v in state_distribution.items()
        },
        "average_cognitive_load_score": round(
            statistics.mean(r["cognitive_load_score"] for r in eligible_task_rows), 1
        ) if eligible_task_rows else None,
        "timeline_file": str(output_csv.name)
    }

    if output_json is None:
        output_json = session_dir / "session_report.json"
    
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    print(f"[Fuse] Laporan sesi kognitif berhasil disimpan: {output_json}")

    return summary_report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Multimodal Eye Tracking + HRV Feature Fusion & Cognitive Load Analysis")
    parser.add_argument("--session-dir", required=True, type=Path, help="Path ke folder sesi yang berisi hrv.csv dan eye_tracking.csv")
    parser.add_argument("--output-csv", type=Path, default=None, help="Path output file CSV timeline")
    parser.add_argument("--output-json", type=Path, default=None, help="Path output file JSON ringkasan kognitif")
    args = parser.parse_args(argv)

    try:
        report = fuse_session(args.session_dir, args.output_csv, args.output_json)
        if report:
            print("\n" + "="*50)
            print("RINGKASAN BEBAN KOGNITIF:")
            print(f"Rata-rata Skor Beban Kognitif: {report['average_cognitive_load_score']} / 100")
            print("Distribusi State:")
            for state, pct in report["state_breakdown_percent"].items():
                print(f"  - {state:<12}: {pct}%")
            print("="*50)
            return 0
        return 1
    except Exception as e:
        print(f"[Fuse Error] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
