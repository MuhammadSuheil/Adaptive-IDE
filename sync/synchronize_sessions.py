"""Join an eye CSV and an HRV hrv.csv in memory by UTC time.

The source logs are never modified.  The output is one row per HRV window,
written to sync/sessions_sync by default, so eye frames are summarized over the
same interval represented by each RMSSD/SDNN value.
"""

import argparse
import csv
import math
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


OUTPUT_ROOT = Path(__file__).resolve().parent / "sessions_sync"
OUTPUT_FIELDS = [
    "phase", "window_start", "window_end", "window_seconds",
    "hrv_status", "hrv_reason", "rr_count", "rr_pairs", "rmssd_ms", "sdnn_ms",
    "lf_power", "z_score",
    "eye_frame_count", "eye_face_detected_ratio", "eye_on_screen_ratio",
    "eye_mean_confidence", "eye_mean_gaze_x", "eye_mean_gaze_y",
    "eye_blink_count_max", "eye_mean_fps", "eye_mean_transition_rate",
]


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("HRV timestamp harus memiliki timezone")
    return parsed.astimezone(timezone.utc).timestamp()


def _mean(rows, name):
    values = [_number(row.get(name)) for row in rows]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else ""


def latest_session_logs():
    """Find the newest completed eye and HRV logs in their source folders."""
    project_root = Path(__file__).resolve().parents[1]
    eye_logs = sorted(
        (project_root / "eye_tracking_prototype" / "sessions").glob("*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    hrv_logs = sorted(
        (project_root / "hrv_monitor_prototype" / "sessions").glob("*/hrv.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not eye_logs:
        raise FileNotFoundError("Belum ada CSV eye tracking di eye_tracking_prototype/sessions")
    if not hrv_logs:
        raise FileNotFoundError("Belum ada hrv.csv di hrv_monitor_prototype/sessions")
    return eye_logs[0], hrv_logs[0]


def synchronize(eye_csv, hrv_csv, output=None):
    eye_csv, hrv_csv = Path(eye_csv), Path(hrv_csv)
    default_output = OUTPUT_ROOT / hrv_csv.parent.name / "synchronized_features.csv"
    output = Path(output) if output else default_output
    if output.exists() and not output.is_file():
        raise IsADirectoryError(f"Output bukan file: {output}")
    if output.exists() and output == default_output:
        stem, suffix, version = output.stem, output.suffix, 2
        while output.exists():
            output = output.with_name(f"{stem}_{version}{suffix}")
            version += 1
    output.parent.mkdir(parents=True, exist_ok=True)

    with eye_csv.open(newline="", encoding="utf-8") as stream:
        eye_rows = list(csv.DictReader(stream))
    eye_times = []
    for row in eye_rows:
        timestamp_ms = _number(row.get("timestamp_ms"))
        if timestamp_ms is not None:
            eye_times.append((_number(timestamp_ms / 1000.0), row))
    eye_times = [(stamp, row) for stamp, row in eye_times if stamp is not None]

    with hrv_csv.open(newline="", encoding="utf-8") as stream:
        hrv_rows = list(csv.DictReader(stream))

    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for hrv in hrv_rows:
            start = _timestamp(hrv["window_start"])
            end = _timestamp(hrv["window_end"])
            selected = [row for stamp, row in eye_times if start <= stamp <= end]
            face_ratio = (sum(str(row.get("face_detected", "")).lower() == "true"
                              for row in selected) / len(selected) if selected else "")
            screen_ratio = (sum(row.get("gaze_status") == "on_screen"
                                for row in selected) / len(selected) if selected else "")
            writer.writerow({
                "phase": hrv.get("phase", ""),
                "window_start": hrv.get("window_start", ""),
                "window_end": hrv.get("window_end", ""),
                "window_seconds": hrv.get("window_seconds", ""),
                "hrv_status": hrv.get("status", ""),
                "hrv_reason": hrv.get("reason", ""),
                "rr_count": hrv.get("rr_count", ""),
                "rr_pairs": hrv.get("rr_pairs", ""),
                "rmssd_ms": hrv.get("rmssd_ms", ""),
                "sdnn_ms": hrv.get("sdnn_ms", ""),
                "lf_power": hrv.get("lf_power", ""),
                "z_score": hrv.get("z_score", ""),
                "eye_frame_count": len(selected),
                "eye_face_detected_ratio": face_ratio,
                "eye_on_screen_ratio": screen_ratio,
                "eye_mean_confidence": _mean(selected, "confidence"),
                "eye_mean_gaze_x": _mean(selected, "gaze_x_smooth"),
                "eye_mean_gaze_y": _mean(selected, "gaze_y_smooth"),
                "eye_blink_count_max": max((_number(row.get("total_blinks")) or 0 for row in selected), default=""),
                "eye_mean_fps": _mean(selected, "fps_actual"),
                "eye_mean_transition_rate": _mean(selected, "transition_rate"),
            })
    return output


def run_synchronized_session(config_path=None):
    """Launch both recorders and derive the synchronized log after Stop."""
    project_root = Path(__file__).resolve().parents[1]
    session_id = uuid4().hex
    eye_script = project_root / "eye_tracking_prototype" / "eye_tracking_prototype.py"
    hrv_script = project_root / "hrv_monitor_prototype" / "hrv_monitor.py"
    config_path = Path(config_path) if config_path else project_root / "eye_tracking_prototype" / "config.yaml"
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config eye tracking tidak ditemukan: {config_path}")

    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    hrv = subprocess.Popen([sys.executable, str(hrv_script), "--session-id", session_id],
                           cwd=project_root, creationflags=flags)
    eye = subprocess.Popen([sys.executable, str(eye_script), "--config", str(config_path),
                            "--session-id", session_id], cwd=project_root)
    print(f"[Sync] Shared session ID: {session_id}")
    print("[Sync] Webcam dan HRV aktif. Tekan Q pada jendela eye tracking untuk Stop.")
    try:
        eye.wait()
    finally:
        if hrv.poll() is None:
            try:
                hrv.send_signal(signal.CTRL_BREAK_EVENT)
                hrv.wait(timeout=8)
            except (AttributeError, OSError, subprocess.TimeoutExpired):
                if hrv.poll() is None:
                    hrv.terminate()
                    hrv.wait(timeout=5)

    eye_files = sorted((project_root / "eye_tracking_prototype" / "sessions").glob(f"session_{session_id}_*.csv"))
    hrv_file = project_root / "hrv_monitor_prototype" / "sessions" / session_id / "hrv.csv"
    if eye.returncode != 0:
        raise RuntimeError(f"Eye tracking berhenti dengan kode {eye.returncode}")
    if not eye_files:
        raise RuntimeError("Eye tracking tidak menghasilkan CSV; sinkronisasi dibatalkan")
    if not hrv_file.is_file():
        raise RuntimeError("HRV tidak menghasilkan hrv.csv; sinkronisasi dibatalkan")
    output = synchronize(eye_files[-1], hrv_file)
    print(f"[Sync] Sesi selesai. Log sinkronisasi: {output}")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eye_csv", type=Path, nargs="?",
                        help="CSV eye tracking dari sesi yang baru selesai")
    parser.add_argument("hrv_csv", type=Path, nargs="?",
                        help="hrv.csv dari sesi yang sama")
    parser.add_argument("--latest", action="store_true",
                        help="Pilih log eye dan HRV terbaru secara eksplisit")
    parser.add_argument("--sync-only", action="store_true",
                        help="Jangan menyalakan sensor; hanya buat log sinkronisasi")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path,
                        help="Config eye tracking untuk sesi baru")
    args = parser.parse_args(argv)
    if not args.sync_only and not args.latest and args.eye_csv is None and args.hrv_csv is None:
        run_synchronized_session(args.config)
        return
    if (args.eye_csv is None) != (args.hrv_csv is None):
        parser.error("eye_csv dan hrv_csv harus diberikan bersama-sama")
    if args.latest and args.eye_csv is not None:
        parser.error("gunakan --latest atau dua path file, bukan keduanya")
    if not args.latest and args.eye_csv is None:
        parser.error("gunakan --sync-only --latest, atau berikan dua path file")
    eye_csv, hrv_csv = (latest_session_logs() if args.latest
                        else (args.eye_csv, args.hrv_csv))
    print(f"Eye log : {eye_csv}")
    print(f"HRV log : {hrv_csv}")
    print(f"Synchronized log: {synchronize(eye_csv, hrv_csv, args.output)}")


if __name__ == "__main__":
    main()
