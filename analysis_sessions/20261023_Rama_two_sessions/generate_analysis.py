"""Generate minute-level analysis exports for the two Rama sessions."""

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
WIB = timezone(timedelta(hours=7))
SESSIONS = [
    ("Sesi_1", "20261023_142709_P01_coding_task"),
    ("Sesi_2_Rama02", "20261023_154026_rama02"),
]
GAZE_STATES = [
    "on_screen", "gaze_outside_screen", "gaze_unconfirmed",
    "eyes_invalid_or_blink", "face_missing",
]
HRV_STATES = ["ready", "insufficient_data", "gap", "recovering"]
COGNITIVE_STATES = ["FOCUSED", "CONFUSED", "SCANNING", "OVERLOADED", "DISTRACTED"]


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def average(values):
    return statistics.mean(values) if values else ""


def median(values):
    return statistics.median(values) if values else ""


def percentile(values, fraction):
    if not values:
        return ""
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def iso_datetime(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"Timestamp tidak memiliki zona waktu: {value}")
    return dt


def minute_wib(dt):
    return dt.astimezone(WIB).strftime("%Y-%m-%d %H:%M")


def epoch_minute_wib(value_ms):
    return minute_wib(datetime.fromtimestamp(float(value_ms) / 1000, timezone.utc))


def as_bool(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def eye_analysis(session_dir):
    source = read_csv(session_dir / "eye_tracking.csv")
    grouped = defaultdict(list)
    for row in source:
        row["_minute"] = epoch_minute_wib(row["timestamp_ms"])
        grouped[row["_minute"]].append(row)

    output = []
    for minute, rows in sorted(grouped.items()):
        n = len(rows)
        faces = [row for row in rows if as_bool(row.get("face_detected"))]
        state_counts = Counter(row.get("gaze_status", "") for row in rows)
        confidence = [number(row.get("confidence")) for row in faces]
        confidence = [value for value in confidence if value is not None]
        yaw = [abs(number(row.get("head_yaw"))) for row in faces]
        yaw = [value for value in yaw if value is not None]
        pitch = [abs(number(row.get("head_pitch"))) for row in faces]
        pitch = [value for value in pitch if value is not None]
        q = [number(row.get("calibration_quality")) for row in rows]
        q = [value for value in q if value is not None]
        blink_rate = [number(row.get("blink_rate_bpm")) for row in rows]
        blink_rate = [value for value in blink_rate if value is not None]
        fps = [number(row.get("fps_actual")) for row in rows]
        fps = [value for value in fps if value is not None]
        capture_fps = [number(row.get("capture_fps")) for row in rows]
        capture_fps = [value for value in capture_fps if value is not None]
        output.append({
            "minute_wib": minute,
            "frame_count": n,
            "first_frame_wib": datetime.fromtimestamp(float(rows[0]["timestamp_ms"]) / 1000, timezone.utc).astimezone(WIB).isoformat(),
            "last_frame_wib": datetime.fromtimestamp(float(rows[-1]["timestamp_ms"]) / 1000, timezone.utc).astimezone(WIB).isoformat(),
            "observed_fps": round((n - 1) / ((float(rows[-1]["timestamp_ms"]) - float(rows[0]["timestamp_ms"])) / 1000), 3) if n > 1 and float(rows[-1]["timestamp_ms"]) > float(rows[0]["timestamp_ms"]) else "",
            "face_detected_count": len(faces),
            "face_detected_rate": round(len(faces) / n, 5) if n else "",
            "face_missing_count": n - len(faces),
            "face_missing_rate": round((n - len(faces)) / n, 5) if n else "",
            "gaze_on_screen_count": state_counts["on_screen"],
            "gaze_on_screen_rate": round(state_counts["on_screen"] / n, 5) if n else "",
            "gaze_outside_count": state_counts["gaze_outside_screen"],
            "gaze_outside_rate": round(state_counts["gaze_outside_screen"] / n, 5) if n else "",
            "gaze_unconfirmed_count": state_counts["gaze_unconfirmed"],
            "eyes_invalid_or_blink_count": state_counts["eyes_invalid_or_blink"],
            "mean_gaze_confidence_face_frames": round(average(confidence), 5) if confidence else "",
            "p95_abs_head_yaw_deg": round(percentile(yaw, .95), 3) if yaw else "",
            "p95_abs_head_pitch_deg": round(percentile(pitch, .95), 3) if pitch else "",
            "frames_abs_yaw_gt_15deg": sum(value > 15 for value in yaw),
            "frames_abs_pitch_gt_12deg": sum(value > 12 for value in pitch),
            "frames_either_pose_threshold_exceeded": sum(value > 15 or p > 12 for value, p in zip(yaw, pitch)),
            "head_pose_shifted_flag_count": sum(as_bool(row.get("head_pose_shifted")) for row in rows),
            "mean_calibration_quality": round(average(q), 5) if q else "",
            "blink_rate_mean_bpm": round(average(blink_rate), 3) if blink_rate else "",
            "blinking_frame_count": sum(as_bool(row.get("is_blinking")) for row in rows),
            "iris_valid_rate": round(sum(as_bool(row.get("iris_valid")) for row in rows) / n, 5) if n else "",
            "processing_fps_mean": round(average(fps), 3) if fps else "",
            "capture_fps_mean": round(average(capture_fps), 3) if capture_fps else "",
        })
    columns = list(output[0]) if output else ["minute_wib"]
    return output, columns, source


def hrv_analysis(session_dir):
    source = read_csv(session_dir / "hrv.csv")
    grouped = defaultdict(list)
    for row in source:
        row["_minute"] = minute_wib(iso_datetime(row["window_end"]))
        grouped[(row.get("phase", ""), row["_minute"])].append(row)
    output = []
    for (phase, minute), rows in sorted(grouped.items()):
        statuses = Counter(row.get("status", "") for row in rows)
        coverage = [number(row.get("coverage")) for row in rows]
        coverage = [value for value in coverage if value is not None]
        rmssd = [number(row.get("rmssd_ms")) for row in rows]
        rmssd = [value for value in rmssd if value is not None]
        sdnn = [number(row.get("sdnn_ms")) for row in rows]
        sdnn = [value for value in sdnn if value is not None]
        rr = [number(row.get("rr_count")) for row in rows]
        rr = [value for value in rr if value is not None]
        pairs = [number(row.get("rr_pairs")) for row in rows]
        pairs = [value for value in pairs if value is not None]
        age = [number(row.get("data_age_seconds")) for row in rows]
        age = [value for value in age if value is not None]
        n = len(rows)
        item = {
            "phase": phase,
            "minute_wib": minute,
            "window_count": n,
            "ready_count": statuses["ready"],
            "ready_rate": round(statuses["ready"] / n, 5) if n else "",
            "insufficient_data_count": statuses["insufficient_data"],
            "gap_count": statuses["gap"],
            "recovering_count": statuses["recovering"],
            "coverage_mean": round(average(coverage), 5) if coverage else "",
            "coverage_min": round(min(coverage), 5) if coverage else "",
            "coverage_max": round(max(coverage), 5) if coverage else "",
            "rmssd_valid_count": len(rmssd),
            "rmssd_mean_ms": round(average(rmssd), 4) if rmssd else "",
            "rmssd_median_ms": round(median(rmssd), 4) if rmssd else "",
            "sdnn_valid_count": len(sdnn),
            "sdnn_mean_ms": round(average(sdnn), 4) if sdnn else "",
            "rr_count_mean": round(average(rr), 3) if rr else "",
            "rr_pairs_mean": round(average(pairs), 3) if pairs else "",
            "data_age_mean_seconds": round(average(age), 4) if age else "",
            "data_age_max_seconds": round(max(age), 4) if age else "",
        }
        output.append(item)
    columns = list(output[0]) if output else ["phase", "minute_wib"]
    return output, columns, source


def multimodal_analysis(session_dir):
    source = read_csv(session_dir / "multimodal_timeline.csv")
    grouped = defaultdict(list)
    for row in source:
        row["_minute"] = minute_wib(iso_datetime(row["window_end"]))
        grouped[row["_minute"]].append(row)
    output = []
    for minute, rows in sorted(grouped.items()):
        cognitive = Counter(row.get("cognitive_state", "") for row in rows)
        hrv = Counter(row.get("hrv_status", "") for row in rows)
        scores = [number(row.get("cognitive_load_score")) for row in rows]
        scores = [value for value in scores if value is not None]
        on_screen = [number(row.get("on_screen_ratio")) for row in rows]
        on_screen = [value for value in on_screen if value is not None]
        face = [number(row.get("face_detected_ratio")) for row in rows]
        face = [value for value in face if value is not None]
        frame_counts = [number(row.get("eye_frames_count")) for row in rows]
        frame_counts = [value for value in frame_counts if value is not None]
        coverage = [number(row.get("hrv_coverage")) for row in rows]
        coverage = [value for value in coverage if value is not None]
        item = {
            "minute_wib": minute,
            "window_count": len(rows),
            "hrv_ready_count": hrv["ready"],
            "hrv_insufficient_data_count": hrv["insufficient_data"],
            "hrv_gap_count": hrv["gap"],
            "hrv_recovering_count": hrv["recovering"],
            "mean_hrv_coverage": round(average(coverage), 5) if coverage else "",
            "mean_eye_frames_per_window": round(average(frame_counts), 2) if frame_counts else "",
            "mean_eye_face_detected_ratio": round(average(face), 5) if face else "",
            "mean_eye_on_screen_ratio": round(average(on_screen), 5) if on_screen else "",
            "mean_cognitive_load_score": round(average(scores), 3) if scores else "",
            "median_cognitive_load_score": round(median(scores), 3) if scores else "",
            "focused_count": cognitive["FOCUSED"],
            "confused_count": cognitive["CONFUSED"],
            "scanning_count": cognitive["SCANNING"],
            "overloaded_count": cognitive["OVERLOADED"],
            "distracted_count": cognitive["DISTRACTED"],
        }
        output.append(item)
    columns = list(output[0]) if output else ["minute_wib"]
    return output, columns, source


def rr_diagnostics(session_dir):
    source = read_csv(session_dir / "rr_raw.csv")
    ordered = sorted(source, key=lambda row: iso_datetime(row["received_at"]))
    gaps = []
    for previous, current in zip(ordered, ordered[1:]):
        seconds = (iso_datetime(current["received_at"]) - iso_datetime(previous["received_at"])).total_seconds()
        if seconds > 5:
            gaps.append((seconds, iso_datetime(previous["received_at"]).astimezone(WIB), iso_datetime(current["received_at"]).astimezone(WIB)))
    return ordered, gaps


def attempt_analysis(attempt_source, eye_source, joint_origin):
    min_epoch = min(float(row["timestamp_ms"]) for row in eye_source)
    max_epoch = max(float(row["timestamp_ms"]) for row in eye_source)
    start = datetime.fromtimestamp(min_epoch / 1000, timezone.utc)
    end = datetime.fromtimestamp(max_epoch / 1000, timezone.utc)
    attempts = []
    for row in attempt_source:
        if not row.get("name", "").strip().lower().startswith("rama"):
            continue
        timestamp = iso_datetime(row["timestamp"])
        if not start <= timestamp <= end:
            continue
        enriched = dict(row)
        enriched["timestamp_wib"] = timestamp.astimezone(WIB).isoformat()
        enriched["minute_wib"] = minute_wib(timestamp)
        enriched["elapsed_from_joint_task_start_seconds"] = round((timestamp - joint_origin).total_seconds(), 3)
        attempts.append(enriched)
    attempts.sort(key=lambda row: row["timestamp_wib"])
    return attempts


def integrated_minute(eye_rows, hrv_rows, multi_rows, attempts):
    maps = {
        "eye": {row["minute_wib"]: row for row in eye_rows},
        "hrv": {row["minute_wib"]: row for row in hrv_rows if row["phase"] == "task"},
        "multi": {row["minute_wib"]: row for row in multi_rows},
    }
    attempt_map = defaultdict(list)
    for attempt in attempts:
        attempt_map[attempt["minute_wib"]].append(attempt)
    minutes = sorted(set(maps["eye"]) | set(maps["hrv"]) | set(maps["multi"]) | set(attempt_map))
    output = []
    for minute in minutes:
        eye = maps["eye"].get(minute, {})
        hrv = maps["hrv"].get(minute, {})
        multi = maps["multi"].get(minute, {})
        atts = attempt_map.get(minute, [])
        output.append({
            "minute_wib": minute,
            "eye_frame_count": eye.get("frame_count", ""),
            "eye_face_detected_rate": eye.get("face_detected_rate", ""),
            "eye_face_missing_rate": eye.get("face_missing_rate", ""),
            "eye_on_screen_rate_unvalidated": eye.get("gaze_on_screen_rate", ""),
            "eye_mean_gaze_confidence": eye.get("mean_gaze_confidence_face_frames", ""),
            "eye_yaw_gt15_frames": eye.get("frames_abs_yaw_gt_15deg", ""),
            "eye_pitch_gt12_frames": eye.get("frames_abs_pitch_gt_12deg", ""),
            "hrv_window_count": hrv.get("window_count", ""),
            "hrv_ready_count": hrv.get("ready_count", ""),
            "hrv_ready_rate": hrv.get("ready_rate", ""),
            "hrv_coverage_mean": hrv.get("coverage_mean", ""),
            "hrv_rmssd_median_ms": hrv.get("rmssd_median_ms", ""),
            "multimodal_window_count": multi.get("window_count", ""),
            "multimodal_mean_load_score": multi.get("mean_cognitive_load_score", ""),
            "multimodal_confused_count": multi.get("confused_count", ""),
            "multimodal_distracted_count": multi.get("distracted_count", ""),
            "attempt_count": len(atts),
            "attempt_correct_count": sum(row.get("correct", "").lower() == "true" for row in atts),
            "attempts_in_minute": " | ".join(f"{row['timestamp_wib'][11:19]} {row['puzzle_title']} ({row['score_percent']}%, {row['correct']})" for row in atts),
        })
    return output


def main():
    all_attempts = read_csv(ROOT / "codearrange-web" / "logs" / "attempts.csv")
    summary_rows = []
    report_sections = []
    for key, folder_name in SESSIONS:
        session_dir = ROOT / "sessions" / folder_name
        if not session_dir.is_dir():
            raise FileNotFoundError(f"Folder sesi tidak ditemukan: {session_dir}")
        eye_rows, eye_columns, eye_source = eye_analysis(session_dir)
        hrv_rows, hrv_columns, hrv_source = hrv_analysis(session_dir)
        multi_rows, multi_columns, multi_source = multimodal_analysis(session_dir)
        rr_rows, rr_gaps = rr_diagnostics(session_dir)
        status = json.loads((session_dir / "hrv_calib_status.json").read_text(encoding="utf-8"))
        joint_origin = iso_datetime(status["both_calibrated_at_utc"])
        attempts = attempt_analysis(all_attempts, eye_source, joint_origin)
        prefix = f"{key}_{folder_name}"

        write_csv(OUT / f"{prefix}_eye_tracking_per_minute.csv", eye_rows, eye_columns)
        write_csv(OUT / f"{prefix}_hrv_per_minute.csv", hrv_rows, hrv_columns)
        write_csv(OUT / f"{prefix}_multimodal_per_minute.csv", multi_rows, multi_columns)
        attempt_columns = list(attempts[0]) if attempts else ["timestamp_wib"]
        write_csv(OUT / f"{prefix}_codearrange_attempts.csv", attempts, attempt_columns)
        integrated = integrated_minute(eye_rows, hrv_rows, multi_rows, attempts)
        integrated_columns = list(integrated[0]) if integrated else ["minute_wib"]
        write_csv(OUT / f"{prefix}_minute_alignment.csv", integrated, integrated_columns)

        summary = json.loads((session_dir / "session_report.json").read_text(encoding="utf-8"))
        eye_meta = json.loads((session_dir / "eye_summary.json").read_text(encoding="utf-8"))
        diag = eye_meta["summary"].get("calibration_diagnostics", {})
        task_hrv = [row for row in hrv_source if row.get("phase") == "task"]
        task_ready = sum(row.get("status") == "ready" for row in task_hrv)
        face_total = sum(as_bool(row.get("face_detected")) for row in eye_source)
        rr_times = [iso_datetime(row["received_at"]).astimezone(WIB) for row in rr_rows]
        total_attempts = len(attempts)
        correct_attempts = sum(row.get("correct", "").lower() == "true" for row in attempts)
        eye_summary = eye_meta["summary"]
        summary_rows.append({
            "analysis_session_key": key,
            "source_folder": folder_name,
            "embedded_report_session_name": summary.get("session_name", ""),
            "joint_task_start_wib": joint_origin.astimezone(WIB).isoformat(),
            "eye_first_frame_wib": datetime.fromtimestamp(min(float(row["timestamp_ms"]) for row in eye_source) / 1000, timezone.utc).astimezone(WIB).isoformat(),
            "eye_last_frame_wib": datetime.fromtimestamp(max(float(row["timestamp_ms"]) for row in eye_source) / 1000, timezone.utc).astimezone(WIB).isoformat(),
            "eye_frames": len(eye_source),
            "eye_face_detection_rate": round(face_total / len(eye_source), 6),
            "eye_mean_fps": round(eye_summary.get("avg_fps", 0), 4),
            "eye_dropped_frame_ratio": round(eye_summary.get("dropped_frame_ratio", 0), 6),
            "eye_calibration_quality": diag.get("quality", ""),
            "eye_validation_passed": diag.get("held_out_validation", {}).get("passed_thresholds", ""),
            "eye_validation_median_error_px": diag.get("validation_median_error_px", ""),
            "eye_validation_p95_error_px": diag.get("validation_p95_error_px", ""),
            "hrv_baseline_status": next((row.get("status") for row in hrv_source if row.get("phase") == "baseline"), ""),
            "hrv_baseline_accepted_rr_seconds": next((row.get("accepted_rr_seconds") for row in hrv_source if row.get("phase") == "baseline"), ""),
            "hrv_baseline_rmssd_ms": next((row.get("rmssd_ms") for row in hrv_source if row.get("phase") == "baseline"), ""),
            "hrv_task_windows": len(task_hrv),
            "hrv_task_ready_windows": task_ready,
            "hrv_task_ready_rate": round(task_ready / len(task_hrv), 5) if task_hrv else "",
            "hrv_task_mean_coverage": round(average([number(row.get("coverage")) for row in task_hrv if number(row.get("coverage")) is not None]), 5) if task_hrv else "",
            "rr_rows": len(rr_rows),
            "rr_gaps_over_5s": len(rr_gaps),
            "rr_longest_gap_seconds": round(max((gap[0] for gap in rr_gaps), default=0), 3),
            "rr_longest_gap_start_wib": rr_gaps[max(range(len(rr_gaps)), key=lambda i: rr_gaps[i][0])][1].isoformat() if rr_gaps else "",
            "rr_longest_gap_end_wib": rr_gaps[max(range(len(rr_gaps)), key=lambda i: rr_gaps[i][0])][2].isoformat() if rr_gaps else "",
            "codearrange_attempts": total_attempts,
            "codearrange_correct_attempts": correct_attempts,
            "multimodal_windows": summary.get("total_windows", ""),
            "multimodal_mean_cognitive_load_score": summary.get("average_cognitive_load_score", ""),
        })

        attempts_lines = [
            f"| {row['timestamp_wib'][11:19]} | {row['puzzle_title']} | {row['score_percent']}% | {row['correct']} | {row['submission_type']} |"
            for row in attempts
        ]
        if key == "Sesi_1":
            key_findings = (
                "### Interval penting\n\n"
                "- **14:30–14:36:** HRV hanya 12/82 window ready (14,6%; coverage rata-rata 46%). Eye stream lengkap, tetapi gaze calibration quality 0,225 dan validasi gagal.\n"
                "- **14:37–14:42:** HRV membaik menjadi 53/72 ready (73,6%; coverage 85%). Gaze tetap perlu dianggap tidak tervalidasi.\n"
                "- **14:43–14:49:** periode gabungan terburuk: HRV 6/84 ready (7,1%), gaze confidence rata-rata 0,126, dan hanya 26,7% frame diberi label on-screen. Di sini tercatat dua submit Bubble Sort dan Binary Search timeout (14:44:58–14:48:03).\n"
                "- **14:50–14:56:** HRV 22/84 ready (26,2%; coverage 63%); gaze confidence 0,204. Fibonacci dimulai pada 14:53:45.\n"
                "- **14:57–15:01:** HRV relatif pulih (36/60 ready; 60%), tetapi gaze masih tak tervalidasi.\n"
                "- **15:02–15:05:** HRV turun menjadi 7/42 ready (16,7%) dan gaze confidence 0,174; submit terakhir tercatat 15:02:36, sensor berlanjut sampai 15:05.\n\n"
                "Gerakan berdiri tidak dapat disimpulkan dari sesi ini; kamera dan pose lebih stabil daripada sesi kedua.\n\n"
            )
        else:
            key_findings = (
                "### Interval penting dan indikasi gerakan\n\n"
                "- **15:43–15:55:** HRV relatif baik (118/148 window ready; 79,7% untuk gabungan rentang ini). Wajah mayoritas terdeteksi. Gaze calibration tetap gagal validasi.\n"
                "- **15:56–15:57:** anomali paling kuat: wajah hilang pada 21,9% frame, ada beberapa rentang tanpa wajah (terpanjang sekitar 12,5 detik), dan tidak ada RR selama **15:56:44.540–15:57:50.510** (66 detik). HRV hanya 1/24 window ready. Terjadi di antara timeout Menghitung Huruf Vokal (15:55:49) dan submit pertama Fibonacci (15:57:40). Konsisten dengan tester berpindah jauh dari kamera/beranjak, tetapi tidak membuktikan berdiri.\n"
                "- **15:58–15:59:** sudut kepala melewati batas 15° yaw atau 12° pitch di sekitar 30–36% frame wajah; Fibonacci dikirim 15:58:12 dan benar 15:58:45.\n"
                "- **16:02–16:08:** indikasi perubahan pose kepala berulang, paling tinggi 16:04 (56,5% frame), 16:05 (45,0%), dan 16:06 (51,4%). HRV hanya 8/48 window ready (16,7%) dan ada jeda RR 14–18 detik. Ini bertepatan dengan submit Menggabungkan Array Terurut.\n"
                "- **16:09–16:12:** HRV pulih sebagian (27/46 ready; 58,7%), tetapi gaze confidence rata-rata 0,113. Tiga submit terakhir adalah Pemeriksaan Anagram; yang terakhir timeout pada 16:12:19.\n\n"
                "`head_pose_shifted` tersimpan false karena gate posisi kepala dinonaktifkan. Angka di atas dihitung ulang dari sudut yaw/pitch terhadap batas konfigurasi dan hanya menjadi indikator gerakan kepala. Frame tanpa wajah dan jeda RR memperkuat indikasi bergerak, tetapi kamera wajah saja tidak dapat memastikan orang berdiri. `contact_detected` kosong di semua baris RR sehingga kemungkinan sensor terlepas tidak dapat diuji dari kolom itu.\n\n"
            )
        report_sections.append(
            f"## {key}: `{folder_name}`\n\n"
            f"Embedded report ID: `{summary.get('session_name', '')}`. Joint task origin: `{joint_origin.astimezone(WIB).isoformat()}`. "
            f"Eye coverage: {len(eye_source):,} frames at {eye_summary.get('avg_fps', 0):.2f} FPS; face detection {face_total / len(eye_source):.2%}; "
            f"dropped frames {eye_summary.get('dropped_frame_ratio', 0):.2%}. Gaze calibration quality {diag.get('quality', 'n/a')}; "
            f"validation passed: `{diag.get('held_out_validation', {}).get('passed_thresholds', 'unknown')}` "
            f"(median error {diag.get('validation_median_error_px', 'n/a')} px; p95 {diag.get('validation_p95_error_px', 'n/a')} px).\n\n"
            f"HRV baseline: `{next((row.get('status') for row in hrv_source if row.get('phase') == 'baseline'), 'n/a')}`; "
            f"{next((row.get('accepted_rr_seconds') for row in hrv_source if row.get('phase') == 'baseline'), 'n/a')} s accepted RR; "
            f"RMSSD {next((row.get('rmssd_ms') for row in hrv_source if row.get('phase') == 'baseline'), 'n/a')} ms. "
            f"Task windows: {len(task_hrv)}, with {task_ready} ready ({task_ready / len(task_hrv):.1%}); "
            f"mean coverage {statistics.mean(float(row['coverage']) for row in task_hrv):.1%}. "
            f"Raw RR rows: {len(rr_rows)}; gaps >5 s: {len(rr_gaps)}; longest gap: "
            f"{max((gap[0] for gap in rr_gaps), default=0):.1f} s. CodeArrange attempts: {total_attempts}, "
            f"correct: {correct_attempts}. Fused report mean score: {summary.get('average_cognitive_load_score', 'n/a')}; "
            f"the score is a derived output, not a validated result when gaze calibration fails and HRV windows are missing.\n\n"
            + key_findings
            + f"| WIB | Puzzle | Score | Correct | Submission |\n|---|---|---:|:---:|---|\n"
            + "\n".join(attempts_lines)
            + "\n"
        )

    write_csv(OUT / "two_session_quality_summary.csv", summary_rows, list(summary_rows[0]))
    (OUT / "ANALYSIS.md").write_text(
        "# Analisis dua sesi Rama — 23 Oktober 2026\n\n"
        "Paket ini berisi ringkasan turunan per menit agar dapat dianalisis lebih lanjut. File sumber mentah tidak diubah. "
        "Semua cap waktu di keluaran diberi zona WIB (`+07:00`) atau dikelompokkan menurut menit WIB.\n\n"
        "## Keluaran\n\n"
        "Untuk tiap sesi tersedia: `*_eye_tracking_per_minute.csv`, `*_hrv_per_minute.csv`, `*_multimodal_per_minute.csv`, "
        "`*_codearrange_attempts.csv`, dan `*_minute_alignment.csv`. `two_session_quality_summary.csv` memuat ringkasan lintas sesi.\n\n"
        "## Metode dan batasan\n\n"
        "- Eye Tracking dikelompokkan dari `timestamp_ms` epoch; laju gaze, wajah, confidence, blink, FPS, dan sudut kepala diringkas per menit. "
        "Hitungan sudut memakai batas absolut 15° yaw dan 12° pitch sebagai indikator gerak kepala; bukan bukti seseorang berdiri.\n"
        "- HRV dikelompokkan menurut `window_end` WIB; baris baseline tetap ditandai `phase=baseline`. Window 25 detik dengan langkah 5 detik "
        "saling tumpang tindih, sehingga banyak window bukan banyak pengamatan independen.\n"
        "- Multimodal diringkas menurut `window_end` WIB. Label cognitive state dan rasio gaze mengikuti sumber multimodal; validitasnya bergantung pada gaze calibration dan coverage HRV.\n"
        "- Attempt dipilih dari `codearrange-web/logs/attempts.csv` dengan `name` yang diawali `Rama`, dan timestamp di dalam rentang frame Eye Tracking sesi. "
        "Timestamp sumber UTC dikonversi ke WIB; `elapsed_from_joint_task_start_seconds` mengacu ke penanda kalibrasi bersama.\n"
        "- `head_pose_shifted` hanya dilaporkan sebagaimana tersimpan. Jika gate posisi kepala dinonaktifkan, nilai false tidak membuktikan kepala tidak bergerak. "
        "Kolom sudut yaw/pitch dan hilangnya wajah lebih berguna sebagai indikator, tetapi tidak dapat memastikan tester berdiri.\n"
        "- Folder sesi kedua diberi nama `20261023_154026_rama02`, sedangkan ID di `session_report.json` dan `eye_summary.json` masih menyebut "
        "`20261023_154026_P01_coding_task`. Kecocokan dengan log Rama didasarkan pada rentang waktu, bukan ID yang identik.\n\n"
        "## Ringkasan per sesi\n\n"
        + "\n\n".join(report_sections)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
