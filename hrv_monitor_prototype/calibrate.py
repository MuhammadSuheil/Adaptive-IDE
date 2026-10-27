import argparse
import asyncio
import csv
from datetime import datetime, timedelta, timezone
from itertools import groupby
import json
import math
from pathlib import Path
import sys


INDONESIA_WESTERN_TIME = timezone(timedelta(hours=7))


def to_wib(timestamp):
    """Format timestamp ISO-8601 sebagai waktu Indonesia Barat (UTC+7)."""
    return datetime.fromisoformat(timestamp).astimezone(INDONESIA_WESTERN_TIME).isoformat()

if __package__:
    from .config import (BASELINE_TARGET_SECONDS, BASELINE_MAX_SECONDS,
                         MEDIAN_THRESHOLD_MS, WINDOW_SECONDS, STEP_SECONDS, RR_GAP_SECONDS)
    from .record_rr import SESSION_ROOT, record_sensor
    from .preprocess_rr import RRPreprocessor, preprocess_rows
    from .calculate_hrv import HRV_COLUMNS, LiveHRV, calculate_hrv
else:
    from config import (BASELINE_TARGET_SECONDS, BASELINE_MAX_SECONDS,
                        MEDIAN_THRESHOLD_MS, WINDOW_SECONDS, STEP_SECONDS, RR_GAP_SECONDS)
    from record_rr import SESSION_ROOT, record_sensor
    from preprocess_rr import RRPreprocessor, preprocess_rows
    from calculate_hrv import HRV_COLUMNS, LiveHRV, calculate_hrv


def baseline_data(processed, elapsed, maximum_seconds):
    """Pilih RR utuh di dalam masa kalibrasi dan hitung durasi yang diwakilinya.

    Jeda tidak menambah durasi. Rentang bertumpuk tidak dihitung dua kali.
    RR sebelum awal sesi, dari masa depan, atau diterima lewat deadline tidak dipakai.
    """
    cutoff = min(elapsed, maximum_seconds)
    used = []
    duration = 0.0
    covered_until = 0.0
    for row in processed:
        rr = row["rr_processed_ms"]
        if rr is None or row["elapsed_seconds"] > cutoff:
            continue
        end = row["beat_elapsed_seconds"]
        start = end - rr / 1000
        if start < -1e-9 or end > cutoff + 1e-9:
            continue
        duration += max(0, end - max(0, start, covered_until))
        covered_until = max(covered_until, end)
        used.append(row)
    return used, duration


class SessionCalibration:
    """Satu sesi: kalibrasi -> rekam tugas, atau gagal -> ulang sesi secara manual."""

    def __init__(self, task_duration=None, target_seconds=BASELINE_TARGET_SECONDS,
                 maximum_seconds=BASELINE_MAX_SECONDS,
                 window_seconds=WINDOW_SECONDS, step_seconds=STEP_SECONDS,
                 wait_for_eye=False):
        if (not math.isfinite(target_seconds) or not math.isfinite(maximum_seconds)
                or not 0 < target_seconds <= maximum_seconds):
            raise ValueError("Kalibrasi harus memenuhi 0 < target <= batas maksimum")
        if task_duration is not None and (not math.isfinite(task_duration) or task_duration <= 0):
            raise ValueError("Durasi perekaman tugas harus positif dan finite")
        if not math.isfinite(MEDIAN_THRESHOLD_MS) or MEDIAN_THRESHOLD_MS <= 0:
            raise ValueError("Ambang filter harus positif dan finite")
        if (not math.isfinite(window_seconds) or not math.isfinite(step_seconds)
                or not 0 < step_seconds <= window_seconds):
            raise ValueError("Durasi harus finite dan memenuhi 0 < step <= window")
        self.target_seconds = target_seconds
        self.maximum_seconds = maximum_seconds
        self.task_duration = task_duration
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds
        self.task_preprocessor = RRPreprocessor()
        self.task_hrv = LiveHRV(None, window_seconds, step_seconds)
        self.rows = []
        self.processed = []
        self.result = None
        self.hrv_file = None
        self.phase = "waiting_for_start" if wait_for_eye else "calibration"
        self.wait_for_eye = wait_for_eye
        self.calibration_start_elapsed = None if wait_for_eye else 0.0
        self.calibration_started_at = None
        self.recording_origin = None
        self.last_rr_elapsed = None
        self.current_elapsed = 0.0
        self.start_error = ""
        self.last_print = -1
        self.last_status_update = -1.0
        self.task_start = None
        self.session = None

    def update_calib_status(self, status, elapsed, duration):
        if self.session is None:
            return
        status_file = self.session / "hrv_calib_status.json"
        temp_file = self.session / "hrv_calib_status.json.tmp"
        remaining = max(0.0, self.target_seconds - duration)
        text = f"Kalibrasi: {min(elapsed, self.maximum_seconds):.0f}/{self.maximum_seconds:g} detik | RR diterima: {duration:.1f}/{self.target_seconds:g} detik"
        if self.phase == "waiting_for_start":
            text = "HW9 terhubung. Baseline menunggu Q pada preview kamera."
        payload = {
            "phase": self.phase,
            "status": status,
            "elapsed_seconds": round(float(elapsed), 1),
            "maximum_seconds": float(self.maximum_seconds),
            "accepted_rr_seconds": round(float(duration), 1),
            "target_seconds": float(self.target_seconds),
            "remaining_seconds": round(float(remaining), 1),
            "text": text,
            "start_ready": (self.phase == "waiting_for_start" and self.last_rr_elapsed is not None
                            and 0 <= self.current_elapsed - self.last_rr_elapsed <= RR_GAP_SECONDS),
            "start_error": self.start_error,
        }
        if self.calibration_started_at is not None:
            payload["calibration_started_at_utc"] = self.calibration_started_at.isoformat()
        if self.result is not None:
            payload["hrv_calibration_completed_at_utc"] = self.result.get("calibration_completed_at_utc")
        if getattr(self, "both_calibrated_at_utc", None):
            payload["both_calibrated_at_utc"] = self.both_calibrated_at_utc
        try:
            temp_file.write_text(json.dumps(payload), encoding="utf-8")
            temp_file.replace(status_file)
        except OSError:
            pass

    def start(self, session):
        self.session = Path(session)
        self.started_at = datetime.now(timezone.utc)
        if not self.wait_for_eye:
            self.calibration_started_at = self.started_at
        self.hrv_file = (self.session / "hrv.csv").open("w", newline="", encoding="utf-8")
        self.hrv_writer = csv.DictWriter(
            self.hrv_file,
            fieldnames=["phase", *HRV_COLUMNS, "session_id", "study_phase", "window_phase_mix"],
        )
        self.hrv_writer.writeheader()

        self.hrv_file.flush()
        self.update_calib_status("waiting_for_start" if self.wait_for_eye else "calibrating", 0.0, 0.0)
        if self.wait_for_eye:
            print("HW9 terhubung. Menunggu aliran RR dan Q pada preview kamera untuk mulai baseline.")
        print(f"Duduk tenang. Target {self.target_seconds:g} detik RR diterima, "
              f"batas waktu {self.maximum_seconds:g} detik.")

    def receive(self, rows):
        if not rows:
            return
        self.current_elapsed = float(rows[0]["elapsed_seconds"])
        if self.recording_origin is None:
            self.recording_origin = (datetime.fromisoformat(rows[0]["received_at"])
                                     - timedelta(seconds=self.current_elapsed))
            if not self.wait_for_eye:
                self.calibration_started_at = self.recording_origin.astimezone(timezone.utc)
        for row in rows:
            rr = float(row["rr_original_ms"])
            if math.isfinite(rr) and 300 <= rr <= 2000 and str(row.get("contact_detected", "")).lower() != "false":
                self.last_rr_elapsed = self.current_elapsed
        if self.calibration_start_elapsed is None and not self.try_start_calibration(self.current_elapsed):
            self.last_status_update = -1.0
            return
        if self.task_start is not None:
            for _, packet_rows in groupby(rows, key=lambda row: row["packet_id"]):
                # Waktu UTC, ID, dan RR tetap asli. Elapsed dimulai ulang dari
                # akhir kalibrasi agar window tugas tidak mencampur baseline.
                packet = [dict(row, elapsed_seconds=row["elapsed_seconds"] - self.task_start)
                          for row in packet_rows]
                arrived = packet[0]["elapsed_seconds"]
                self.publish_hrv(arrived, include_boundary=False)
                if self.task_duration is not None and arrived > self.task_duration + 1e-9:
                    continue
                processed = self.task_preprocessor.process_packet(packet)
                self.task_hrv.add(processed, arrived, self.task_preprocessor.recovering)
        elif self.result is None:
            # Only internal baseline copies use time since Q; raw timestamps stay intact.
            self.rows.extend(dict(row, elapsed_seconds=row["elapsed_seconds"] - self.calibration_start_elapsed)
                             for row in rows
                             if 0 <= row["elapsed_seconds"] - self.calibration_start_elapsed <= self.maximum_seconds)

    def try_start_calibration(self, elapsed):
        """Consume the shared trigger once, without restarting on eye retries."""
        if self.calibration_start_elapsed is not None:
            return True
        if self.recording_origin is None:
            return False
        marker = self.session / "calibration_start.json"
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            started = datetime.fromisoformat(payload["started_at_utc"])
            if started.utcoffset() is None:
                raise ValueError("Timestamp pemicu harus menyertakan zona waktu")
            offset = (started - self.recording_origin).total_seconds()
            if not math.isfinite(offset) or offset < 0 or offset > elapsed + 0.1:
                raise ValueError("Waktu pemicu berada di luar rentang sesi")
        except FileNotFoundError:
            return False
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.start_error = f"Penanda mulai kalibrasi tidak valid: {error}"
            return False
        self.calibration_started_at = started.astimezone(timezone.utc)
        self.calibration_start_elapsed = offset
        self.phase = "calibration"
        self.last_status_update = -1.0
        self.start_error = ""
        print("\nQ diterima. Kalibrasi baseline HRV dimulai bersamaan dengan kalibrasi mata.")
        return True

    def progress(self, elapsed):
        # Maksimal lima menit data: gunakan ulang fungsi offline yang sama agar
        # aturan filter identik. Hanya proses ulang jika ada RR baru.
        if len(self.processed) != len(self.rows):
            self.processed = list(preprocess_rows(self.rows))
        if self.calibration_start_elapsed is None:
            return [], 0.0
        return baseline_data(self.processed, elapsed - self.calibration_start_elapsed, self.maximum_seconds)

    def make_result(self, status, reason, elapsed, used, duration):
        origin = self.recording_origin or self.started_at
        calibration_completed_at = origin + timedelta(seconds=elapsed)
        return {
            "status": status, "reason": reason, "condition": "seated_rest",
            "target_rr_seconds": self.target_seconds,
            "maximum_elapsed_seconds": self.maximum_seconds,
            "calibration_end_elapsed_seconds": elapsed,
            "calibration_elapsed_seconds": elapsed - self.calibration_start_elapsed,
            "calibration_started_at_utc": self.calibration_started_at.isoformat(),
            "calibration_completed_at_utc": calibration_completed_at.astimezone(timezone.utc).isoformat(),
            "accepted_rr_seconds": duration,
            # Baseline satu RMSSD/SDNN dari RR kalibrasi, bukan rata-rata window.
            "hrv": calculate_hrv(used) if status == "ready" else None,
        }

    def save_baseline(self, used):
        """Baseline juga hasil HRV, sehingga disimpan pada CSV HRV yang sama."""
        elapsed = self.result["calibration_elapsed_seconds"]
        origin = self.calibration_started_at
        features = calculate_hrv(used)
        if self.result["status"] != "ready":
            features["rmssd_ms"] = features["sdnn_ms"] = None
        duration = self.result["accepted_rr_seconds"]
        self.hrv_writer.writerow({
            "phase": "baseline", "window_start": origin.astimezone(INDONESIA_WESTERN_TIME).isoformat(),
            "window_end": (origin + timedelta(seconds=elapsed)).astimezone(INDONESIA_WESTERN_TIME).isoformat(),
            # Baseline ends before the shared task origin is necessarily known.
            "timestamp_relative": "", "timestamp_relative_ms": "",
            "window_seconds": elapsed, **features,
            "status": self.result["status"], "reason": self.result["reason"],
            "accepted_rr_seconds": duration, "coverage": min(1, duration / elapsed) if elapsed > 0 else 0,
            "data_age_seconds": max(0, elapsed - used[-1]["beat_elapsed_seconds"]) if used else None,
            "session_id": self.session.name,
            "study_phase": "baseline",
            "window_phase_mix": "",
        })
        self.hrv_file.flush()

    def tick(self, elapsed):
        """Dipanggil timer perekam, termasuk ketika sensor sama sekali tidak mengirim RR."""
        self.current_elapsed = elapsed
        if self.calibration_start_elapsed is None and not self.try_start_calibration(elapsed):
            if elapsed - self.last_status_update >= .5:
                self.last_status_update = elapsed
                self.update_calib_status("waiting_for_start", 0.0, 0.0)
            return False
        if self.result is not None:
            if self.result["status"] == "unavailable":
                return True
            if self.task_start is None:
                if (self.session / "eye_calibration_ready").is_file():
                    self.start_task(elapsed)
                return False
            self.publish_hrv(elapsed - self.task_start)
            return self.task_duration is not None and elapsed - self.task_start >= self.task_duration

        used, duration = self.progress(elapsed)
        baseline_elapsed = elapsed - self.calibration_start_elapsed
        if baseline_elapsed - self.last_status_update >= 0.5 or int(baseline_elapsed) != self.last_print:
            self.last_status_update = baseline_elapsed
            self.update_calib_status("calibrating", baseline_elapsed, duration)
        if int(baseline_elapsed) != self.last_print:
            self.last_print = int(baseline_elapsed)
            print(f"\rKalibrasi: {min(baseline_elapsed, self.maximum_seconds):.0f}/{self.maximum_seconds:g} detik "
                  f"| RR diterima: {duration:.1f}/{self.target_seconds:g} detik   ", end="", flush=True)

        features = calculate_hrv(used)
        if (duration + 1e-9 >= self.target_seconds
                and features["rmssd_ms"] is not None and features["sdnn_ms"] is not None):
            self.result = self.make_result("ready", "target_reached", elapsed, used, duration)
            self.save_baseline(used)
            print(f"\nBaseline siap: RMSSD {features['rmssd_ms']:.2f} ms, SDNN {features['sdnn_ms']:.2f} ms.")
            if self.wait_for_eye and not (self.session / "eye_calibration_ready").is_file():
                self.phase = "waiting_for_eye"
                self.update_calib_status("ready", baseline_elapsed, duration)
                print("Baseline HRV siap. Menunggu kalibrasi mata selesai sebelum mulai tugas.")
            else:
                self.start_task(elapsed)
        elif baseline_elapsed >= self.maximum_seconds:
            self.result = self.make_result("unavailable", "timeout", elapsed, used, duration)
            self.save_baseline(used)
            self.update_calib_status("unavailable", baseline_elapsed, duration)
            print("\nBaseline belum tersedia. Jalankan ulang perintah untuk mengulang kalibrasi dalam sesi baru.")
            return True
        return False

    def start_task(self, elapsed):
        """Mulai analisis tugas di memori, terpisah dari riwayat kalibrasi."""
        self.task_start = elapsed
        self.phase = "task"
        origin = (self.recording_origin or self.started_at) + timedelta(seconds=elapsed)
        self.task_hrv.origin = origin
        hrv_ready = datetime.fromisoformat(self.result["calibration_completed_at_utc"])
        eye_marker = self.session / "eye_calibration_ready"
        try:
            eye_ready = datetime.fromisoformat(eye_marker.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            eye_ready = origin
        both_ready = max(hrv_ready, eye_ready).astimezone(timezone.utc)
        self.both_calibrated_at_utc = both_ready.isoformat()
        self.update_calib_status("ready", self.result["calibration_elapsed_seconds"],
                                 self.result["accepted_rr_seconds"])
        print("Perekaman tugas dimulai. Silakan mulai coding; Ctrl+C untuk berhenti.")
        print(f"Preprocessing otomatis. HRV pertama setelah {self.window_seconds:g} detik, "
              f"lalu setiap {self.step_seconds:g} detik.")
        print(f"Log sesi: {self.session / 'rr_raw.csv'} dan {self.session / 'hrv.csv'}")

    def publish_hrv(self, elapsed, include_boundary=True):
        if self.task_duration is not None and elapsed > self.task_duration:
            elapsed = self.task_duration
            include_boundary = True
        for features in self.task_hrv.advance(elapsed, include_boundary):
            joint_origin = datetime.fromisoformat(self.both_calibrated_at_utc)
            relative_ms = (datetime.fromisoformat(features["window_end"]) - joint_origin).total_seconds() * 1000.0
            features["timestamp_relative"] = relative_ms / 1000.0
            features["timestamp_relative_ms"] = round(relative_ms, 3)
            row = dict(features, phase="task", session_id=self.session.name,
                       study_phase="", window_phase_mix="")
            row["window_start"] = to_wib(row["window_start"])
            row["window_end"] = to_wib(row["window_end"])
            self.hrv_writer.writerow(row)
            self.hrv_file.flush()
            rmssd = features["rmssd_ms"]
            sdnn = features["sdnn_ms"]
            rmssd_text = f"{rmssd:.2f} ms" if rmssd is not None else "tidak tersedia"
            sdnn_text = f"{sdnn:.2f} ms" if sdnn is not None else "tidak tersedia"
            label = {"ready": "siap", "gap": "jeda RR", "recovering": "pemulihan",
                     "insufficient_data": "data belum cukup"}[features["status"]]
            line = (f"[HRV] {to_wib(features['window_end'])} | window {self.window_seconds:g}s "
                    f"| status: {label} "
                    f"| RMSSD: {rmssd_text} | SDNN: {sdnn_text} "
                    f"| RR: {features['rr_count']} | pasangan: {features['rr_pairs']} "
                    f"| durasi RR: {features['accepted_rr_seconds']:.2f}/{self.window_seconds:g}s "
                    f"| alasan: {features['reason'] or '-'}")
            print(line, flush=True)

    def timestamp_relative(self, elapsed, phase):
        """HRV sample seconds relative to completion of HRV calibration."""
        if phase != "task" or self.result is None:
            return ""
        return max(0.0, float(elapsed) - self.result["calibration_end_elapsed_seconds"])

    def finish(self, elapsed, reason):
        try:
            if self.task_start is not None and reason != "recording_error":
                self.publish_hrv(elapsed - self.task_start)
            if self.result is None and self.hrv_file is not None:
                if self.calibration_start_elapsed is None:
                    self.update_calib_status("not_started", 0.0, 0.0)
                    print("\nSesi dihentikan sebelum Q; baseline HRV belum dimulai.")
                    return
                used, duration = self.progress(elapsed)
                self.result = self.make_result("unavailable", reason, elapsed, used, duration)
                self.save_baseline(used)
                self.update_calib_status("unavailable", elapsed - self.calibration_start_elapsed, duration)
                print("\nKalibrasi terhenti sebelum selesai. Ulangi dengan menjalankan sesi baru.")
        finally:
            if self.hrv_file is not None:
                self.hrv_file.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="HW9", help="Potongan nama sensor")
    parser.add_argument("--address", help="Alamat BLE jika sensor lebih dari satu")
    parser.add_argument("--duration", type=float, help="Durasi perekaman SETELAH baseline siap; default sampai Ctrl+C")
    parser.add_argument("--output", type=Path, default=SESSION_ROOT, help="Folder induk sesi")
    parser.add_argument("--session-dir", type=Path, help="Folder sesi spesifik")
    parser.add_argument("--wait-for-eye", action="store_true",
                        help="Tunggu pemicu Q dari preview mata; mulai tugas setelah kedua kalibrasi selesai")
    parser.add_argument("--mock", action="store_true", help="Gunakan mock/simulasi sensor HRV tanpa perangkat fisik")
    parser.add_argument("--target-baseline", type=float, default=None,
                        help="Target durasi RR kalibrasi baseline dalam detik (default: 120)")
    args = parser.parse_args(argv)
    if args.wait_for_eye and args.session_dir is None:
        parser.error("--wait-for-eye memerlukan --session-dir")
    if args.target_baseline is not None and (not math.isfinite(args.target_baseline) or args.target_baseline <= 0):
        parser.error("--target-baseline harus positif dan finite")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration harus positif dan finite")

    target_baseline = args.target_baseline if args.target_baseline is not None else BASELINE_TARGET_SECONDS
    max_baseline = max(BASELINE_MAX_SECONDS, target_baseline * 2)
    calibration = SessionCalibration(task_duration=args.duration, target_seconds=target_baseline,
                                     maximum_seconds=max_baseline, wait_for_eye=args.wait_for_eye)
    asyncio.run(record_sensor(args.name, args.address, output=args.output, monitor=calibration, session_dir=args.session_dir, mock=args.mock))
    return 0 if calibration.result and calibration.result["status"] == "ready" else 1



if __name__ == "__main__":
    import signal
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nSesi dihentikan. Rekaman yang sudah ditulis tetap tersimpan.")
        sys.exit(1)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Gagal menjalankan sesi: {error}", file=sys.stderr)
        sys.exit(1)
