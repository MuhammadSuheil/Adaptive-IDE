import argparse
import asyncio
import csv
from datetime import datetime, timedelta, timezone
from itertools import groupby
import json
import math
from pathlib import Path
import sys

if __package__:
    from .config import (BASELINE_TARGET_SECONDS, BASELINE_MAX_SECONDS,
                         MEDIAN_THRESHOLD_MS, WINDOW_SECONDS, STEP_SECONDS)
    from .record_rr import SESSION_ROOT, record_sensor
    from .preprocess_rr import RRPreprocessor, preprocess_rows
    from .calculate_hrv import HRV_COLUMNS, LiveHRV, calculate_hrv
else:
    from config import (BASELINE_TARGET_SECONDS, BASELINE_MAX_SECONDS,
                        MEDIAN_THRESHOLD_MS, WINDOW_SECONDS, STEP_SECONDS)
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
                 window_seconds=WINDOW_SECONDS, step_seconds=STEP_SECONDS):
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
        self.phase = "calibration"
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
        payload = {
            "phase": self.phase,
            "status": status,
            "elapsed_seconds": round(float(elapsed), 1),
            "maximum_seconds": float(self.maximum_seconds),
            "accepted_rr_seconds": round(float(duration), 1),
            "target_seconds": float(self.target_seconds),
            "remaining_seconds": round(float(remaining), 1),
            "text": text,
        }
        try:
            temp_file.write_text(json.dumps(payload), encoding="utf-8")
            temp_file.replace(status_file)
        except OSError:
            pass

    def start(self, session):
        self.session = Path(session)
        self.started_at = datetime.now(timezone.utc)
        self.hrv_file = (self.session / "hrv.csv").open("w", newline="", encoding="utf-8")
        self.hrv_writer = csv.DictWriter(self.hrv_file, fieldnames=["phase", *HRV_COLUMNS])
        self.hrv_writer.writeheader()

        self.hrv_file.flush()
        self.update_calib_status("calibrating", 0.0, 0.0)
        print(f"Duduk tenang. Target {self.target_seconds:g} detik RR diterima, "
              f"batas waktu {self.maximum_seconds:g} detik.")

    def receive(self, rows):
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
            self.rows.extend(row for row in rows if row["elapsed_seconds"] <= self.maximum_seconds)

    def progress(self, elapsed):
        # Maksimal lima menit data: gunakan ulang fungsi offline yang sama agar
        # aturan filter identik. Hanya proses ulang jika ada RR baru.
        if len(self.processed) != len(self.rows):
            self.processed = list(preprocess_rows(self.rows))
        return baseline_data(self.processed, elapsed, self.maximum_seconds)

    def make_result(self, status, reason, elapsed, used, duration):
        return {
            "status": status, "reason": reason, "condition": "seated_rest",
            "target_rr_seconds": self.target_seconds,
            "maximum_elapsed_seconds": self.maximum_seconds,
            "calibration_end_elapsed_seconds": elapsed,
            "accepted_rr_seconds": duration,
            # Baseline satu RMSSD/SDNN dari RR kalibrasi, bukan rata-rata window.
            "hrv": calculate_hrv(used) if status == "ready" else None,
        }

    def save_baseline(self, used):
        """Baseline juga hasil HRV, sehingga disimpan pada CSV HRV yang sama."""
        elapsed = self.result["calibration_end_elapsed_seconds"]
        origin = self.started_at
        if self.rows:
            first = self.rows[0]
            origin = datetime.fromisoformat(first["received_at"]) - timedelta(seconds=first["elapsed_seconds"])
        features = calculate_hrv(used)
        if self.result["status"] != "ready":
            features["rmssd_ms"] = features["sdnn_ms"] = None
        duration = self.result["accepted_rr_seconds"]
        self.hrv_writer.writerow({
            "phase": "baseline", "window_start": origin.isoformat(),
            "window_end": (origin + timedelta(seconds=elapsed)).isoformat(),
            "window_seconds": elapsed, **features,
            "status": self.result["status"], "reason": self.result["reason"],
            "accepted_rr_seconds": duration, "coverage": min(1, duration / elapsed) if elapsed > 0 else 0,
            "data_age_seconds": max(0, elapsed - used[-1]["beat_elapsed_seconds"]) if used else None,
        })
        self.hrv_file.flush()

    def tick(self, elapsed):
        """Dipanggil timer perekam, termasuk ketika sensor sama sekali tidak mengirim RR."""
        if self.result is not None:
            if self.result["status"] == "unavailable":
                return True
            self.publish_hrv(elapsed - self.task_start)
            return self.task_duration is not None and elapsed - self.task_start >= self.task_duration

        used, duration = self.progress(elapsed)
        if elapsed - self.last_status_update >= 0.5 or int(elapsed) != self.last_print:
            self.last_status_update = elapsed
            self.update_calib_status("calibrating", elapsed, duration)
        if int(elapsed) != self.last_print:
            self.last_print = int(elapsed)
            print(f"\rKalibrasi: {min(elapsed, self.maximum_seconds):.0f}/{self.maximum_seconds:g} detik "
                  f"| RR diterima: {duration:.1f}/{self.target_seconds:g} detik   ", end="", flush=True)

        features = calculate_hrv(used)
        if (duration + 1e-9 >= self.target_seconds
                and features["rmssd_ms"] is not None and features["sdnn_ms"] is not None):
            self.result = self.make_result("ready", "target_reached", elapsed, used, duration)
            self.save_baseline(used)
            self.start_task(elapsed)
            self.update_calib_status("ready", elapsed, duration)
            print(f"\nBaseline siap: RMSSD {features['rmssd_ms']:.2f} ms, SDNN {features['sdnn_ms']:.2f} ms.")
            print("Perekaman tugas dimulai. Silakan mulai coding; Ctrl+C untuk berhenti.")
            print(f"Preprocessing otomatis. HRV pertama setelah {self.window_seconds:g} detik, "
                  f"lalu setiap {self.step_seconds:g} detik.")
            print(f"Log sesi: {self.session / 'rr_raw.csv'} dan {self.session / 'hrv.csv'}")
        elif elapsed >= self.maximum_seconds:
            self.result = self.make_result("unavailable", "timeout", elapsed, used, duration)
            self.save_baseline(used)
            self.update_calib_status("unavailable", elapsed, duration)
            print("\nBaseline belum tersedia. Jalankan ulang perintah untuk mengulang kalibrasi dalam sesi baru.")
            return True
        return False

    def start_task(self, elapsed):
        """Mulai analisis tugas di memori, terpisah dari riwayat kalibrasi."""
        self.task_start = elapsed
        self.phase = "task"
        first = self.rows[0]
        origin = (datetime.fromisoformat(first["received_at"])
                  - timedelta(seconds=first["elapsed_seconds"]) + timedelta(seconds=elapsed))
        self.task_hrv.origin = origin

    def publish_hrv(self, elapsed, include_boundary=True):
        if self.task_duration is not None and elapsed > self.task_duration:
            elapsed = self.task_duration
            include_boundary = True
        for features in self.task_hrv.advance(elapsed, include_boundary):
            self.hrv_writer.writerow(dict(features, phase="task"))
            self.hrv_file.flush()
            rmssd = features["rmssd_ms"]
            sdnn = features["sdnn_ms"]
            rmssd_text = f"{rmssd:.2f} ms" if rmssd is not None else "tidak tersedia"
            sdnn_text = f"{sdnn:.2f} ms" if sdnn is not None else "tidak tersedia"
            label = {"ready": "siap", "gap": "jeda RR", "recovering": "pemulihan",
                     "insufficient_data": "data belum cukup"}[features["status"]]
            line = (f"[HRV] {features['window_end']} | window {self.window_seconds:g}s "
                    f"| status: {label} "
                    f"| RMSSD: {rmssd_text} | SDNN: {sdnn_text} "
                    f"| RR: {features['rr_count']} | pasangan: {features['rr_pairs']} "
                    f"| durasi RR: {features['accepted_rr_seconds']:.2f}/{self.window_seconds:g}s "
                    f"| alasan: {features['reason'] or '-'}")
            print(line, flush=True)

    def finish(self, elapsed, reason):
        try:
            if self.task_start is not None and reason != "recording_error":
                self.publish_hrv(elapsed - self.task_start)
            if self.result is None and self.hrv_file is not None:
                used, duration = self.progress(elapsed)
                self.result = self.make_result("unavailable", reason, elapsed, used, duration)
                self.save_baseline(used)
                self.update_calib_status("unavailable", elapsed, duration)
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
    parser.add_argument("--mock", action="store_true", help="Gunakan mock/simulasi sensor HRV tanpa perangkat fisik")
    parser.add_argument("--target-baseline", type=float, default=None,
                        help="Target durasi RR kalibrasi baseline dalam detik (default: 120)")
    args = parser.parse_args(argv)
    if args.target_baseline is not None and (not math.isfinite(args.target_baseline) or args.target_baseline <= 0):
        parser.error("--target-baseline harus positif dan finite")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration harus positif dan finite")

    target_baseline = args.target_baseline if args.target_baseline is not None else BASELINE_TARGET_SECONDS
    max_baseline = max(BASELINE_MAX_SECONDS, target_baseline * 2)
    calibration = SessionCalibration(task_duration=args.duration, target_seconds=target_baseline, maximum_seconds=max_baseline)
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
