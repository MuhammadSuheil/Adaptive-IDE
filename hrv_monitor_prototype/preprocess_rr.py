import argparse
from collections import deque
import csv
from datetime import datetime, timedelta
from itertools import groupby
import math
from pathlib import Path
from statistics import median
import sys

if __package__:
    from .config import FILTER_ENABLED, MEDIAN_THRESHOLD_MS, RR_GAP_SECONDS, RECOVERY_RR_COUNT
else:
    from config import FILTER_ENABLED, MEDIAN_THRESHOLD_MS, RR_GAP_SECONDS, RECOVERY_RR_COUNT

MIN_RR_MS = 300
MAX_RR_MS = 2000
HISTORY_SIZE = 11
MIN_HISTORY = 5
TIMESTAMP_TOLERANCE_SECONDS = 2

REQUIRED_COLUMNS = {
    "raw_id", "packet_id", "received_at", "elapsed_seconds", "rr_original_ms",
}
PROCESSED_COLUMNS = [
    "raw_id", "packet_id", "received_at", "elapsed_seconds", "rr_original_ms",
    "beat_time", "beat_elapsed_seconds", "rr_processed_ms", "segment_id",
    "reason", "reference_median_ms", "filter_enabled", "threshold_ms",
    "available_seconds", "recovery_state",
]


class RRPreprocessor:
    """Filter kausal per paket. Riwayat dipertahankan di antara paket live."""

    def __init__(self, filter_enabled=FILTER_ENABLED, threshold_ms=MEDIAN_THRESHOLD_MS,
                 gap_seconds=RR_GAP_SECONDS, recovery_count=RECOVERY_RR_COUNT):
        if not math.isfinite(threshold_ms) or threshold_ms <= 0:
            raise ValueError("Threshold harus positif dan finite")
        self.filter_enabled = filter_enabled
        self.threshold_ms = threshold_ms
        if not math.isfinite(gap_seconds) or gap_seconds <= 0:
            raise ValueError("Batas jeda harus positif dan finite")
        if type(recovery_count) is not int or recovery_count < 2:
            raise ValueError("Jumlah RR pemulihan harus bilangan bulat minimal 2")
        self.gap_seconds = gap_seconds
        self.recovery_count = recovery_count
        self.recovering = False
        self.pending = []
        self.history = deque(maxlen=HISTORY_SIZE)
        self.cursor = None
        self.origin = None
        self.previous_packet_id = 0
        self.previous_raw_id = 0
        self.previous_elapsed = 0.0
        self.segment_id = 0

    def process_packet(self, packet):
        """Terima satu paket lengkap; tidak perlu membaca ulang RR sebelumnya."""
        if not packet:
            return []
        packet_id = int(packet[0]["packet_id"])
        if any(int(row["packet_id"]) != packet_id for row in packet):
            raise ValueError("process_packet hanya menerima satu paket BLE")
        elapsed = float(packet[0]["elapsed_seconds"])
        if packet_id <= self.previous_packet_id or not math.isfinite(elapsed) or elapsed < self.previous_elapsed:
            raise ValueError("Urutan paket/waktu tidak valid; gunakan CSV asli tanpa mengurutkan ulang")
        received_at = datetime.fromisoformat(packet[0]["received_at"])
        if received_at.utcoffset() is None:
            raise ValueError("Timestamp harus menyertakan zona waktu")
        if self.origin is None:
            self.origin = received_at - timedelta(seconds=elapsed)

        released = []
        gap = self.previous_packet_id > 0 and elapsed - self.previous_elapsed > self.gap_seconds
        if gap:
            released = self.finish_pending(elapsed)
            self.recovering = True
            self.history.clear()
            self.segment_id += 1

        rr_values = [float(row["rr_original_ms"]) for row in packet]
        in_range = [math.isfinite(rr) and MIN_RR_MS <= rr <= MAX_RR_MS for rr in rr_values]
        # Pembatasan ini hanya untuk rekonstruksi waktu; RR asli tidak diubah.
        durations = [min(MAX_RR_MS, max(MIN_RR_MS, rr)) / 1000 if math.isfinite(rr)
                     else MIN_RR_MS / 1000 for rr in rr_values]
        batch_duration = sum(durations)
        reanchor = (gap or self.cursor is None or not all(in_range)
                    or abs(self.cursor + batch_duration - elapsed) > TIMESTAMP_TOLERANCE_SECONDS)
        packet_start = elapsed - batch_duration if reanchor else self.cursor
        overlap = self.cursor is not None and packet_start < self.cursor - 1e-6
        timing_valid = all(in_range) and not overlap
        if reanchor:
            self.history.clear()
            self.segment_id += 1  # RMSSD tidak boleh menyambung dua bagian yang terputus.

        beat_end = packet_start
        result = []
        for row, rr, seconds, plausible in zip(packet, rr_values, durations, in_range):
            raw_id = int(row["raw_id"])
            if raw_id <= self.previous_raw_id or float(row["elapsed_seconds"]) != elapsed:
                raise ValueError("Urutan RR atau timestamp dalam satu paket tidak valid")
            beat_end += seconds
            reasons = []
            if beat_end - seconds < -1e-9:
                reasons.append("before_recording_start")
            if not plausible:
                reasons.append("rr_out_of_range")
            if not timing_valid:
                reasons.append("uncertain_beat_timing")
            if str(row.get("contact_detected", "")).lower() == "false":
                reasons.append("sensor_contact_lost")

            hard_invalid = bool(reasons)
            if hard_invalid or raw_id != self.previous_raw_id + 1:
                self.history.clear()
                self.segment_id += 1
            reference = median(self.history) if len(self.history) >= MIN_HISTORY else None
            outlier = reference is not None and abs(rr - reference) > self.threshold_ms
            if outlier and self.filter_enabled and not self.recovering:
                reasons.append("median_threshold")

            processed = None if reasons else rr
            if processed is None:
                self.segment_id += 1
            if not hard_invalid:
                self.history.append(rr)

            result.append({
                "raw_id": raw_id, "packet_id": packet_id,
                "received_at": row["received_at"], "elapsed_seconds": elapsed,
                "rr_original_ms": row["rr_original_ms"],
                "beat_time": (self.origin + timedelta(seconds=beat_end)).isoformat(),
                "beat_elapsed_seconds": beat_end, "rr_processed_ms": processed,
                "segment_id": self.segment_id, "reason": ";".join(reasons),
                "reference_median_ms": reference,
                "filter_enabled": self.filter_enabled, "threshold_ms": self.threshold_ms,
                "available_seconds": elapsed,
                "recovery_state": "recovering" if self.recovering else "normal",
            })
            self.previous_raw_id = raw_id

        self.cursor = max(self.cursor, beat_end) if self.cursor is not None else beat_end
        self.previous_packet_id = packet_id
        self.previous_elapsed = elapsed
        if self.recovering:
            return released + self.recover(result, elapsed)
        return released + result

    def recover(self, rows, elapsed):
        """Tahan RR sampai median kelompok pemulihan dapat dibentuk.
        RR awal diperiksa kembali; hasil yang sudah diterbitkan tidak diubah.
        """
        self.pending.extend(rows)
        candidates = [row["rr_processed_ms"] for row in self.pending if row["rr_processed_ms"] is not None]
        reference = median(candidates[-max(HISTORY_SIZE, self.recovery_count):]) if candidates else None
        accepted = [rr for rr in candidates
                    if not self.filter_enabled or abs(rr - reference) <= self.threshold_ms]
        if len(accepted) >= self.recovery_count:
            self.segment_id += 1
            for row in self.pending:
                rr = row["rr_processed_ms"]
                row["available_seconds"] = elapsed
                row["reference_median_ms"] = reference
                if rr is not None and self.filter_enabled and abs(rr - reference) > self.threshold_ms:
                    row["rr_processed_ms"] = None
                    row["reason"] = "median_threshold"
                row["segment_id"] = self.segment_id
                if row["rr_processed_ms"] is None:
                    self.segment_id += 1
            self.pending[-1]["recovery_state"] = "normal"
            result, self.pending = self.pending, []
            self.history.clear()
            self.history.extend(accepted[-HISTORY_SIZE:])
            self.recovering = False
            return result
        # Batasi memori saat pemulihan tidak kunjung berhasil.
        excess = max(0, len(self.pending) - max(HISTORY_SIZE, 2 * self.recovery_count))
        result, self.pending = self.pending[:excess], self.pending[excess:]
        return self.reject_pending(result, elapsed)

    @staticmethod
    def reject_pending(rows, elapsed):
        for row in rows:
            row["rr_processed_ms"] = None
            row["available_seconds"] = elapsed
            row["reason"] = ";".join(filter(None, [row["reason"], "recovery_incomplete"]))
        return rows

    def finish_pending(self, elapsed=None):
        """Simpan juga RR pemulihan yang belum lengkap saat jeda baru/sesi berakhir."""
        result, self.pending = self.pending, []
        return self.reject_pending(result, self.previous_elapsed if elapsed is None else elapsed)


def preprocess_rows(rows, filter_enabled=FILTER_ENABLED, threshold_ms=MEDIAN_THRESHOLD_MS):
    """Jalur offline memakai filter yang sama dengan perekaman live."""
    processor = RRPreprocessor(filter_enabled, threshold_ms)
    for _, packet_rows in groupby(rows, key=lambda row: row["packet_id"]):
        yield from processor.process_packet(list(packet_rows))
    yield from processor.finish_pending()


def preprocess_csv(source, output=None, filter_enabled=FILTER_ENABLED, threshold_ms=MEDIAN_THRESHOLD_MS):
    """Simpan hasil terpisah. File yang sudah ada tidak pernah ditimpa."""
    source = Path(source)
    output = Path(output) if output else source.with_name("rr_processed.csv")
    with source.open(newline="", encoding="utf-8") as raw_file:
        reader = csv.DictReader(raw_file)
        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV raw harus memuat: {', '.join(sorted(REQUIRED_COLUMNS))}")
        with output.open("x", newline="", encoding="utf-8") as processed_file:
            writer = csv.DictWriter(processed_file, fieldnames=PROCESSED_COLUMNS)
            writer.writeheader()
            writer.writerows(preprocess_rows(reader, filter_enabled, threshold_ms))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="File rr_raw.csv")
    parser.add_argument("--output", type=Path, help="Default: rr_processed.csv di folder sumber")
    parser.add_argument("--threshold-ms", type=float, default=MEDIAN_THRESHOLD_MS,
                        help="Ambang median dalam ms; default dari config.py")
    filter_options = parser.add_mutually_exclusive_group()
    filter_options.add_argument("--filter", dest="filter_enabled", action="store_true",
                                help="Aktifkan filter median untuk proses ini")
    filter_options.add_argument("--no-filter", dest="filter_enabled", action="store_false",
                                help="Lewati filter median saja untuk proses ini")
    parser.set_defaults(filter_enabled=FILTER_ENABLED)
    args = parser.parse_args(argv)
    if not math.isfinite(args.threshold_ms) or args.threshold_ms <= 0:
        parser.error("--threshold-ms harus positif dan finite")
    output = preprocess_csv(args.source, args.output, args.filter_enabled, args.threshold_ms)
    print(f"Hasil preprocessing: {output}")
    return output


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(f"Gagal preprocessing: {error}", file=sys.stderr)
        sys.exit(1)
