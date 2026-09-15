import argparse
import csv
from datetime import datetime, timedelta
import math
from pathlib import Path
from statistics import mean, stdev
import sys


# Mendukung perintah python -m maupun menjalankan file secara langsung.
if __package__:
    from .config import (WINDOW_SECONDS, STEP_SECONDS, RR_GAP_SECONDS,
                         HRV_MIN_COVERAGE, HRV_MIN_PAIRS, HRV_MAX_DATA_AGE_SECONDS)
else:
    from config import (WINDOW_SECONDS, STEP_SECONDS, RR_GAP_SECONDS,
                        HRV_MIN_COVERAGE, HRV_MIN_PAIRS, HRV_MAX_DATA_AGE_SECONDS)


HRV_COLUMNS = [
    "window_start", "window_end", "window_seconds", "rr_count", "rr_pairs",
    "rmssd_ms", "sdnn_ms",
    "status", "reason", "accepted_rr_seconds", "coverage", "data_age_seconds",
]


def calculate_hrv(intervals):
    """RMSSD: akar rata-rata kuadrat selisih RR. SDNN: standar deviasi sampel."""
    usable = [row for row in intervals if row["rr_processed_ms"] is not None]
    rr_values = [row["rr_processed_ms"] for row in usable]
    squared_differences = []

    for previous, current in zip(usable, usable[1:]):
        consecutive = current["raw_id"] == previous["raw_id"] + 1
        same_segment = current["segment_id"] == previous["segment_id"]
        # Jangan membuat pasangan melintasi RR terbuang atau jeda rekaman.
        if consecutive and same_segment:
            difference = current["rr_processed_ms"] - previous["rr_processed_ms"]
            squared_differences.append(difference ** 2)

    rmssd = math.sqrt(mean(squared_differences)) if squared_differences else None
    sdnn = stdev(rr_values) if len(rr_values) >= 2 else None
    return {
        "rr_count": len(rr_values), "rr_pairs": len(squared_differences),
        "rmssd_ms": rmssd, "sdnn_ms": sdnn,
    }


def window_features(rows, start, end, last_event=None, min_coverage=HRV_MIN_COVERAGE,
                    min_pairs=HRV_MIN_PAIRS, max_age=HRV_MAX_DATA_AGE_SECONDS):
    """Pisahkan rumus matematis dari keputusan apakah hasil boleh ditampilkan."""
    if (not math.isfinite(min_coverage) or not 0 <= min_coverage <= 1
            or type(min_pairs) is not int or min_pairs < 1
            or not math.isfinite(max_age) or max_age <= 0):
        raise ValueError("Periksa minimum coverage, pasangan, dan batas umur data")
    visible = [row for row in rows if row["elapsed_seconds"] <= end + 1e-9]
    if last_event is None and visible:
        latest = max(visible, key=lambda row: (row["elapsed_seconds"], row["raw_id"]))
        last_event = (latest["elapsed_seconds"], latest.get("recovery_state", "normal"))
    selected = [row for row in visible
                if start + 1e-9 < row["beat_elapsed_seconds"] <= end + 1e-9
                and row.get("available_seconds", row["elapsed_seconds"]) <= end + 1e-9]
    features = calculate_hrv(selected)
    usable = sorted((row for row in selected if row["rr_processed_ms"] is not None),
                    key=lambda row: row["beat_elapsed_seconds"])
    covered_until = start
    duration = 0.0
    for row in usable:
        beat_end = row["beat_elapsed_seconds"]
        beat_start = max(start, beat_end - row["rr_processed_ms"] / 1000)
        duration += max(0, beat_end - max(beat_start, covered_until))
        covered_until = max(covered_until, beat_end)
    coverage = min(1.0, duration / (end - start))
    age = max(0, end - usable[-1]["beat_elapsed_seconds"]) if usable else None
    reasons = []
    if coverage + 1e-9 < min_coverage:
        reasons.append("insufficient_coverage")
    if features["rr_pairs"] < min_pairs:
        reasons.append("insufficient_pairs")
    if age is None or age > max_age + 1e-9:
        reasons.append("stale_or_missing_data")
    if last_event is None or end - last_event[0] > RR_GAP_SECONDS + 1e-9:
        status = "gap"
        reasons.insert(0, "rr_gap")
    elif last_event[1] == "recovering":
        status = "recovering"
        reasons.insert(0, "recovery_pending")
    else:
        status = "insufficient_data" if reasons else "ready"
    if status != "ready":
        features["rmssd_ms"] = features["sdnn_ms"] = None
    return {**features, "status": status, "reason": ";".join(reasons),
            "accepted_rr_seconds": duration, "coverage": coverage, "data_age_seconds": age}


class LiveHRV:
    """Window berjalan sesuai timer, termasuk saat tidak ada RR baru."""

    def __init__(self, origin, window_seconds=WINDOW_SECONDS, step_seconds=STEP_SECONDS,
                 min_coverage=HRV_MIN_COVERAGE, min_pairs=HRV_MIN_PAIRS,
                 max_age=HRV_MAX_DATA_AGE_SECONDS):
        if (not math.isfinite(window_seconds) or not math.isfinite(step_seconds)
                or not 0 < step_seconds <= window_seconds):
            raise ValueError("Durasi harus finite dan memenuhi 0 < step <= window")
        self.origin = origin
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds
        self.next_end = window_seconds
        self.rows = []
        self.events = []
        self.requirements = dict(min_coverage=min_coverage, min_pairs=min_pairs, max_age=max_age)
        window_features([], 0, window_seconds, **self.requirements)  # Validasi sebelum merekam.

    def add(self, processed, received_elapsed=None, recovering=False):
        self.rows.extend(processed)
        if received_elapsed is not None:
            self.events.append((received_elapsed, "recovering" if recovering else "normal"))
        else:
            self.events.extend((row["elapsed_seconds"], row.get("recovery_state", "normal")) for row in processed)

    def advance(self, elapsed, include_boundary=True):
        """Terbitkan setiap window satu kali. Sebelum paket masuk, selesaikan
        window yang berakhir SEBELUM paket diterima agar data terlambat tidak
        mengubah hasil yang seharusnya sudah tersedia.
        """
        limit = elapsed + 1e-9 if include_boundary else elapsed - 1e-9
        while self.next_end <= limit:
            end = self.next_end
            start = end - self.window_seconds
            past_events = [event for event in self.events if event[0] <= end + 1e-9]
            last_event = past_events[-1] if past_events else None
            features = {
                "window_start": (self.origin + timedelta(seconds=start)).isoformat(),
                "window_end": (self.origin + timedelta(seconds=end)).isoformat(),
                "window_seconds": self.window_seconds,
                **window_features(self.rows, start, end, last_event, **self.requirements),
            }
            self.next_end += self.step_seconds
            # Simpan hanya RR yang masih mungkin masuk window berikutnya.
            keep_after = self.next_end - self.window_seconds
            self.rows = [row for row in self.rows if row["beat_elapsed_seconds"] > keep_after + 1e-9]
            self.events = ([last_event] if last_event else []) + [event for event in self.events if event[0] > end + 1e-9]
            yield features


def read_processed_csv(source):
    """Ubah teks CSV menjadi angka. Nilai RR kosong tetap None, bukan nol."""
    intervals = []
    required = {"raw_id", "segment_id", "received_at", "elapsed_seconds",
                "beat_elapsed_seconds", "rr_processed_ms"}
    with Path(source).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Input harus rr_processed.csv; jalankan preprocessing dahulu")
        previous_id = 0
        for row in reader:
            row["raw_id"] = int(row["raw_id"])
            row["segment_id"] = int(row["segment_id"])
            row["elapsed_seconds"] = float(row["elapsed_seconds"])
            row["beat_elapsed_seconds"] = float(row["beat_elapsed_seconds"])
            rr = float(row["rr_processed_ms"]) if row["rr_processed_ms"] else None
            if rr is not None and (not math.isfinite(rr) or rr <= 0):
                raise ValueError("RR processed harus positif dan finite, atau kosong")
            if (row["raw_id"] <= previous_id or row["segment_id"] < 0
                    or not math.isfinite(row["elapsed_seconds"]) or row["elapsed_seconds"] < 0
                    or not math.isfinite(row["beat_elapsed_seconds"])):
                raise ValueError("ID atau waktu pada CSV processed tidak valid")
            row["rr_processed_ms"] = rr
            row["available_seconds"] = float(row.get("available_seconds") or row["elapsed_seconds"])
            if not math.isfinite(row["available_seconds"]) or row["available_seconds"] < row["elapsed_seconds"]:
                raise ValueError("Waktu ketersediaan preprocessing tidak valid")
            intervals.append(row)
            previous_id = row["raw_id"]
    return intervals


def hrv_windows(intervals, window_seconds=WINDOW_SECONDS, step_seconds=STEP_SECONDS, **requirements):
    """Window (awal, akhir]: data tepat pada batas akhir tidak dihitung dua kali
    pada window bersebelahan yang tidak overlap. Window parsial terakhir dilewati.
    """
    if (not math.isfinite(window_seconds) or not math.isfinite(step_seconds)
            or not 0 < step_seconds <= window_seconds):
        raise ValueError("Durasi harus finite dan memenuhi 0 < step <= window")
    if not intervals:
        return

    first = intervals[0]
    received_at = datetime.fromisoformat(first["received_at"])
    if received_at.utcoffset() is None:
        raise ValueError("Timestamp harus menyertakan zona waktu")
    origin = received_at - timedelta(seconds=first["elapsed_seconds"])
    last_elapsed = max(row["elapsed_seconds"] for row in intervals)
    end = window_seconds

    while end <= last_elapsed + 1e-9:
        start = end - window_seconds
        features = window_features(intervals, start, end, **requirements)
        yield {
            "window_start": (origin + timedelta(seconds=start)).isoformat(),
            "window_end": (origin + timedelta(seconds=end)).isoformat(),
            "window_seconds": window_seconds, **features,
        }
        end += step_seconds


def calculate_csv(source, output=None, window_seconds=WINDOW_SECONDS, step_seconds=STEP_SECONDS):
    source = Path(source)
    output = Path(output) if output else source.with_name("hrv.csv")
    intervals = read_processed_csv(source)
    count = 0
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HRV_COLUMNS)
        writer.writeheader()
        for features in hrv_windows(intervals, window_seconds, step_seconds):
            writer.writerow(features)
            count += 1
    print(f"Hasil HRV: {output} ({count} window)")
    if count == 0:
        print("Belum ada window lengkap sampai timestamp RR terakhir.")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="File rr_processed.csv")
    parser.add_argument("--output", type=Path, help="Default: hrv.csv di folder sumber")
    parser.add_argument("--window", type=float, default=WINDOW_SECONDS,
                        help="Panjang window dalam detik; default dari config.py")
    parser.add_argument("--step", type=float, default=STEP_SECONDS,
                        help="Pergeseran window dalam detik; default dari config.py")
    args = parser.parse_args(argv)
    if (not math.isfinite(args.window) or not math.isfinite(args.step)
            or not 0 < args.step <= args.window):
        parser.error("Durasi harus finite dan memenuhi 0 < step <= window")
    return calculate_csv(args.source, args.output, args.window, args.step)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(f"Gagal menghitung HRV: {error}", file=sys.stderr)
        sys.exit(1)
