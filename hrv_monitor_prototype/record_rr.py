import argparse
import asyncio
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import time
from uuid import uuid4


HR_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
SESSION_ROOT = Path(__file__).with_name("sessions")
RAW_COLUMNS = [
    "raw_id", "packet_id", "received_at", "elapsed_seconds",
    "rr_units", "rr_original_ms", "contact_detected", "phase",
]


def parse_rr_packet(data):
    """Ambil RR (unit 1/1024 detik) dan status kontak dari paket standar BLE."""
    if len(data) < 2:
        raise ValueError("Paket BLE terlalu pendek")

    flags = data[0]
    if flags & 0xE0:
        raise ValueError("Flag BLE tidak sesuai format Heart Rate Measurement")

    offset = 1 + (2 if flags & 0x01 else 1)
    if flags & 0x08:
        offset += 2  
    if len(data) < offset:
        raise ValueError("Field sebelum RR terpotong")

    contact = bool(flags & 0x02) if flags & 0x04 else None
    if not flags & 0x10:
        if len(data) != offset:
            raise ValueError("Ada byte tambahan tanpa flag RR")
        return [], contact

    remaining = len(data) - offset
    if remaining == 0 or remaining % 2:
        raise ValueError("Data RR harus berisi pasangan byte yang lengkap")
    rr_units = [int.from_bytes(data[i:i + 2], "little")
                for i in range(offset, len(data), 2)]
    return rr_units, contact


class RRRecorder:
    """Tulis RR asli tanpa filtering atau perhitungan HRV."""

    def __init__(self, raw_file):
        self.raw_file = raw_file
        self.writer = csv.DictWriter(raw_file, fieldnames=RAW_COLUMNS)
        self.writer.writeheader()
        self.raw_file.flush()
        self.packet_id = 0
        self.raw_id = 0
        self.last_rows = []

    def record(self, data, received_at, elapsed_seconds, phase="raw"):
        self.packet_id += 1
        self.last_rows = []
        try:
            rr_units, contact = parse_rr_packet(data)
        except ValueError as error:
            print(f"Paket {self.packet_id}: {error}; RR tidak dapat dibaca.", file=sys.stderr)
            return 0

        for units in rr_units:
            self.raw_id += 1
            row = {
                "raw_id": self.raw_id, "packet_id": self.packet_id,
                "received_at": received_at, "elapsed_seconds": elapsed_seconds,
                "rr_units": units, "rr_original_ms": units * 1000 / 1024,
                "contact_detected": contact, "phase": phase,
            }
            self.writer.writerow(row)
            self.last_rows.append(row)
        self.raw_file.flush()
        return len(rr_units)


async def record_sensor(name="HW9", address=None, duration=None, output=SESSION_ROOT, monitor=None):
    """Rekam RR. Monitor opsional menerima start, receive, tick, dan finish;
    logika kalibrasi tetap berada di file pemanggil, bukan di perekam raw.
    """
    from bleak import BleakClient, BleakScanner

    device = address
    if not device:
        print(f"Mencari sensor {name!r}...")
        devices = await BleakScanner.discover(timeout=5)
        matches = [item for item in devices if item.name and name.lower() in item.name.lower()]
        if len(matches) != 1:
            found = ", ".join(f"{item.name}: {item.address}" for item in matches)
            raise RuntimeError(f"Ditemukan {len(matches)} sensor yang cocok. "
                               f"Gunakan --address. Hasil: {found or 'tidak ada'}")
        device = matches[0]

    disconnected = asyncio.Event()
    async with BleakClient(device, disconnected_callback=lambda _: disconnected.set()) as client:
        session_name = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid4().hex[:8]}"
        session = Path(output) / session_name
        session.mkdir(parents=True, exist_ok=False)
        origin = time.monotonic()
        with (session / "rr_raw.csv").open("x", newline="", encoding="utf-8") as raw_file:
            recorder = RRRecorder(raw_file)
            failure = None
            end_reason = "stopped"

            def on_packet(_sender, data):
                nonlocal failure
                if failure is not None:
                    return
                try:
                    timestamp = datetime.now(timezone.utc).isoformat()
                    count = recorder.record(data, timestamp, time.monotonic() - origin,
                                            phase=getattr(monitor, "phase", "raw"))
                    if monitor is not None:
                        monitor.receive(recorder.last_rows)
                    else:
                        print(f"\rPaket: {recorder.packet_id} | RR tersimpan: {recorder.raw_id} "
                              f"| RR di paket terakhir: {count}   ", end="", flush=True)
                except Exception as error:
                    failure = error
                    disconnected.set()

            print(f"Tersambung. Menyimpan RR mentah ke: {session}")
            print("Tekan Ctrl+C untuk berhenti.")
            try:
                if monitor is not None:
                    monitor.start(session)
                await client.start_notify(HR_MEASUREMENT_UUID, on_packet)
                while not disconnected.is_set():
                    elapsed = time.monotonic() - origin
                    if monitor is not None and monitor.tick(elapsed):
                        end_reason = "monitor_finished"
                        break
                    if duration is not None and elapsed >= duration:
                        end_reason = "duration_reached"
                        break
                    await asyncio.sleep(0.1)
                if failure is not None:
                    raise failure
                if not client.is_connected:
                    end_reason = "disconnected"
                    print("\nSensor terputus; perekaman dihentikan.")
            except (asyncio.CancelledError, KeyboardInterrupt):
                end_reason = "interrupted"
                raise
            except Exception:
                end_reason = "recording_error"
                raise
            finally:
                stopped_elapsed = time.monotonic() - origin
                try:
                    if client.is_connected:
                        await client.stop_notify(HR_MEASUREMENT_UUID)
                finally:
                    try:
                        if monitor is not None:
                            monitor.finish(stopped_elapsed, end_reason)
                    finally:
                        print(f"\nPerekaman ditutup: {recorder.raw_id} RR ditulis ke {session}")
    return session


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="HW9", help="Potongan nama sensor")
    parser.add_argument("--address", help="Alamat BLE jika sensor lebih dari satu")
    parser.add_argument("--duration", type=float, help="Durasi rekaman dalam detik; default sampai Ctrl+C")
    parser.add_argument("--output", type=Path, default=SESSION_ROOT, help="Folder induk sesi")
    args = parser.parse_args(argv)
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration harus positif dan finite")
    return asyncio.run(record_sensor(args.name, args.address, args.duration, args.output))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPerekaman dihentikan. Data yang sudah ditulis tetap tersimpan.")
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Gagal merekam: {error}", file=sys.stderr)
        sys.exit(1)
