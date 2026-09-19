#!/usr/bin/env python3
"""run_multimodal.py - Master Orchestrator for Multimodal Adaptive IDE Research

Menjalankan Eye Tracking Prototype dan HRV Monitor Prototype secara paralel dan tersinkronisasi
ke dalam satu sesi eksperimen terpadu untuk mengukur Cognitive Load programmer.

Contoh Penggunaan:
    # Uji coba sesi standar:
    python run_multimodal.py --participant P01 --task debug_recursion

    # Membatasi durasi tugas selama 5 menit (300 detik):
    python run_multimodal.py --participant P01 --task hard_task --duration 300

    # Menentukan alamat BLE sensor spesifik:
    python run_multimodal.py --name HW9 --address AA:BB:CC:DD:EE:FF
"""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

# Impor pipeline fusi data
try:
    from fuse_and_analyze import fuse_session
except ImportError:
    fuse_session = None


def setup_session_directory(output_root: Path, participant: str, task: str) -> Path:
    """Membuat folder sesi terpadu dengan format timestamp_participant_task."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_p = "".join(c if c.isalnum() or c in "-_" else "_" for c in participant)
    clean_t = "".join(c if c.isalnum() or c in "-_" else "_" for c in task)
    session_name = f"{timestamp}_{clean_p}_{clean_t}"
    
    session_dir = output_root / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def print_banner(session_dir: Path, participant: str, task: str, duration: float):
    print("=" * 65)
    print("      ADAPTIVE IDE — MULTIMODAL COGNITIVE LOAD SUITE")
    print("=" * 65)
    print(f"  Partisipan : {participant}")
    print(f"  Tugas      : {task}")
    print(f"  Durasi     : {f'{duration} detik' if duration else 'Hingga Ctrl+C / selesai'}")
    print(f"  Folder Sesi: {session_dir}")
    print("=" * 65)
    print("Alur Eksperimen:")
    print("  1. Pastikan sensor detak jantung (HW9) terpasang.")
    print("  2. Kalibrasi Baseline HRV (~120s): Partisipan duduk tenang.")
    print("  3. Kalibrasi Eye Tracking (9-titik): Ikuti titik merah di layar.")
    print("  4. Pengerjaan Tugas: Mulai ngoding.")
    print("  Tekan Ctrl+C kapan saja untuk mengakhiri sesi lebih awal.")
    print("=" * 65 + "\n")


def main(argv=None):
    repo_root = Path(__file__).resolve().parent
    default_sessions = repo_root / "sessions"
    default_eye_config = repo_root / "eye_tracking_prototype" / "config.yaml"

    parser = argparse.ArgumentParser(description="Adaptive IDE Multimodal Suite Runner")
    parser.add_argument("--participant", default="P01", help="ID Partisipan (contoh: P01)")
    parser.add_argument("--task", default="coding_task", help="Nama/label tugas koding (contoh: debug_easy)")
    parser.add_argument("--duration", type=float, default=None, help="Durasi perekaman tugas (detik)")
    parser.add_argument("--name", default="HW9", help="Potongan nama sensor BLE (default: HW9)")
    parser.add_argument("--address", default=None, help="Alamat BLE sensor jika ada beberapa perangkat")
    parser.add_argument("--output-root", type=Path, default=default_sessions, help="Folder induk seluruh sesi")
    parser.add_argument("--config-eye", type=Path, default=default_eye_config, help="Path ke config.yaml eye tracking")
    parser.add_argument("--skip-hrv", action="store_true", help="Jalankan hanya eye tracking (mode uji)")
    parser.add_argument("--skip-eye", action="store_true", help="Jalankan hanya HRV monitor (mode uji)")
    args = parser.parse_args(argv)

    session_dir = setup_session_directory(args.output_root, args.participant, args.task)
    print_banner(session_dir, args.participant, args.task, args.duration)

    processes = []
    python_bin = sys.executable

    # 1. Siapkan proses HRV Monitor
    hrv_proc = None
    if not args.skip_hrv:
        hrv_script = repo_root / "hrv_monitor_prototype" / "calibrate.py"
        hrv_cmd = [
            python_bin, str(hrv_script),
            "--name", args.name,
            "--session-dir", str(session_dir)
        ]
        if args.address:
            hrv_cmd.extend(["--address", args.address])
        if args.duration:
            hrv_cmd.extend(["--duration", str(args.duration)])

        print("[Orchestrator] Menyiapkan proses HRV Monitor...")
        try:
            hrv_proc = subprocess.Popen(hrv_cmd, cwd=str(repo_root))
            processes.append(("HRV Monitor", hrv_proc))
        except Exception as e:
            print(f"[Orchestrator Error] Gagal menjalankan HRV: {e}", file=sys.stderr)
            return 1

    # Beri jeda singkat agar BLE scanning mulai berjalan
    time.sleep(1.0)

    # 2. Siapkan proses Eye Tracker
    eye_proc = None
    if not args.skip_eye:
        eye_script = repo_root / "eye_tracking_prototype" / "eye_tracking_prototype.py"
        eye_cmd = [
            python_bin, str(eye_script),
            "--config", str(args.config_eye),
            "--session-dir", str(session_dir),
            "--session-id", session_dir.name
        ]

        print("[Orchestrator] Menyiapkan proses Eye Tracking...")
        try:
            eye_proc = subprocess.Popen(eye_cmd, cwd=str(repo_root / "eye_tracking_prototype"))
            processes.append(("Eye Tracker", eye_proc))
        except Exception as e:
            print(f"[Orchestrator Error] Gagal menjalankan Eye Tracker: {e}", file=sys.stderr)
            # Hentikan HRV jika eye tracker gagal
            if hrv_proc:
                hrv_proc.terminate()
            return 1

    print("\n[Orchestrator] Kedua proses aktif. Memantau sesi...")

    session_interrupted = False
    start_time = time.monotonic()

    try:
        while True:
            # Cek jika proses selesai sendiri
            all_done = True
            for name, proc in processes:
                ret = proc.poll()
                if ret is None:
                    all_done = False
                else:
                    # Salah satu proses selesai (misal user menekan 'q' di window eye tracking)
                    pass

            if all_done:
                print("\n[Orchestrator] Semua proses telah selesai secara normal.")
                break

            # Cek limit durasi jika ditentukan
            if args.duration and (time.monotonic() - start_time) > (args.duration + 140): # +140s untuk baseline
                print(f"\n[Orchestrator] Batas waktu sesi ({args.duration}s) tercapai.")
                break

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[Orchestrator] Sinyal Ctrl+C diterima. Menghentikan semua proses secara anggun...")
        session_interrupted = True

    finally:
        # Hentikan semua child process dengan aman
        for name, proc in processes:
            if proc.poll() is None:
                print(f"[Orchestrator] Menutup {name}...")
                try:
                    proc.send_signal(signal.SIGINT)
                    proc.wait(timeout=3)
                except Exception:
                    proc.terminate()
                    proc.wait(timeout=2)

        print("[Orchestrator] Semua file log telah di-flush ke disk.")

    # 3. Jalankan Fusi Data dan Analisis Kognitif Otomatis
    hrv_csv = session_dir / "hrv.csv"
    eye_csv = session_dir / "eye_tracking.csv"
    
    if hrv_csv.exists() and eye_csv.exists() and fuse_session is not None:
        print("\n[Orchestrator] Menjalankan Pipeline Fusi Fitur & Analisis Beban Kognitif...")
        try:
            fuse_session(session_dir)
        except Exception as e:
            print(f"[Orchestrator] Gagal melakukan fusi otomatis: {e}", file=sys.stderr)
            print(f"Anda dapat mencoba menjalankan manual: python fuse_and_analyze.py --session-dir {session_dir}")
    else:
        print(f"\n[Orchestrator] File log tersimpan di: {session_dir}")
        if not hrv_csv.exists():
            print("  (Catatan: hrv.csv tidak terbentuk / sesi dihentikan sebelum data tersimpan)")
        if not eye_csv.exists():
            print("  (Catatan: eye_tracking.csv tidak terbentuk)")

    print("\n[Orchestrator] Sesi selesai.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
