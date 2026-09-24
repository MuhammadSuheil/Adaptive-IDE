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
from datetime import datetime
import os
import math
from pathlib import Path
import signal
import subprocess
import sys
import time

from fuse_and_analyze import fuse_session


def setup_session_directory(output_root: Path, participant: str, task: str) -> Path:
    """Membuat folder sesi terpadu dengan format timestamp_participant_task."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_p = "".join(c if c.isalnum() or c in "-_" else "_" for c in participant)
    clean_t = "".join(c if c.isalnum() or c in "-_" else "_" for c in task)
    session_name = f"{timestamp}_{clean_p}_{clean_t}"
    
    session_dir = output_root / session_name
    suffix = 0
    while True:
        try:
            session_dir.mkdir(parents=True, exist_ok=False)
            return session_dir
        except FileExistsError:
            suffix += 1
            session_dir = output_root / f"{session_name}_{suffix}"


def print_banner(session_dir: Path, participant: str, task: str, duration: float | None, mock_hrv: bool = False):
    print("=" * 65)
    print("      ADAPTIVE IDE — MULTIMODAL COGNITIVE LOAD SUITE")
    print("=" * 65)
    print(f"  Partisipan : {participant}")
    print(f"  Tugas      : {task}")
    print(f"  Durasi     : {f'{duration} detik' if duration else 'Hingga Ctrl+C / selesai'}")
    print(f"  Sensor HRV : {'HW9 (SIMULATOR / MOCK)' if mock_hrv else 'HW9 (BLE)'}")
    print(f"  Folder Sesi: {session_dir}")
    print("=" * 65)
    print("Alur Eksperimen:")
    if mock_hrv:
        print("  1. Simulator sensor HRV aktif (tidak memerlukan armband fisik).")
    else:
        print("  1. Pastikan sensor detak jantung (HW9) terpasang.")
    print("  2. Kalibrasi mata dan baseline HRV berjalan bersamaan. Kurangi gerakan tubuh.")
    print("  3. Setelah mata selesai, layar HRV menampilkan sisa baseline jika belum siap.")
    print("  4. Pengerjaan Tugas: Mulai ngoding.")
    print("  Tekan Ctrl+C kapan saja untuk mengakhiri sesi lebih awal.")
    print("=" * 65 + "\n")


def stop_processes(processes) -> bool:
    """Stop every child; report failures rather than skipping remaining children."""
    clean_shutdown = True
    for name, proc in processes:
        if proc.poll() is not None:
            continue
        print(f"[Orchestrator] Menutup {name}...")
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
            proc.wait(timeout=15)
        except (OSError, subprocess.TimeoutExpired) as error:
            clean_shutdown = False
            print(f"[Orchestrator] Gagal menutup {name} secara normal: {error}", file=sys.stderr)
            try:
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired) as kill_error:
                print(f"[Orchestrator] Gagal menghentikan {name}: {kill_error}", file=sys.stderr)
    return clean_shutdown


def main(argv=None):
    repo_root = Path(__file__).resolve().parent
    default_sessions = repo_root / "sessions"
    default_eye_config = repo_root / "eye_tracking_prototype" / "config.yaml"

    parser = argparse.ArgumentParser(description="Adaptive IDE Multimodal Suite Runner")
    parser.add_argument("--participant", default="P01", help="ID Partisipan (contoh: P01)")
    parser.add_argument("--task", default="coding_task", help="Nama/label tugas koding (contoh: debug_easy)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Durasi tugas setelah baseline HRV; dengan --skip-hrv, durasi total termasuk kalibrasi mata (detik)")
    parser.add_argument("--name", default="HW9", help="Potongan nama sensor BLE (default: HW9)")
    parser.add_argument("--address", default=None, help="Alamat BLE sensor jika ada beberapa perangkat")
    parser.add_argument("--output-root", type=Path, default=default_sessions, help="Folder induk seluruh sesi")
    parser.add_argument("--config-eye", type=Path, default=default_eye_config, help="Path ke config.yaml eye tracking")
    parser.add_argument("--skip-hrv", action="store_true", help="Jalankan hanya eye tracking (mode uji)")
    parser.add_argument("--skip-eye", action="store_true", help="Jalankan hanya HRV monitor (mode uji)")
    parser.add_argument("--mock-hrv", action="store_true",
                        help="Gunakan simulator/mock sensor HRV (tanpa perangkat sensor fisik)")
    parser.add_argument("--target-baseline", type=float, default=None,
                        help="Target durasi kalibrasi baseline HRV dalam detik (default: 120)")
    args = parser.parse_args(argv)

    if args.skip_eye and args.skip_hrv:
        parser.error("--skip-eye dan --skip-hrv tidak boleh dipakai bersamaan")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration harus positif dan finite")
    if args.target_baseline is not None and (not math.isfinite(args.target_baseline) or args.target_baseline <= 0):
        parser.error("--target-baseline harus positif dan finite")
    eye_config = args.config_eye.resolve()
    if not args.skip_eye and not eye_config.is_file():
        parser.error(f"Konfigurasi eye tracking tidak ditemukan: {eye_config}")

    try:
        session_dir = setup_session_directory(args.output_root.resolve(), args.participant, args.task)
    except OSError as error:
        print(f"[Orchestrator] Gagal membuat folder sesi: {error}", file=sys.stderr)
        return 1
    print_banner(session_dir, args.participant, args.task, args.duration, mock_hrv=args.mock_hrv)

    processes = []
    python_bin = sys.executable
    # Pass an explicit integer, not **dict[str, int] (invalid for other Popen keywords).
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    exit_code = 0
    try:
        if not args.skip_hrv:
            hrv_cmd = [
                python_bin, str(repo_root / "hrv_monitor_prototype" / "calibrate.py"),
                "--name", args.name, "--session-dir", str(session_dir),
            ]
            if not args.skip_eye:
                hrv_cmd.append("--wait-for-eye")
            if args.mock_hrv:
                hrv_cmd.append("--mock")
            if args.target_baseline is not None:
                hrv_cmd.extend(["--target-baseline", str(args.target_baseline)])
            if args.address:
                hrv_cmd.extend(["--address", args.address])
            if args.duration is not None:
                hrv_cmd.extend(["--duration", str(args.duration)])
            print("[Orchestrator] Menyiapkan proses HRV Monitor" + (" (MOCK)..." if args.mock_hrv else "..."))
            hrv_proc = subprocess.Popen(hrv_cmd, cwd=str(repo_root), creationflags=creationflags)
            processes.append(("HRV Monitor", hrv_proc))
            # Startup is inside try/finally so Ctrl+C cannot leave HRV running.
            time.sleep(1.0)

        if not args.skip_eye:
            eye_cmd = [
                python_bin, str(repo_root / "eye_tracking_prototype" / "eye_tracking_prototype.py"),
                "--config", str(eye_config), "--session-dir", str(session_dir),
                "--session-id", session_dir.name,
            ]
            if not args.skip_hrv:
                eye_cmd.append("--wait-for-hrv")
            print("[Orchestrator] Menyiapkan proses Eye Tracking...")
            eye_proc = subprocess.Popen(
                eye_cmd, cwd=str(repo_root / "eye_tracking_prototype"), creationflags=creationflags)
            processes.append(("Eye Tracker", eye_proc))

        print(f"\n[Orchestrator] {len(processes)} proses aktif. Memantau sesi...")
        start_time = time.monotonic()
        while True:
            completed = []
            for name, proc in processes:
                code = proc.poll()
                if code is not None:
                    completed.append((name, code))
            if completed:
                for name, code in completed:
                    print(f"[Orchestrator] {name} selesai (exit {code}). Mengakhiri sesi.")
                    if code != 0:
                        exit_code = 1
                break

            # HRV owns task duration after baseline (which can take up to 300s).
            # Eye-only mode limits total runtime, including eye calibration.
            if (args.skip_hrv and args.duration is not None
                    and time.monotonic() - start_time >= args.duration):
                print(f"\n[Orchestrator] Batas waktu eye tracking ({args.duration}s) tercapai.")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[Orchestrator] Ctrl+C diterima. Menghentikan perekam...")
    except OSError as error:
        print(f"[Orchestrator] Gagal menjalankan perekam: {error}", file=sys.stderr)
        exit_code = 1
    finally:
        if not stop_processes(processes):
            exit_code = 1

    if any(proc.poll() is None for _, proc in processes):
        print("[Orchestrator] Masih ada perekam aktif; fusi ditunda.", file=sys.stderr)
        return 1

    # 3. Jalankan Fusi Data dan Analisis Kognitif Otomatis
    hrv_csv = session_dir / "hrv.csv"
    eye_csv = session_dir / "eye_tracking.csv"
    
    if not args.skip_hrv and not args.skip_eye and hrv_csv.exists() and eye_csv.exists():
        print("\n[Orchestrator] Menjalankan Pipeline Fusi Fitur & Analisis Beban Kognitif...")
        try:
            report = fuse_session(session_dir)
            print(f"[Orchestrator] CSV gabungan: {session_dir / report['timeline_file']}")
        except Exception as e:
            print(f"[Orchestrator] Gagal melakukan fusi otomatis: {e}", file=sys.stderr)
            print(f'Coba manual: python fuse_and_analyze.py --session-dir "{session_dir}"')
            return 1
    else:
        print(f"\n[Orchestrator] File log tersimpan di: {session_dir}")
        if not args.skip_hrv and not hrv_csv.exists():
            exit_code = 1
            print("  (Catatan: hrv.csv tidak terbentuk / sesi dihentikan sebelum data tersimpan)")
        if not args.skip_eye and not eye_csv.exists():
            exit_code = 1
            print("  (Catatan: eye_tracking.csv tidak terbentuk)")

    print("\n[Orchestrator] Sesi selesai.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
