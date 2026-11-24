# Dataset v4: rating kesulitan peserta

Dataset utama: `window_15s_step_5s/good_acceptable.csv`. Target: `difficulty_rating` (1–10). Lihat [QUALITY_REPORT.md](QUALITY_REPORT.md) untuk cakupan, asal label, eksklusi, dan batas data.

Gunakan `episode_ratings.csv` untuk bukti label, `rating_matches.csv` untuk audit log, dan `window_assignments.csv` untuk pemetaan window–episode. `manifest.json` menyimpan hash sumber dan keluaran. Distribusi 25/45 detik hanya tersedia sebagai ringkasan cakupan.

Build: `python -m analysis_sessions.dataset_fusion.build_dataset_v4`.

Untuk memverifikasi bundle, gunakan modul yang sama dengan `--verify <folder-v4>`. Untuk menyalin paket Batch 06, tambahkan `--package models/batch-06/dataset/v4_self_reported_rating`. Tambahkan `--replace` hanya bila memperbarui bundle atau paket v4 yang sudah terverifikasi.
