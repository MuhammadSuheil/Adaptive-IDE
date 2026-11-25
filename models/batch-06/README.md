# Batch 06 — Eksperimen `v3_behavioral_proxy` & `v4_self_reported_rating`

Batch 06 menjalankan dua rangkaian eksperimen terpisah dan bebas kebocoran pada dua paket dataset baru window 15 detik / step 5 detik di [dataset/DATASET_README.md](dataset/DATASET_README.md).

## Ringkasan Dokumen & Notebook

[Laporan hasil v3 dan perbandingan v4 (PDF)](v3/report/Laporan_v3.pdf) memuat sebaran data, kurva training, dan hasil LOPO, validasi serta Test resmi. Di Test v3, exp11 benar 11/18 episode dan exp13 benar 9/18. Keduanya mendeteksi 1/2 High.

| Berkas | Isi |
|---|---|
| [PLAN.md](PLAN.md) | Rancangan protokol eksperimen, aturan anti-kebocoran, profil fitur, normalisasi, dan seleksi model |
| [SUMMARY.md](SUMMARY.md) | Laporan lengkap hasil eksekusi V3 dan V4, temuan kunci, serta implikasi penelitian |
| [exp01_v3_behavioral_proxy/EyeHeartCognitiveLoad_v3.ipynb](exp01_v3_behavioral_proxy/EyeHeartCognitiveLoad_v3.ipynb) | Notebook mandiri klasifikasi 3-kelas pada `v3_behavioral_proxy` (sudah dieksekusi & terverifikasi) |
| [exp02_v4_self_reported_rating/EyeHeartCognitiveLoad_v4.ipynb](exp02_v4_self_reported_rating/EyeHeartCognitiveLoad_v4.ipynb) | Notebook mandiri regresi, 3-kelas, biner ekstrem, & diagnostik dalam-peserta pada `v4_self_reported_rating` (sudah dieksekusi & terverifikasi) |
| [exp03_v4_binary_overload/EyeHeartCognitiveLoad_v4_binary.ipynb](exp03_v4_binary_overload/EyeHeartCognitiveLoad_v4_binary.ipynb) | Notebook mandiri deep-dive deteksi biner overload (`1–3` vs `7–10`, `1–5` vs `6–10`, `1–6` vs `7–10`) dengan LOPO 17 peserta, bootstrap 95% CI, ablasi grup fitur, audit perancu, & audit zona netral (sudah dieksekusi & terverifikasi) |
| [outputs/exp01_v3_behavioral_proxy/RESULTS.md](outputs/exp01_v3_behavioral_proxy/RESULTS.md) | Ringkasan metrik, confusion matrix, dan grafik hasil V3 |
| [outputs/exp02_v4_self_reported_rating/RESULTS.md](outputs/exp02_v4_self_reported_rating/RESULTS.md) | Ringkasan metrik 4 track dan grafik hasil V4 |
| [outputs/exp03_v4_binary_overload/RESULTS.md](outputs/exp03_v4_binary_overload/RESULTS.md) | Ringkasan LOPO 17 peserta, 95% CI, audit perancu, dan grafik hasil Exp03 |

## Menjalankan Ulang (Lokal atau Google Colab)

Kedua notebook bersifat **mandiri** (tidak memerlukan file `.py` tambahan) dan otomatis mendeteksi lingkungan lokal maupun Google Colab:

1. **Lokal:** Buka notebook dari root repositori (atau folder notebook) dan jalankan **Run All**. Notebook otomatis menemukan `models/batch-06/dataset` dan menyimpan hasil ke `models/batch-06/outputs/`.
2. **Google Colab:** Salin folder `models/batch-06/dataset/` dan notebook ke Google Drive pada `/content/drive/MyDrive/Penelitian-Adaptive-IDE/models/batch-06/`, buka di Colab, lalu jalankan **Run All**.
3. **Verifikasi otomatis:** Setiap notebook memverifikasi SHA-256 seluruh CSV masukan, mengecek konsistensi kunci (`window_id`, `episode_id`, `participant_id`), memastikan tidak ada kebocoran antar-split, serta mengunci tepat satu pemenang per-window sebelum mengevaluasi Test.

## Struktur Output

```text
models/batch-06/outputs/
├── exp01_v3_behavioral_proxy/
│   ├── run.json
│   ├── cv_leaderboard.csv
│   ├── validation_leaderboard.csv
│   ├── test_evaluation.json
│   ├── test_predictions.csv
│   ├── episode_diagnostics.csv
│   ├── RESULTS.md
│   └── figures/
│       ├── v3_leaderboard_comparison.png
│       └── v3_confusion_matrices.png
└── exp02_v4_self_reported_rating/
    ├── run.json
    ├── participant_splits_v4.csv
    ├── trackA_regression_cv_leaderboard.csv
    ├── trackA_test_predictions.csv
    ├── trackB_3class_cv_leaderboard.csv
    ├── trackB_test_predictions.csv
    ├── trackC1_binary_cv_leaderboard.csv
    ├── trackC2_within_participant_loeo.csv
    ├── test_evaluation.json
    ├── RESULTS.md
    └── figures/
        ├── v4_experiment_overview.png
        └── v4_ablation_trackB.png
└── exp03_v4_binary_overload/          # Dibuat otomatis saat EyeHeartCognitiveLoad_v4_binary.ipynb dijalankan
    ├── lopo_all_tasks_leaderboard.csv
    ├── feature_coefficients_lopo.csv
    ├── binary_deepdive_summary.json
    ├── RESULTS.md
    └── figures/
        ├── v4_binary_deepdive_overview.png
        └── v4_binary_lopo_confusion_matrices.png
```
