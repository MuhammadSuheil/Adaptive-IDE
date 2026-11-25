# Batch 06 Exp02

Exp02 memisahkan pengaruh normalisasi RMSSD dan eye pada empat peserta, baris, baseline, serta fold subset eye Exp01 yang sama. [Run pertama selesai](SUMMARY.md): gabungan tertinggi pada OOF peserta dengan accuracy 41,43%, RMSSD-only 38,15%, dan eye-only 31,74%. Analisis memakai output notebook; ZIP mentah Exp02 belum disertakan. Model tetap Extra Trees 200 trees dengan parameter tetap.

## Jalankan di Colab

1. Salin [EyeHeartCognitiveLoad.ipynb](EyeHeartCognitiveLoad.ipynb) ke `MyDrive/Penelitian-Adaptive-IDE/models/batch-06/exp02/`.
2. Gunakan tiga CSV pada root Drive yang sama seperti Exp01.
3. Buka notebook di Google Colab dan pilih **Run all**. Mount Drive saat diminta. CPU cukup.
4. Hasil otomatis tersimpan di `models/batch-06/outputs/exp02/ablation_<waktu>_<id>/`. Baca `RESULTS.md`, tabel akhir, dan grafik.

```text
MyDrive/Penelitian-Adaptive-IDE/
  Dataset/v2_session_phase/
    window_15s_step_5s/good_acceptable.csv
    window_25s_step_5s/good_acceptable.csv
    window_45s_step_5s/good_acceptable.csv
  models/batch-06/
    exp02/EyeHeartCognitiveLoad.ipynb
    outputs/exp02/                  # dibuat otomatis
```

**Cukup salin notebook.** Tidak perlu menjalankan ulang Exp01 atau menyalin file Python, snapshot JSON, output Exp01, maupun rekaman mentah ke Drive. Semua definisi serta snapshot sudah tertanam. Ubah `PROJECT_ROOT` hanya jika lokasi/nama proyek berbeda.

Hash tiga CSV harus sama dengan run Exp01 yang dianalisis. Jika data berubah, notebook berhenti dengan pesan mismatch; jangan mengganti split atau nilai hash hanya untuk meneruskan fitting. Ablasi pada data baru memerlukan persiapan snapshot baru agar perbandingan tetap dapat ditelusuri.

## Empat varian

| Varian | RMSSD | Eye | SDNN |
|---|---|---|---|
| `raw` | Asli | Asli | Asli |
| `rmssd_only` | Relatif terhadap baseline recorder | Asli | Asli |
| `eye_only` | Asli | Relatif terhadap acuan pre_task awal | Asli |
| `combined` | Relatif terhadap baseline recorder | Relatif terhadap acuan pre_task awal | Asli |

Kata only menunjukkan **normalisasi yang diaktifkan**, bukan membuang modalitas lain. Seluruh varian menerima sepuluh fitur core. Identitas bukan predictor. Baseline, transformasi eye, dan urutan fitting mengikuti Exp01; tidak ada acuan yang dipilih ulang berdasarkan skor. SDNN tetap mentah karena baseline independennya tidak tersedia pada CSV.

## Data dan evaluasi tetap

Sumber: `baseline_20261123T192338Z_8d8cee5e` pada Exp01. Train mempunyai 671 window dari Baithar, Fabian, Mon, dan kaka: 264 Low, 180 Medium, 227 High. Empat fold peserta menahan seluruh data satu orang; lima fold waktu memakai guard 45 detik. Seluruh varian memakai daftar fit/held-out yang persis sama.

Validation historis mempunyai 99 window: **43 Low, nol Medium, dan 56 High**. Bagian ini sudah pernah diperiksa dan bukan hold-out baru. Split tidak diubah untuk mengejar skor. Recall Medium ditampilkan **N/A** dan support nol; Macro-F1 tetap memakai tiga label untuk konsistensi dengan Exp01. Accuracy validation bukan evaluasi tiga kelas lengkap ataupun pengguna baru.

Test tidak dievaluasi atau masuk fitting. Window acuan awal dan interval yang bertabrakan tetap dipisahkan dari endpoint supervised. Scaler dihitung dari fitting masing-masing fold.

## Budget dan pembacaan hasil

Budget tetap **45 fitting CV + empat fitting Train penuh**. Empat representasi Extra Trees dan satu Dummy dievaluasi pada sembilan fold. Semua empat varian dikunci untuk validation sebelum endpoint itu dievaluasi; tidak ada pemilihan pemenang menurut validation, tuning, ensemble, atau refit Train + validation.

Kontras dibakukan sebelum fitting:

- `rmssd_only vs raw`: manfaat transformasi RMSSD ketika eye tetap asli.
- `eye_only vs raw`: manfaat transformasi eye ketika RMSSD tetap asli.
- `combined vs rmssd_only`: manfaat tambahan transformasi eye ketika RMSSD sudah relatif.
- `combined vs eye_only`: manfaat tambahan transformasi RMSSD ketika eye sudah relatif.
- `combined vs raw`: perubahan gabungan sebagai pembanding Exp01.

Laporkan accuracy/Macro-F1 OOF bersama rata-rata per orang dan support kelas. Perbedaan antar-kontras menunjukkan ketergantungan antarrepresentasi, bukan bukti kausal fisiologis. Empat peserta membatasi kesimpulan; window overlap bukan observasi independen dan belum ada pengukuran beban mental independen.

Raw dan combined dijalankan ulang sebagai kontrol. `exp01_control_checks.csv` membandingkannya dengan metrik Exp01. Jika versi paket/numerik menghasilkan perbedaan, notebook memberi peringatan dan tetap menghitung semua kontras dari hasil Exp02 yang baru. Tidak ada pencarian seed atau penggantian parameter untuk memaksakan kecocokan.

## Output

- `plan.json`, `cohort_snapshot.json`: hash, baseline, peserta, window/fold, model, dan budget.
- `cv_leaderboard.csv`, `fold_metrics.csv`, `oof_predictions.csv`: hasil dua protokol.
- `paired_contrasts.csv`, `participant_metrics.csv`: perbandingan transformasi serta bobot per orang.
- `comparison_lock.json`: empat kontrol dibakukan sebelum validation; bukan pilihan model dari validation.
- `validation_metrics.csv`, `validation_predictions.csv`: hasil dengan support kelas dan recall N/A bila tidak tersedia.
- `exp01_control_checks.csv`: kecocokan kontrol raw/combined dengan Exp01.
- `models/*.joblib`: empat bundle pembanding; bukan satu pemenang resmi berdasarkan validation.
- `RESULTS.md`, `figures/`, `status.json`: laporan, grafik, dan status akhir.

Setiap Run all membuat folder baru. Tidak ada overwrite atau resume otomatis; hasil parsial dipertahankan bila runtime terputus. Bobot mengikuti kebijakan ignore repositori, sedangkan metadata, prediksi, dan grafik dapat masuk Git.

## Pemeriksaan lokal

[preparation/readiness.json](preparation/readiness.json) mencatat preflight tanpa training penelitian. [cohort_snapshot.json](cohort_snapshot.json) adalah sumber snapshot yang juga tertanam di notebook, bukan file wajib Colab.

Untuk pemeliharaan dari root repositori:

```powershell
python models/batch-06/build_ablation_notebook.py
python -m unittest discover -s tests -p 'test_batch06*.py' -v
```

Generator memakai arsip lokal Exp01 saat membangun notebook. Saat notebook dijalankan di Colab, arsip tersebut tidak diperlukan.
