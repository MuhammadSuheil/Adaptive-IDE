# Batch 4: Optimasi Model Cognitive Load

Empat notebook membandingkan representasi sensor, acuan relatif, deep learning tabular, dan model temporal. Semua menggunakan tiga dataset `good_acceptable` terbaru serta pembagian data yang sama. Setiap notebook mencakup training sampai evaluasi test dan penyimpanan model.

| Notebook | Fokus | Runtime disarankan |
|---|---|---|
| [exp01](exp01/EyeHeartCognitiveLoad.ipynb) | Model tabular, fitur turunan, seleksi fitur, balancing, ordinal/hierarchy, dan ensemble | CPU |
| [exp02](exp02/EyeHeartCognitiveLoad.ipynb) | Normalisasi, acuan sensor, dan ringkasan riwayat | CPU |
| [exp03](exp03/EyeHeartCognitiveLoad.ipynb) | MLP, residual MLP, TabM, dan Transformer tabular | GPU T4 atau lebih tinggi |
| [exp04](exp04/EyeHeartCognitiveLoad.ipynb) | GRU dan TCN kausal, dengan pembanding satu-window dan ringkasan riwayat | GPU T4 atau lebih tinggi |

Tidak ada urutan wajib antar-notebook. Exp04 tidak memerlukan hasil exp01–exp03. Anggota tim dapat menjalankan notebook berbeda secara paralel menggunakan folder run yang berbeda.

## Persiapan Google Drive

```text
MyDrive/Penelitian-Adaptive-IDE/
├── Dataset/
│   └── v2_session_phase/
│       ├── window_15s_step_5s/good_acceptable.csv
│       ├── window_25s_step_5s/good_acceptable.csv
│       └── window_45s_step_5s/good_acceptable.csv
└── models/
    └── batch-04/
        ├── exp01/EyeHeartCognitiveLoad.ipynb
        ├── exp02/EyeHeartCognitiveLoad.ipynb
        ├── exp03/EyeHeartCognitiveLoad.ipynb
        └── exp04/EyeHeartCognitiveLoad.ipynb
```

Yang perlu disalin hanya **tiga CSV dan empat notebook**. README dan PLAN boleh ikut. Tidak diperlukan folder shared, file `.py`, requirements, feature JSON, rekaman mentah, paket sensor, atau output batch sebelumnya. Dataset dibaca dari path Drive; tidak ada upload CSV pada setiap run.

Gunakan tiga CSV dari release yang sama, setelah pemeriksaan RR terbaru. Jumlah awal saat penyiapan:

| Window | Low | Medium | High | Total |
|---|---:|---:|---:|---:|
| 15 detik | 1.687 | 1.763 | 1.803 | 5.253 |
| 25 detik | 1.549 | 1.487 | 1.572 | 4.608 |
| 45 detik | 1.394 | 966 | 1.149 | 3.509 |

Jika dataset berubah, gunakan run baru. Distribusi dan hash yang ditampilkan notebook menjadi acuan run tersebut. Jangan menggunakan resume dari release lama.

Untuk Drive bersama, semua operator memerlukan izin Editor dan pintasan folder proyek di My Drive. Path tetap `/content/drive/MyDrive/Penelitian-Adaptive-IDE`. Jangan menggantinya dengan path internal `.shortcut-targets-by-id`.

## Menjalankan notebook

1. Buka notebook yang dipilih di Colab. Gunakan Python 3.12 atau lebih baru; environment penyiapan diuji pada Python 3.13.
2. Untuk exp03/exp04, pilih GPU melalui pengaturan runtime. CPU tetap didukung tetapi lebih lambat.
3. Biarkan `PROJECT_ROOT`, `DATASET_ROOT`, dan `RESUME_DIR = None` seperti yang disediakan, kecuali proyek berada di lokasi berbeda.
4. Pilih **Run all**. Tidak ada mode smoke/full.
5. Cell setup memasang dependency yang diperlukan. Jika dependency yang sudah dimuat berubah dan notebook meminta restart, restart runtime lalu jalankan kembali dari atas.
6. Izinkan akses Drive. Periksa distribusi data, tabel pembagian, hash split, dan path `Run:`.
7. Tunggu hingga hasil test serta status `complete` tersimpan. Simpan notebook setelah running agar tabel/gambar output cell ikut tersimpan.

Anggaran pencarian memang luas. Durasi bergantung runtime, keluarga model yang lolos, dan jumlah trial yang dipruning. Tidak ada batas waktu singkat yang dijanjikan. Pekerjaan tersimpan per fold/trial/seed; tidak perlu menyelesaikan satu notebook dalam satu sesi Colab.

## Cara evaluasi

Ini **evaluasi internal Train–validation–test**, bukan LOPO atau training bergantian per peserta. Orang yang sama dapat muncul pada ketiga bagian.

- Blok waktu 240 detik ditargetkan ke 70% Train, 15% validation akhir, dan 15% test.
- Satu dari 128 penempatan dipilih berdasarkan distribusi kelas/fase/sumber dan retensi data. Skor model serta nilai fitur tidak digunakan untuk memilih pembagian.
- Purge 105 detik mencegah overlap rentang sensor dan riwayat di antara bagian. Persentase sesudah purge tidak harus tepat 70/15/15.
- Di dalam Train ada lima fold bergantian sebagai validation internal. Shortlist diperiksa pada pengulangan lima fold dengan seed kedua. Ini bukan satu fold per peserta.
- Validation akhir membandingkan shortlist dan ensemble. Pemenang per-window serta model utama dikunci sebelum test dibuka.
- Model test adalah model pilihan yang sama, tanpa refit setelah seleksi.
- Dummy kelas mayoritas Train dilaporkan sebagai kontrol, bukan sebagai model utama sensor.

Urutan pemilihan adalah accuracy, Macro-F1, recall kelas terlemah, variasi fold, lalu kompleksitas input. Accuracy 75% merupakan sasaran. Nilai tinggi pada training atau validation saja tidak menetapkan bahwa sasaran test sudah tercapai.

Semua notebook harus menampilkan **hash split yang sama** ketika ketiga CSV sama. Perbedaan hash perlu diperiksa sebelum membandingkan hasil.

## Fitur dan riwayat

Input utama adalah 10 fitur core: RMSSD, SDNN, blink rate, mean/std EAR kiri/kanan, mean/std estimasi ukuran iris, dan mean perubahan iris. Core-head menambahkan mean/std pitch dan yaw bila seluruh nilainya tersedia. Jika head tidak lengkap, cabang core tetap digunakan; gaze yang gagal validasi tidak diimputasi.

Target langsung memakai `cognitive_load_label`. Identitas, fase, difficulty, soal, kualitas, dan timestamp absolut tidak menjadi predictor classifier. Identitas sesi boleh memilih acuan normalisasi training dan membatasi riwayat.

Riwayat 15/30/60 detik hanya membaca window sebelumnya pada sesi yang sama. Celah endpoint lebih dari 5 detik mereset urutan. Riwayat tersebut merupakan urutan window valid dalam CSV, bukan rekonstruksi seluruh rekaman mentah. Acuan median/statistik memerlukan minimal dua window sebelumnya; jika belum cukup, representasi saat ini menjadi fallback. GRU/TCN menerima urutan pendek tanpa membuang baris. Padding dimasking, bukan diperlakukan sebagai pengukuran nol.

Imputasi core tidak dilakukan. Nilai negatif setelah pemusatan tetap sah. Resampling hanya dilakukan pada fitting data; CSV sumber tidak bertambah dan validation/test tidak dioversampling.

## Membaca hasil

Output dibuat otomatis di `models/batch-04/outputs/<exp>/<run_id>/` pada Drive.

| Artefak | Isi |
|---|---|
| `run.json`, `split.json`, `split_counts.csv` | Identitas data, environment, implementasi, anggaran, dan pembagian data |
| `progress.json` | Tahap, window, kandidat, fold, trial, atau seed yang sedang dikerjakan |
| `validation_leaderboard.csv` | Hasil CV internal dan pemeriksaan pengulangan |
| `trials_*.csv` | Riwayat tuning Optuna, termasuk trial yang dipruning |
| `fold_*/` | Skor fold, prediksi validation internal, dan history neural bila ada |
| `cv_*/` | Gabungan prediksi out-of-fold dan variasi fold |
| `fit_*/` | Model yang dilatih pada seluruh Train serta schema fitur |
| `selection_leaderboard.csv`, `selection/` | Hasil validation akhir dan pilihan model yang dikunci |
| `seed_stability.csv` | Hasil validation menurut seed, tanpa memilih seed menggunakan test |
| `selection_*/` | Bundle model dan prediksi validation akhir |
| `test_*/` | Prediksi test, classification report, confusion matrix, dan kelompok fase/riwayat |
| `test_summary.csv`, `RESULTS.md`, `complete/` | Rekap test dan status selesai |

`primary` pada test summary menandai model utama yang dipilih lewat validation. `window_winner` menandai pemenang masing-masing window. Model kontrol tetap dicatat terpisah. Perbandingan utama tidak mengganti pemenang berdasarkan test tertinggi.

`progress.json` diperbarui pada batas pekerjaan, bukan setiap epoch. Neural mencetak epoch 1 dan setiap 20 epoch. Waktu tanpa perubahan status selama satu fitting bukan otomatis proses macet.

## Resume setelah runtime terputus

Salin path `Run:` dari eksekusi sebelumnya, misalnya:

```python
RESUME_DIR = "/content/drive/MyDrive/Penelitian-Adaptive-IDE/models/batch-04/outputs/exp03/<run_id>"
```

Ganti `<run_id>` dengan nama folder run yang sebenarnya, lalu jalankan notebook yang sama dari atas. Pastikan tipe runtime dan versi dependency kompatibel dengan run sebelumnya.

Fold/trial/seed lengkap dibaca kembali tanpa fitting ulang. Trial yang terputus menggunakan recipe yang sudah disimpan; fold lengkapnya tidak diulang. Fitting yang terhenti di tengah epoch dimulai kembali dari awal fold tersebut. Pemilihan model yang sudah dikunci juga dipertahankan.

Resume ditolak bila dataset, split, implementasi, anggaran, atau environment berubah. Gunakan `RESUME_DIR = None` untuk memulai run baru. Jangan menjalankan satu folder run dari dua runtime sekaligus.

Checkpoint diselesaikan di disk lokal Colab sebelum disalin ke Drive. Salinan diverifikasi dan completion marker ditulis terakhir. Folder dibuat ulang pada setiap retry. Checkpoint dengan checksum tidak cocok tidak dianggap selesai; jangan menghapus marker untuk menyembunyikan kerusakan.

## Model tersimpan

Cell terakhir memuat model utama dan memprediksi beberapa endpoint test tanpa training ulang. Muat hanya model terpercaya dengan cell definisi serta environment yang sama. Normalisasi dan schema tersimpan di bundle. XGBoost/CatBoost diserialisasi native di dalam bundle; network neural menyimpan bobot CPU dan konfigurasi arsitektur.

Bobot `.pkl` tetap diabaikan Git; pertahankan salinan Drive. CSV, JSON, Markdown, dan figure mengikuti aturan Git proyek. CSV prediksi memuat metadata penelitian, sehingga aksesnya perlu dibatasi sesuai kebijakan tim.
