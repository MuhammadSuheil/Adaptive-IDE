# Batch-04: Normalisasi Sensor dan Seleksi Model

| Notebook | Perlakuan | Runtime |
|---|---|---|
| [Exp01](exp01/EyeHeartCognitiveLoad.ipynb) | Model tabular, normalisasi, tuning, kontrol sensor, fitur relatif, class weight, dan ensemble | CPU |
| [Exp02](exp02/EyeHeartCognitiveLoad.ipynb) | MLP, residual MLP, Transformer kecil, normalisasi, regularisasi, fitur relatif, dan class weight | GPU T4 |

Masing-masing notebook memproses tiga window dan menyimpan hasilnya sendiri. Keduanya dapat dijalankan terpisah atau paralel pada run yang berbeda. [PLAN.md](PLAN.md) memuat parameter dan aturan seleksi. [DIAGNOSIS.md](DIAGNOSIS.md) membahas hasil run pertama.

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
        └── exp02/EyeHeartCognitiveLoad.ipynb
```

Salin kedua notebook terbaru dan tiga CSV terbaru. Yang dibutuhkan di Drive hanya file pada struktur tersebut; output akan dibuat oleh notebook. Untuk folder bersama, berikan izin Editor dan buat pintasan proyek pada My Drive akun yang menjalankan Colab. Gunakan path MyDrive di atas, bukan path internal `.shortcut-targets-by-id`.

## Menjalankan

1. Buka notebook pilihan di Colab. Gunakan GPU T4 untuk exp02.
2. Biarkan `PROJECT_ROOT` dan `DATASET_ROOT` bila struktur Drive sesuai contoh.
3. Biarkan `RESUME_DIR = None` untuk memulai run baru. Kedua notebook aktif memakai pipeline `batch04-2.0.0`.
4. Pilih **Run all** dan izinkan akses Drive.
5. Jika cell dependency meminta restart, pilih **Runtime > Restart session**, lalu jalankan kembali dari awal.
6. Catat path setelah `Run:`. Tunggu sampai tabel test, laporan, dan contoh prediksi model tersimpan selesai.
7. Simpan notebook agar output cell ikut tersimpan.

Tidak ada pengaturan smoke/full. Run biasa menjalankan evaluasi lengkap. Jangan melanjutkan folder run versi 1 dengan notebook revisi ini; pembagian development dan seleksi modelnya berbeda. Output versi 1 tetap disimpan untuk dokumentasi.

## Alur evaluasi

1. Pertahankan test dari evaluasi sebelumnya.
2. Bagi sisa data menjadi Train dan validation akhir.
3. Bandingkan konfigurasi pada lima fold di dalam Train. Setiap baris Train mendapat satu prediksi out-of-fold.
4. Latih kandidat terpilih pada seluruh Train, kemudian bandingkan bentuk model individual/ensemble pada validation akhir.
5. Kunci model per-window dan model utama, lalu evaluasi test dengan model yang sama.

Kelompok split merupakan blok waktu 120 detik dalam sesi. Ini bukan LOPO atau training satu model per peserta. Orang yang sama boleh muncul pada Train, validation, dan test. Purge 45 detik mencegah dukungan sensor yang overlap masuk ke bagian berbeda, termasuk antarwindow.

| Window | Baris sumber | Train | Validation akhir | Test | Purged |
|---|---:|---:|---:|---:|---:|
| 15 detik | 3.579 | 2.069 | 560 | 464 | 486 |
| 25 detik | 3.214 | 1.845 | 489 | 423 | 457 |
| 45 detik | 2.527 | 1.444 | 373 | 338 | 372 |

Angka ini berlaku untuk CSV saat revisi disiapkan. Tabel pada notebook dan manifest run menjadi acuan bila CSV berubah. Dataset dan test sudah digunakan pada eksperimen sebelumnya, sehingga hasil merupakan evaluasi internal eksploratif.

### Membaca tabel hasil

- `validation_leaderboard.csv`: accuracy gabungan prediksi out-of-fold, Macro-F1, mean/std/minimum accuracy fold, dan training accuracy. Tabel ini membandingkan konfigurasi di dalam Train.
- `selection_leaderboard.csv`: hasil model yang sudah dilatih pada seluruh Train, termasuk bentuk ensemble, pada validation akhir. Tabel ini menentukan pemenang.
- `test_summary.csv`: hasil test setelah pilihan dikunci. Kolom `primary` menandai window utama yang dipilih dari validation akhir, bukan window dengan skor test tertinggi.

Accuracy menjadi kriteria utama. Macro-F1 dan hasil per kelas tetap perlu diperiksa agar accuracy tidak menutupi recall yang rendah pada satu kelas. Ensemble tidak otomatis lebih baik dari model individual.

## Neural: pemantau epoch dan durasi training

Dalam setiap fold, exp02 mengambil bagian pemantau dari training. Accuracy pemantau, Macro-F1, lalu cross-entropy menentukan epoch terbaik. Validation penilaian fold tidak menentukan kapan training berhenti.

Setelah epoch dipilih, jaringan dilatih ulang pada seluruh training fold. Untuk kandidat akhir, durasi fitting seluruh Train memakai median epoch terbaik lima fold. Model seed 42 dibandingkan dengan rata-rata probabilitas seed 42/43/44.

Riwayat pemantauan tersimpan pada `monitor_history.csv`. Riwayat fitting ulang tersimpan pada `history.csv`. Loss training adalah objective yang dapat menggunakan class weight atau smoothing; loss pemantau adalah cross-entropy tanpa keduanya. Karena itu nilainya tidak selalu dapat dibandingkan langsung sebagai selisih loss.

Exp01 memiliki maksimum 51 konfigurasi per window, termasuk Dummy. Exp02 memiliki maksimum 30 konfigurasi neural per window, ditambah Dummy. Konfigurasi identik tidak diulang. Neural menjalankan tahap pemantauan dan fitting ulang pada setiap fold, sehingga waktu training lebih panjang daripada satu fitting per konfigurasi. Gunakan resume bila runtime terputus.

## Normalisasi dan prediksi

Classifier menerima fitur sensor saja. `session_id` memilih acuan normalisasi training, bukan menjadi predictor numerik atau kategori. Sesi yang belum mempunyai acuan memakai statistik global training. Notebook tidak menghitung ulang acuan dari test atau seluruh sesi baru.

Fitur relatif dihitung saat preprocessing, sehingga CSV tidak perlu ditambah kolom. Schema model menyimpan fitur sumber yang diperlukan dan urutan input setiap komponen. Nilai negatif setelah pemusatan merupakan deviasi yang valid.

SVM memakai keputusan kelas langsung. `decision_score` bukan probabilitas. `vote_fraction` adalah proporsi suara. Neural dan model yang mendukung `predict_proba` menghasilkan probabilitas; hanya skor jenis ini yang boleh dirata-ratakan.

## Output

Setiap notebook membuat folder run baru:

```text
models/batch-04/outputs/exp01/<run_id>/
models/batch-04/outputs/exp02/<run_id>/
```

| Artefak | Isi |
|---|---|
| `run.json`, `run_config.json` | Hash dataset, versi kode, parameter dasar, dan environment |
| `artifact_io.json` | Revisi penulisan checkpoint; terpisah dari identitas training agar perbaikan I/O tetap dapat melanjutkan run yang kompatibel |
| `split.json`, `split_counts.csv` | Pembagian luar, lima fold internal, dan pemantau neural |
| `progress.json` | Status terakhir, kandidat, window, fold atau seed, dan waktu UTC |
| `validation_leaderboard.csv`, `validation_folds.csv` | Agregat out-of-fold serta hasil tiap fold |
| `w.../oof_predictions.csv` | Prediksi gabungan out-of-fold suatu konfigurasi |
| `w..._f0/` sampai `w..._f4/` | Model, prediksi validation internal, dan riwayat neural bila ada |
| `normalization_comparison.csv`, figure | Perbandingan normalisasi pada screening |
| `fit_w..._s42/`, `s43/`, `s44/` | Fitting kandidat pada seluruh Train; seed 43/44 hanya neural |
| `selection_w.../` | Model yang dinilai pada validation akhir beserta prediksinya |
| `selection_leaderboard.csv`, `selection/` | Perbandingan akhir dan pilihan model yang dikunci |
| `final_selection_w.../` | Model identik dengan hasil seleksi, prediksi test, classification report, confusion matrix, dan schema |
| `test_summary.csv`, `RESULTS.md`, `complete/` | Hasil test dan penanda penyelesaian |

`progress.json` diperbarui sebelum training dan setelah fold selesai, bukan setiap epoch. Output epoch tampil pada notebook neural. Status yang belum berubah selama fitting bukan bukti bahwa proses berhenti.

XGBoost memakai serialisasi native JSON dan CatBoost memakai CBM, disertai bundle preprocessing. Muat hanya model terpercaya dengan environment yang sesuai `run.json`. Simpan bobot model di Drive; CSV, metadata, laporan, dan figure dapat masuk Git.

## Resume

Ubah `RESUME_DIR` menjadi path lengkap run versi 2 yang terputus:

```python
RESUME_DIR = "/content/drive/MyDrive/Penelitian-Adaptive-IDE/models/batch-04/outputs/exp02/<run_id>"
```

Gunakan `exp01` untuk notebook exp01. Jalankan kembali dari awal. Bagian yang sudah lengkap dan lolos checksum dilewati; fold atau seed yang belum selesai diulang. Resume tidak memulihkan posisi di tengah epoch.

Gunakan `RESUME_DIR = None` bila dataset, implementasi, split, atau environment berubah. Jangan menjalankan satu folder run dari dua runtime bersamaan. Bila penulisan Drive gagal, periksa izin dan koneksi, lalu resume folder yang sama. Jangan menghapus completion marker untuk mengabaikan kerusakan artefak.

Writer checkpoint revisi 2 membuat ulang folder pada setiap retry dan memverifikasi salinan hingga lima kali, dengan jeda 0,5/1/2/4 detik. Cache lokal menyimpan seluruh artefak beserta marker terverifikasi sebelum unggahan dimulai. Marker Drive baru ditulis setelah seluruh salinan berhasil diverifikasi. Bila unggahan gagal dan runtime masih hidup, cache lengkap dapat dipakai kembali tanpa fitting ulang. Bila runtime sudah berakhir, hanya checkpoint lengkap yang tersimpan di Drive yang dapat dilanjutkan tanpa fitting ulang.

Perbaikan writer tidak mengubah parameter model, dataset, atau split. Run `batch04-2.0.0` sebelum revisi writer tetap dapat di-resume; identitas training pada `run.json` dipertahankan, sedangkan `artifact_io.json` mencatat revisi penyimpanan. Run versi 1 tetap tidak kompatibel dengan pipeline versi 2.

## Arsip dan validasi

Notebook yang telah dijalankan dengan pipeline versi 1 disimpan di folder run lamanya sebagai `notebook_executed_v1.ipynb`. Hasil run tersebut tidak ditimpa oleh revisi ini.

Validasi lokal mencakup syntax, struktur notebook, audit split pada CSV terbaru tanpa training penelitian, dan tes sintetis kecil untuk preprocessing, model, seleksi, serialisasi, serta resume. Hasil penelitian versi 2 baru tersedia setelah notebook dijalankan di Colab.
