# Batch-03: Model Tabular dan Deep Learning

Batch ini memakai dua notebook mandiri. Masing-masing membaca tiga dataset dari Drive, menjalankan training, memilih kandidat menggunakan validation, mengevaluasi test, dan menyimpan model. Tidak ada notebook khusus rekap atau persiapan.

| Notebook | Isi | Runtime yang disarankan |
|---|---|---|
| [exp01](exp01/EyeHeartCognitiveLoad.ipynb) | Delapan algoritma tabular, sensor ablation, tuning, seleksi fitur, balancing, interaksi, hierarchy, soft voting, dan late fusion | CPU |
| [exp02](exp02/EyeHeartCognitiveLoad.ipynb) | MLP, MLP residual, Transformer tabular, dua preset, eye/fusion, pengulangan seed, dan ensemble neural | GPU T4 |

Exp01 tidak membutuhkan exp02, dan sebaliknya. Keduanya boleh dijalankan dengan akun berbeda secara paralel. Perbandingan kedua notebook dilakukan setelah hasil tersedia, bukan melalui notebook ketiga.

## Hasil eksekusi

Tim sudah menjalankan kedua notebook di Google Colab. [SUMMARY.md](SUMMARY.md) merangkum enam finalist, model utama pilihan validation, skor test tertinggi yang teramati, hasil per kelas, dan analisis gap training–validation–test. Laporan juga mencocokkan metrik dengan prediksi yang tersimpan.

## Persiapan Google Drive

```text
MyDrive/Penelitian-Adaptive-IDE/
├── Dataset/
│   └── v2_session_phase/
│       ├── window_15s_step_5s/good_acceptable.csv
│       ├── window_25s_step_5s/good_acceptable.csv
│       └── window_45s_step_5s/good_acceptable.csv
└── models/
    └── batch-03/
        ├── exp01/EyeHeartCognitiveLoad.ipynb
        └── exp02/EyeHeartCognitiveLoad.ipynb
```

Hanya tiga CSV dan dua notebook yang diperlukan. README boleh ikut. Tidak perlu mengunggah `shared`, `.py`, file requirements, feature JSON, folder sesi, paket sensor, output batch lama, atau file pengujian repositori. Tidak ada upload CSV dari cell notebook.

Untuk folder bersama, setiap anggota memerlukan izin Editor dan pintasan folder proyek di My Drive. Path tetap `/content/drive/MyDrive/Penelitian-Adaptive-IDE`; jangan mengganti path menjadi ID internal pintasan. Ketiga CSV harus berasal dari salinan dataset terbaru yang sama.

## Menjalankan

1. Buka salah satu notebook di Colab. Gunakan Python 3.12 atau lebih baru. Untuk exp02 pilih GPU T4 jika tersedia.
2. Biarkan `PROJECT_ROOT`, `DATASET_ROOT`, dan `RESUME_DIR = None` seperti yang disediakan, kecuali lokasi proyek berbeda.
3. Pilih **Run all** pada notebook tersebut. Ini menjalankan satu eksperimen, bukan launcher untuk semua notebook.
4. Jika cell Environment memperbarui dependency dan meminta restart, pilih **Runtime > Restart session**, lalu jalankan kembali dari atas. PyTorch/CUDA bawaan tidak dipasang ulang.
5. Izinkan akses Drive. Perhatikan tabel distribusi split dan path `Run:` yang dicetak.
6. Tunggu sampai bagian evaluasi test selesai. Simpan notebook setelah running agar output cell ikut tersimpan.

Tidak ada mode smoke/full. Notebook memproses tiga window secara otomatis. Exp01 melakukan penyaringan dahulu sebelum optimasi dua keluarga terbaik per-window, bukan mencari seluruh kombinasi. Exp02 menjalankan 36 konfigurasi awal, enam pengulangan seed, lalu refit finalist. Early stopping dapat menghentikan neural sebelum batas 100 epoch. Durasi bergantung runtime Colab; tidak ada jaminan waktu tetap.

## Cara split dan pemilihan model

- Target langsung memakai `cognitive_load_label`, bukan memetakan ulang difficulty. Metadata peserta, soal, fase, waktu, dan kualitas tidak menjadi fitur prediksi.
- Blok waktu 120 detik per sesi dibagi mendekati 70% training, 15% validation, dan 15% test, dengan seed 42 dan pengelompokan kelas dominan blok.
- Data orang yang sama boleh muncul pada ketiga bagian. Ini evaluasi internal dengan blok waktu campuran, bukan LOPO atau TimeSeriesSplit.
- Rentang pendukung 45 detik dipakai pada semua window. Test dipertahankan; validation yang overlap test dibuang; training yang overlap validation/test dibuang. Ini juga mencegah overlap lintas pilihan window. Persentase baris sesudah purge dapat berbeda dari persentase blok.
- Imputasi, transformasi, seleksi fitur, dan resampling hanya dipelajari dari training selama pencarian kandidat. CSV sumber tidak berubah. Tiga window tidak ditumpuk menjadi satu dataset.
- Accuracy validation menjadi kriteria pemilihan utama, diikuti Macro-F1 dan kompleksitas. Finalist tiap window serta pemenang utama dikunci sebelum evaluasi test.
- Finalist di-fit ulang pada training + validation, kemudian diuji pada test. Untuk neural, epoch refit sudah ditentukan dari checkpoint terbaik sebelumnya. Model yang disimpan adalah model yang diuji tersebut; test tidak ikut fitting.

Sebagai pemeriksaan dataset saat penyiapan, jumlah baris setelah purge adalah:

| Window | Training | Validation | Test | Dibuang karena overlap |
|---|---:|---:|---:|---:|
| 15 detik | 2131 | 550 | 464 | 434 |
| 25 detik | 1878 | 510 | 423 | 403 |
| 45 detik | 1424 | 435 | 338 | 330 |

Ketiga kelas tersedia pada setiap bagian. Angka ini dapat berubah jika CSV diperbarui. Gunakan tabel yang dicetak notebook sebagai sumber untuk run baru; split tidak dipilih ulang berdasarkan skor.

## Hasil yang disimpan

```text
models/batch-03/outputs/exp01/<run_id>/
models/batch-03/outputs/exp02/<run_id>/
```

Setiap run menyimpan identitas data/environment, split, distribusi kelas, proporsi fitur kosong, leaderboard validation, parameter kandidat, dan checkpoint kandidat selesai. Pada exp02, riwayat loss/accuracy serta grafik learning curve ikut disimpan.

Bagian akhir menghasilkan:

- `selection/`: kandidat yang dikunci menggunakan validation, sebelum penilaian test.
- `final_<candidate_id>/`: model evaluasi, prediksi test, classification report, confusion matrix, dan schema fitur.
- `test_summary.csv`: hasil test finalist tiap window, dengan penanda model utama.
- `RESULTS.md`: hasil model utama dan status target accuracy 75%.
- `complete/`: penanda bahwa semua evaluasi akhir berhasil tersimpan.

Tabel selama pencarian adalah **validation**, bukan test. Tabel test mempunyai accuracy, Macro-F1, balanced accuracy, dan jumlah baris. Precision, recall, F1 per kelas, serta confusion matrix ditampilkan langsung. Model utama tetap yang dipilih lewat validation, bukan diganti berdasarkan angka test tertinggi.

Target accuracy 75% adalah sasaran eksperimen, bukan hasil yang dijamin. Jika belum tercapai, notebook tetap menyimpan hasil sebenarnya. Bobot model diabaikan Git mengikuti aturan folder models; CSV, JSON, Markdown, dan figure tetap dapat dibagikan. CSV prediksi berisi metadata peserta, sehingga aksesnya tetap mengikuti ketentuan penelitian.

## Melanjutkan jika runtime terputus

Salin path setelah teks `Run:` dari eksekusi sebelumnya, misalnya:

```python
RESUME_DIR = "/content/drive/MyDrive/Penelitian-Adaptive-IDE/models/batch-03/outputs/exp01/<run_id>"
```

Ganti `<run_id>` dengan nama folder run sebenarnya. Untuk exp02 gunakan folder exp02. Jalankan kembali notebook dari atas.

Kandidat yang telah lengkap dan checksum-nya cocok dibaca kembali tanpa training ulang. Kandidat yang terputus sebelum selesai diulang dari awal; resume di tengah epoch tidak dijanjikan. Refit/evaluasi akhir yang sudah selesai juga tidak diulang. Jangan memakai satu run yang sama dari dua runtime bersamaan.

Resume ditolak jika CSV, split, implementasi, versi Python, atau dependency berbeda. Jika menggunakan dataset baru atau mengubah pipeline, gunakan `RESUME_DIR = None` untuk membuat run baru. Jangan menyalin checkpoint dari run lama untuk melewati pemeriksaan.

File kerja dibuat di disk sementara Colab. Artefak disalin ke Drive dan diverifikasi sebelum ditandai selesai, tanpa mengandalkan atomic rename pada pintasan Drive. Jika penyalinan gagal, notebook berhenti; perbaiki koneksi/izin lalu resume. Jika checkpoint terdeteksi rusak, gunakan run baru atau pulihkan salinan lengkap yang benar, jangan menghapus penanda kegagalan agar run tampak selesai.

## Memakai model yang tersimpan

Cell terakhir menunjukkan cara memuat model utama dan memprediksi ulang beberapa baris tanpa training. Jalankan cell definisi pipeline pada notebook yang sesuai sebelum memuat model. Semua definisi sudah ada dalam notebook; tidak perlu file Python pendamping. Gunakan urutan fitur/schema yang disimpan. Muat hanya file model terpercaya yang dibuat tim.

Pada penyiapan awal, notebook telah diperiksa dengan syntax/schema validation, audit split pada CSV, dan pengujian ringan menggunakan data sintetis. Tim kemudian menjalankan training penuh di Colab; hasil penelitian berada pada folder outputs dan dirangkum pada SUMMARY.md. Rekap hasil tidak melatih ulang model atau mengubah dataset sumber.
