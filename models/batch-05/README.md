# Batch 05

[Ringkasan batch-05](SUMMARY.md) memuat hasil tiga tahap dan audit penutup yang sudah selesai. [Diagnosis](DIAGNOSIS.md) menjelaskan bukti sensor, ledger tugas, dan kontrol fitur. Tidak perlu menjalankan notebook lagi untuk menyelesaikan batch ini. Petunjuk di bawah berlaku jika ingin mereproduksi atau membuat run baru; output lama dipertahankan.

## Tahap 3: neural terarah

Notebook: [exp01/EyeHeartCognitiveLoad.ipynb](exp01/EyeHeartCognitiveLoad.ipynb).

1. Simpan tiga CSV terbaru di Drive sesuai struktur pada [PLAN.md](PLAN.md#dataset-dan-fitur).
2. Salin notebook ke `MyDrive/Penelitian-Adaptive-IDE/models/batch-05/exp01/`.
3. Buka di Colab dan Run all. MLP memakai GPU CUDA jika tersedia, atau CPU jika tidak. Kontrol Extra Trees tetap menggunakan CPU.
4. Baca tabel akhir dan `RESULTS.md` pada folder run `stage3_...`. Grafik tersimpan pada `figures` dan tampil di notebook ketika menggunakan Colab.

Path proyek default adalah `/content/drive/MyDrive/Penelitian-Adaptive-IDE`. Jika memakai shortcut Drive, pastikan akun memiliki izin Editor dan shortcut berada di My Drive dengan nama tersebut. Ubah `PROJECT_ROOT` bila nama/lokasi proyek berbeda.

Yang perlu disalin hanya notebook dan tiga CSV. Manifest referensi/development serta monitor berada di notebook. Run all memverifikasi snapshot, menjalankan tahap 3 pada window 25 detik, lalu menyimpan evaluasi. Tidak perlu output tahap 1/2, folder shared, atau file Python pendamping. Tidak ada pilihan smoke/full.

Tahap 3 membandingkan satu MLP, tiga normalisasi, tiga aturan checkpoint, dan tiga seed. Probabilitas seed digabung tanpa memilih seed terbaik. Extra Trees menjadi kontrol matched-data dan baseline penuh. Batasnya 48 lintasan neural ditambah 24 fitting tabular, tanpa search tambahan. Run all tidak mengulang 71 fitting tahap 2.

Pemenang dipilih melalui OOF sebelum membaca validation utama. Model serta bobot dikunci sebelum satu evaluasi Test referensi. Test ini sudah pernah dianalisis; hasilnya tidak diperlakukan sebagai pengujian independen baru. Skor validation/Test tidak mengganti pemenang.

## Melanjutkan run terputus

Biarkan `RESUME_DIR = None` untuk run baru. Untuk melanjutkan, isi dengan path lengkap folder run yang terputus, misalnya:

```python
RESUME_DIR = PROJECT_ROOT / 'models/batch-05/outputs/exp01/stage3_<nama-run>'
```

Gunakan nama folder yang benar-benar ada di Drive. Jalankan ulang dari awal notebook; pekerjaan selesai dilewati. Neural yang terputus melanjutkan checkpoint epoch tersimpan, maksimal kehilangan empat epoch yang belum disimpan. Resume memerlukan data, manifest, monitor, kode, versi Python/pustaka, dan device yang kompatibel. Jangan menjalankan dua runtime pada satu run. Run CPU tidak dilanjutkan sebagai run GPU.

Jika hash atau lingkungan tidak cocok, notebook berhenti dengan pesan yang sesuai. Jangan menonaktifkan pemeriksaan; gunakan run baru jika lingkungan berubah.

## Isi output

- `cv_leaderboard.csv`: seluruh kandidat berdasarkan OOF.
- `seed_fold_metrics.csv`: hasil neural tiap seed, fold, dan aturan checkpoint.
- `checkpoint_choices.csv`: epoch dan metrik monitor checkpoint terpilih.
- `history.csv`: loss serta metrik fitting/monitor CV setiap epoch.
- `validation_metrics.csv`: hasil kandidat final yang telah ditetapkan.
- `test_reference.json`: metrik Test untuk model terkunci.
- `error_breakdown.csv`: rincian validation/Test menurut kelas, fase, dan dukungan sumber.
- `model_lock.json`: kandidat terpilih dan checksum bobot/preprocessing.
- `progress.json` dan `status.json`: kemajuan serta hasil akhir run.
- `RESULTS.md` dan `figures`: ringkasan serta grafik.

Folder `jobs` menyimpan checkpoint terpilih, prediksi, statistik preprocessing, dan dukungan fitting. `partial` menyimpan dua slot resume neural. File `.pt` dan `.joblib` mengikuti aturan ignore model terlatih; metadata, CSV, grafik, dan laporan tetap dapat dipush. Untuk melanjutkan run di Drive, bobot/checkpoint run tersebut harus tetap ada di Drive, bukan hanya metadata yang diambil dari Git.

Kolom `policy` kosong pada kontrol tabular karena Extra Trees tidak memilih checkpoint epoch. Itu bukan label cognitive load yang hilang.

## Arsip tahap 1 dan 2

Folder output tanpa prefix `stage2_` atau `stage3_` merupakan persiapan tanpa training classifier. Folder `stage2_` berisi training tabular, kontrol pembagian, dan learning curve yang sudah selesai sebelumnya. Hasil tersebut tetap menjadi arsip; notebook versi tahap 3 langsung menjalankan tahap 3.

[Rencana tiga tahap](PLAN.md) menjelaskan kriteria dan batas evaluasinya.
