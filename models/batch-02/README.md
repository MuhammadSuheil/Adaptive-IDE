# Batch-02: Google Colab

Batch-02 dimulai dari awal memakai tiga dataset terbaru setelah penggantian selektif Mon dan Alif. Hasil, snapshot, status, dan persetujuan training lokal lama tidak digunakan. Ada 14 notebook eksperimen (exp01–exp14) dan satu notebook persiapan (exp00). Tujuan dan tahapan tiap eksperimen dijelaskan pada notebook masing-masing.

## Rekap hasil

Hasil ekspor exp01–exp13, perbandingan kandidat, grafik, serta penilaian kualitas model tersedia di [SUMMARY.md](SUMMARY.md). Rekap membedakan eksperimen yang menghasilkan prediksi, TabPFN yang dilewati pada exp12, dan exp14 yang tidak diselesaikan. Kedua bagian unduhan Drive tetap tersimpan pada folder outputs aslinya.

## Melanjutkan setelah perbaikan penyimpanan exp03

Jika exp00–exp02 sudah selesai di Colab, **jangan hapus inputs, registry, atau outputs yang sudah dibuat di Drive**. Lakukan pembaruan berikut:

1. Hentikan notebook yang sedang berjalan sebelum memperbarui kode shared bersama.
2. Perbarui seluruh folder `shared` di Drive, termasuk `shared/colab_bootstrap.py` dan isi `shared/batch02`. Perbarui juga notebook exp03–exp14 dan README ini. Dataset, paket sensor, dan requirements tidak berubah pada perbaikan ini.
3. Gunakan runtime baru, lalu jalankan exp03 terbaru dari atas dengan `RESUME_DIR=None`. Run exp03 yang gagal sebelumnya tetap disimpan, tetapi tidak dilanjutkan memakai kode baru. Hasil exp01–exp02 tidak perlu diulang.
4. Lanjutkan notebook berikutnya. Jangan membawa salinan notebook lama yang masih menulis semua file training langsung ke Drive.

## Folder bersama dan akun anggota tim

Anggota tim boleh memakai akun berbeda. Bagikan folder proyek dengan izin **Editor**, buat pintasan folder `Penelitian-Adaptive-IDE` di My Drive masing-masing, dan mount Drive dengan akun yang memiliki akses tersebut. Membagikan notebook saja tidak otomatis memberikan izin menulis pada folder hasil.

Path tetap `/content/drive/MyDrive/Penelitian-Adaptive-IDE`. Path hasil resolve dapat menampilkan `.shortcut-targets-by-id`; jangan menggantinya dengan ID folder yang di-hard-code. Exp00 harus selesai sebelum anggota lain memulai training. Setelah itu akun A boleh menjalankan exp03 bersamaan dengan akun B yang menjalankan exp04. Satu eksperimen/run hanya dikerjakan satu runtime; ini bukan pembagian fold satu eksperimen ke banyak akun. Jangan memperbarui input, registry, requirements, atau shared selama eksperimen aktif.

## Folder yang disalin ke Drive satu kali

```text
MyDrive/Penelitian-Adaptive-IDE/
├── Dataset/
│   └── v2_session_phase/                 salinan output dataset terbaru
│       ├── manifest.json
│       ├── replacement_selection.json
│       ├── replacement_pairs.csv
│       ├── window_15s_step_5s/
│       │   ├── good_acceptable.csv
│       │   └── feature_columns.json
│       ├── window_25s_step_5s/
│       │   ├── good_acceptable.csv
│       │   └── feature_columns.json
│       └── window_45s_step_5s/
│           ├── good_acceptable.csv
│           └── feature_columns.json
└── models/
    └── batch-02/
        ├── exp00/EyeHeartCognitiveLoad.ipynb
        ├── exp01/EyeHeartCognitiveLoad.ipynb
        ├── ...
        ├── exp14/EyeHeartCognitiveLoad.ipynb
        ├── shared/
        │   ├── colab_bootstrap.py
        │   └── batch02/                  seluruh isi folder kode
        ├── requirements.txt
        ├── sensor_streams/               seluruh isi paket pendamping
        │   ├── window_15s_step_5s.csv
        │   ├── window_25s_step_5s.csv
        │   ├── window_45s_step_5s.csv
        │   ├── stream_manifest.json
        │   └── verification.json
        └── README.md
```

Untuk persiapan pertama, salin folder lokal `analysis_sessions/dataset_fusion/output/v2_session_phase` ke `Dataset/v2_session_phase`. Pada batch-02, salin exp00–exp14, shared, requirements.txt, sensor_streams, dan README pada struktur di atas. File laporan dan view dataset lain boleh ikut; training utama hanya memakai tiga good_acceptable. Jangan mengunggah inputs, registry, atau hasil training lokal lama. Jika release Colab sudah dibuat melalui exp00, pertahankan hasilnya seperti petunjuk pembaruan di atas. PLAN.md dan folder tests tidak diperlukan di Drive.

Tidak ada `files.upload()` atau upload tiga CSV pada setiap run. Notebook membaca path Drive. Jangan menambahkan tingkat folder ganda seperti `batch-02/batch-02` atau `v2_session_phase/v2_session_phase`.

`inputs`, `registry`, dan `outputs` dibuat otomatis oleh exp00/notebook. Jangan menyalin snapshot input, registry release, atau hasil dari eksekusi lokal lama. Folder `.venv`, source sensor, `sessions`, dan seluruh repositori tidak diperlukan di Drive.

## Apa itu sensor_streams?

Paket ini berisi fitur seluruh grid sensor sebelum pemilihan target, termasuk endpoint tanpa label tunggal. History dan GRU membentuk konteks dari aliran tersebut, bukan rolling atas CSV curated yang waktunya bisa terputus. Paket tidak menambah contoh training atau membuat label baru.

Input telah direkonstruksi dari rekaman terbaru, termasuk Mon dan Alif, dengan rumus kanonis. Nilainya dicocokkan dengan seluruh row curated. Pada interval timestamp eye ambigu, fitur eye konteks dikosongkan; HRV independen tetap dipertahankan. Frame dan timestamp mentah tidak diubah.

Paket terikat pada hash manifest dataset. Exp00 menolak pasangan dataset/paket yang berbeda. Setelah dataset diperbarui lagi, paket sensor dan persetujuan snapshot perlu diperbarui; jangan mengganti path untuk meneruskan run lama.

## Pemeriksaan sebelum Run All

- Pastikan ketiga `good_acceptable.csv` beserta `feature_columns.json`, manifest, dan dua file seleksi berasal dari salinan dataset terbaru yang sama; jangan hanya mengganti CSV-nya.
- Pastikan kelima file `sensor_streams` ikut disalin. Paket ini dibutuhkan meskipun tiga CSV training sudah ada; folder `inputs` lokal tidak perlu diunggah sebagai penggantinya.
- Pada exp00, hash persetujuan sudah terisi untuk manifest terbaru: `5bc566a20c7aa67388da4cf19a0587791af94ba650c33eac785a6ea9989e4d52`. Biarkan nilai tersebut untuk dataset saat ini.
- Biarkan konfigurasi awal notebook seperti yang disediakan, termasuk `INSTALL_DEPENDENCIES=True` dan `RESUME_DIR=None`. Tidak perlu mode smoke/full.

## Menjalankan notebook

1. Buka exp00 melalui Google Colab. Pilih runtime versi **2026.10** bila tersedia (Python 3.13), atau runtime Python minimal 3.12. GPU T4 cukup sebagai pilihan awal untuk cabang neural. TPU tidak digunakan.
2. Jalankan exp00 dari atas sampai selesai. Izinkan akses Drive saat diminta. Instalasi dependency dilakukan oleh cell Environment. Jika diminta restart setelah instalasi, restart runtime sekali lalu jalankan kembali dari awal.
3. Exp00 memeriksa dataset, membuat release baru, memverifikasi paket sensor, dan menyimpan split. Tidak ada model yang dilatih di exp00. Jangan mengubah nilai hash persetujuan untuk melewati dataset yang tidak sesuai.
4. Setelah exp00 selesai, buka exp01 lalu jalankan dari atas. Lanjutkan exp02 sampai exp14. Satu runtime menjalankan satu notebook pada satu waktu; anggota tim boleh menjalankan eksperimen berbeda secara paralel. Tidak perlu mode smoke/full atau launcher seluruh eksperimen.
5. Simpan notebook yang sudah dijalankan ke Drive agar output cell ikut tersimpan. Artefak evaluasi otomatis ditulis ke `models/batch-02/outputs/expXX/<run_id>`.

Semua notebook memproses window 15/25/45 detik. Path utama sudah diisi `/content/drive/MyDrive/Penelitian-Adaptive-IDE`. Nama folder huruf besar/kecil harus sama. Jangan menjalankan notebook training sebelum exp00 selesai.

## Workspace Colab dan checkpoint otomatis

Pada exp03–exp14, cell Lokasi proyek menyalin kode dan satu release input yang sudah disahkan ke `/content/batch02-work/<release_id>`, kemudian memeriksa checksum-nya. Workspace yang sama dipakai kembali setelah restart kernel jika file runtime masih ada. Input atau kode yang berubah tidak menimpa workspace aktif; gunakan runtime baru setelah pembaruan paket. Ini disk VM Colab, bukan komputer operator.

Selama training, seluruh file kerja ditulis ke workspace tersebut. Checkpoint dikirim otomatis ke `outputs/expXX/<run_id>/checkpoints` setelah outer fold, pergantian tahap utama, dan pada batas kandidat yang selesai jika minimal lima menit telah lewat. Satu fitting yang masih berlangsung tidak dipotong untuk upload. Checkpoint berisi arsip perubahan file beserta manifest commit dan checksum; file setengah terunggah tidak dianggap checkpoint valid. Jangan menghapus arsip lama dalam folder checkpoints karena commit terbaru masih dapat merujuknya.

Laporan utama, CSV OOF/metrik, dan grafik akhir disalin ke struktur output Drive yang dapat dibaca langsung. Detail per-fold, cache inner search, dan bobot model tetap lengkap di arsip checkpoint; semuanya dipulihkan otomatis ketika resume. `status.json` dan `progress.json` di Drive merupakan status saat sinkronisasi terakhir, bukan pembaruan setiap detik. `last_checkpoint_sequence` menunjukkan urutan penyimpanan yang terverifikasi.

Jika akses Drive gagal, publikasi mencoba maksimal lima kali untuk gangguan I/O sementara; izin ditolak atau disk penuh dihentikan lebih awal. Training tidak melanjutkan pekerjaan baru setelah kegagalan penyimpanan. Workspace yang sudah ada tidak dihapus. Status `storage_pending` berarti komputasi selesai tetapi penyimpanan final belum lengkap. Run baru dianggap tersimpan lengkap setelah laporan serta checkpoint akhir berhasil diverifikasi.

## GPU dan TabPFN

MLP, FT-Transformer, dan GRU memakai CUDA bila tersedia, serta tetap mempunyai fallback CPU. Dependency neural dipasang otomatis pada notebook yang memerlukannya. CPU threads mengikuti perangkat, dibatasi maksimal empat. Kebutuhan dukungan kelas/grup untuk early stopping tetap berlaku.

TabPFN adalah cabang opsional. Default `ENABLE_TABPFN=False` tidak mengunduh paket/checkpoint atau menerima lisensi otomatis. Cabang ini akan tercatat skipped; itu bukan hasil evaluasi TabPFN dan bukan error run.

Untuk mengaktifkannya pada exp12 atau exp14, sediakan checkpoint TabPFN V2 yang kompatibel di `models/batch-02/checkpoints/tabpfn-v2-classifier.ckpt`, isi hash SHA-256 yang benar dan konfirmasi lisensinya di cell konfigurasi, lalu ubah `ENABLE_TABPFN=True`. Setelah konfirmasi tersebut, bobot diverifikasi dan disalin sekali ke workspace. GPU dan batas ukuran data juga harus memenuhi protokol. Semua file checkpoint ini opsional; 14 notebook tidak otomatis berarti seluruh cabang model eligible.

## Hasil, waktu, dan resume

`RESUME_DIR=None` memulai run baru. Untuk melanjutkan exp03–exp14 yang memakai sistem checkpoint ini, isi RESUME_DIR dengan folder run **di Drive**, misalnya `BATCH_DIR / 'outputs' / 'exp03' / '<run_id>'`, kemudian jalankan notebook dari atas. Jangan menunjuk file tar.gz atau folder `/content` dari runtime yang sudah hilang. Sistem memilih commit valid terbaru, memverifikasi arsip, dan memulihkan workspace. Jika runtime yang sama masih hidup, progres lokal yang belum terunggah tidak ditimpa. Resume mensyaratkan data, kode, seed, konfigurasi, dependency, dan perangkat yang kompatibel; jangan mengedit identity.json untuk melewati pemeriksaan ini. Run exp03 sebelum perbaikan harus dimulai ulang karena kode dan format penyimpanannya berbeda.

Jika hanya upload yang gagal sementara runtime masih hidup:

1. Periksa akses Editor, kuota penyimpanan, dan mount Drive. Jangan menghapus atau me-remount paksa runtime ketika ada proses lain yang memakainya.
2. Di bagian **Pemulihan Penyimpanan** notebook, ubah `RETRY_PUBLISH=True` dan jalankan cell itu saja. Cell memanggil `persistence.retry_publish()` dan tidak menjalankan training. Setelah berhasil, kembalikan nilainya ke False.
3. Jika komputasi sudah selesai, tampilkan kembali cell Hasil Tersimpan. Jika belum, isi `RESUME_DIR` dengan folder run Drive yang dicetak sebelumnya dan jalankan kembali cell training; cache pekerjaan yang selesai dipakai kembali.

Disk `/content` bersifat sementara. Jika VM hilang sebelum upload berhasil, hanya checkpoint terakhir yang terverifikasi di Drive yang dapat dipulihkan. Pekerjaan setelah checkpoint tersebut perlu diulang. Tidak ada EXECUTION_STATUS global atau worker training lokal. Untuk ekspor ulang laporan/bobot, restore menggunakan `persistence.restore(path_run_drive)` pada runtime yang telah disiapkan tanpa memanggil cell training.

Nested evaluation dapat memerlukan waktu lama, terutama pada exp02 dan exp14. GPU tidak mempercepat semua algoritma tabular. Colab tetap mempunyai batas runtime, kuota, penyimpanan, dan kemungkinan putus; resume serta artefak Drive diperlukan. Baca [FAQ Colab](https://research.google.com/colaboratory/faq.html) dan [versi runtime](https://research.google.com/colaboratory/runtime-version-faq.html).

Setiap run menyimpan prediksi OOF, metrik kelas, confusion matrix, metadata, grafik, serta laporan SUMMARY. `complete_oof=false` tidak boleh dianggap hasil lengkap. Exp14 juga menyimpan evaluasi pemilihan window dan model refit. Skor refit bukan skor test baru.

## Pemeriksaan paket

Pemeriksaan ulang mencakup JSON notebook, syntax seluruh cell kode dan modul shared, konfigurasi path Drive, checksum dataset/paket sensor, serta kecocokan 26 fitur pada seluruh row curated. Hasil pencocokan adalah 3.579 row untuk window 15 detik, 3.214 row untuk 25 detik, dan 2.527 row untuk 45 detik. Metadata status eksekusi lokal lama dibersihkan; notebook disiapkan tanpa output tersimpan.

Tes penyimpanan memakai release fixture dan fungsi fitting deterministik pengganti. Cakupannya meliputi pembuatan/rename file sementara, delta checkpoint, upload gagal, arsip korup, pemulihan progres, isolasi output eksperimen, serta kesamaan prediksi sebelum/sesudah persistence. Tes tidak melatih estimator penelitian. Instalasi dependency serta eksekusi pada hosted Colab belum diuji langsung, sehingga keberhasilan instalasi, ketersediaan GPU, kuota, dan putusnya runtime tetap perlu diperhatikan saat pelaksanaan. Bila cell Environment meminta restart setelah instalasi, lakukan restart lalu jalankan lagi dari atas; jangan melewati pesan tersebut.

Pemeriksaan 18 November 2026: 27 tes lolos, termasuk simulasi jalur exp03–exp14. Seluruh 15 notebook (117 cell kode) lolos nbformat/syntax; lint dan type check paket lolos; tidak ada temuan keamanan kategori medium/high pada pemeriksaan statis. Persiapan workspace juga diuji pada proses terpisah dengan tiga input terbaru dan seluruh row/fiturnya cocok dengan paket sensor. Pemeriksaan input menggunakan snapshot temporer khusus validasi, bukan release training baru pada folder proyek atau Drive tim.
