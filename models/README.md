# Model Cognitive Load

## Mulai batch-02 di Google Colab

Panduan operasional ada di [batch-02/README.md](batch-02/README.md), termasuk struktur Google Drive, urutan notebook, dan cara melanjutkan run yang terputus.

1. Salin dataset terbaru dari `analysis_sessions/dataset_fusion/output/v2_session_phase` ke `MyDrive/Penelitian-Adaptive-IDE/Dataset/v2_session_phase`.
2. Salin `models/batch-02` ke `MyDrive/Penelitian-Adaptive-IDE/models/batch-02`. Yang diperlukan adalah folder `exp00` sampai `exp14`, seluruh `shared`, seluruh `sensor_streams`, dan `requirements.txt`. README boleh ikut. Jangan membawa `inputs`, `registry`, atau `outputs` dari run lama.
3. Jalankan notebook exp00 sampai selesai, kemudian exp01 sampai exp14. Setiap runtime menjalankan satu notebook pada satu waktu; anggota tim boleh memakai akun berbeda untuk eksperimen berbeda secara paralel. Setiap notebook sudah menangani tiga window; tidak perlu membuat salinan per-window atau mengunggah CSV pada setiap run.

`inputs`, `registry`, dan `outputs` dibuat otomatis di Drive. Untuk run baru, biarkan `RESUME_DIR=None`. TabPFN pada exp12 merupakan cabang opsional yang default-nya dinonaktifkan; baca panduan sebelum mengaktifkannya.

Pada exp03–exp14, pekerjaan training berlangsung di disk sementara Colab dan checkpoint disimpan otomatis ke Drive bersama. Semua akun perlu izin Editor pada folder proyek dan pintasan di My Drive masing-masing. Bila exp00–exp02 sudah selesai, cukup perbarui shared serta notebook exp03–exp14, gunakan runtime baru, dan mulai ulang exp03 yang gagal; pertahankan inputs/registry/output Colab yang sudah ada. Ikuti bagian pembaruan pada panduan batch-02, bukan menyalin ulang seluruh folder di atas hasil yang sudah selesai.

## Konteks penelitian dan peran kolom

Tujuan penelitian adalah satu model klasifikasi cognitive load (Low/Medium/High) dari fitur eye tracking dan HRV selama aktivitas pemrograman. Satu baris mewakili satu window sensor. Penelitian tidak menargetkan pengenalan identitas atau satu model terpisah untuk setiap peserta. CodeArrange merupakan sarana pengambilan data dan sumber penanda tugas, bukan ketergantungan wajib saat model utama digunakan pada aktivitas pemrograman lain.

Protokol menetapkan batas waktu easy **5 menit**, medium **12 menit**, dan hard **15 menit**. Batas ini bukan durasi aktual yang wajib dihabiskan peserta. Operator juga mengonfirmasi bahwa sebagian peserta sengaja langsung mengerjakan medium/high karena keterbatasan waktu pelaksanaan. Perbedaan durasi dan ketidaklengkapan kelas tersebut bukan otomatis kegagalan recorder atau cleaning. Konsekuensi statistiknya tetap perlu dilaporkan, tanpa menghapus peserta atau membuat data untuk tugas yang tidak mereka kerjakan.

| Peran | Kolom/contoh | Penggunaan |
|---|---|---|
| Fitur prediksi `X` | RMSSD, SDNN, EAR, kedipan, iris, head pose, gaze, eye state | Daftar eksplisit fitur sensor dan transformasinya; bukan seluruh kolom CSV |
| Label `y` | V2: `cognitive_load_label` (`Low`, `Medium`, `High`) | Target langsung; `difficulty_level` tetap metadata, bukan target kedua atau input prediksi |
| Metadata identitas | `participant_id`, `session_id`, `window_id` | Pengelompokan train/test, deduplikasi, dan penelusuran sumber; bukan fitur model utama |
| Metadata konteks/kualitas | Timestamp, `puzzle_id`, `puzzle_set`, fase, coverage, validasi gaze | Alignment, filtering, batas riwayat, audit, dan pelaporan; bukan otomatis fitur prediksi |

Label v2 mengikuti protokol fase/difficulty: active easy/practice/demo, pre-task setelah kalibrasi, feedback, rating, jeda, dan post-task diberi Low; active medium diberi Medium dan hard diberi High. Rating/feedback setelah medium/hard juga Low. Kalibrasi dan interval tanpa konteks yang dapat ditelusuri dikeluarkan. Seluruh window harus mempunyai satu target yang sama, meskipun fase/soalnya boleh berbeda. Jangan memasukkan nama set, ID soal, kesulitan, atau batas waktu set ke `X` untuk menaikkan skor dengan informasi yang mengungkap label.

Output dipisahkan menjadi `v1_active_task` (13 artefak lama byte-identik, termasuk dataset batch-01) dan `v2_session_phase` (window 15/25/45 detik, step 5). Ketiga view CSV di setiap varian menyimpan fitur, target, dan metadata bersama. [Allowlist v2 25 detik](../analysis_sessions/dataset_fusion/output/v2_session_phase/window_25s_step_5s/feature_columns.json) menetapkan 26 fitur sensor, tidak termasuk ID, difficulty, fase, atau target. [Laporan v2](../analysis_sessions/dataset_fusion/output/v2_session_phase/QUALITY_REPORT.md) mencatat distribusi dan rekonstruksi RR. Jangan memakai pemetaan difficulty milik notebook lama pada CSV v2; target Low dapat berada pada rating setelah hard. Kode pemodelan lama `models/shared` tetap dihapus.

Pemisahan berbasis peserta merupakan pengaman evaluasi model sensor, bukan perubahan tujuan menjadi analisis perorangan. Nilai sensor tetap dapat membawa variasi antarrekaman walaupun nama peserta tidak menjadi input. Menghapus ID peserta tidak menghilangkan variasi itu dan akan menghilangkan kunci pengelompokan jika tidak disimpan terpisah.

Untuk batch berikutnya, model utama per-window harus dapat memprediksi tanpa mengetahui nama peserta, soal, set, atau batas waktu CodeArrange. Bila memakai riwayat temporal, definisikan kebutuhan waktu dan batas stream/sesi yang tersedia pada penggunaan nyata. Arsip exp09 memakai metadata soal untuk reset riwayat; itu tidak menjadikan ID soal fitur numerik, tetapi ketergantungan operasional tersebut perlu dipisahkan dari rancangan model umum. Normalisasi dari baseline sensor yang valid boleh diuji dan tidak sama dengan melatih classifier personal.

Perubahan subset fitur atau sensor pada batch berikutnya harus merupakan eksperimen terkontrol. Jangan menghapus peserta yang sengaja tidak mengerjakan easy, membuang metadata dari sumber, mengubah label berdasarkan prediksi, atau menyamakan durasi set hanya agar jumlah kelas sama. Konteks pada bagian ini menjadi acuan bagi laporan dan plan batch berikutnya.

## Pengelompokan eksperimen

Satu `batch` adalah satu rangkaian penelitian/pemodelan; `exp` adalah percobaan di dalam rangkaian itu. Nomor batch dimulai dari 00. Dengan penamaan ini, exp01 pada batch yang berbeda tetap memiliki konteks yang jelas.

| Batch | Isi | Ringkasan |
|---|---|---|
| batch-00 | Eksperimen awal tim; sebelumnya `exp0` | [Dataset lama](batch-00/old-dataset/SUMMARY.md), [dataset baru](batch-00/new-dataset/SUMMARY.md) |
| batch-01 | Baseline dan rangkaian exp01 sampai exp12 | [Dataset lama](batch-01/old-dataset/SUMMARY.md), [dataset baru](batch-01/new-dataset/SUMMARY.md) |
| batch-02 | 14 eksperimen pada dataset terbaru setelah Mon/Alif; dijalankan dari awal di Colab | [Panduan Colab](batch-02/README.md) |

Notebook batch-02 sudah disiapkan: exp00 untuk persiapan, exp01–exp14 untuk eksperimen. Dataset dan paket sensor dibaca melalui path Google Drive. Training lokal lama tidak dilanjutkan. Rangkaian setelahnya dapat memakai nama batch-03 bila disepakati.

Batch-02 tidak membandingkan skor dengan batch-00/batch-01 atau menguji sebelum–sesudah penambahan Low. Seluruh dataset terbaru menjadi sumber evaluasi dan baseline batch-02. Manifest fold yang sama digunakan untuk membandingkan konfigurasi di dalam batch tersebut; hasil batch lama tetap menjadi arsip.

## Hasil evaluasi dan Git

[Diagnosis batch-01 dataset lama](batch-01/old-dataset/DIAGNOSIS.md) dan [dataset baru](batch-01/new-dataset/DIAGNOSIS.md) menelusuri keterwakilan Low, variasi antarrekaman, mutu sensor, dan batas hasil model, disertai bukti audit serta prioritas batch berikutnya.

Hasil batch-01 berada di folder `outputs` pada masing-masing arsip `old-dataset` dan `new-dataset`. Baca [laporan dataset lama](batch-01/old-dataset/SUMMARY.md) atau [laporan dataset baru](batch-01/new-dataset/SUMMARY.md) untuk rincian metode, hasil per kelas, grafik, dan cakupan pengujian. Notebook yang dieksekusi melalui Python tidak harus memiliki output cell tersimpan; hasilnya dapat diperiksa melalui CSV dan metadata run.

[`.gitignore`](.gitignore) berlaku untuk seluruh batch, termasuk batch berikutnya:

- CSV evaluasi/prediksi, JSON konfigurasi/manifest, TXT environment, Markdown, dan grafik tetap dapat masuk Git.
- Bobot model (`.joblib`, `.pkl`, `.onnx`, dan format bobot lain), cache training, environment lokal, serta file sementara tidak masuk Git.
- Subfolder `models/` atau `folds/` di dalam hasil tidak diabaikan seluruhnya, karena berisi metadata fitting/preprocessing yang berguna.

Meng-clone repositori cukup untuk membaca laporan dan hasil evaluasi yang disertakan. Untuk inference dari model terlatih, bagikan bobot melalui penyimpanan artefak terpisah. Metadata hasil tetap dapat menyebut path atau checksum bobot yang tidak disertakan di Git.

Seluruh 17 file kode/allowlist pada `models/shared` telah dihapus atas keputusan tim setelah batch-01 selesai. Hasil, grafik, notebook, dan laporan tetap disimpan. Notebook batch-01 serta script audit diagnosis yang bergantung pada kode tersebut kini hanya menjadi arsip dan tidak dapat dijalankan tanpa memulihkan dependensinya. File yang dihapus belum tercatat di Git lokal saat pembersihan; pemulihan memerlukan salinan lain atau riwayat editor yang tersedia.

Arsip notebook juga masih memiliki sebagian path lama dan rujukan setup yang sebelumnya dihapus. Penyertaan hasil evaluasi tidak memulihkan kesiapan Run All. Batch-02 akan memakai paket kode dan dependensinya sendiri di `models/batch-02`, bukan `models/shared` lama.

CSV hasil memuat identitas peserta, ID sesi, dan timestamp. Alias dalam grafik laporan tidak menganonimkan CSV. Periksa akses repositori dan persetujuan penggunaan data sebelum membagikannya di luar tim.
