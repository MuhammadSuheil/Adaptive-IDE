# Model Cognitive Load

## Konteks penelitian dan peran kolom

Tujuan penelitian adalah satu model klasifikasi cognitive load (Low/Medium/High) dari fitur eye tracking dan HRV selama aktivitas pemrograman. Satu baris mewakili satu window sensor. Penelitian tidak menargetkan pengenalan identitas atau satu model terpisah untuk setiap peserta. CodeArrange merupakan sarana pengambilan data dan sumber penanda tugas, bukan ketergantungan wajib saat model utama digunakan pada aktivitas pemrograman lain.

Protokol menetapkan batas waktu easy **5 menit**, medium **12 menit**, dan hard **15 menit**. Batas ini bukan durasi aktual yang wajib dihabiskan peserta. Operator juga mengonfirmasi bahwa sebagian peserta sengaja langsung mengerjakan medium/high karena keterbatasan waktu pelaksanaan. Perbedaan durasi dan ketidaklengkapan kelas tersebut bukan otomatis kegagalan recorder atau cleaning. Konsekuensi statistiknya tetap perlu dilaporkan, tanpa menghapus peserta atau membuat data untuk tugas yang tidak mereka kerjakan.

| Peran | Kolom/contoh | Penggunaan |
|---|---|---|
| Fitur prediksi `X` | RMSSD, SDNN, EAR, kedipan, iris, head pose, gaze, eye state | Daftar eksplisit fitur sensor dan transformasinya; bukan seluruh kolom CSV |
| Label `y` | `difficulty_level`: `easy`, `medium`, `hard` | Dipetakan menjadi Low, Medium, High saat training/evaluasi; tidak menjadi input prediksi |
| Metadata identitas | `participant_id`, `session_id`, `window_id` | Pengelompokan train/test, deduplikasi, dan penelusuran sumber; bukan fitur model utama |
| Metadata konteks/kualitas | Timestamp, `puzzle_id`, `puzzle_set`, fase, coverage, validasi gaze | Alignment, filtering, batas riwayat, audit, dan pelaporan; bukan otomatis fitur prediksi |

Label CL saat ini mengikuti kesulitan tugas sebagai proxy. Tujuan aplikasi tetap klasifikasi CL dalam pemrograman, dengan batas validitas label tersebut dicatat. Jangan memasukkan nama set, ID soal, kesulitan, atau batas waktu set ke `X` untuk menaikkan skor dengan informasi yang mengungkap label.

Tiga CSV di `analysis_sessions/dataset_fusion/output` tetap menyimpan fitur, target, dan metadata bersama. Keberadaan metadata di CSV tidak berarti model memakainya. [Allowlist dataset](../analysis_sessions/dataset_fusion/output/feature_columns.json) menetapkan 26 fitur sensor yang juga dipakai sebagai allowlist dasar batch-01. Salinan allowlist pada kode pemodelan lama sudah dihapus bersama `models/shared`. Tidak diperlukan penghapusan metadata, penambahan label duplikat, atau perubahan CSV hanya untuk pelurusan konteks ini.

Pemisahan berbasis peserta merupakan pengaman evaluasi model sensor, bukan perubahan tujuan menjadi analisis perorangan. Nilai sensor tetap dapat membawa variasi antarrekaman walaupun nama peserta tidak menjadi input. Menghapus ID peserta tidak menghilangkan variasi itu dan akan menghilangkan kunci pengelompokan jika tidak disimpan terpisah.

Untuk batch berikutnya, model utama per-window harus dapat memprediksi tanpa mengetahui nama peserta, soal, set, atau batas waktu CodeArrange. Bila memakai riwayat temporal, definisikan kebutuhan waktu dan batas stream/sesi yang tersedia pada penggunaan nyata. Arsip exp09 memakai metadata soal untuk reset riwayat; itu tidak menjadikan ID soal fitur numerik, tetapi ketergantungan operasional tersebut perlu dipisahkan dari rancangan model umum. Normalisasi dari baseline sensor yang valid boleh diuji dan tidak sama dengan melatih classifier personal.

Perubahan subset fitur atau sensor pada batch berikutnya harus merupakan eksperimen terkontrol. Jangan menghapus peserta yang sengaja tidak mengerjakan easy, membuang metadata dari sumber, mengubah label berdasarkan prediksi, atau menyamakan durasi set hanya agar jumlah kelas sama. Konteks pada bagian ini menjadi acuan bagi laporan dan plan batch berikutnya.

## Pengelompokan eksperimen

Satu `batch` adalah satu rangkaian penelitian/pemodelan; `exp` adalah percobaan di dalam rangkaian itu. Nomor batch dimulai dari 00. Dengan penamaan ini, exp01 pada batch yang berbeda tetap memiliki konteks yang jelas.

| Batch | Isi | Ringkasan |
|---|---|---|
| batch-00 | Eksperimen awal tim; sebelumnya `exp0` | [SUMMARY.md](batch-00/SUMMARY.md) |
| batch-01 | Baseline dan rangkaian exp01 sampai exp12 | [SUMMARY.md](batch-01/SUMMARY.md) |
| batch-02 | Evaluasi mandiri dataset terbaru: fitur sensor, window, balancing, dan model; belum training | [PLAN.md](batch-02/PLAN.md) |

Rencana batch-02 sudah tersedia. Dataset tambahannya masih menunggu pengambilan/penerimaan data; notebook dan training baru belum dibuat. Rangkaian setelahnya dapat memakai nama `batch-03` bila nanti disepakati.

Batch-02 tidak membandingkan skor dengan batch-00/batch-01 atau menguji sebelum–sesudah penambahan Low. Seluruh dataset terbaru menjadi sumber evaluasi dan baseline batch-02. Manifest fold yang sama digunakan untuk membandingkan konfigurasi di dalam batch tersebut; hasil batch lama tetap menjadi arsip.

## Hasil evaluasi dan Git

[Diagnosis batch-01](batch-01/DIAGNOSIS.md) menelusuri keterwakilan Low, variasi antarrekaman, mutu sensor, dan batas hasil model, disertai bukti audit serta prioritas batch berikutnya.

Hasil lengkap batch-01 berada di `batch-01/outputs/exp01` sampai `exp12`. Baca [laporan batch-01](batch-01/SUMMARY.md) untuk rincian metode, hasil per kelas, grafik, dan cakupan pengujian. Notebook yang dieksekusi melalui Python tidak harus memiliki output cell tersimpan; hasilnya dapat diperiksa melalui CSV dan metadata run.

[`.gitignore`](.gitignore) berlaku untuk seluruh batch, termasuk batch berikutnya:

- CSV evaluasi/prediksi, JSON konfigurasi/manifest, TXT environment, Markdown, dan grafik tetap dapat masuk Git.
- Bobot model (`.joblib`, `.pkl`, `.onnx`, dan format bobot lain), cache training, environment lokal, serta file sementara tidak masuk Git.
- Subfolder `models/` atau `folds/` di dalam hasil tidak diabaikan seluruhnya, karena berisi metadata fitting/preprocessing yang berguna.

Meng-clone repositori cukup untuk membaca laporan dan hasil evaluasi yang disertakan. Untuk inference dari model terlatih, bagikan bobot melalui penyimpanan artefak terpisah. Metadata hasil tetap dapat menyebut path atau checksum bobot yang tidak disertakan di Git.

Seluruh 17 file kode/allowlist pada `models/shared` telah dihapus atas keputusan tim setelah batch-01 selesai. Hasil, grafik, notebook, dan laporan tetap disimpan. Notebook batch-01 serta script audit diagnosis yang bergantung pada kode tersebut kini hanya menjadi arsip dan tidak dapat dijalankan tanpa memulihkan dependensinya. File yang dihapus belum tercatat di Git lokal saat pembersihan; pemulihan memerlukan salinan lain atau riwayat editor yang tersedia.

Arsip notebook juga masih memiliki sebagian path lama dan rujukan setup yang sebelumnya dihapus. Penyertaan hasil evaluasi tidak memulihkan kesiapan Run All. Batch-02 akan memakai paket kode dan dependensinya sendiri di `models/batch-02`, bukan `models/shared` lama.

CSV hasil memuat identitas peserta, ID sesi, dan timestamp. Alias dalam grafik laporan tidak menganonimkan CSV. Periksa akses repositori dan persetujuan penggunaan data sebelum membagikannya di luar tim.
