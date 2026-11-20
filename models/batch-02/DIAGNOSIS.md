# Diagnosis Hasil Batch-03

Tanggal pemeriksaan: 20 November 2026.

## Kesimpulan

Pemeriksaan menemukan empat masalah pipeline pada run batch-03:

1. Normalisasi global belum mengatasi pergeseran level fitur antarsesi. Uji dengan Logistic Regression yang sama menunjukkan kenaikan accuracy test 19,18–23,40 poin persentase setelah pemusatan fitur berdasarkan statistik training tiap sesi.
2. Kalibrasi probabilitas SVM menurunkan accuracy keputusan model dasarnya. Pada fusion 25 detik, accuracy test turun dari 43,50% menjadi 25,30% setelah kalibrasi, tanpa perubahan bobot SVM dasar.
3. Pemilihan berdasarkan satu validation partition melewatkan kandidat dengan hasil test lebih tinggi. Contoh tabular 45 detik: kandidat pilihan mencapai 40,24% sebelum refit, sedangkan HistGradientBoosting yang sudah tersedia mencapai 64,50% pada test yang sama. Angka kedua merupakan pemeriksaan post-hoc, bukan hasil pemilihan model yang sah untuk mengganti pemenang resmi.
4. Refit training + validation menurunkan accuracy test keenam finalist sebesar 2,37–10,17 poin persentase. Perbandingan menggunakan checkpoint sebelum dan sesudah refit pada baris test yang identik.

Model neural juga menunjukkan gap generalisasi: ensemble 45 detik sebelum refit mencapai 97,89% pada training, 61,84% pada validation, dan 50,59% pada test. Gap ini merupakan hasil pengukuran; pemeriksaan belum mengisolasi seluruh penyebab pembentuk gap tersebut.

Keputusan normalisasi, kalibrasi SVM, pemilihan kandidat, dan refit berada dalam pipeline yang disiapkan untuk batch ini. Menyalahkan jumlah Low atau kualitas sensor saja tidak sesuai hasil pemeriksaan. Penambahan data dan pembenahan dataset sebelumnya tidak menjamin bahwa pipeline sudah memanfaatkan informasi sensor dengan baik.

## Cakupan dan cara pemeriksaan

Pemeriksaan menggunakan dua run lengkap:

- [Exp01: tabular](outputs/exp01/20261119T185251108907Z_2bc268ab/selection/record.json).
- [Exp02: neural](outputs/exp02/20261119T191445996307Z_f399c8da/selection/record.json).

Dataset, pembagian blok, target Low/Medium/High, dan fitur predictor tetap sama. Pemeriksaan tidak mengubah CSV, notebook, model tersimpan, atau hasil evaluasi resmi dalam [SUMMARY.md](SUMMARY.md). Tidak ada training ulang notebook atau pencarian hyperparameter baru. Pemeriksaan mencakup inference checkpoint dan dua jenis probe kecil: normalisasi Logistic Regression serta fold kalibrasi SVM.

Hash CSV yang dipakai kedua run cocok dengan CSV lokal saat diagnosis:

| Window | SHA-256 good_acceptable.csv |
|---|---|
| 15 detik | `b9a534b8de6c27d8888eb48ddc9ec5ab91d5f6712f59371de85ebc7473dbb1f5` |
| 25 detik | `478d080ab031cd06c59fa1791d9c81f1ca8845c9c62a96cdc942f1475e995a0a` |
| 45 detik | `028df02c14422d8ad469a1df0acfa57091ad367440f5607f6d89293081ee83ea` |

Hash split kedua run: `e05f94e6098828da10f64b5c39ec59e73c92978b01ff7cdc8d2241420f7e8bb7`.

Pemeriksaan memakai Python 3.13.16 dan versi dependency yang tercatat pada run. Untuk neural, inference memakai Torch 2.14.1 CPU, sedangkan Colab memakai Torch 2.14.1 CUDA. Enam finalist menghasilkan seluruh label validation dan test yang sama dengan artefak tersimpan. Selisih probabilitas maksimum pada finalist tabular kurang dari 1e-14; pada finalist neural sekitar 0,00122. Karena itu, perbandingan accuracy finalist mereproduksi keputusan Colab, meskipun probabilitas neural tidak identik bit-per-bit.

## 1. Pergeseran level fitur antarsesi belum tertangani

StandardScaler global menyetarakan skala kolom pada kumpulan training. Langkah ini tidak mengurangi rata-rata khusus tiap rekaman dari fitur rekaman tersebut.

Pada dataset 25 detik, pengelompokan sesi menjelaskan variasi fitur jauh lebih besar daripada pengelompokan kelas:

| Fitur | Variasi antar-rerata sesi / total | Variasi antar-rerata kelas / total |
|---|---:|---:|
| RMSSD | 70,38% | 0,54% |
| SDNN | 52,91% | 0,44% |
| EAR kiri, mean | 89,03% | 0,10% |
| Estimasi ukuran iris, mean | 90,44% | 1,42% |
| Head yaw, mean | 92,38% | 5,64% |

Perhitungan memakai rasio jumlah kuadrat antar-rerata kelompok terhadap jumlah kuadrat total, atau eta-squared satu faktor. Kedua kolom bukan komponen yang boleh dijumlahkan. Nilai kecil untuk kelas tidak membuktikan tidak ada informasi nonlinear atau interaksi fitur.

### Uji terkontrol

Probe mempertahankan 23 fitur fusion, Logistic Regression `C=10`, label, training/validation/test, imputer, dan StandardScaler. Probe mengubah satu perlakuan: mengurangi nilai fitur dengan rata-rata sesi yang dihitung dari **baris training saja**. Untuk sesi tanpa training, probe menggunakan rata-rata global training. Probe tidak membaca nilai validation/test untuk menghitung pusat fitur dan tidak memakai label untuk menghitung rata-rata.

| Window | Accuracy test, normalisasi global | Accuracy test, pemusatan per sesi | Selisih |
|---|---:|---:|---:|
| 15 detik | 35,34% | 54,53% | +19,18 poin |
| 25 detik | 34,99% | 58,39% | +23,40 poin |
| 45 detik | 39,35% | 60,36% | +21,01 poin |

Koreksi saat penyiapan batch-04: probe awal melewatkan hasil pemusatan melalui pemeriksaan nilai HRV mentah, sehingga sebagian deviasi negatif yang sah berubah menjadi missing. Tabel di atas memakai probe yang memvalidasi sensor mentah terlebih dahulu dan mempertahankan nilai negatif setelah pemusatan. Angka lama 50,22%/56,97%/59,47% tidak digunakan sebagai hasil pemusatan murni. Koreksi ini tidak mengubah checkpoint atau hasil resmi batch-03.

Probe ini menunjukkan kontribusi penanganan level fitur antarsesi terhadap kesalahan Logistic Regression pada split tersebut. Probe tidak menetapkan sumber biologis atau teknis pergeseran itu, dan tidak membuktikan kenaikan yang sama untuk seluruh algoritma.

Validation juga tidak selalu memilih perlakuan yang lebih baik pada test: untuk fusion 25 detik, accuracy validation turun dari 54,90% menjadi 50,20% setelah pemusatan, sementara accuracy test naik dari 34,99% menjadi 58,39%. Pemusatan belum menjadi metode final yang tervalidasi.

`session_id` dipakai untuk mengelompokkan statistik preprocessing, bukan sebagai predictor classifier. Uji ini sesuai konteks batch-03 yang mengizinkan rekaman yang sama hadir di training dan test. Penerapan pada sesi baru memerlukan prosedur baseline/kalibrasi yang tersedia saat inference; jangan menghitung pusat fitur dari seluruh sesi test.

## 2. Kalibrasi probabilitas SVM membalik kegunaan skor dasar

Pada cell `tabular`, fungsi `make_estimator` membungkus SVC dengan `CalibratedClassifierCV(method="sigmoid", cv=3, ensemble=False)`. Kode prediksi memakai argmax probabilitas hasil kalibrasi, bukan keputusan langsung SVC.

Pemeriksaan memuat satu bundle tersimpan, menjalankan preprocessor yang sama, lalu membandingkan keputusan SVC dasar dengan probabilitas classifier terkalibrasi. Tidak ada fitting tambahan untuk tabel berikut.

| Fusion | Accuracy training, SVC langsung | Accuracy training, terkalibrasi | Accuracy test, SVC langsung | Accuracy test, terkalibrasi |
|---|---:|---:|---:|---:|
| 15 detik | 84,51% | 3,10% | 46,55% | 27,16% |
| 25 detik | 89,56% | 1,76% | 43,50% | 25,30% |
| 45 detik | 93,40% | 1,12% | 44,08% | 24,56% |

Calibrator fusion ketiga window mempunyai slope sigmoid positif pada tiga kelas. Dengan bentuk sigmoid yang dipakai library, probabilitas kelas menurun ketika skor dasar kelas itu meningkat. Pemeriksaan juga menemukan penurunan pada eye-only. Urutan label `classes_` tetap `[0, 1, 2]`; pemeriksaan ini tidak menemukan pertukaran nama kelas pada output.

### Isolasi fold kalibrasi

Probe fusion 25 detik mempertahankan preprocessing, training, dan parameter SVC. Probe mengganti `cv=3` dengan `StratifiedKFold(3, shuffle=True, random_state=42)` hanya untuk menguji pengaruh pembagian internal kalibrasi.

- Keputusan dasar SVC sebelum/sesudah probe identik; selisih maksimum decision score adalah 0.
- Slope calibrator berubah dari sekitar `[0,168; 0,364; 0,309]` menjadi `[-2,354; -1,977; -2,576]`.
- Accuracy test setelah kalibrasi berubah dari 25,30% menjadi 44,68%.

Probe mengisolasi kalibrasi sebagai penyebab penurunan keputusan SVM tersebut. Probe ini tidak menetapkan random inner CV sebagai solusi final: window training saling overlap, sehingga desain kalibrasi final tetap harus mengendalikan kedekatan temporal. SVM langsung juga masih mempunyai accuracy test sekitar 43–47%, sehingga memperbaiki kalibrasi saja belum mencapai 75%.

Keenam finalist resmi tidak memakai SVM. Temuan ini menjelaskan kerusakan hasil kandidat SVM dalam pencarian, bukan penyebab langsung prediksi salah pada finalist non-SVM.

## 3. Validation belum memberi pemeringkatan kandidat yang dapat diandalkan

Pemeriksaan tambahan menjalankan checkpoint kandidat sebelum refit pada test yang sama. Ini merupakan **audit oracle/post-hoc** untuk memeriksa apakah pencarian kandidat melewatkan model yang lebih baik. Audit ini tidak boleh dipakai untuk mengganti pemilihan model resmi atau mengklaim skor test baru yang bebas bias seleksi.

Sebanyak 201 kandidat tabular dan 53 kandidat neural lolos pemeriksaan reproduksi dan dapat diperiksa. Sembilan checkpoint XGBoost tidak dapat dimuat pada runtime lokal karena error deserialisasi `input stream corrupted`; satu checkpoint neural berbeda satu keputusan validation antara CPU dan CUDA dan dikeluarkan dari audit bank. Kegagalan memuat XGBoost lokal ini bukan bukti bahwa training Colab gagal atau bahwa sembilan kandidat tersebut telah diaudit pada test.

| Window | Kandidat tabular tertinggi di test audit | Accuracy validation | Accuracy test audit |
|---|---|---:|---:|
| 15 detik | HistGradientBoosting fusion | 46,73% | 57,33% |
| 25 detik | Random Forest fusion | 47,84% | 58,16% |
| 45 detik | HistGradientBoosting fusion | 52,18% | 64,50% |

Contoh paling jelas ada pada 45 detik. Seleksi resmi memilih late fusion dengan validation 60,92% dan test sebelum refit 40,24%. HistGradientBoosting mempunyai validation 52,18% dan test audit 64,50%. Keduanya sudah ada sebelum diagnosis. Hasil ini mengukur kegagalan ranking pada partition tersebut; hasil ini tidak membuktikan seluruh split waktu buruk atau LOPO wajib digunakan kembali.

Pada 66 kandidat tabular non-dummy yang dapat direproduksi per window, korelasi peringkat Spearman antara accuracy validation dan test hanya 0,028 pada 25 detik dan -0,055 pada 45 detik. Pemeriksaan ini mendukung perlunya validation yang lebih stabil daripada memilih berdasarkan satu partition setelah banyak percobaan.

Tidak ada kandidat dalam bank yang berhasil diperiksa mencapai 75% accuracy test. Angka 64,50% juga berasal dari pemilihan setelah melihat test, sehingga bukan jaminan performa yang akan berulang.

## 4. Refit menurunkan hasil keenam finalist

Notebook memilih recipe memakai validation, kemudian melatih ulang model beserta preprocessing pada training + validation. Untuk neural, refit memakai jumlah epoch checkpoint terbaik sebelumnya, tanpa validation terpisah untuk memantau refit.

Tabel membandingkan checkpoint asli dan checkpoint hasil refit pada baris test yang sama. Perbandingan ini tidak mencampur skor validation dengan test.

| Finalist | Window | Accuracy test sebelum refit | Accuracy test sesudah refit | Selisih |
|---|---|---:|---:|---:|
| Tabular | 15 detik | 51,29% | 46,77% | -4,53 poin |
| Tabular | 25 detik | 34,99% | 31,21% | -3,78 poin |
| Tabular | 45 detik | 40,24% | 35,21% | -5,03 poin |
| Neural | 15 detik | 49,78% | 47,41% | -2,37 poin |
| Neural | 25 detik | 52,25% | 42,08% | -10,17 poin |
| Neural | 45 detik | 50,59% | 45,86% | -4,73 poin |

Pada neural 25 detik, refit mengubah 70 jawaban test yang sebelumnya benar menjadi salah dan memperbaiki 27 jawaban yang sebelumnya salah. Kerugian bersih adalah 43 jawaban dari 423 window test.

Refit mencakup perubahan preprocessing, bobot model, serta ukuran training. Uji ini membuktikan dampak negatif keseluruhan langkah refit pada keenam finalist, belum memisahkan kontribusi setiap perubahan internalnya. Menghapus refit saja tidak menjamin 75%; checkpoint sebelum refit juga masih rendah.

## 5. Gap generalisasi pada neural sudah ada sebelum refit

| Checkpoint sebelum refit | Accuracy training | Accuracy validation | Accuracy test |
|---|---:|---:|---:|
| MLP fusion 15 detik | 87,47% | 51,64% | 49,78% |
| Ensemble neural 25 detik | 93,34% | 60,98% | 52,25% |
| Ensemble neural 45 detik | 97,89% | 61,84% | 50,59% |

Training accuracy tinggi tidak menunjukkan bahwa penambahan epoch atau arsitektur yang lebih besar akan memperbaiki test. Pada MLP 15 detik, training loss turun dari 1,096 pada epoch awal menjadi 0,421 pada checkpoint epoch 75, sedangkan validation loss naik dari 1,094 menjadi 1,249. Notebook memilih checkpoint berdasarkan accuracy dan Macro-F1, sehingga kenaikan loss tidak otomatis menghentikan pemilihan checkpoint tersebut.

Gap ini menunjukkan overfitting pada model dan representasi yang diuji. Pemeriksaan belum menguji bahwa mengganti early stopping menjadi loss, menaikkan dropout, atau mengecilkan jaringan akan menghasilkan kenaikan tertentu. Langkah-langkah itu tetap memerlukan perbandingan terkontrol, bukan janji perbaikan.

## Hal yang tidak terbukti sebagai penjelasan utama

### Kekurangan jumlah Low bukan penjelasan yang cukup untuk run ini

| Window | Low | Medium | High | Rasio kelas terbesar / terkecil |
|---|---:|---:|---:|---:|
| 15 detik | 1062 | 1266 | 1251 | 1,192 |
| 25 detik | 984 | 1093 | 1137 | 1,155 |
| 45 detik | 908 | 728 | 891 | 1,247 |

Proporsi saat ini berbeda dari dataset awal yang hanya mempunyai 249 Low. Pada enam hasil test resmi, balanced accuracy tetap sekitar 32,25–47,57%. Kinerja rendah tetap terlihat ketika ketiga kelas mendapat bobot evaluasi setara. Ini tidak membuktikan balancing tidak berguna, tetapi tidak mendukung klaim bahwa menambah Low saja akan menyelesaikan hasil batch-03.

### CSV atau urutan prediksi yang rusak

Hash dataset/split sesuai dengan run. Pemeriksaan menghitung ulang metric dari probabilitas tersimpan dan mereproduksi keputusan enam finalist. Pemeriksaan tidak menemukan penyebab berupa label output yang tertukar, atau penggunaan CSV berbeda tanpa tercatat.

Pada 23 fitur fusion bersih, tidak ada kolom yang konstan seluruhnya atau pasangan kolom duplikat persis. Fitur dengan nilai kosong hanya blink rate: 6 baris pada 15 detik, 5 pada 25 detik, dan 10 pada 45 detik; pipeline memakai imputer training. Kondisi ini tidak sesuai dugaan bahwa model gagal karena sebagian besar predictor kosong.

### Pemisahan peserta baru

Batch-03 mengizinkan rekaman yang sama hadir pada training dan test. Karena itu, evaluasi lintas peserta bukan penjelasan untuk hasil batch ini. Kebanyakan kesalahan test juga berasal dari kombinasi sesi–kelas yang sudah mempunyai contoh training.

### Kualitas gaze

Dataset saat ini masih memuat 74,2–76,8% window dari rekaman yang gagal validasi gaze spasial. Lihat [laporan kondisi dataset](../../analysis_sessions/dataset_fusion/output/v2_session_phase/DATASET_CONDITION_REPORT.md). Status tersebut adalah kondisi rekaman sumber, bukan pengukuran baru per window.

Namun, subset test yang lolos gaze tidak konsisten mempunyai accuracy lebih tinggi. Contohnya, MLP 15 detik mencapai 39,09% pada 110 window test dari rekaman yang lolos, dibanding 50,00% pada 354 window dari rekaman yang gagal. Komposisi sesi dan kelas kedua subset berbeda; tabel ini tidak membuktikan gaze buruk lebih baik atau gaze baik merugikan.

Pemeriksaan belum mengisolasi besarnya pengaruh kegagalan gaze terhadap skor model. Karena itu, laporan tidak menempatkan kegagalan gaze sebagai penyebab tunggal yang sudah terbukti dan tidak menyatakan seluruh data eye/HRV buruk.

### Fase Low dan arti label

Pemeriksaan mempertahankan protokol v2 yang menetapkan fase sesi Low serta easy/practice sebagai Low dan pengerjaan medium/hard sebagai Medium/High. Pemeriksaan tidak mengganti label atau mengubah tujuan penelitian. Audit ini juga tidak mempunyai pengukuran independen yang cukup untuk menyatakan label fisiologis tiap window salah.

## Prioritas tindak lanjut

1. **Perbaiki jalur keputusan SVM.** Bandingkan keputusan langsung dengan kalibrasi yang mempunyai desain fold sesuai data temporal. Tambahkan pemeriksaan agar kalibrasi yang merusak keputusan dasar tidak lolos tanpa terlihat. Jangan menetapkan random inner CV dari probe sebagai solusi final.
2. **Uji penanganan level fitur antarsesi sebagai perubahan preprocessing utama.** Gunakan statistik training atau baseline yang tersedia sebelum inference. Pertahankan classifier dan split pembanding agar pengaruh normalisasi terukur. Metadata sesi tetap bukan predictor.
3. **Perbaiki pemilihan model.** Gunakan beberapa pembagian blok waktu dengan purge pada bagian development, lalu agregasikan hasilnya. Evaluasi internal tetap dapat mengizinkan orang yang sama di training dan test; tidak perlu mengubahnya menjadi LOPO untuk langkah ini. Jangan memilih kandidat berdasarkan tabel oracle diagnosis.
4. **Pisahkan keputusan memakai checkpoint terpilih dari keputusan refit.** Sediakan kontrol tanpa refit. Jika ingin refit, tetapkan prosedur pemantauan dan aturan refit melalui development data sebelum melihat evaluasi akhir.
5. **Uji kontrol overfitting setelah normalisasi dan seleksi diperbaiki.** Bandingkan model sederhana dengan neural yang lebih kecil atau regularisasi berbeda. Nilai hasil held-out, bukan kenaikan training accuracy.

Test batch-03 sudah dipakai untuk diagnosis ini. Hasil probe dan audit oracle di atas tidak boleh dianggap evaluasi akhir baru yang belum pernah dilihat. Tetapkan evaluasi akhir yang belum digunakan untuk memilih perlakuan berikutnya.

Laporan ini memastikan hasil uji pada run yang diperiksa. Laporan tidak mengklaim seluruh penyebab biologis, kualitas perangkat, dan hubungan sensor–CL sudah teridentifikasi atau bahwa satu perubahan akan menjamin accuracy 75%.
