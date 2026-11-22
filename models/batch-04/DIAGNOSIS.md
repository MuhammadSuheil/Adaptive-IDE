# Diagnosis hasil Batch 04

Tanggal pemeriksaan: 22 November 2026. Cakupan hasil resmi: exp01, exp03, dan exp04. Exp02 belum selesai dan tidak digunakan untuk menyimpulkan performa.

## Kesimpulan

Belum ada bukti bahwa dataset dan pipeline saat ini dapat menghasilkan accuracy 70% pada window waktu yang terpisah dari fitting. Hasil test tertinggi yang tersimpan adalah 44,80%. Angka itu bukan batas maksimal teoretis; pemeriksaan ini tidak membuktikan bahwa 70% mustahil.

Pemeriksaan menemukan masalah yang berbeda pada beberapa bagian:

- Fitur membedakan window yang berdekatan jauh lebih baik daripada blok waktu yang terpisah. Ribuan window overlap memberikan keteraturan lokal, tetapi belum menghasilkan pengenalan kelas yang stabil pada interval lain.
- Perbedaan level dasar antarsesi besar. Normalisasi global tidak menghilangkannya; kontrol normalisasi sesi membantu sebagian model, tetapi belum menghasilkan pemisahan yang mendekati 70%.
- Pembagian data menjaga kelengkapan tiga kelas secara global, tetapi sebagian pola sesi–kelas pada test tidak mempunyai contoh pada Train atau validation akhir.
- Pemantau epoch neural dapat mempunyai komposisi kelas yang berbeda jauh dari fitting. Model akhir sering memakai checkpoint epoch 1–2 karena loss monitor memburuk saat training berlanjut.
- Aturan seleksi accuracy-first tetap menerima model yang tidak mengenali satu kelas, termasuk GRU utama exp04.

Temuan tersebut tidak mendukung kesimpulan bahwa hanya jumlah Low yang menghambat hasil, bahwa seluruh sensor rusak, atau bahwa cukup mengganti arsitektur sekali lagi. Prioritas berikutnya adalah pemeriksaan terkontrol atas representasi, pembagian development/monitor, dan kriteria pemilihan. Perlu juga memeriksa bukti hubungan label protokol dengan respons sensor; keberhasilan cleaning tidak mengukur hubungan itu.

## 1 Bukti dan batas pemeriksaan

Rekap hasil resmi tersedia di [SUMMARY.md](SUMMARY.md). Audit numerik tersimpan dalam [diagnostic_checks.json](diagnostic_checks.json), mencakup hash data, hasil rekonsiliasi, prediksi model resmi, metadata monitor, dan kontrol diagnostik tambahan.

Pemeriksaan memverifikasi ketiga CSV terbaru dan hash split yang sama, menghitung ulang metrik dari 26 berkas prediksi test, serta memeriksa 1.457 artefak fitting/selection/test/complete. Tidak ditemukan perbedaan pada metrik, label sumber, atau checksum dalam cakupan tersebut. Tidak ada training ulang notebook secara penuh, pencarian hyperparameter baru, atau penggantian hasil resmi.

Pemeriksaan tambahan terdiri dari:

1. Tiga puluh fitting classifier sederhana pada outer Train, dinilai pada validation akhir yang sudah ada, untuk membandingkan normalisasi.
2. Uji 1-nearest-neighbor dengan lima fold acak dan lima fold blok purged. Kedua skema menghasilkan OOF atas kumpulan outer Train yang sama. Kontrol tidak memakai Test resmi untuk fitting atau memilih konfigurasi.
3. Rekonstruksi metadata monitor dari kode dan split asli, tanpa melatih jaringan baru.

Seluruh skor kontrol merupakan diagnosis setelah eksperimen, bukan evaluasi akhir yang independen. Pemeriksaan tidak memilih ulang pemenang berdasarkan test.

## 2 Cleaning selesai tidak berarti keterpisahan label sudah terbukti

Ketiga dataset tidak mempunyai missing, infinity, atau duplikasi ID pada fitur core. Proporsi terbesar/terkecil kelas masing-masing sekitar 1,07; 1,06; dan 1,44. Kelas pada test 15 detik bahkan hampir sama besar: 263 Low, 253 Medium, dan 252 High. Walaupun demikian, pemenang exp01 hanya mencapai 28,78% pada test 15 detik.

Perbaikan dataset menyelesaikan masalah nilai tidak layak, penggunaan fitur spasial, provenance, dan beberapa artefak RR. Langkah itu tidak membuat pola fisiologis ketiga label menjadi terpisah. Penilaian bahwa setelah cleaning hanya pipeline yang tersisa terlalu kuat: yang telah lulus adalah pemeriksaan kelayakan teknis, bukan bukti kecukupan informasi untuk klasifikasi 70%.

Sasaran model tetap satu classifier dari sensor, bukan classifier identitas. Identitas sesi pada diagnosis ini hanya mengukur sumber variasi, cakupan data, dan batas waktu; identitas bukan predictor yang dimasukkan agar skor naik.

### Makna label dan pengukuran yang tersedia

Builder menetapkan `cognitive_load_label` melalui protokol difficulty dan fase. Protokol tersebut tetap dipertahankan. CSV tidak berisi penilaian mental effort atau pengukuran cognitive load independen untuk setiap window. Kolom `difficulty_level` berasal dari konteks tugas, bukan pengukuran fisiologis tambahan.

CodeArrange dapat meminta rating tingkat kesulitan soal 1–10. Rating itu dapat membantu memeriksa kesesuaian pengalaman peserta dengan protokol, tetapi tidak otomatis sama dengan skala mental effort tervalidasi. Audit ini tidak menemukan rating pada tiga CSV model dan tidak mengklaim bahwa rating tidak ada pada seluruh arsip tim.

Karena itu, belum ada hasil dalam eksperimen ini yang membuktikan setiap interval berlabel Medium atau High mempunyai respons sensor berbeda dari Low. Ini batas bukti, bukan alasan untuk mengubah label mengikuti prediksi. Audit rating/performa harus dipisahkan dari fitting classifier; jangan melabel ulang baris hanya untuk mengejar accuracy.

Ukuran `iris_size_estimate` juga merupakan estimasi geometri landmark iris pada mode `eye_width_ratio`, bukan pengukuran diameter pupil. Perbaikannya tidak dapat menciptakan sinyal pupil dilation yang belum direkam. Kode sumber mendefinisikan makna ini pada [builder](../../analysis_sessions/dataset_fusion/build_dataset.py) dan [recorder eye](../../eye_tracking_prototype/eye_tracking_prototype.py).

## 3 Kontrol baru menunjukkan ketergantungan pada kedekatan waktu

Kontrol menggunakan core-head 14 fitur, StandardScaler yang fit hanya pada fitting fold, dan 1-nearest-neighbor tanpa tuning. Seluruh baris outer Train memperoleh tepat satu prediksi OOF pada masing-masing skema. Skema acak memakai StratifiedKFold lima bagian, shuffle, seed 42. Skema blok memakai lima fold repeat pertama dari manifest resmi beserta purge.

| Window | Baris OOF yang sama | Accuracy OOF acak | Accuracy OOF blok purged | Baris acak dengan window fitting overlap |
|---|---:|---:|---:|---:|
| 15 detik | 3.019 | 84,17% | 41,40% | 2.947 atau 97,62% |
| 25 detik | 2.677 | 93,16% | 43,48% | 2.643 atau 98,73% |
| 45 detik | 2.049 | 98,39% | 49,00% | 2.022 atau 98,68% |

Pada skema blok, tidak ada baris scoring dengan window fitting yang overlap. Kelas dan kumpulan baris OOF sama di antara dua skema; assignment fold dan jumlah fitting berubah karena purge. Angka tersebut mengukur sensitivitas terhadap rancangan pembagian, bukan mengisolasi efek overlap saja dengan semua faktor lain identik.

Uji ini menunjukkan bahwa angka di atas 70% dapat diperoleh pada dataset sekarang melalui kedekatan window. Angka acak tersebut tidak dapat dilaporkan sebagai keberhasilan model pada interval baru. Window yang berbagi pengukuran mentah memberikan informasi yang terlalu dekat dengan contoh latihan. Penjelasan umum tentang ketergantungan sampel waktu tersedia pada [dokumentasi cross-validation scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-of-time-series-data).

Sebaliknya, hasil blok 41–49% dari satu algoritma bukan bukti bahwa classifier apa pun maksimal 49%. Kesimpulan yang didukung adalah keteraturan lokal lebih kuat daripada pola kelas yang dapat dipindahkan antarblok dengan kontrol tersebut. Mengubah split menjadi acak untuk memperoleh 70% akan menjawab pertanyaan yang berbeda dan berisiko memberi estimasi optimistis.

## 4 Level dasar antarsesi masih mendominasi beberapa fitur

Audit menghitung ulang rasio variasi antar-rerata kelompok terhadap variasi total. Pada window 45 detik:

| Fitur | Variasi antar-rerata sesi | Variasi antar-rerata kelas |
|---|---:|---:|
| RMSSD | 78,99% | 0,12% |
| EAR kiri mean | 88,31% | 2,56% |
| Estimasi ukuran iris mean | 91,56% | 1,27% |

Kedua kolom bukan komponen yang dijumlahkan. Perbedaan rerata kelas yang kecil tidak membuktikan bahwa tidak ada interaksi nonlinear. Namun, besarnya variasi antarsesi menjelaskan mengapa standardisasi global belum cukup sebagai pengendalian level dasar. Menghapus nama peserta dari X tidak menghapus variasi yang tetap ada dalam nilai sensor.

### Kontrol normalisasi yang mempertahankan model

Kontrol berikut mempertahankan Logistic Regression C=1, core-head 14 fitur, Train/validation resmi, dan seed. Acuan median/IQR sesi hanya memakai fitting Train. Seluruh nilai negatif yang sah setelah pemusatan dipertahankan.

| Window | Accuracy validation global | Session centered | Session robust |
|---|---:|---:|---:|
| 15 detik | 27,91% | 40,39% | 44,66% |
| 25 detik | 27,50% | 42,63% | 41,45% |
| 45 detik | 32,55% | 38,32% | 44,62% |

Kontrol core 10 fitur memberikan validation tertinggi 46,46% pada Logistic Regression session robust 45 detik. Extra Trees 200 pohon dengan global atau session-centered tidak menunjukkan kenaikan konsisten. Seluruh rincian 30 fitting ada dalam audit numerik.

Hasil tersebut mendukung manfaat normalisasi pada sebagian model linear. Hasil itu tidak mendukung janji bahwa normalisasi saja akan menaikkan accuracy menjadi 70%. Exp02 seharusnya menguji variasi acuan lebih luas; karena belum selesai, performanya tetap tidak diketahui.

## 5 Cakupan sesi dan kelas belum terjaga di seluruh bagian

Algoritma pembagian mensyaratkan tiga kelas secara global dan memakai penalti distribusi sumber, tetapi tidak mewajibkan setiap sel sesi–kelas yang dinilai pada test mempunyai dukungan pada Train atau validation.

| Window | Seluruh test | Baris test dari sel sesi–kelas tanpa Train | Baris test dari sel sesi–kelas tanpa validation akhir |
|---|---:|---:|---:|
| 15 detik | 768 | 30 atau 3,91% | 352 atau 45,83% |
| 25 detik | 683 | 93 atau 13,62% | 319 atau 46,71% |
| 45 detik | 529 | 94 atau 17,77% | 274 atau 51,80% |

Ini tidak berarti seluruh kelas Medium/High hilang dari Train. Contohnya, ada pola Low dari satu sesi pada test yang tidak pernah hadir sebagai Low dari sesi tersebut pada Train, walaupun Low dari sesi lain tersedia. Sel yang sama dapat masuk kedua kolom; jumlahnya tidak dijumlahkan.

Untuk model sensor dengan level dasar antarsesi yang berbeda besar, keadaan ini menyulitkan transfer pola dan membuat validation kurang mewakili sebagian test. Audit ini belum mengisolasi berapa poin penurunan yang disebabkan oleh kekurangan cakupan tersebut.

Pemecahannya juga tidak selalu dapat berupa pemaksaan setiap sel muncul pada tiga bagian. Sebagian fase pendek tidak mempunyai cukup blok terpisah yang tetap memenuhi purge. Rancangan berikutnya perlu memeriksa kelayakan cakupan terlebih dahulu dan melaporkan sel yang tidak dapat dibagi. Jangan menghapus baris test atau mencari seed berdasarkan accuracy.

## 6 Pemantau neural memberi sinyal yang tidak stabil

Monitor untuk early stopping berasal dari sebagian blok fitting, bukan validation akhir. Kode menerima pembagian monitor ketika tiga kelas hadir. Kode tidak mensyaratkan keseimbangan proporsi kelas, fase, atau sumber sesi.

Contoh repeat pertama, fold kedua, window 45 detik:

| Kelas | Fitting optimizer | Monitor |
|---|---:|---:|
| Low | 284 atau 24,76% | 180 atau 83,33% |
| Medium | 357 atau 31,12% | 27 atau 12,50% |
| High | 506 atau 44,12% | 9 atau 4,17% |

Monitor mempunyai tiga kelas, tetapi distribusinya berbeda jauh dari fitting. Sebagai pemeriksaan risiko desain, ini lebih informatif daripada sekadar mengecek tidak ada kelas yang kosong.

### Epoch 1 berarti checkpoint terbaik, bukan training berhenti setelah satu epoch

Pada recipe GRU utama 45 detik, sembilan dari sepuluh fold memilih checkpoint epoch pertama. Contoh fold pertama menjalankan 21 epoch: train loss turun dari 1,0846 menjadi 0,8341, tetapi monitor loss naik dari 1,0433 menjadi 1,3093 dan monitor accuracy turun dari 48,18% menjadi 14,09%. Kode kemudian memulihkan checkpoint epoch pertama.

Pada TabM utama 45 detik, contoh fold pertama menjalankan 22 epoch. Train loss turun dari 1,0985 menjadi 0,7174, sedangkan monitor loss naik dari 1,0915 menjadi 1,4495. Checkpoint terbaik berada pada epoch kedua.

Median epoch fold menentukan durasi fitting seluruh Train. Hasilnya:

- Exp03: 16 dari 24 recipe finalist neural dilatih akhir selama satu epoch; tiga recipe selama dua epoch.
- Exp04: 12 dari 15 recipe finalist neural dilatih akhir selama satu epoch.

Data tersebut membuktikan bahwa model menyesuaikan fitting ketika loss monitor justru memburuk. Distribusi monitor yang tidak seimbang dan perbedaan sumber merupakan masalah desain yang perlu diuji. Audit belum membuktikan bahwa salah satunya menjadi satu-satunya penyebab pemburukan loss.

Karena itu, menaikkan batas maksimum epoch atau memaksa minimum epoch belum merupakan perbaikan yang teruji. Periksa representativitas monitor, kestabilan checkpoint antarblok, dan konsistensi learning-rate schedule pada CV versus fitting akhir. Early stopping yang mempertahankan checkpoint awal juga tidak sama dengan memastikan model akhir telah belajar pola yang bermanfaat.

## 7 Seleksi accuracy menerima model yang kehilangan kelas

GRU utama exp04 memperoleh validation accuracy 52,76% dengan memprediksi seluruh 138 Low dengan benar, 63 dari 115 Medium, dan 0 dari 128 High. Pada test, model menghasilkan 394 prediksi Low, 135 Medium, dan 0 High.

Ini kegagalan yang sudah terlihat sebelum test. Macro-F1 validation GRU 43,04%, sedangkan HistGradientBoosting history 60 detik mencapai 43,93% dengan recall minimum 32,17%. Accuracy validation model tabular lebih rendah, 44,62%, sehingga aturan accuracy-first tetap memilih GRU yang recall High-nya nol.

Pada test yang telah tersimpan, kontrol HistGradientBoosting tersebut mencapai 44,42% dan Macro-F1 42,02%, dibandingkan GRU 40,08% dan 26,21%. Perbandingan ini menunjukkan akibat pilihan pada kandidat yang sudah diuji, bukan izin mengganti pemenang setelah melihat test. Selisih itu juga masih jauh dari sasaran 70%.

TabM pemenang 15 detik di exp03 mempunyai masalah serupa pada Medium: tidak ada prediksi Medium pada validation maupun test. Kesalahan ini tidak memerlukan diagnosis sensor baru untuk diketahui; aturan seleksi sudah mempunyai informasi bahwa satu kelas tidak dikenali.

Rancangan berikutnya perlu menetapkan kelayakan pengenalan tiga kelas sebelum accuracy menjadi pembeda kandidat. Ambang dan metrik harus dibakukan pada development, bukan dicari menggunakan test yang sudah diketahui. Memilih metrik lain setelah test dibaca tidak menghasilkan evaluasi baru yang independen.

## 8 Hasil TabM tertinggi belum mendekati sasaran

TabM 45 detik menjawab benar 237 dari 529 test. Untuk mencapai 70% pada kumpulan baris tersebut, model memerlukan setidaknya 371 jawaban benar, atau tambahan 134. Itu sekitar 46% dari 292 kesalahan sekarang.

Model mengenali 144 dari 215 Low, 34 dari 122 Medium, dan 59 dari 192 High. Masalahnya tidak terbatas pada satu kelas minoritas. Sebagai ilustrasi hitungan, membetulkan seluruh Low yang salah sambil mempertahankan hasil kelas lain hanya menghasilkan 58,22%. Ilustrasi ini bukan prediksi model baru.

Dari 350 window test yang berfase active_task, TabM benar pada 101 atau 28,86%. Dari 174 window mixed yang tetap berlabel Low sesuai protokol, model benar pada 132 atau 75,86%. Diagnosis ini tidak mengganti protokol Low atau menyatakan mixed tidak sah. Temuan itu menunjukkan bahwa accuracy gabungan 44,80% menutupi perbedaan kemampuan antarkonteks. Keberhasilan pada beberapa bagian Low belum menjadi pengenalan beban pada seluruh aktivitas yang ditargetkan.

GRU utama lebih berat pada pola tersebut: accuracy active_task 13,43%, sedangkan mixed Low 91,95%. Karena itu, score gabungan atau recall Low yang tinggi saja tidak cukup untuk menilai model berguna.

## 9 Alasan hasil belum membaik setelah beberapa batch

Rangkaian sebelumnya menyelesaikan sejumlah kesalahan nyata: keputusan SVM terkalibrasi, refit setelah selection, penggunaan gaze yang tidak layak, redundansi fitur, dan artefak RR. Perbaikan tersebut mengurangi sumber kesalahan tertentu. Perbaikan itu tidak membuktikan bahwa informasi sensor yang tersisa dapat memisahkan label protokol di luar neighborhood waktu tempat model belajar.

Batch 04 juga mengubah dataset, jumlah/profil fitur, panjang blok, purge, cakupan test, dan beberapa aturan training dibandingkan batch sebelumnya. Selisih skor antarbatch bukan uji terkontrol efek cleaning saja. Skor lama yang lebih tinggi tidak membuktikan dataset baru lebih buruk; skor baru yang rendah tidak membuktikan seluruh perbaikan tidak berguna.

Memperluas keluarga model tetap mencoba memetakan input dan label yang sama di bawah batas bukti tersebut. Jumlah eksperimen tidak menambah informasi pengukuran, bukti validitas label, atau interval independen. Di sisi pipeline, memperbaiki satu mekanisme seleksi juga tidak menjamin masalah generalisasi sudah selesai.

## 10 Keputusan berikutnya

Jangan langsung membuka batch besar berisi banyak arsitektur. Mulai dengan audit development yang kecil, dengan dataset tetap, agar perubahan dapat dijelaskan.

| Prioritas | Pekerjaan | Bukti yang perlu dihasilkan sebelum memperluas eksperimen |
|---|---|---|
| 1 | Bekukan versi dataset, profil fitur, dan evaluasi; audit cakupan blok/sesi–kelas serta komposisi monitor | Tabel dukungan yang layak, pemisahan waktu tetap sah, dan monitor tidak hanya lolos karena tiga kelas sekadar hadir |
| 2 | Replikasi classifier sederhana dan kontrol normalisasi di development; tampilkan learning curve dan confusion matrix | Perbaikan yang berulang pada blok berbeda, bukan hanya satu skor atau satu kelas |
| 3 | Uji mekanisme monitor dan durasi fitting neural dengan satu arsitektur terlebih dahulu | Penentuan epoch stabil serta pengenalan tiga kelas membaik tanpa memakai test untuk memilih epoch |
| 4 | Tetapkan kelayakan per kelas sebelum peringkat accuracy; laporkan Macro-F1 dan recall minimum | Kandidat dengan kelas tidak dikenali tidak dinyatakan model akhir yang memadai |
| 5 | Audit kesesuaian label protokol dengan rating kesulitan/performa yang tersedia dan kebutuhan sinyal sensor | Bukti eksternal tentang manipulasi tugas serta kejelasan fitur yang benar-benar diukur; tanpa relabel berdasarkan prediksi |
| 6 | Jika development tetap sekitar 40–50%, bahas perubahan fitur/protokol atau pengambilan data terarah | Keputusan sumber informasi yang perlu ditambah, bukan pengulangan search pada input yang sama |

Pemeriksaan normalisasi tambahan di sini belum menggantikan seluruh exp02. Menghentikan exp02 dapat menjadi keputusan biaya/waktu, tetapi performanya tidak boleh ditebak lalu dicatat sebagai gagal.

Target 70% tetap dapat dipakai sebagai sasaran tim. Bukti sekarang belum cukup untuk menjanjikan pencapaiannya pada dataset yang sama. Jika tim memperoleh angka 70% dengan pembagian yang membiarkan window overlap antarbagian, angka itu belum menjawab kemampuan pada interval baru. Jika perbaikan development memberi hasil lebih tinggi, lakukan evaluasi akhir yang belum digunakan untuk memilih perubahan. Test Batch 04 kini sudah menjadi bagian diagnosis dan tidak lagi sepenuhnya belum pernah dilihat.

## Referensi pemeriksaan

- [SUMMARY.md](SUMMARY.md), [audit numerik dan rancangan kontrol](diagnostic_checks.json).
- [Run exp01](outputs/exp01/20261121T025639248779Z_9443342741/run.json), [run exp03](outputs/exp03/20261121T073428389875Z_8be6ef1c10/run.json), [run exp04](outputs/exp04/20261121T090223169904Z_62687ffa6e/run.json).
- [Pembagian data dan monitor](exp01/EyeHeartCognitiveLoad.ipynb); [training neural exp03](exp03/EyeHeartCognitiveLoad.ipynb), [training temporal exp04](exp04/EyeHeartCognitiveLoad.ipynb).
- [Histori TabM 45 detik](outputs/exp03/20261121T073428389875Z_8be6ef1c10/fold_w45_447044003f18e07a_r0f0/history.csv), [histori GRU 45 detik](outputs/exp04/20261121T090223169904Z_62687ffa6e/fold_w45_4efa8c8fa2e4222b_r0f0/history.csv).
- [Laporan kualitas dataset](../../analysis_sessions/dataset_fusion/output/v2_session_phase/QUALITY_REPORT.md), [makna fitur iris pada builder](../../analysis_sessions/dataset_fusion/build_dataset.py), [rating CodeArrange](../../codearrange-web/server.js).
- [Cross-validation of time series data](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-of-time-series-data), [data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage), dokumentasi scikit-learn.
