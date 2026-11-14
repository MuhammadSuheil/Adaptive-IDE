# Batch 02: Rencana Eksperimen Cognitive Load

## Status dan tujuan

Dokumen ini merencanakan pekerjaan setelah [diagnosis batch-01](../batch-01/DIAGNOSIS.md). Notebook, pipeline baru, dan training batch-02 belum dibuat atau dijalankan. Dataset tambahan Low juga belum tersedia pada saat penyusunan rencana. Tahap pertama adalah menerima dan memvalidasi pembaruan data, bukan membuat baris sintetis sebagai pengganti rekaman baru.

**Batch-02 mengevaluasi dataset terbaru secara mandiri.** Data yang sudah diperbarui menjadi sumber seluruh eksperimen batch ini. Tidak ada eksperimen sebelum–sesudah penambahan Low, kewajiban mengulang dataset lama, atau perbandingan skor dengan batch-00/batch-01. Diagnosis lama hanya membantu menetapkan hipotesis; data dan hasil lama tetap menjadi arsip. Perbandingan model, fitur, window, dan balancing dilakukan di dalam batch-02 dengan kontrak evaluasi yang sama.

Tujuan tetap **satu classifier Low/Medium/High per window dari fitur eye tracking dan HRV selama aktivitas pemrograman**. Kita tidak membuat model pengenal peserta, model pengenal soal, atau classifier terpisah untuk setiap orang. CodeArrange merupakan instrumen pengambilan data. Label saat ini mengikuti kesulitan tugas sebagai proxy CL.

Rencana operator adalah sekitar **5 anggota tim sebagai peserta baru terhadap cohort batch-01**, dengan kemungkinan tambahan **2–3 peserta lama**, semuanya mengerjakan easy. Jumlah akhir dan identitas yang berulang dikonfirmasi melalui daftar sesi saat data diterima. Tidak ada kewajiban mengambil ulang medium/high pada batch ini.

Identitas peserta tetap metadata, bukan fitur prediksi. Metadata itu diperlukan untuk mencegah rekaman lama dan rekaman baru orang yang sama berada pada sisi train/test yang berbeda. Menjaga pembagian data bukan perubahan tujuan menjadi pemodelan perorangan.

## 1. Keputusan dasar

1. Selesaikan audit data tambahan sebelum full training utama.
2. Pertahankan data, kode, dan hasil batch-01 sebagai arsip yang tidak ditimpa; tidak perlu melatih ulang atau membandingkan skornya.
3. Setelah dataset terbaru disahkan, pisahkan pengaruh representasi fitur, window, riwayat, balancing, dan algoritma.
4. Gunakan model referensi dan manifest test yang sama untuk membandingkan konfigurasi di dalam batch-02.
5. Pilih parameter dan recipe menggunakan inner validation, bukan outer test atau peserta yang menghasilkan skor terbaik.
6. Utamakan macro-F1, dengan precision/recall tiap kelas dan coverage. Accuracy tinggi dengan Low yang tidak dikenali tidak cukup.
7. Batasi pencarian kandidat. Model yang lebih kompleks dan kelas yang sama besar tidak menjamin hasil lebih baik.
8. Model utama harus dapat memprediksi tanpa nama peserta, ID soal, nama set, difficulty, atau batas waktu CodeArrange.

Protokol easy 5 menit, medium 12 menit, dan hard 15 menit tetap menjadi konteks pengambilan data. Sebagian peserta lama memang sengaja melewati easy. Kita tidak menyebutnya kegagalan recorder, membuang peserta tersebut, atau menyamakan durasi set hanya untuk menyeimbangkan jumlah baris.

## 2. Masalah yang hendak diuji

| Temuan batch-01 | Yang masih perlu diketahui | Pekerjaan batch-02 |
|---|---|---|
| Low sebelumnya hanya 249 window dan kesempatan perekamannya lebih sedikit | Bagaimana distribusi, kualitas, dan kemampuan mengenali Low pada dataset terbaru? | Audit dataset gabungan, lalu ukur ketiga kelas pada evaluasi batch-02 |
| Model Logistic sudah sering menebak Low, tetapi banyak salah | Apakah masalahnya representasi fitur, penalti training, atau keduanya? | Ablation fitur dan normalisasi sebelum mengulang balancing |
| Sejumlah fitur memiliki perbedaan antarrekaman yang kuat | Apakah fitur yang lebih relatif/dinamis mentransfer lebih baik? | Uji pengurangan mean geometris dan normalisasi yang tersedia sebelum prediksi |
| Validasi gaze asli gagal pada 12/13 peserta cohort | Apakah model memerlukan gaze/eye-state atau bisa mengandalkan kelompok lain? | Pisahkan sensor/kelompok fitur tanpa membuang peserta secara otomatis |
| Baseline ready recorder belum tentu usable menurut builder | Bagaimana menangani rasio/baseline yang tidak layak secara konsisten? | Audit baseline, kontrol tanpa rasio, serta fallback yang terdefinisi |
| Window 25 detik mengeluarkan banyak blok Low pendek | Apakah kompromi mutu HRV dan keterwakilan data lebih baik pada window lain? | Hitung ulang RR/eye mentah dan laporkan coverage, bukan skor pada subset yang lebih mudah saja |
| Riwayat exp09 dan dua pilihan fold exp12 memakai reset soal | Apakah riwayat generik yang kausal membantu tanpa konteks CodeArrange? | Uji reset berdasarkan stream, waktu, dan gap |
| Pemilihan 154 kandidat belum memperbaiki hasil outer | Apakah bank kecil dengan kontrol lebih mudah dipilih secara stabil? | Batasi recipe dan evaluasi keseluruhan prosedur pemilihan |
| Label berasal dari kesulitan tugas dan urutannya tetap | Seberapa jauh klaim CL dan transfer tugas dapat dibuat? | Pertahankan batas proxy; jangan mengubah label berdasarkan prediksi |

Temuan tersebut tidak menetapkan satu penyebab tunggal. Batch-02 menguji perbaikan yang terarah; tidak menjanjikan angka accuracy tertentu.

## 3. Urutan pekerjaan dan syarat melanjutkan

| Tahap | Pekerjaan | Syarat melanjutkan |
|---|---|---|
| A | Bekukan data lama dan terima rekaman Low baru | Sumber, identitas lintas sesi, dan provenance lengkap |
| B | Bangun dataset baru dan kontrak evaluasi | Tidak ada duplikasi/kebocoran; label, fase, dan kualitas lolos pemeriksaan |
| C | Bangun baseline pada dataset terbaru | Prediksi dan metrik dapat ditelusuri; manifest fold dibekukan untuk eksperimen batch-02 |
| D | Uji representasi sensor dan baseline | Recipe fitur serta fallback jelas; tidak menggunakan label/future test |
| E | Uji window dan riwayat generik | Fitur benar-benar dihitung ulang; coverage dan kausalitas dilaporkan |
| F | Uji objective/balancing dan algoritma terbatas | Preprocessing dan pencarian hanya di training |
| G | Uji opsi lanjutan yang memenuhi syarat | Ada alasan dari inner validation dan kualitas data, bukan keinginan menambah eksperimen |
| H | Evaluasi prosedur seleksi, refit, dan laporkan | Semua hasil utama lengkap, identitas run cocok, keterbatasan tercatat |

Jika suatu opsi tidak memenuhi syarat, simpan status `skipped_not_eligible` beserta alasannya. Jangan membuat skor nol palsu atau mengganti metode gagal secara diam-diam. Audit dan prototipe dapat disiapkan memakai data lama, tetapi hasil utama batch-02 menunggu dataset baru yang lolos penerimaan.

## 4. Tahap A–B: persiapan dataset

### 4.1 Arsipkan data lama dan tetapkan versi dataset terbaru

Dataset batch-01 adalah `good_acceptable.csv` dengan 2.294 window dan SHA256:

`bbb4c93eca30aab6b896db887dfbb981abffe118fce385d0f9bd3d28e786247e`

Sebelum memperbarui path dataset aktif, simpan salinan byte-identik CSV lama beserta manifest, konfigurasi, dan allowlist di penyimpanan versi yang tidak ditimpa. Verifikasi hash setelah menyalin. Jangan mengandalkan nama file yang sama untuk membuktikan versi data sama.

Istilah dalam rencana:

| Nama | Isi |
|---|---|
| D0 | Snapshot dataset lama pada window 25 detik/step 5 detik; hanya arsip, bukan kondisi training batch-02 |
| D1 | Dataset terbaru: rekaman lama dan Low tambahan yang lolos deduplikasi serta pemeriksaan kualitas |

Jika rebuild mengubah nilai sensor/label baris lama, catat perubahan beserta alasannya sebagai versi reprocessing. Terapkan aturan builder secara konsisten pada sumber yang mendukungnya. Tidak perlu membuat kondisi training D0 atau kontrol sebelum–sesudah; catatan perubahan dipakai untuk audit dan reproduksi D1.

### 4.2 Satukan identitas tanpa menjadikannya fitur

Buat registry yang menghubungkan `session_id` dan folder sumber ke ID orang yang konsisten. Semua rekaman orang yang sama mendapat grup yang sama, termasuk perbedaan kapitalisasi/alias yang sudah dikonfirmasi operator.

Kode [participant_group](../../analysis_sessions/session_inventory.py) saat ini terutama memakai nama folder, menghapus akhiran Copy, dan menangani pengulangan Ramadhan secara khusus. Aturan tersebut belum menjadi pemetaan umum untuk sesi easy tambahan. Nama seperti `Nama_easy_baru` tidak boleh otomatis dianggap orang baru.

Jangan mengambil identitas orang hanya dari token `P01` pada nama sesi: token itu berulang pada beberapa rekaman lama. `session_id` menandai rekaman, bukan selalu identitas orang lintas sesi. Pemetaan memerlukan daftar operator, bukan dugaan berdasarkan kemiripan sensor.

Registry juga mencatat asal pengambilan lama/tambahan, perangkat, kondisi baseline, dan status tugas yang dikerjakan. Informasi ini untuk audit/evaluasi, bukan masuk `X`. Status anggota tim tidak membuktikan tingkat kemampuan tertentu; pengalaman atau keterbiasaan terhadap soal hanya dicatat jika informasinya tersedia.

### 4.3 Pemeriksaan rekaman baru

- Ada eye tracking, RR mentah, HRV, ledger tugas, serta metadata kalibrasi/baseline yang diperlukan.
- Timestamp dapat disejajarkan pada waktu absolut; origin elapsed yang berbeda tidak disamakan secara buta.
- Label Low hanya berasal dari interval easy yang benar-benar `active_task`. Practice, rating, feedback, dan istirahat tidak diubah menjadi Low.
- Identitas sesi antarfile cocok dan `window_id` unik. Salinan folder tidak menambah jumlah pengukuran.
- Baseline diperiksa berdasarkan status, coverage, durasi, dan kondisi pengambilannya. Jangan menyamakan baseline HRV yang bersamaan dengan kalibrasi mata dengan istirahat terpisah tanpa catatan.
- Gaze diperiksa terpisah dari kelengkapan frame. Gagal gaze tidak otomatis membatalkan HRV atau semua fitur mata nonspasial.
- Gunakan standar penerimaan yang sama pada rekaman lama/baru. Jika aturan berubah, dokumentasikan versinya dan periksa penerapannya pada seluruh sumber, tanpa eksperimen skor sebelum–sesudah.
- Laporkan jumlah Low yang benar-benar lolos, durasi unik, blok aktif, dan jumlah rekaman. Jumlah baris atau tercapainya rasio 1:1:1 bukan satu-satunya ukuran kecukupan.

Data Low tambahan dari tim dapat dipakai. Risiko yang perlu diaudit adalah keterikatan label dengan asal pengambilan: sebagian besar label Low baru berasal dari sesi/perangkat/kelompok baru, sedangkan Medium/High berasal dari rekaman lama. Menyimpan asal data di luar `X` tidak otomatis menghilangkan perbedaan nilai sensor akibat kondisi tersebut.

### 4.4 Pembaruan CSV dan skema

Pertahankan bentuk sensor + target + metadata. Allowlist predictor tetap eksplisit. `difficulty_level` dipetakan menjadi Low/Medium/High pada pemodelan; tidak perlu mengganti arti label atau menambahkan kolom target duplikat agar nama terlihat seperti CL.

Bangun versi baru `all_data`, `good_acceptable`, dan `good_only` secara konsisten bila memakai builder tiga view yang sama. Fokus training utama tetap good+acceptable. View good-only tidak dianggap validasi spasial gaze dan tidak otomatis menggantikan cohort test.

Rekaman baru harus ditinjau dalam kebijakan admission, karena builder lama memakai daftar nama peserta. Jangan memasukkan anggota tim ke kelompok kualitas tertentu hanya berdasarkan perannya, atau membiarkan rekaman valid tidak ikut D1 hanya karena namanya belum terdaftar. Perubahan admission harus terdokumentasi dan tidak ditentukan dari skor model.

Pertahankan metadata asal dalam registry/sidecar atau kolom metadata baru yang terdokumentasi. Metadata tambahan bukan fitur default. Tidak ada interpolasi sensor atau Low sintetis pada tahap penerimaan dataset.

**Keluaran penerimaan:** snapshot D0, versi D1, registry lintas sesi, hash sumber/hasil, laporan distribusi/kualitas, daftar perubahan baris lama, dan manifest evaluasi. Full training dihentikan jika identitas tidak jelas, label tidak dapat ditelusuri, atau sumber baru belum lengkap.

## 5. Kontrak evaluasi

### 5.1 Evaluasi utama: seluruh dataset terbaru

Gunakan leave-one-participant-out (LOPO) pada seluruh D1. Semua orang yang lolos penerimaan, termasuk anggota tim baru, bergiliran menjadi grup test. Seluruh sesi orang test dikeluarkan dari training, baik rekaman lama maupun easy tambahan. Tidak ada grup test khusus dari cohort lama atau jalur evaluasi sekunder untuk data baru.

Contoh: saat orang A menjadi test, seluruh window D1 milik A menjadi test dan window orang lain menjadi training. Jika A mempunyai dua sesi, keduanya tetap satu grup. Identitas hanya menentukan pemisahan; classifier menerima fitur sensor.

Bekukan manifest outer setelah D1 dan registry disahkan. Jumlah outer fold mengikuti jumlah orang unik pada D1, bukan angka 13 dari batch-01 atau jumlah folder sesi. Gunakan manifest yang sama untuk seluruh konfigurasi batch-02. Untuk eksperimen window, pertahankan grup yang sama dan cocokkan waktu prediksi sesuai bagian Exp04.

Pemisahan grup mengikuti prinsip [grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data). Seluruh D1 berpartisipasi dalam evaluasi bergilir; kita tidak melatih pada seluruh D1 lalu mengujinya pada baris yang sama.

### 5.2 Inner validation untuk pemilihan konfigurasi

Pada training setiap outer fold, buat tiga inner fold berbasis grup. Gunakan seluruh grup training yang tersedia, tanpa membedakan asal lama atau baru. Upayakan distribusi kelas melalui stratified group split dan simpan manifest yang sama untuk semua kandidat pada outer fold tersebut. Semua sesi orang validasi harus keluar dari fitting inner.

Periksa cakupan kelas dan kelayakan split sebelum training. Setiap training partition harus memuat tiga kelas. Catat kelas yang tidak ada pada validation partition dan gunakan macro-F1 OOF inner gabungan dengan urutan tiga label tetap sebagai skor seleksi, bukan rata-rata skor fold dengan definisi kelas berbeda. Jika tiga inner fold tidak feasible, revisi jumlah fold secara terdokumentasi sebelum full run; jangan menggantinya dengan split baris acak.

Semua fit imputer, scaler, pemilih fitur, sampler, threshold, dan parameter dilakukan pada training inner yang bersangkutan. Outer test hanya untuk evaluasi setelah konfigurasi dipilih.

### 5.3 Cakupan kelas dan arti evaluasi mandiri

Test orang yang hanya memiliki Low menguji pengenalan Low pada rekaman tersebut, bukan kemampuan tiga kelas lengkap pada orang itu. Laporkan support dan metrik kelas yang tersedia. Evaluasi utama menggabungkan prediksi OOF seluruh D1 untuk menilai ketiga kelas; laporan per orang merupakan audit pelengkap.

Mandiri berarti batch-02 memakai dataset, baseline, manifest, dan hasilnya sendiri. Diagnosis batch-01 boleh menjadi dasar hipotesis, tetapi skor batch-00/batch-01 tidak menjadi pembanding atau syarat keberhasilan. Tidak ada kesimpulan kuantitatif tentang manfaat penambahan Low sebelum–sesudah karena eksperimen tersebut tidak dijalankan.

### 5.4 Metrik dan ketidakpastian

Metrik utama adalah macro-F1 OOF gabungan tiga kelas. Pelengkap wajib: accuracy, balanced accuracy, precision/recall/F1/support per kelas, confusion matrix, prediksi OOF, coverage, dan hasil per rekaman/orang untuk audit.

Per-kelas yang tidak mempunyai support dilaporkan tidak tersedia. Macro-F1 gabungan tiga kelas berbeda dari rata-rata macro-F1 per orang dengan kelas yang hadir. Tampilkan nama agregasi secara eksplisit.

Untuk perbandingan konfigurasi di dalam batch-02, resample grup orang yang sama dari dua set prediksi OOF dan hitung selisih metrik, bukan resample window independen. Interval bootstrap atas prediksi yang sudah terbentuk bersifat deskriptif/bersyarat dan tidak mencakup semua ketidakpastian pemilihan serta fitting ulang model. Jangan mengklaim signifikansi hanya karena satu skor lebih tinggi.

Ini evaluasi internal pada dataset terbaru, bukan test tiga kelas eksternal yang baru. Sebagian rekaman sudah dianalisis sebelumnya; perubahan rencana tidak membuatnya menjadi data yang belum pernah dilihat. Pilihan final membutuhkan validasi lebih lanjut sebelum klaim umum di IDE.

## 6. Peta eksperimen

Nama exp berikut merupakan pemetaan keluarga eksperimen untuk notebook terpisah. Belum ada notebook yang diimplementasikan. Exp01–exp07 merupakan inti dengan cabang bersyarat; exp08–exp09 opsional; exp10 merupakan seleksi/validasi akhir, bukan kewajiban menambah algoritma.

| Eksperimen | Pertanyaan | Perlakuan utama | Pembanding yang dijaga |
|---|---|---|---|
| exp01 | Bagaimana kemampuan awal pada dataset terbaru? | Dummy, Logistic fixed, XGBoost fixed; tanpa balancing/riwayat | Seluruh D1, fitur sensor, window 25/5, dan manifest batch-02 yang sama |
| exp02 | Kelompok fitur mana yang mentransfer lebih baik? | Original/clean, dinamis, nonspasial, sensor inti, HRV-only, eye-only | Dataset D1, window 25/5, model referensi tetap |
| exp03 | Apakah baseline/normalisasi relatif membantu? | Absolut, tanpa rasio, relatif pada baseline valid | Baris test tetap; fallback terdefinisi |
| exp04 | Apakah window lain memberi kompromi lebih baik? | 15/25/45 detik dari RR/eye mentah; kontrol step 5 | Endpoint waktu D1, mutu sensor, dan coverage eksplisit |
| exp05 | Apakah riwayat sensor generik membantu? | Tanpa riwayat, riwayat kausal 15/30 detik | Tidak memakai batas soal/label saat inference |
| exp06 | Apakah objective/balancing membantu pada data baru? | Tanpa balancing, class weight, ROS, SMOTE bersyarat | Test asli dan recipe sensor yang sama |
| exp07 | Algoritma mana yang paling sesuai setelah representasi dibenahi? | Logistic, SVM, XGBoost, kandidat baru HistGradientBoosting | Bank parameter kecil, inner group validation |
| exp08 | Apakah fusi prediksi memperbaiki gabungan fitur? | Early fusion vs late fusion sensor | Cabang sensor dan kalibrasi hanya dari training |
| exp09 | Apakah memanfaatkan urutan label membantu? | Multiclass langsung vs ordinal supervised | Label/fitur/test sama; bukan novelty detection |
| exp10 | Bagaimana keseluruhan prosedur memilih model bekerja? | Seleksi recipe terbatas, audit, refit, uji inference | Outer test tidak memilih recipe atau seed |

Pada seluruh tahap, kata “fitur/recipe terpilih” berarti pemilihan di training inner pada masing-masing outer fold. Jangan mengambil pemenang global dari outer leaderboard exp02, lalu menganggap evaluasi exp03–exp10 atas pemenang itu bebas bias.

## 7. Rincian eksperimen inti

### Exp01: baseline mandiri pada dataset terbaru

Gunakan Dummy majority-training, Logistic C=1 dengan StandardScaler, dan XGBoost fixed sebagai kontrol pada D1. Logistic/XGBoost tidak memakai balancing atau riwayat. Gunakan allowlist sensor original dan window 25 detik/step 5 detik; perbandingan clean ditempatkan di exp02.

Tujuannya memverifikasi pipeline, mengenali pola kesalahan ketiga kelas, dan menyediakan baseline batch-02 untuk eksperimen berikutnya. Semua rekaman D1 mengikuti evaluasi bagian 5. Tidak ada cabang dataset lama, tambahan tim saja, atau tambahan peserta kembali saja.

Dummy memilih kelas terbanyak pada training tiap fold, bukan dipaksa selalu High. Distribusi training menentukan pilihan tersebut.

Bekukan environment, parameter fixed, serta manifest batch-02. Tidak perlu menjalankan ulang kontrol D0 atau mencocokkan hasil dengan skor arsip batch-01.

**Keluaran:** macro-F1, accuracy, precision/recall/F1 ketiga kelas, confusion matrix, coverage, dan prediksi OOF D1. Model sensor harus dibandingkan dengan Dummy pada dataset yang sama; banyaknya tebakan Low tidak cukup tanpa melihat ketepatannya.

### Exp02: representasi fitur dan kontribusi sensor

Mulai dari definisi fitur sensor batch-01. Jangan memilih fitur berdasarkan korelasi seluruh dataset, identitas orang test, atau pemenang outer test.

| Kode | Definisi awal |
|---|---|
| F0 original | Allowlist 26 fitur; kosong/konstan ditangani di training |
| F1 clean | Hapus blink_count, iris_size_delta_std, dan eye_state_other_ratio yang redundan/konstan pada data lama; validasi kembali hubungannya pada D1 |
| F2 dinamis | Dari clean, hapus mean EAR kiri/kanan, mean estimasi iris dan delta iris, serta mean head pitch/yaw; tetap pertahankan statistik perubahan/variasi yang tersedia |
| F3 nonspasial | Dari clean, keluarkan gaze dan eye-state; tetap memakai HRV dan fitur mata nonspasial |
| F4 sensor inti | HRV tiga fitur, blink rate, serta mean/std EAR kiri/kanan |
| F5 HRV-only | RMSSD, SDNN, rasio baseline yang tersedia |
| F6 eye-only | Fitur mata clean, tanpa HRV |

F2 menguji pengurangan mean geometris yang kuat berbeda antarrekaman. F3 menguji kelompok yang bergantung pada pemetaan gaze. Keduanya merupakan hipotesis, bukan penghapusan fitur permanen karena hasil audit deskriptif. Iris tidak disebut diameter pupil.

Pakai Logistic dan XGBoost fixed sebagai referensi agar perbedaan fitur tidak tercampur tuning algoritma. Fitur korelasi/MI otomatis bukan fokus utama karena sudah dicoba tanpa perbaikan; dapat menjadi kontrol terbatas di training bila D1 menunjukkan kebutuhan baru.

Tetap pertahankan satu kontrol dengan semua fitur. Quality tier/perangkat/ID tidak otomatis ditambahkan sebagai predictor. Jumlah fitur aktual setelah filter training dicatat per fold.

### Exp03: baseline, skala, dan ketidaklengkapan

Pertanyaan pertama adalah perbedaan absolut versus tanpa rasio baseline HRV pada dataset baru. Rasio yang ditolak aturan kualitas tetap kosong. Jangan menurunkan syarat coverage hanya untuk menghilangkan NaN.

Pertanyaan kedua, bersyarat, adalah representasi relatif dari baseline sensor yang tersedia sebelum tugas/prediksi. Audit terlebih dahulu apakah baseline mata/HRV dapat dibentuk secara konsisten dari sumber lama dan baru. Jika rekaman tidak mendukungnya, catat opsi itu tidak eligible untuk klaim utama; jangan membentuk baseline dari seluruh sesi test.

Kontrak baseline relatif sebelum implementasi:

- Segmen baseline berasal dari waktu yang tersedia sebelum prediksi dan tidak memakai label CL target.
- Definisi segmen, statistik, minimum coverage, serta waktu tunggunya dibekukan sebelum melihat outer score.
- Baseline HRV dan baseline geometri mata tidak dianggap sama hanya karena keduanya memakai nama baseline.
- Normalisasi pembagian memakai penyebut positif yang layak; z-score hanya memakai skala baseline yang tidak mendekati nol. Median/selisih atau rasio dipilih sesuai arti fitur, bukan satu rumus untuk semua kolom.
- Baseline tidak tersedia memakai fallback yang telah ditetapkan, misalnya fitur absolut atau nilai kosong dengan imputasi training. Jangan diam-diam menghapus test yang sulit.
- Tidak ada centering dengan mean seluruh orang test, quantile seluruh test, atau adaptasi memakai label test.

Seluruh normalisasi tetap mendukung satu classifier umum. Status baseline disimpan untuk audit. Penggunaan flag missing/quality sebagai fitur merupakan eksperimen terpisah jika nanti diperlukan, bukan perubahan default yang bisa memperkenalkan kembali ciri sesi.

Bandingkan StandardScaler dan RobustScaler secara terbatas pada satu recipe referensi. Native missing-value handling XGBoost/HGB dapat menjadi kontrol implementasi, tetapi tidak boleh sekaligus mengubah fitur dan cohort tanpa pencatatan.

### Exp04: penghitungan ulang window mentah

Builder lama secara eksplisit mengunci 25 detik/step 5 detik. Karena itu, perlu jalur builder versi baru yang terpisah dari arsip. Mengubah `window_seconds` pada CSV, merata-ratakan RMSSD lama, atau sekadar mengganti config bukan eksperimen window yang sah.

Kandidat awal adalah 15, 25, dan 45 detik dengan step 5 detik. Window 25 menjadi kontrol. Panjang 10/60 detik hanya boleh ditambahkan melalui perubahan registry yang terdokumentasi sebelum hasil outer dibuka, bila audit sensor menunjukkan alasan. Jumlah baris terbesar tidak menjadi kriteria memilih window.

Rekonstruksi RR dari sumber dengan aturan timing, artefak, gap, adjacency pasangan RMSSD, dan ketersediaan kausal yang konsisten. Hitung SDNN dari RR yang masuk window, bukan dari SDNN window lain. Sejajarkan agregasi eye pada batas absolut yang sama. Gunakan endpoint yang konsisten dan jangan memakai sampel yang baru tersedia setelah akhir window.

Kualitas HRV dilaporkan per panjang: jumlah RR/pasangan, coverage, stale data, serta kestabilan RMSSD dan SDNN masing-masing. Kestabilan/agreement pada segmen yang dapat dibandingkan tidak ditentukan dari korelasi saja. Analisis kelayakan tidak boleh memakai outer CL score untuk memilih aturan sensor. Referensi HRV ultra-short berbeda menurut kondisi dan metode; penelitian ECG bukan jaminan untuk HW9 dalam tugas ini. [Muñoz dkk.](https://pmc.ncbi.nlm.nih.gov/articles/PMC4586373/), [Burma dkk.](https://pubmed.ncbi.nlm.nih.gov/33856258/).

Pisahkan dua perbandingan:

| Perbandingan | Cara menjaga keadilan | Batas klaim |
|---|---|---|
| Efek representasi window | Train/test memakai decision endpoint yang sama dan eligible pada kandidat yang dibandingkan; hitung ulang fitur untuk tiap panjang | Berlaku pada cohort endpoint bersama, yang bisa mengutamakan span panjang |
| Pemakaian operasional | Training dapat memakai seluruh window layak tiap kandidat; test tetap endpoint waktu D1 yang telah dibekukan | Fitur dan kuantitas training berubah bersama; coverage harus ikut dibaca |

Pada test operasional, endpoint yang tidak dapat menghasilkan fitur/prediksi tidak boleh menghilang dari laporan. Simpan `prediction_available`, alasannya, coverage total/per kelas, skor bersyarat pada prediksi yang tersedia, dan skor dengan abstention dihitung sebagai kesalahan. Untuk F1 per kelas, abstention merupakan false negative pada kelas acuannya; bukan label CL keempat pada data sumber.

Bekukan endpoint evaluasi dari kontrol 25 detik D1, mencakup rekaman lama dan baru yang lolos penerimaan. Endpoint ini hanya menyamakan waktu prediksi antarkandidat window di batch-02; bukan test warisan batch-01.

Guard awal untuk kandidat model utama: minimal 80% coverage total dan Low terhadap endpoint kontrol 25 detik D1. Ini pengaman operasional yang diusulkan, bukan ambang fisiologis yang tervalidasi. Bekukan sebelum skor model dibaca; perubahan memerlukan versi protokol baru. Kandidat yang gagal boleh dilaporkan sebagai diagnostik, tetapi tidak disebut pemenang utama hanya karena subset tersisanya mendapat skor tinggi.

Window/step dipilih di inner validation dari kandidat yang telah memenuhi kontrak. Penurunan step menambah overlap, bukan otomatis informasi independen. Jika menguji step 10 detik sebagai pengurangan redundansi, cocokkan endpoint test dan pisahkan eksperimen itu dari perubahan panjang window.

Multi-resolution eye/HRV tetap opsi lanjutan, bukan solusi otomatis kehilangan Low. HRV panjang yang harus murni active_task tetap membutuhkan span panjang. Membolehkan HRV melewati feedback/rest akan mengubah definisi konteks dan memerlukan rancangan terpisah, bukan jalan pintas untuk menambah baris.

### Exp05: riwayat kausal tanpa konteks soal

Bandingkan tanpa riwayat dengan riwayat berbasis waktu 15/30 detik, memakai fitur dasar HRV, blink rate, dan EAR yang sudah didefinisikan. Gunakan mean/median dan delta yang terdefinisi dari data saat ini atau masa lalu. Jangan memakai rolling terpusat atau future fill.

Reset hanya pada perubahan stream/sesi, waktu yang tidak meningkat, dan gap melebihi batas yang dibekukan berdasarkan step. `puzzle_id`, difficulty, label, dan event submit tidak menjadi syarat inference. Peserta digunakan untuk memastikan stream training tidak tercampur, bukan fitur prediksi.

Durasi riwayat dinyatakan dalam detik agar tidak berubah arti saat step berubah. Catat perilaku warm-up, jumlah riwayat minimum, dan keluaran ketika gap. Waktu tunggu atau riwayat parsial tidak disembunyikan sebagai fitur lengkap.

Wajib ada uji prefix invariance: menambahkan baris masa depan tidak mengubah fitur/prediksi masa lalu. Batch inference dan stream satu window per langkah harus konsisten pada input yang sama.

### Exp06: objective dan balancing setelah tambahan data

Gunakan tanpa balancing sebagai kontrol. Class weight dihitung dari training saat itu, bukan dari seluruh CSV. ROS/SMOTE hanya dipertimbangkan bila Low masih minoritas dalam training fold. Jika target resampling sudah tercapai, tandai no-op dan deduplikasi konfigurasi; jangan melakukan undersampling terselubung.

Kandidat awal yang terbatas:

- Class weight: none versus balanced.
- ROS Low-only: target 0,50/0,75 dari kelas training terbesar, bila memang menambah Low.
- SMOTE Low-only: target 0,50/0,75 dan k=3; kandidat k=5 hanya bila jumlah/keragaman Low training mendukung.

Imputasi/scaling dan tetangga SMOTE dibentuk di training partition. Validation/test tetap asli. Sampel sintetis tidak mendapat identitas peserta baru atau ditulis kembali sebagai pengukuran sensor pada CSV sumber. SMOTE yang gagal tidak diam-diam diganti ROS. [Panduan imbalanced-learn](https://imbalanced-learn.org/stable/common_pitfalls.html).

Periksa domain fitur setelah resampling, termasuk rasio dan batas numerik. Lolos batas numerik tidak membuktikan validitas fisiologis sampel sintetis. Bila precision Low memburuk terhadap kontrol batch-02, jangan menganggap recall yang naik saja sebagai keberhasilan.

Pengimbangan kontribusi blok rekaman yang sangat tidak merata dapat menjadi analisis sensitivitas tambahan, bukan default atau model personal. Cara tersebut tidak memasukkan ID sebagai fitur dan harus dipisahkan dari pembobotan kelas agar pengaruhnya tidak tercampur.

### Exp07: perbandingan algoritma terbatas

Mulai dari keluarga yang masuk akal pada data tabular kecil, bukan mengulang pencarian besar enam algoritma pada seluruh kombinasi fitur.

| Model | Bank awal yang diusulkan |
|---|---|
| Logistic | C = 0,1 / 1 / 10 |
| SVM RBF | C = 0,1 / 1 / 10; gamma = scale / 0,1 |
| XGBoost | Kontrol fixed; enam kombinasi depth 2/3/5 dan learning rate 0,03/0,10, 300 pohon |
| HistGradientBoosting | Enam kombinasi learning rate 0,05/0,10 dan max_leaf_nodes 7/15/31; 300 iterasi |

Semua parameter lain dibekukan, termasuk regularisasi/minimum leaf yang disepakati pada implementasi. Kontrol fixed disimpan sebagai kontrol, bukan selalu ditambahkan ke tuning bila efektif sama dengan kandidat yang sudah ada. GPU tidak menjadi kebutuhan awal.

HGB merupakan kandidat baru, bukan klaim akan lebih baik. Matikan early stopping acak bawaan untuk perbandingan awal. Bila early stopping diperlukan, validation-nya harus berasal dari grup training terpisah, bukan baris acak atau outer test. [Dokumentasi HGB](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html).

Data scaling, missing-value handling, dan seleksi fitur harus berada pada pipeline training yang benar. Jangan menafsirkan keberhasilan algoritma ketika jumlah fitur, cohort test, dan aturan preprocessing ikut berubah tanpa kontrol.

## 8. Opsi lanjutan yang tidak wajib dijalankan

### Exp08: late fusion yang dapat dievaluasi

Bandingkan satu early-fusion reference dengan dua classifier cabang HRV/eye. Bobot gabungan dipilih di inner validation. Jangan menyamakan decision score SVM dengan probabilitas.

Jika probabilitas dikalibrasi, gunakan prediksi out-of-group dari training untuk kalibrasi, tanpa label outer test. Biaya fitting dan split tambahan dihitung dalam budget. Simpan hasil masing-masing cabang untuk mengetahui apakah fusi hanya mengulangi sensor dominan. [Dokumentasi calibration](https://scikit-learn.org/stable/modules/calibration.html).

Cabang ini dilanjutkan bila kedua sensor menyediakan informasi yang dapat dipakai pada inner validation. Jika satu cabang gagal kualitas atau fusi tidak memperbaiki inner score, model sederhana tetap menjadi kandidat. Jangan menambah fusi hanya agar terlihat multimodal; kontribusi sensor harus dilaporkan dari hasil.

### Exp09: target berurutan sebagai hipotesis tambahan

Low/Medium/High mempunyai urutan, tetapi belum tentu jarak yang sama. Kandidat ordinal supervised dapat menguji keputusan kumulatif di atas Low dan di atas Medium, lalu membentuk probabilitas tiga kelas yang valid.

Gunakan kontrol multiclass dengan fitur dan test sama. Tetapkan penanganan probabilitas kumulatif yang tidak monoton sebelum evaluasi. Jangan mengganti target menjadi skor buatan atau membalik label agar HRV mengikuti arah tertentu. Ini berbeda dari novelty detection batch-00 maupun Low/Elevated dua tahap batch-01.

Prioritasnya lebih rendah daripada data, representasi, dan window. Klaim manfaat harus tetap berdasarkan macro-F1 dan seluruh kelas, bukan hanya berkurangnya kesalahan Low-ke-High.

## 9. Exp10: seleksi akhir dan kriteria hasil

### 9.1 Recipe yang dipilih harus end-to-end

Recipe mencakup versi dataset, fitur/normalisasi, window, riwayat, missing-value handling, sampler/weight, model/parameter, serta aturan inference. Menyeleksi hanya estimator setelah fitur dipilih dari outer test bukan nested selection yang lengkap.

Sebelum full run, tulis registry final yang berisi paling banyak **24 recipe efektif** dari rencana inti dan cabang yang eligible. Isi tuple lengkapnya ditetapkan pada plan implementasi setelah audit D1; tidak diisi dari pemenang outer leaderboard. Cabang opsional yang masuk menggantikan slot budget, bukan memperluas pencarian tanpa batas.

Kelayakan berdasarkan ketersediaan data, domain sensor, dan kompatibilitas inference dipisahkan dari pemilihan berbasis metrik. Jika ada keputusan berbasis skor, keputusan itu dilakukan di inner training pada tiap outer fold. Bila tim mengubah bank setelah melihat outer score, beri versi baru dan laporkan sebagai eksplorasi tambahan, bukan pengujian final yang belum tersentuh.

Pilih recipe di inner, latih pada seluruh training outer, lalu evaluasi sekali pada outer test. Setelah pelaporan OOF selesai, lakukan pemilihan group-validation pada seluruh data yang disepakati dan refit untuk artefak penggunaan. Skor pemilihan refit bukan skor generalisasi baru. [Nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

Untuk artefak penggunaan, lakukan seleksi melalui group-validation pada seluruh D1 dengan prinsip bagian 5.2. Rekaman lama dan baru mempunyai peran yang sama: semua sesi orang validasi dikeluarkan dari fitting. Setelah recipe terpilih, fit pada seluruh D1 dan simpan `selection_protocol`. Tidak ada jalur refit khusus cohort lama. Skor pemilihan pada tahap ini tidak menggantikan hasil outer OOF.

### 9.2 Definisi hasil yang lebih baik

- Macro-F1 lebih tinggi daripada kontrol atau kandidat lain pada evaluasi batch-02 yang sama, dengan selisih dan ketidakpastian dilaporkan.
- Recall/precision Low tidak dibeli dengan runtuhnya Medium/High; sajikan perubahan per kelas.
- Hasil tidak hanya berasal dari satu rekaman atau dari penghapusan endpoint sulit.
- Coverage sensor, warm-up, dan kebutuhan baseline dapat dipenuhi pada penggunaan yang didefinisikan.
- Model tidak membutuhkan label, identitas orang, atau konteks soal untuk memprediksi.
- Kesimpulan membandingkan konfigurasi batch-02; tidak mengklaim peningkatan sebelum–sesudah penambahan Low atau kemenangan atas batch sebelumnya.

Jika selisih kecil dan tidak meyakinkan, laporkan tie/ketidakpastian dan pertimbangkan recipe yang lebih sederhana, stabil, serta mudah dijalankan. Jangan memaksa kesimpulan menang. Tidak ada janji accuracy 70%, 80%, atau angka lain sebelum bukti tersedia.

### 9.3 Batas yang belum terselesaikan

Tambahan Low-only tidak menciptakan test tiga kelas independen pada sesi baru, tidak menghapus confounding urutan tugas lama, dan tidak memvalidasi label CL secara independen. Perubahan kondisi perekaman baru dapat tetap memengaruhi sensor walaupun metadata tidak menjadi fitur.

Pengujian coding/debugging di luar CodeArrange, validasi target dengan ukuran independen, dan konfirmasi pada cohort baru tiga kelas merupakan pekerjaan sesudah bukti internal memadai. Tidak memaksakan pengambilan medium/high ulang sekarang bukan berarti batas tersebut hilang.

## 10. Pengendalian jumlah eksperimen dan biaya

Perubahan utama dievaluasi dengan model fixed lebih dahulu. Tuning luas tidak dijalankan pada setiap ablation, panjang window, panjang riwayat, dan sampler sekaligus.

Sebelum menjalankan full notebook, tampilkan jumlah grup, outer/inner fold, kandidat efektif, estimasi fit, serta dependency. Kandidat dengan data/fit yang sama dapat berbagi cache hanya bila hash data, registry grup, window, fitur, parameter, seed, dan environment cocok.

Bank final 24 recipe dengan tiga inner fold pada N orang unik D1 memerlukan sampai 72 × N fit kandidat ditambah N refit outer, sebelum kalibrasi/cabang fusi dan seleksi refit penggunaan. Misalnya, N=18 memerlukan sampai 1.296 fit kandidat dan 18 refit outer. Ini ilustrasi biaya, bukan jumlah peserta final atau training yang sudah dilakukan. Audit menentukan N dari orang unik, bukan jumlah sesi tambahan. Budget notebook rinci harus menghitung cabang dan biaya ulang fitur, bukan hanya jumlah estimator di tabel.

Seed utama dibekukan, misalnya 42. Seed tambahan hanya untuk sensitivitas, bukan memilih skor tertinggi. Jika budget habis, simpan hasil parsial berstatus belum lengkap dan jangan membandingkannya dengan full run seolah-olah selesai.

## 11. Rancangan implementasi dan keluaran

### 11.1 Notebook dan dependensi

Setiap keluarga eksperimen mendapat notebook biasa di foldernya sendiri. Tidak ada notebook atau command Run All lintas eksperimen. Operator boleh memakai Run All di dalam notebook yang sedang dikerjakan setelah dependensinya terpenuhi.

Siapkan satu paket batch-02 yang memuat notebook, shared code versi batch-02, serta satu requirements tervalidasi. Paket diunggah sekali; tidak ada kewajiban mengganti shared/requirements untuk setiap eksperimen. Tempatkan kode di `models/batch-02/shared`. Tim sudah menghapus `models/shared` lama; batch-02 tidak boleh bergantung pada folder tersebut atau menganggap kode batch-01 masih tersedia.

Notebook full disiapkan untuk eksekusi penelitian. Pemeriksaan teknis kecil dan tes sintetis dilakukan pengembang sebelum serah-terima, bukan kewajiban operator menjalankan seluruh notebook dua kali smoke/full. Tidak menjanjikan bebas semua error eksternal Colab, tetapi validasi input, dependency, resume dan kegagalan harus diuji.

Notebook dan script diagnosis batch-01 masih merujuk kode shared serta setup root yang sudah dihapus. Keduanya disimpan sebagai arsip, bukan paket runnable atau dependency batch-02. Gunakan laporan, konfigurasi, dan keluaran tersimpan sebagai catatan metode; implementasi batch-02 memerlukan setup mandiri beserta tesnya sendiri.

### 11.2 Path lokal dan Drive

Root Drive yang dipakai tim:

`drive/MyDrive/Penelitian-Adaptive-IDE`

Root dataset:

`drive/MyDrive/Penelitian-Adaptive-IDE/Dataset`

Pada implementasi, letakkan snapshot/version dataset di bawah root tersebut dengan hash yang jelas. Dataset aktif boleh bernama `good_acceptable.csv`, tetapi setiap run harus menunjuk versi dan hash tertentu. Salinan D0 harus sudah aman sebelum path aktif diperbarui.

Struktur berikut merupakan rancangan, bukan file yang sudah dibuat:

```text
models/batch-02/
  PLAN.md
  shared/
  requirements.txt
  registry/
  exp00_data_audit/
  exp01/ ... exp10/
  outputs/expXX/<run_id>/
```

Root output Drive direncanakan `drive/MyDrive/Penelitian-Adaptive-IDE/models/batch-02/outputs`. Notebook menerima konfigurasi root agar tidak menyisipkan path mesin tertentu ke banyak cell.

### 11.3 Artefak wajib

Setiap run menyimpan config, status, hash data/kode/environment, manifest grup/split, allowlist, statistik preprocessing, kandidat/pilihan inner, prediksi OOF, metrik, confusion matrix dan grafik. Untuk window/riwayat tambahkan coverage, alasan unavailable, warm-up, dan audit prefix/endpoint.

Untuk provenance, simpan tabel sumber lama/tambahan tim/peserta kembali dan bukti seluruh sesi orang held-out dikeluarkan. Tabel sumber hanya untuk audit, bukan kondisi eksperimen sebelum–sesudah. Untuk baseline/quality, simpan keputusan dan fallback tanpa menganggap nilai kosong sebagai nol sensor.

CSV/JSON/Markdown dan figure dapat masuk Git sesuai [aturan models](../.gitignore). Bobot model/cache tidak masuk Git dan dibagikan melalui penyimpanan artefak bila diperlukan. Metadata identitas tetap memerlukan kontrol akses; alias grafik tidak menganonimkan seluruh hasil.

## 12. Validasi sebelum notebook diserahkan

| Area | Uji minimal |
|---|---|
| Identitas | Orang yang sama dengan beberapa sesi/alias tetap satu grup; salinan sesi tidak dihitung dua kali |
| Split | Seluruh orang D1 bergiliran test; tidak ada identitas/window overlap train/test; seluruh sesi orang held-out dikeluarkan dari inner dan outer training |
| Data | Hash/target/schema D1 konsisten, NaN tetap NaN, sumber dan arsip tidak ditimpa |
| Dataset terbaru | Sumber lama dan Low tambahan yang valid masuk D1; baseline hanya memakai D1; tidak ada kewajiban training D0 atau test cohort lama |
| Preprocessing | Median/scaler/seleksi hanya fit training; fitur seluruhnya kosong/konstan ditangani eksplisit |
| Window | Rumus RR/pasangan/gap, batas tepat awal/akhir, availability data terlambat, alignment eye, dan label mixed diuji |
| Riwayat | Tidak melintasi stream/gap; prefix invariance; batch dan streaming konsisten; tidak membutuhkan ID soal/label |
| Balancing | Hanya training, kandidat no-op/invalid jelas, validation/test tidak disintesis |
| Seleksi | Tidak memakai outer score; tie-break deterministik; recipe lengkap; cache tidak tertukar |
| Evaluasi | Metrik dapat dihitung ulang dari prediksi; absent class dan abstention ditangani; partial run tidak dianggap complete |
| Inference | Input minimal sensor berhasil tanpa nama/soal; baseline/histori optional mempunyai kontrak dan fallback |
| Reproduksi | Notebook dapat dijalankan dari runtime bersih dengan satu paket dependency; resume tidak menggabungkan identitas run berbeda |

## 13. Serah-terima ke tim atau LLM berikutnya

1. Baca [konteks penelitian](../README.md#konteks-penelitian-dan-peran-kolom), [ringkasan batch-01](../batch-01/SUMMARY.md), dan [diagnosis](../batch-01/DIAGNOSIS.md) sebelum mengubah pipeline.
2. Konfirmasi penerimaan sekitar lima anggota tim baru dan sesi easy peserta lama yang benar-benar tersedia. Jangan mengarang identitas/kelas untuk melengkapi tabel.
3. Bekukan D0 dan buat registry lintas sesi sebelum rebuild. Jangan merusak reproduksi batch-01 ketika CSV aktif diganti.
4. Selesaikan tahap A–B, lalu turunkan rencana ini menjadi definisi notebook, registry kandidat, dan tes implementasi yang lengkap. Belum ada otorisasi dari dokumen ini untuk menganggap kode atau training sudah selesai.
5. Jalankan baseline D1 pada exp01 sebelum menilai perubahan lain. Gunakan manifest batch-02 yang sama antarkonfigurasi; jangan menambahkan evaluasi sebelum–sesudah atau pembandingan batch-00/batch-01.
6. Jangan menambahkan ID/difficulty/set/time limit ke fitur untuk mengejar accuracy. Peserta dipakai untuk menjaga pemisahan data, bukan sasaran classifier.
7. Jangan menghapus peserta tanpa Low, mengubah protokol 5/12/15 menit, mengganti label, atau melonggarkan mutu sensor diam-diam untuk meningkatkan skor.
8. Jangan menjalankan lagi 12 eksperimen lama tanpa perubahan yang jelas, membuat launcher Run All lintas eksperimen, atau mengubah source batch-01 demi batch baru.
9. Catat keputusan, kandidat yang tidak eligible, hasil negatif, dan perubahan versi. Model optimal berarti terbaik dalam cakupan pengujian yang sah, bukan skor maksimum setelah banyak percobaan pada test yang sama.

## Referensi metode

Rencana memakai bukti lokal pada diagnosis dan kode sumber yang ditautkan, serta dokumentasi primer untuk batas metode. Dokumentasi web dapat lebih baru daripada environment arsip; versi dependency implementasi nanti harus dipin dan diuji, bukan mengikuti versi terbaru secara buta.

- [Grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data): pemisahan grup tidak sama dengan memasukkan identitas sebagai fitur.
- [Nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html): pemilihan parameter/prosedur dipisahkan dari test luar.
- [Resampling pitfalls](https://imbalanced-learn.org/stable/common_pitfalls.html): resampling seluruh dataset sebelum split menyebabkan kebocoran.
- [Muñoz dkk., validitas HRV ultra-short](https://pmc.ncbi.nlm.nih.gov/articles/PMC4586373/) dan [Burma dkk., reliabilitas/validitas HRV ultra-short](https://pubmed.ncbi.nlm.nih.gov/33856258/): durasi perlu dinilai menurut fitur dan kondisi, bukan diasumsikan universal.
- [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html): score dan probabilitas tidak dapat dipertukarkan tanpa dasar.

Tidak ada hasil eksperimen batch-02 pada dokumen ini. Pemilihan kandidat, kelayakan window/baseline, dan jumlah fold evaluasi seluruh D1 baru dapat difinalkan setelah dataset serta registry sesi tersedia.
