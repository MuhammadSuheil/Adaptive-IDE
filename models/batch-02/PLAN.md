# Batch-02: Rencana Eksperimen Cognitive Load

Tanggal revisi: 16 November 2026.

## 1. Status, tujuan, dan syarat mulai

**Full training batch-02 belum dimulai.** Notebook exp00–exp14 dan paket kode bersama telah disiapkan, beserta pemeriksaan teknis tanpa fitting classifier penelitian. Bobot dan hasil evaluasi batch-02 belum tersedia. Baca [README](README.md) dan [catatan validasi](technical_validation/CHECKS.md) untuk penggunaan dan batas pengujian.

Batch ini menghasilkan satu classifier umum Low/Medium/High dari fitur eye tracking dan HRV per window. Identitas hanya membantu pengelompokan evaluasi dan penelusuran sumber. Model tidak menerima nama peserta, difficulty, soal, set, fase, tanggal pengambilan, atau batas waktu CodeArrange sebagai predictor. Model temporal boleh memerlukan waktu dan kunci stream yang tidak mengandung arti identitas.

Tiga kandidat utama adalah good_acceptable v2 pada window 15, 25, dan 45 detik, semuanya step 5 detik. Ketiganya merupakan representasi alternatif dari sumber yang sama, bukan tiga kumpulan data yang boleh ditumpuk untuk memperbanyak sampel.

### 1.1 Gate penerimaan dataset

Training penelitian hanya boleh berjalan setelah empat kondisi berikut terpenuhi:

| Syarat | Status saat revisi ini | Bukti penerimaan |
|---|---|---|
| Builder dan integritas dataset diperbaiki | Selesai untuk sumber yang tersedia | Manifest v2, tes builder, validasi sensor/label/subset/checksum |
| Pengambilan tambahan selesai dan tergabung | Dua rekaman tambahan tergabung; keputusan menutup pengambilan belum ditetapkan | Ihsan dan bangsur_easy_medium masuk manifest; bangsur_easy_only dikecualikan karena tidak terhubung ke CodeArrange |
| Proporsi kelas setelah penambahan ditinjau dan disetujui tim | Belum selesai | Jumlah kelas, rasio terbesar/terkecil, durasi unik, dan cakupan rekaman pada ketiga window |
| Input batch-02 dibekukan | Belum selesai | Release dataset, registry identitas, hash tiga CSV/allowlist, serta manifest split |

Simpan keputusan tersebut dalam `registry/dataset_release.json`: status setiap gate, daftar sesi tambahan, tanggal penerimaan, hash manifest/CSV/allowlist, distribusi aktual, dan persetujuan penanggung jawab data. Default gate tambahan, proporsi, dan pembekuan adalah pending; notebook full harus menolak gate yang belum approved.

Peninjauan proporsi tidak mensyaratkan rasio persis 1:1:1 dan tidak boleh dilakukan dengan membuang data layak dari dataset utama. Penambahan yang direncanakan mencakup easy dan, bila tersedia, medium. Catat hasil pengambilan aktual; jangan menganggap seluruh data baru otomatis Low atau menjanjikan jumlah peserta tertentu. Resampling training merupakan eksperimen terpisah setelah gate ini selesai.

Audit input, tes sintetis, dan prototipe teknis boleh dikerjakan lebih dahulu. Skornya tidak masuk hasil penelitian batch-02.

### 1.2 Acuan dataset terkini, setelah tambahan 16 November 2026

| good_acceptable | Low | Medium | High | Total | Orang |
|---|---:|---:|---:|---:|---:|
| 15 detik / step 5 | 1.062 | 1.266 | 1.251 | 3.579 | 19 |
| 25 detik / step 5 | 984 | 1.093 | 1.137 | 3.214 | 19 |
| 45 detik / step 5 | 908 | 728 | 891 | 2.527 | 18 |

Rasio kelas terbesar/terkecil masing-masing sekitar 1,192; 1,155; dan 1,247. Angka ini belum menjadi release final batch-02. Medium masih merupakan kelas terkecil pada varian 45 detik; kebutuhan pengambilan berikutnya tidak boleh dirancang sebagai Low-only. Proporsi tidak perlu dipaksa sama dengan membuang window yang layak.

Sumber yang diproses saat revisi ini mencakup 21 rekaman dari 20 orang. Tambahan berasal dari `ihsan` dan `bangsur_easy_medium`; Ihsan berbeda dari Ikhsan. Kedua folder Bangsur tetap memakai satu identitas orang. Sesi `bangsur_easy_only` dengan ID `20261116_151757_P01_coding_task` tidak memiliki ledger aktivitas karena operator belum menghubungkan CodeArrange, sehingga dikecualikan melalui `session_exclusions` tanpa mengubah sumber atau membuat label pengganti.

Semua window tambahan yang memenuhi aturan dipertahankan, tanpa kuota per kelas maupun panjang window. Admission v2 memakai kualitas window tanpa pembatasan daftar nama. `team_review_group` hanya metadata historis. Ambang kualitas tetap; validasi gaze asli tetap terpisah dari tier kelengkapan/clipping. Dua rekaman tambahan yang dipakai mempunyai validasi gaze terakhir diterima manual, tetapi tidak melewati ambang validasi. Rebuild tidak memperbaiki akurasi pengukuran gaze yang gagal.

Penggabungan ini memperbarui sembilan view dataset dan dokumentasinya, bukan hasil model. Gate pada `registry/dataset_release.json` tetap pending sampai tim menetapkan kelengkapan pengambilan, menerima proporsi/kualitas, dan membekukan input. Tidak ada training atau persetujuan release otomatis.

Baca [pilihan dataset](../../analysis_sessions/dataset_fusion/output/README.md) dan [laporan v2](../../analysis_sessions/dataset_fusion/output/v2_session_phase/QUALITY_REPORT.md) untuk definisi fitur, jumlah seluruh view, serta prosedur menambah sumber. Perbarui bundle aktif dengan builder `--replace`, sesuai prosedur yang sudah diuji. V1 tetap arsip.

## 2. Dasar revisi dan penggunaan eksperimen sebelumnya

Bukti acuan: [batch-00 lama](../batch-00/old-dataset/SUMMARY.md), [batch-00 pada sembilan dataset v2](../batch-00/new-dataset/SUMMARY.md), [batch-01 lama](../batch-01/old-dataset/DIAGNOSIS.md), serta [batch-01 v2](../batch-01/new-dataset/SUMMARY.md) dan [diagnosis lanjutannya](../batch-01/new-dataset/DIAGNOSIS.md).

Evaluasi batch-01 v2 menggunakan snapshot sebelum pembaruan admission terbaru: 3.011/2.711/2.118 row dan 13 orang. Hasil tersebut bukan hasil model pada tiga CSV aktif di tabel bagian 1.2. Runtime, beberapa script audit, dan sebagian artefak pendukung lama telah dibersihkan; laporan dan output yang masih tersedia menjadi catatan metode, bukan dependency runnable batch-02.

| Temuan yang relevan | Keputusan batch-02 |
|---|---|
| Macro-F1 nested v2 sebelumnya sekitar 0,314/0,328/0,355 | Buat baseline baru setelah release final; jangan memakai skor arsip sebagai baseline dataset terbaru |
| Reset/panjang history tanpa sensor mencapai macro-F1 sekitar 0,475 pada 25 detik | Uji kebebasan fitur dari metadata; history harus berasal dari aliran sensor sebelum filter target |
| XGBoost 25 detik: history soal/fase sekitar 0,363, history aliran sekitar 0,304 | Pertahankan uji temporal, tetapi jangan menyalin reset soal/fase yang lama |
| Menghapus rasio baseline membantu kedua kontrol pada tiga window | Jadikan kontrol tanpa rasio wajib; jangan menghapus rasio dari CSV sumber |
| Tahap Medium–High tetap lemah | Tinjau ketiga kelas, bukan menganggap penambahan Low menyelesaikan seluruh kesalahan |
| Accuracy training XGBoost history sekitar 94–98%, test sekitar 34% | Uji regularisasi, representasi, learning curve, dan model sederhana sebelum menambah kapasitas |
| Perbedaan antarrekaman dan kegagalan validasi gaze masih ada | Uji kelompok sensor dan fitur relatif/dinamis tanpa menghapus peserta dari hasil model |

Eksperimen lama boleh diulang atau dikombinasikan bila hipotesis dan implementasinya sesuai kontrak baru. Yang dipertahankan: kontrol tabular, ablation sensor, rasio baseline, seleksi fitur training-only, class weight, ROS/SMOTE, fusion, dan klasifikasi supervised bertahap. Yang tidak disalin: split row acak/blok tanpa pemisahan rekaman, imputasi sebelum split, rolling lintas sesi atau history yang direset oleh soal/fase, pemilihan dari outer leaderboard, serta anggapan Low adalah anomaly tanpa evaluasi tiga kelas end-to-end.

Batch-02 mengevaluasi dataset terbaru secara mandiri. Tidak ada kewajiban melatih ulang v1, mengalahkan skor batch lama, atau melakukan eksperimen sebelum–sesudah penambahan data. Membandingkan konfigurasi di dalam batch-02 tetap wajib.

## 3. Kontrak data dan input model

### 3.1 Target dan pemisahan peran kolom

Gunakan `cognitive_load_label` langsung, dengan urutan kelas tetap Low=0, Medium=1, High=2. Pertahankan protokol v2: active easy/practice/demo dan fase pre-task setelah kalibrasi, feedback, rating, rest, post-task/interrupted yang terlacak menjadi Low; active medium menjadi Medium; active hard menjadi High. Window berlabel harus mempunyai target tunggal sepanjang interval. Kalibrasi/tidak terlacak dan campuran target tidak menjadi row supervised.

Allowlist awal tetap 26 fitur sensor. Metadata sumber tetap berada dalam CSV/registry, terpisah dari matriks predictor. Jangan membuat ulang target dari difficulty. Jangan mengubah label berdasarkan nilai HRV, prediksi, kemampuan peserta, atau asal kelompok pengambilan.

`all_data` dan `good_only` dipakai untuk audit/sensitivitas kualitas, bukan otomatis memperluas sembilan jalur training. Eksperimen utama menggunakan ketiga good_acceptable final.

### 3.2 Registry, identitas, dan pembekuan

Buat registry sesi dengan session ID, ID orang lintas sesi yang dikonfirmasi operator, folder sumber, asal gelombang pengambilan, perangkat, dan baseline. Beberapa sesi orang yang sama tetap satu grup, termasuk peserta yang kembali. Token P01 dalam session ID bukan ID orang yang unik.

Gunakan [inventory sumber](../../analysis_sessions/session_inventory.py), lalu lengkapi alias yang dikonfirmasi untuk sesi tambahan. Jangan menebak identitas dari kemiripan sensor. Tidak adanya Low/Medium/High pada seseorang bukan alasan membuangnya.

Sesudah gate lulus, salin input yang dipakai ke snapshot run batch-02: tiga CSV, allowlist, manifest, registry, dan hash sumber turunan. Ini snapshot input pemodelan, bukan folder versi dataset publik tambahan. Builder boleh memperbarui path v2 aktif pada kesempatan berikutnya; run yang sudah dimulai harus membaca snapshot yang dibekukan. Perubahan hash menolak resume atau menghasilkan run ID baru.

### 3.3 Aliran sensor sebelum filter target

CSV supervised saja tidak cukup untuk membuat history yang bebas dari pola filtering label. Sediakan cache turunan seluruh grid sensor dari RR/eye mentah untuk setiap sesi dan panjang window:

- Satu row per endpoint 5 detik, termasuk endpoint yang tidak mempunyai label tunggal.
- Kolom input: kunci stream, timestamp/available_at, fitur sensor yang sama, serta validitas sensor yang diperlukan untuk masking nilai.
- Label dan event CodeArrange disimpan di sidecar untuk joining supervised dan audit; keduanya tidak boleh mengatur pembentukan fitur.
- Rekonstruksi memakai rumus, origin, endpoint, paket, gap/recovery, serta masking sensor builder v2. Fitur pada window yang sama harus cocok dengan CSV published dalam toleransi numerik builder.
- Buat cache sekali per hash sumber/konfigurasi/kode. Sumber mentah tidak diubah, dan cache tidak ditulis ke CSV dataset utama.

Lapisan feature builder menerima stream sensor, bukan dataframe yang sudah disaring menurut label. Setelah fitur statis/temporal selesai, gabungkan label dan pilih endpoint eligible untuk fitting/evaluasi. Gap karena target campuran tidak boleh disamakan dengan sensor terputus.

Jika RR/eye mentah atau cache seluruh grid yang terverifikasi tidak tersedia, jalur history/sequence berstatus `skipped_missing_stream`; jangan menggantinya dengan rolling atas CSV filtered. Jalur stateless tetap dapat diuji, tetapi paket batch final harus mencatat cabang yang belum dievaluasi.

## 4. Evaluasi, pemilihan, dan perbandingan window

Istilah yang dipakai pada seluruh eksperimen:

| Istilah | Arti operasional |
|---|---|
| Recipe | Satu konfigurasi lengkap: fitur, transformasi, history, balancing, estimator, parameter, dan aturan prediksi |
| Outer fold | Satu putaran evaluasi: seluruh rekaman satu orang menjadi test, orang lain menjadi training |
| Inner fold | Pemisahan grup di dalam outer training untuk memilih recipe tanpa membaca outer test |
| OOF | Prediksi pada row yang tidak dipakai melatih model pembuat prediksi tersebut; pooled outer OOF menggabungkan hasil seluruh outer fold |
| Fitting partition | Bagian data yang benar-benar dipakai mempelajari statistik preprocessing atau bobot model pada tahap terkait |

### 4.1 Pemisahan grup

Evaluasi utama memakai leave-one-participant-out pada seluruh orang unik release final. Semua sesi orang held-out dikeluarkan dari fitting. Manifest grup induk merupakan union grup dari ketiga window; setiap window menyimpan daftar train/test window ID-nya sendiri.

Outer fold dengan nol row test pada suatu window diberi `no_test_support`, tidak diberi skor nol dan tidak dihitung sebagai fold teruji. Orang tersebut tetap tercatat pada manifest dan dapat mempunyai support di window lain. Pool OOF tiap window harus mencakup setiap row eligible tepat sekali.

Pada outer training, gunakan StratifiedGroupKFold tiga fold, shuffle=True, seed=42. Bekukan split dan pakai untuk semua kandidat pada window/outer fold yang sama. Setiap fitting partition wajib mempunyai tiga kelas. Jika tiga fold tidak feasible, coba dua fold dengan seed yang sama; jika tetap tidak feasible, hentikan full run dan perbaiki kontrak data. Jangan menggantinya dengan split row acak.

Kelas absent pada validation/test bukan kegagalan: simpan support dan hitung skor seleksi dari pooled inner OOF dengan tiga label tetap. Median/scaler/seleksi fitur/sampler/kalibrasi/early stopping hanya memakai training yang sesuai.

### 4.2 Metrik

Utama: pooled OOF macro-F1 tiga kelas. Wajib melengkapi dengan accuracy, balanced accuracy, precision/recall/F1/support tiap kelas, confusion matrix, log loss bila probabilitas tersedia, serta hasil per stream/orang untuk audit.

Tampilkan metrik pada kelas yang memiliki support; metrik per kelas absent bernilai kosong. Bedakan macro-F1 fixed-three-class pooled dari rata-rata fold/per-orang dengan kelas yang hadir.

Laporkan train versus test, distribusi prediksi, dan kesalahan Medium–High serta Low–High. Untuk Low, tambahkan recall pada pengerjaan saja, campuran pengerjaan/nonpengerjaan, dan nonpengerjaan saja, berdasarkan segmen berdurasi positif. Pengelompokan ini hanya audit, bukan predictor atau perubahan label.

Bandingkan prediksi dua recipe pada test yang sama. Gunakan paired bootstrap 2.000 kali atas grup orang, seed=42, untuk interval 95% dan selisih skor. Bootstrap ini bersyarat pada prediksi OOF yang tersedia; bukan pengganti pengulangan pengambilan atau validasi eksternal.

### 4.3 Dua laporan window yang tidak boleh dicampur

| Laporan | Endpoint dan training | Penggunaan |
|---|---|---|
| Native support | Seluruh row good_acceptable masing-masing window; training juga sesuai window | Menilai classifier pada cakupan dataset window tersebut |
| Paired window | Intersection endpoint absolut yang eligible pada 15/25/45 detik dan labelnya sama | Membandingkan window pada target/test yang sama |

Intersection memakai pasangan session ID dan window_end yang dinormalisasi UTC, bukan window_id yang mencakup panjang interval. Bentuk dari metadata sebelum membaca skor. Simpan jumlah orang, kelas, durasi, dan proporsi retained terhadap masing-masing native support.

Pada exp04, jalankan dua kontrol fixed dengan train dan test sama-sama dibatasi intersection untuk mengisolasi perubahan representasi. Pada seleksi akhir lintas window, training boleh memakai seluruh row eligible kandidat, sedangkan validation/test tetap intersection; tandai bahwa kuantitas training ikut berbeda.

**Jangan menghitung window yang dikeluarkan karena campuran target sebagai kesalahan model atau sensor unavailable.** Itu kehilangan label evaluasi yang tunggal, bukan bukti sensor tidak mampu menghasilkan fitur. Ukur physical feature availability dari stream lengkap menggunakan kualitas sensor saja. Jangan memakai label/difficulty untuk mengatur abstention saat inference.

Simpan sensor coverage dan labelable coverage secara terpisah, per kelas bila label acuan tersedia. Evaluasi prediksi pada transisi memakai target acuan 25 detik hanya boleh menjadi analisis tambahan yang diberi nama anchored-target, bukan dicampur dengan skor native 45 detik.

### 4.4 Aturan seleksi dan penyampaian hasil

Seluruh pilihan berbasis metrik terjadi di inner validation, termasuk fitur, window, history, sampler, parameter, fusion, dan jenis estimator. Exp01–exp13 memberi hasil keluarga/ablation; baris outer tertinggi di antaranya bersifat eksploratif. Exp14 menilai prosedur pemilihan recipe lengkap.

Bekukan registry kandidat dan aturan pencarian sebelum membuka outer results. Bila registry diubah setelah hasil outer dilihat, beri protocol ID baru dan sebut eksplorasi lanjutan, bukan evaluasi final yang belum pernah dipakai memilih.

Primary rank memakai pooled inner macro-F1. Recipe dalam 0,005 dari skor inner terbaik menjadi tie set operasional; pilih tanpa baseline wajib, lalu jumlah fitur lebih sedikit, tanpa history/riwayat lebih pendek, jumlah estimator lebih sedikit, lalu recipe ID alfabetis. Ambang 0,005 adalah aturan kesederhanaan yang ditetapkan sebelum run, bukan uji signifikansi.

Panjang konteks pada tie-break mencakup centering N3 (60 detik) dan GRU (30 detik), bukan hanya kolom history tabular. Pada seleksi lintas window, bila seluruh kriteria termasuk recipe ID sama, pilih window lebih pendek. Aturan ini dibekukan sebelum hasil batch-02 tersedia.

## 5. Preprocessing dan representasi

### 5.1 Kontrol bersama

- Fit seluruh transformasi yang belajar statistik hanya di fitting partition. Jangan menghitung mean, quantile, PCA, MI, atau korelasi dari gabungan train dan test.
- Drop kolom seluruhnya kosong dan kolom konstan menurut training; simpan nama/keputusannya. Jangan menjatuhkan row test untuk menyembunyikan missing baseline.
- Jalur median mengisi nilai dari median training; StandardScaler memakai statistik training. RobustScaler memakai median/IQR training, quantile_range=(25,75).
- Nilai sensor sumber tetap NaN; nol tidak menggantikan pengukuran yang tidak tersedia. Jika tidak ada fitur tersisa, kandidat gagal eksplisit.
- Kontrol native-NaN pada XGBoost/HGB/CatBoost menjadi recipe terpisah; fitur dan test tetap sama. Jangan sekaligus mengubah sensor dan missing handler tanpa kontrol.
- Tidak ada outlier removal berdasarkan seluruh dataset. Winsorization hanya eksperimen training-derived: quantile 1%/99% training, satu recipe kontrol, tanpa mengubah CSV.
- Missing indicator hanya eksperimen terpisah, dibuat dari fitur sensor yang NaN, bukan metadata kualitas/aktivitas. Simpan kontrol missing-only untuk mengetahui apakah prediksi bertumpu pada pola ketersediaan.

### 5.2 Kelompok fitur

| Kode | Definisi |
|---|---|
| F0 | Semua 26 fitur pada allowlist |
| F1 | F0 tanpa blink_count dan eye_state_other_ratio; konstanta training ditangani terpisah |
| F2 | F1 tanpa seluruh fitur eye berakhiran _mean; menguji statistik variasi yang lebih sedikit membawa offset geometris |
| F3 | F1 tanpa gaze_x/y dan seluruh proporsi eye_state |
| F4 | RMSSD, SDNN, rasio baseline, blink rate, mean/std EAR kiri/kanan |
| F5 | Tiga fitur HRV saja |
| F6 | Eye F1 saja, tanpa HRV |
| F7 | F1 tanpa keempat fitur iris |

Rasio baseline yang tidak tersedia tetap kosong. Untuk setiap kelompok yang memuat HRV, buat pasangan dengan/tanpa rmssd_baseline_ratio; eye-only tidak diduplikasi. Simpan daftar fitur efektif per fold. iris_size_delta_std tidak dibuang otomatis hanya karena keputusan pada dataset lama; konstansi/seleksi diperiksa dari training terbaru.

Pilihan normalisasi:

- N0: nilai sumber.
- N1: tanpa rasio baseline, tetap RMSSD/SDNN absolut.
- N2: log1p pada RMSSD/SDNN dan log rasio positif yang baseline-nya valid. Nilai rasio nol pada CSV sumber tetap nol; hanya fitur turunan log yang menjadi NaN karena log(0) tidak terdefinisi. Rasio yang memang tidak valid tetap NaN.
- N3: centering kausal pada mean geometris eye memakai median trailing 60 detik sebelum endpoint sekarang, minimal tiga nilai valid. Belum cukup riwayat menghasilkan NaN pada transformasi relatif; fitur absolut tetap tersedia.

N3 menggunakan aliran sebelum label mask dan hanya masa lalu. Jangan membentuk baseline dari seluruh sesi test. Baseline HRV yang tercatat bersamaan dengan kalibrasi eye tidak dinyatakan sebagai resting baseline terpisah. Baseline eye dari kalibrasi hanya boleh ditambahkan pada revisi registry bila sumber menyediakan statistik dan timestamp yang konsisten; tidak direka dari data test.

## 6. Peta eksperimen dan urutan

Nomor berikut adalah keluarga notebook, bukan kewajiban menjalankan semua kombinasi kartesian. Exp00–exp10 dan exp14 merupakan inti. Exp11 dimulai dari kontrol MLP kecil bila gate dukungan data dan stopping lolos; FT-Transformer, exp12, dan exp13 juga mengikuti gate teknis yang ditetapkan sebelum run. Setiap cabang neural yang eligible dibandingkan dengan MLP kecil pada input yang sama. Tidak ada training pada sembilan view sekaligus.

| ID | Fokus | Perlakuan utama | Window awal |
|---|---|---|---|
| exp00 | Penerimaan dan kontrak | Gate dataset, registry, stream, split, invariant, budget | Ketiganya |
| exp01 | Baseline baru | Dummy, Logistic, XGBoost fixed, Logistic tanpa rasio | Ketiganya |
| exp02 | Kontribusi sensor | F0–F7 dan pasangan rasio; dua model fixed | 25; konfirmasi kandidat di exp14 untuk 15/45 |
| exp03 | Normalisasi/missing | N0–N3, Standard/Robust, median/native NaN, missing indicator | 25 |
| exp04 | Panjang window dan redundansi | Native/paired endpoints, matched training, step 5 versus thinning 10 | Ketiganya |
| exp05 | History generik | Tanpa history, trailing 15/30 detik; sebelum mask label | Ketiganya |
| exp06 | Seleksi fitur | Tanpa seleksi, corr 0,95, MI top-8/top-12, PCA 95% | 25 |
| exp07 | Imbalance/objective | None, class weight, ROS, SMOTE seluruh kelas minoritas; kontrol bobot rekaman | 25 |
| exp08 | Model/regularisasi | Logistic, SVM, RF, Extra Trees, XGBoost, HGB, CatBoost | 25; konfirmasi di exp14 |
| exp09 | Fusion dan ensemble kecil | Early/late fusion sensor dan soft vote lintas estimator | 25 |
| exp10 | Struktur target | Multiclass, ordinal supervised, hierarchy supervised end-to-end | 25 |
| exp11 | Neural tabular | MLP kecil dan FT-Transformer kecil | 25 |
| exp12 | Foundation model tabular | TabPFN-2 lokal, checkpoint dibekukan | 25, bersyarat |
| exp13 | Urutan sensor | GRU kecil dari stream lengkap, tanpa reset soal/fase | 25, bersyarat |
| exp14 | Seleksi akhir, sensitivitas, refit | Nested recipe per-window dan pemilihan window pada intersection | Ketiganya |

Exp02–exp13 pada 25 detik mengisolasi hipotesis dengan jumlah run terkendali. Pilihan fitur/algoritma untuk 15/45 tidak boleh disalin dari pemenang outer 25 detik. Exp14 menjalankan aturan screening inner yang sama pada setiap window; reuse cache hanya bila input/split/spec identik.

### 6.1 Exp01: baseline yang harus tersedia

Kontrol: Dummy most_frequent, Logistic L2 C=1, XGBoost fixed, serta Logistic F1 tanpa rasio. Jalur utama median-training dan StandardScaler untuk Logistic; tanpa balancing atau history. F0 menjadi baseline semua sensor.

XGBoost fixed: multiclass softprob, 300 tree, max_depth=3, learning_rate=0,05, min_child_weight=5, reg_lambda=5, reg_alpha=0, subsample=0,8, colsample_bytree=0,8, tree_method=hist, seed=42, dua thread per fit. Kontrol ini baru untuk batch-02; tidak diklaim identik dengan parameter arsip.

Logistic: solver lbfgs, max_iter=5000, tol=1e-4. Ketidak-konvergenan ditandai sebagai kegagalan kandidat, bukan warning yang diabaikan. Dummy memperoleh kelas mayoritas dari training, bukan dipaksa High.

### 6.2 Exp02–exp03: kontribusi sensor dan representasi

Gunakan Logistic dan XGBoost fixed saat mengubah fitur. Laporkan apakah penambahan eye membantu HRV, dan sebaliknya. Pisahkan penghapusan iris, gaze/state, serta mean geometris. Gagal validasi gaze tidak otomatis menghapus peserta atau semua fitur eye.

Normalisasi, native missing, dan missing indicator dibandingkan dengan pasangan kontrol yang hanya berbeda satu perlakuan. Missing-only bukan kandidat classifier sensor final. Jika informasi penting berasal dari mask, tampilkan hasilnya dan periksa distribusinya menurut fase/gelombang; jangan menyebutnya perbaikan fisiologis.

### 6.3 Exp04: window dan kepadatan row

Konsumsi tiga CSV yang sudah diverifikasi; jangan merata-ratakan RMSSD lama atau mengganti kolom window_seconds. Terapkan kontrak native/paired bagian 4.3.

Kontrol step 10 memakai subset endpoint setiap 10 detik dari origin grid sensor, bukan mengambil setiap row kedua setelah filter. Bandingkan pada test step 5 yang sama untuk menguji pengurangan redundansi training. Jika ingin test step 10 juga, laporkan terpisah sebagai perubahan frekuensi prediksi.

Tambahkan sensitivitas test tanpa overlap dengan pemilihan interval greedy per sesi menggunakan start/end dan tanpa membaca y/prediksi. Laporkan jumlah data yang tersisa; subset ini tidak disebut sampel independen sepenuhnya.

### 6.4 Exp05: history sebelum label mask

Fitur temporal dasar: RMSSD, SDNN, blink rate, EAR kiri/kanan, dan rasio baseline bila recipe memakainya. Untuk horizon 15 atau 30 detik, tambahkan mean, standar deviasi sampel, dan delta terhadap endpoint sebelumnya dari feature yang sama. Jendela trailing memakai endpoint dalam (t-horizon,t], hanya nilai yang sudah tersedia pada t. Minimal satu nilai untuk mean, dua untuk std, dan pasangan valid berjarak maksimal 7,5 detik untuk delta; selain itu NaN.

Reset stream hanya pada rekaman baru atau gap endpoint >7,5 detik. Timestamp tidak meningkat merupakan error input. Validitas/missing diterapkan per fitur; mean lama tidak boleh bertahan melewati horizon. Jangan reset karena soal, submit, rating, difficulty, mixed-label, atau row supervised dibuang. Jangan memakai panjang history/reset flag sebagai predictor default.

Uji wajib: prefix invariance, metadata invariance, mask invariance, batch-versus-streaming parity, dan train/test stream isolation. Mengubah atau menghapus label/annotation tidak boleh mengubah fitur pada sensor/endpoint yang sama. Menambahkan data masa depan tidak mengubah fitur/prediksi masa lalu. Cache transformasi harus memasukkan hash nilai sensor, waktu, stream, serta spec; jangan hanya mengandalkan window_id atau atribut dataframe yang dapat terbawa ke salinan dengan isi berbeda.

### 6.5 Exp06: seleksi fitur training-only

Corr 0,95 menghapus fitur belakangan dalam urutan allowlist jika korelasi Pearson absolut dengan fitur yang sudah dipertahankan melewati 0,95. MI memakai random_state=42, n_neighbors=3, top-8 atau top-12, dibatasi jumlah fitur tersedia. PCA mempertahankan 95% variance training, whiten=False, sesudah imputasi dan scaling.

Simpan fitur/komponen serta stabilitas pilihan antar-fold. Permutation importance dikerjakan sebagai analisis setelah prediksi tersimpan; tidak memakai outer importance untuk memilih fitur lalu menguji ulang pada outer yang sama. Tidak ada keharusan menghapus fitur hanya karena ablation lama tidak menang.

### 6.6 Exp07: balancing yang mengikuti data terbaru

Kontrol tanpa balancing wajib. Class weight memakai n_train/(3*n_class_train), dihitung ulang pada fitting partition dan dinormalisasi rata-rata satu. Untuk estimator tanpa class_weight langsung, gunakan sample_weight; jangan sekaligus menerapkan weight dan resampling pada kontrol awal.

ROS dan SMOTE menambah setiap kelas di bawah jumlah kelas training terbesar sampai jumlah terbesar tersebut. Kelas yang sudah sama tidak berubah. Ini berlaku untuk Low, Medium, atau High; bukan Low-only. Jika tidak ada penambahan, simpan no-op dan deduplikasi.

SMOTE memakai k=3, hanya jika setiap kelas yang akan disintesis memiliki lebih dari tiga sampel asli; jika tidak, kandidat invalid, bukan diam-diam diganti ROS. Tetangga dibentuk setelah median/scaling training, tanpa metadata. Jangan memakai SMOTE pada tensor sequence. Sampel sintetis hanya ada selama fitting; tidak ditulis sebagai rekaman sensor atau diberi identitas orang baru.

Tambahkan kontrol pembobotan rekaman: weight setiap row = 1/jumlah row training pada sesi tersebut, lalu normalisasi rata-rata satu. Jalankan terpisah dari class weight. Tujuannya menguji dominasi rekaman panjang, bukan membuat model personal.

Opsional keputusan kelas: kalikan probabilitas asli dengan salah satu vektor `[1, 1, 1]`, `[1.25, 1, 1]`, `[1, 1.25, 1]`, atau `[1, 1, 1.25]` dalam urutan Low/Medium/High, normalisasi, lalu argmax. Pilih di inner OOF saja. Catat sebagai penyesuaian keputusan, bukan probabilitas terkalibrasi baru. Jangan menyeimbangkan validation/test atau menilai recall tanpa precision dan kelas lain.

### 6.7 Exp08: bank algoritma awal

Parameter yang tidak ditulis mengikuti versi dependency yang dibekukan dan diekspor lengkap pada config. Semua model memakai urutan kelas tetap dan seed 42 jika tersedia.

| Keluarga | Kandidat yang dibekukan |
|---|---|
| Logistic L2 | C=0,1 / 1 / 10; solver/max_iter seperti exp01 |
| SVM RBF | C=0,1/1/10 dengan gamma=scale; tambahan C=1,gamma=0,1; probability=False |
| XGBoost | Fixed exp01; depth=2,min_child_weight=10,lambda=5,lr=0,05; depth=3,min_child_weight=10,lambda=10,lr=0,10; parameter lain mengikuti fixed |
| HGB | max_iter=300, min_samples_leaf=20, l2_regularization=1; pasangan (max_leaf_nodes,learning_rate)=(7;0,05)/(15;0,05)/(7;0,10); early_stopping=False |
| CatBoost | iterations=300, lr=0,05, loss MultiClass, use_best_model=False, allow_writing_files=False, CPU; (depth,l2_leaf_reg)=(4,5)/(6,10) |
| Random Forest | 300 tree, min_samples_leaf=5, max_features=0,7; max_depth=5/None; class_weight=None |
| Extra Trees | 300 tree, min_samples_leaf=5, max_features=0,7; max_depth=5/None; class_weight=None |

SVM memakai decision score pada pelaporan; jangan menamai nilainya probabilitas atau mengaktifkan CV probabilitas row acak tersembunyi. Bila probabilitas SVM diperlukan, gunakan sigmoid calibration dengan split grup di dalam fitting partition dan hitung biaya tambahan.

Utamakan pengurangan gap training/test, bukan menaikkan training accuracy. Early stopping boosting tidak boleh memakai outer test atau validation row acak. Kontrol awal di atas tidak memakai early stopping.

### 6.8 Exp09: fusion dan ensemble

Early fusion memakai gabungan fitur. Late fusion menggabungkan p = alpha*p_HRV + (1-alpha)*p_eye, alpha=0,25/0,50/0,75 dipilih inner. Kontrol cabang memakai dua Logistic fixed atau dua XGBoost fixed. Fitting preprocessing setiap cabang tetap training-only.

Soft vote memakai dua keluarga estimator berprobabilitas asli terbaik pada inner, bobot sama; keluarkan pasangan dengan keluarga sama. Tidak ada stacking dengan meta-model yang dilatih lalu dinilai pada prediksi training-nya sendiri. Jalur SVM tidak masuk soft vote tanpa kalibrasi grup.

Semua recipe harus memprediksi pada test yang sama. Bila input cabang benar-benar tidak tersedia menurut kontrak sensor yang dibekukan, gunakan fallback stateless sensor gabungan fixed tanpa history yang dilatih pada partition yang sama; catat alasan dan frekuensinya. Aturan ketersediaan tidak membaca label atau hasil test. Error fitting, exception implementasi, dan gagalnya validasi bukan alasan mengganti prediksi dengan fallback; kandidat tersebut gagal dan harus diperbaiki atau ditandai invalid.

### 6.9 Exp10: ordinal dan hierarchy supervised

Kontrol adalah multiclass Logistic/XGBoost pada fitur dan test sama.

Ordinal Logistic melatih q1=P(y>Low) dan q2=P(y>Medium). Jika q2>q1, proyeksikan keduanya ke rata-ratanya. Bentuk pLow=1-q1, pMedium=q1-q2, pHigh=q2, lalu argmax. Pastikan probabilitas nonnegatif dan berjumlah satu. Parameter C mengikuti tiga kandidat Logistic.

Hierarchy melatih Low versus Elevated dan Medium versus High pada training sebenarnya. Saat inference, bentuk pLow dari tahap pertama, pMedium=(1-pLow)*pMedium_given_Elevated, pHigh=(1-pLow)*pHigh_given_Elevated; ambil argmax. Jangan merutekan test menggunakan y_true. Evaluasi conditional Medium–High hanya diagnostik; hasil utama tetap end-to-end tiga kelas.

Isolation Forest/One-Class SVM tidak menjadi model utama CL: Low adalah kelas supervised yang tersedia, bukan otomatis anomali. Keduanya dapat diteliti pada studi novelty terpisah, bukan cara mengubah skor tiga kelas batch ini.

### 6.10 Exp11: MLP dan FT-Transformer

Kontrol MLP untuk cabang neural yang eligible: dua hidden layer 64/32, ReLU, dropout=0,2, tiga output softmax. Dua kandidat learning rate AdamW 1e-3/3e-4, weight_decay=1e-3, batch=64, maksimum 200 epoch.

FT-Transformer kecil: numerical feature tokenizer d_token=64, dua block, empat head, FFN dimension=128, attention/residual dropout=0,1, tiga output. AdamW lr=3e-4, weight_decay=1e-3, batch=64, maksimum 200 epoch. Ini Transformer antarfitur tabular, bukan model bahasa dan bukan Transformer atas video mentah. Ikuti implementasi primer [FT-Transformer](https://github.com/yandex-research/rtdl-revisiting-models).

Gunakan median/scaling training dan kontrol F1/F4 tanpa history. Untuk cabang neural, fitting partition yang benar-benar dipakai melatih bobot, termasuk setelah menyisihkan grup stopping, harus memuat minimal delapan grup dan minimal 20 endpoint nonoverlap tiap kelas; ini guard operasional, bukan bukti kecukupan statistik. Jika tidak terpenuhi, simpan skipped_insufficient_support. Penentuan endpoint nonoverlap memakai interval sensor, tanpa membaca prediksi.

Early stopping patience=20 pada log loss grup stopping yang terpisah dari fitting. Di setiap inner fit, ambil fold pertama yang feasible dari StratifiedGroupKFold(3,seed=42) pada inner-training untuk stopping; inner validation tidak menjadi stopping set. Setelah epoch terpilih, refit preprocessing dan model pada seluruh inner-training untuk jumlah epoch itu. Outer refit memakai median epoch terpilih dari inner fits, tanpa membuka outer test. Jika tidak ada stopping split grup yang feasible, cabang neural tidak eligible.

Seed 42 untuk seleksi; 17 dan 2026 hanya untuk sensitivitas recipe terpilih, bukan memilih seed tertinggi. GPU membantu, tetapi MLP kecil dapat berjalan CPU. FT-Transformer boleh skipped_hardware berdasarkan preflight sebelum skor.

### 6.11 Exp12: TabPFN-2 lokal

Tambahkan satu kandidat TabPFN-2 untuk tabular kecil, bukan menganggap pretrained Transformer pasti unggul. Dasarnya adalah [penelitian TabPFN](https://www.nature.com/articles/s41586-024-08328-6) dan [implementasi resmi](https://github.com/PriorLabs/TabPFN).

Pilih versi checkpoint V2 secara eksplisit, jangan mengikuti default package terbaru. Bekukan versi package, revision/hash weights, n_estimators=8, seed=42, dan catat lisensi bobot serta kebutuhan atribusinya. Gunakan inferensi lokal dengan GPU; tidak mengirim fitur/label/identitas ke API atau layanan hosted. Download bobot bukan izin mengunggah data penelitian.

Kandidat memakai F1 dan F4 mentah tanpa scaling eksternal, tanpa SMOTE, dan missing handling resmi checkpoint. Hanya training fold yang menjadi konteks berlabel; validation/test tidak boleh masuk fit-context. Jalur acuan memprediksi satu endpoint dengan konteks training yang tetap. Optimasi prediksi batch hanya boleh dipakai jika prediksinya sama dengan acuan tersebut dalam toleransi runtime yang dibekukan; jika tidak, tetap gunakan jalur satu endpoint dan catat biaya waktunya. Jika implementasi tidak dapat menyediakan inference kausal ini, catat skipped_noncausal_inference.

Gate: checkpoint/lisensi tersedia, GPU lolos preflight, dimensi memenuhi batas V2 yang didokumentasikan (maksimal 10.000 contoh dan 500 fitur), serta uji persist/load/inference lulus. Jika salah satu gagal, catat skipped_checkpoint, skipped_hardware, atau skipped_size. Jangan memotong training secara diam-diam atau berpindah ke cloud. Jalur CPU inti tetap dapat selesai tanpa TabPFN.

### 6.12 Exp13: sequence sensor kecil

Uji satu GRU satu layer, hidden_size=32, dropout pada head=0,2, output tiga kelas. Input adalah urutan sampai tujuh endpoint dalam trailing 30 detik pada grid 5 detik, F4 tanpa rasio dan tanpa history buatan. Target hanya label endpoint sekarang; label terdahulu tidak menjadi input atau syarat mengambil urutan.

Ambil urutan dari stream sebelum label mask. Gunakan masking/packed sequence untuk warm-up 1–7 endpoint; tidak mengisi dengan frame masa depan atau melintasi gap/rekaman. Imputasi dan scaling memakai statistik fitting partition, bukan setiap sequence test secara penuh. AdamW lr=1e-3, weight_decay=1e-3, batch=64, maksimum 200 epoch; stopping dan support guard mengikuti exp11.

Bandingkan dengan MLP yang menerima row sekarang pada F4 yang sama serta history generik exp05. Uji batch/stream parity dan prefix invariance. Jangan menambah LSTM/TCN/Transformer temporal besar pada registry awal; hasil negatif GRU harus tetap disimpan. Arsitektur lanjutan memerlukan revisi registry, bukan pencarian tanpa batas setelah melihat test.

## 7. Exp14: seleksi end-to-end dan keluaran akhir

### 7.1 Aturan pencarian yang tidak memakai outer leaderboard

Terapkan urutan berikut secara terpisah di setiap outer training dan setiap window. Seluruh ranking memakai manifest inner yang dibekukan:

1. Screening fitur: nilai F0–F7 dan pasangan dengan/tanpa rasio menggunakan dua kontrol fixed. Pertahankan dua representasi berbeda dengan inner score terbaik.
2. Model: jalankan bank exp08 pada dua representasi tersebut. Pertahankan dua pipeline statis terbaik.
3. Refinement: untuk keduanya, uji history 0/15/30, seleksi none/corr/MI, balancing none/weight/ROS/SMOTE, dan Standard/Robust yang relevan, satu faktor per kontras. Uji PCA, N2/N3, missing indicator, dan native NaN sebagai kontras satu faktor terhadap dua pipeline statis tersebut, bukan perkalian semua opsi. N2 hanya berlaku pada pipeline dengan HRV, N3 pada pipeline dengan fitur eye yang ditransformasi, native NaN hanya pada estimator yang mendukungnya, dan PCA hanya sesudah imputasi/scaling. Kontras yang tidak berlaku ditandai not_applicable, bukan diubah diam-diam.
4. Interaksi terarah: untuk masing-masing pipeline, gabungkan pilihan history+seleksi terbaik dan history+balancing terbaik; maksimal empat recipe kombinasi. Hindari kombinasi ROS/SMOTE pada GRU dan pretrained TabPFN.
5. Bentuk pool final maksimum 24 recipe unik: enam statis, empat history/interaksi, dua normalisasi kausal, empat fusion/ordinal/hierarchy, dan maksimal delapan neural/pretrained/sequence. Slot tak terpakai tidak perlu diisi. Kontrol Logistic tanpa rasio dan XGBoost fixed selalu ikut dalam slot statis.
6. Pilih dengan aturan bagian 4.4, refit pada seluruh outer training, dan prediksi outer test sekali untuk prosedur terpilih.

Setiap keluarga advanced dipilih/diranking hanya dari training inner. Jangan mengambil pemenang global dari outer exp02–exp13 sebagai input tahap berikutnya. Gunakan input cache inner yang sama bila identitasnya cocok. Pada 15/45, ulangi screening inner; jangan memaksakan fitur yang dipilih dari outer 25.

Registry awal dibatasi 120 recipe efektif per outer/window, termasuk screening, refinement, dan cabang eligible; bukan 120 kombinasi untuk setiap faktor. Registry generator harus menghitung jumlahnya sebelum run. Jika melebihi batas, pangkas menurut prioritas tetap: hapus PCA/native/missing tambahan, kemudian refinement Robust/N3, kemudian cabang advanced menurut urutan exp13→exp12→FT; kontrol, sensor ablation, dan uji tanpa rasio tetap. Catat truncation, dan bekukan hasil registry sebelum outer results dibuka. Kandidat identik/no-op hanya dihitung sekali.

### 7.2 Model per-window dan pilihan window

Keluaran wajib adalah evaluasi nested native untuk setiap window dan tiga kandidat refit per-window. Skor native 15/25/45 tidak langsung diperlakukan sebagai peringkat global karena test-nya berbeda.

Tambahkan evaluasi prosedur seleksi lintas window pada intersection bagian 4.3. Di setiap outer training, lakukan screening/refinement dan pemilihan window menggunakan inner OOF pada intersection training; outer test juga intersection held-out. Jika intersection fitting tidak mendukung tiga kelas, jalur global tidak eligible; tiga jalur per-window tetap dilaporkan.

Untuk artefak penggunaan, pilih recipe/window melalui group validation pada seluruh release final dengan aturan yang sama, lalu refit. Simpan konfigurasi preferred_window dan tiga alternatif. Bila jalur global tidak eligible, default artefak operasional adalah kandidat 25 detik, dengan alasan kompatibilitas kontrol; jangan menyebutnya terbaik di antara window.

Skor refit/seleksi pada seluruh data bukan hasil test baru. Angka kinerja yang dikutip berasal dari OOF prosedur yang sesuai cakupannya: native per-window atau selection-on-intersection.

### 7.3 Pemeriksaan akhir

- Simpan train/test gap, confusion matrix, seluruh metrik kelas, dan kontribusi tiap keluarga sensor.
- Learning curve pada grup training 50%, 75%, 100% untuk Logistic tanpa rasio dan recipe statis terpilih; pilih subset deterministik seed 42 dan tambahkan grup minimum jika diperlukan agar tiga kelas hadir. Outer test tetap sama; subset bukan identitas sebagai fitur.
- Laporkan skor pada test nonoverlap, fase Low, baseline tersedia/tidak, validasi gaze, dan asal gelombang. Subset audit tidak mengganti evaluasi utama atau dipakai memilih peserta yang dianggap bagus.
- Jalankan seed 17/2026 untuk recipe terpilih dengan manifest tetap, bukan mengulang seluruh pencarian. Jangan memilih atau merata-ratakan hasil secara oportunistis; default deployment tetap seed 42.
- Jalankan kontrol diagnostik missing-only serta warm-up/history-length-only dari stream kausal yang tidak membaca label, soal, atau fase. Keduanya tidak masuk kandidat final. Skor tinggi memerlukan inspeksi pola akuisisi dan ketersediaan, tetapi bukan bukti kebocoran dengan sendirinya; pelanggaran invariant tetap harus ditunjukkan. Jangan memakai metadata fase/set yang dapat merekonstruksi target sebagai classifier pembanding fisiologis.
- Uji sensor-only inference, persist/load, cold start, missing baseline, warm-up, gap, serta parity batch/stream. Tidak ada parameter preprocessing dari label test.

## 8. Biaya, implementasi, dan artefak

### 8.1 Paket disiapkan sebelum gate data; training sesudah persetujuan

Pembuatan notebook, kode bersama, dan tes teknis tidak menunggu pengambilan tambahan. Gate bagian 1 membatasi training penelitian, bukan persiapan implementasi. Exp00 memisahkan pemeriksaan kesiapan dari pengesahan release; persetujuan tidak diaktifkan otomatis. Penambahan rekaman dengan skema yang sama tidak memerlukan penulisan ulang notebook.

Setiap keluarga memakai notebook biasa pada foldernya sendiri, dengan markdown berbahasa Indonesia yang menjelaskan tujuan dan tahapan utama. Ikuti gaya notebook awal tim: tanpa emoji atau komentar dekoratif. Tidak ada launcher Run All lintas eksperimen. Operator tidak diwajibkan menjalankan setiap notebook dua kali smoke/full; pengembang melakukan tes teknis sebelum full diserahkan.

Bangun shared code batch-02 yang mandiri, jangan bergantung pada models/shared lama atau source batch-01 yang sudah dibersihkan. Satu requirements.txt yang tervalidasi memuat seluruh dependency CPU dan optional neural pada versi terkunci; package GPU/bobot dapat berstatus optional, tetapi versinya tetap tercatat. Tidak perlu mengganti shared/requirements setiap eksperimen.

Struktur implementasi yang direncanakan:

```text
models/batch-02/
  PLAN.md
  shared/
  requirements.txt
  registry/
  inputs/<release_id>/
  exp00/ ... exp14/
  outputs/expXX/<run_id>/
```

Root Drive: `drive/MyDrive/Penelitian-Adaptive-IDE`. Dataset: `Dataset/v2_session_phase/window_XXs_step_5s/good_acceptable.csv`; output: `models/batch-02/outputs` di bawah root yang sama. Notebook mendeteksi lokal/Colab melalui konfigurasi root, bukan path mesin yang berulang di banyak cell. Untuk jalur temporal di Colab, unggah cache stream turunan yang telah diverifikasi beserta manifest, bukan mengandalkan tiga CSV supervised saja.

### 8.2 Budget dan pelaksanaan

Sebelum full run, tampilkan jumlah grup, split, recipe efektif, jumlah estimator atomik, serta estimasi waktu dari benchmark teknis. Tiga inner fold berarti tiga fit per recipe dasar per outer/window; fusion/hierarchy/calibration dan early-stopping refit menambah fit. Dengan K recipe dan N orang, tiga window memerlukan sampai 9KN fit dasar sebelum tambahan tersebut. Batas K=120 adalah batas atas, bukan target untuk dihabiskan.

Jalankan CPU dengan maksimal satu tingkat paralelisme: paralel kandidat/fold atau paralel estimator, tidak keduanya tanpa pembatasan. Default estimator dua thread. GPU advanced serial. Simpan hasil setiap fold, gunakan resume berbasis hash, dan jangan menandai run parsial sebagai complete.

Cache model/matriks harus mencakup hash data/label/stream, grup/split, spec lengkap, kode, seed, dependency dan checkpoint. Tidak boleh memakai cache hanya karena ID row sama. Perubahan rencana/versi data menolak resume lama. Semua error harus menyebut recipe/fold dan tahap; jangan mengganti model/metode diam-diam.

Ketiadaan GPU tidak memblokir inti CPU. Bekukan status eligible/skipped advanced dari preflight, bukan dari ketidakpuasan pada hasil outer. Tidak ada janji accuracy tertentu atau bahwa Transformer akan lebih baik daripada Logistic/boosting.

### 8.3 Artefak dan laporan

Wajib: dataset_release, config, status, registry, split manifest, dependency/code/checkpoint hashes, feature lists/transform stats, seluruh skor inner, recipe terpilih, prediksi OOF dengan probabilitas/decision score yang dibedakan, metrik kelas, coverage sensor/label, alasan fallback/unavailable, confusion CSV dan grafik.

Tulis SUMMARY dan DIAGNOSIS batch-02 setelah full run, dengan hasil tiap keluarga, efek kombinasi, hasil negatif, biaya, serta daftar cabang skipped. Evaluasi internal dataset ini tidak otomatis membuktikan transfer ke semua aktivitas programming di IDE atau memvalidasi CL setiap window dengan pengukuran rujukan independen.

Hasil CSV/JSON/Markdown/figure mengikuti [aturan Git models](../.gitignore); bobot, environment, snapshot input sensitif, dan cache tidak dipush. Jangan memperluas akses data atau memakai hosted inference tanpa keputusan terpisah. Alias grafik tidak menghilangkan identitas dari CSV.

## 9. Validasi implementasi dan urutan serah-terima

| Area | Kriteria lulus |
|---|---|
| Gate | Full menolak data tambahan/proporsi/release yang belum approved; tidak mengarang jumlah final |
| Input | Tiga hash CSV/allowlist/manifest konsisten; target langsung; admission quality-only; sumber dan v1 tidak diubah |
| Identitas/split | Seluruh sesi orang held-out keluar dari inner/outer fitting; setiap native row mendapat satu prediksi OOF |
| Fitur | Metadata/event/label tidak masuk X; imputer/scaler/MI/corr/PCA/resampler hanya fit training |
| History/sequence | Prefix, metadata, mask, cache dan batch/stream invariant; tidak reset karena label/soal/fase |
| Window | Native versus paired jelas; labelable versus sensor coverage terpisah; tidak menghukum label mixed sebagai sensor failure |
| Baseline/missing | Tanpa rasio tetap dapat inference; tidak ada full-test centering atau nol palsu; empty/constant training columns tercatat |
| Balancing | Mendukung kelas minoritas apa pun, no-op/invalid eksplisit; validation/test tetap asli |
| Neural/pretrained | Group stopping, checkpoint/license, fit-context isolation, seed, batch parity dan ukuran/hardware lulus |
| Seleksi | Registry/search/tie-break deterministik; tidak memakai outer leaderboard; absennya kandidat tidak menjadi skor nol |
| Numerik | Metrik dapat dihitung ulang dari OOF; probabilitas valid; kelas absent dan no_test_support tercatat |
| Operasional | Persist/load dan sensor-only inference lulus; resume tidak mencampur run; file hasil dapat dibuka pengguna di Windows |

Urutan pelaksanaan:

1. Terima dan gabungkan pengambilan tambahan; rebuild v2, tinjau proporsi, lalu sahkan release.
2. Implementasikan exp00 dan tes kontrak sampai lulus. Bekukan registry/group/endpoint sebelum full.
3. Siapkan satu paket notebook/shared/dependency, uji pada runtime bersih, lalu jalankan baseline dan keluarga inti.
4. Jalankan cabang advanced yang eligible dan exp14 sesuai registry yang dibekukan. Periksa parity inference sebelum menyatakan artefak siap digunakan.
5. Laporkan hasil batch-02 sendiri. Pertahankan arsip sebelumnya; tidak ada kewajiban mengulang dataset v1 atau mengubah hasil lama.

Paket implementasi sudah disiapkan tanpa training classifier atau epoch neural. Pemeriksaan teknis tidak menggantikan pemeriksaan saat fitting pada runtime dan dataset final. Dua rekaman tambahan 16 November telah diproses. Pekerjaan berikutnya adalah menetapkan apakah pengambilan sudah cukup, menerima proporsi/kualitas, membekukan release, serta menyiapkan cache sensor sebelum training penelitian.

## Referensi metode

- [Grouped CV](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data), [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html), dan [nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
- [Resampling pitfalls](https://imbalanced-learn.org/stable/common_pitfalls.html): resampling harus berada di training partition.
- [Calibration](https://scikit-learn.org/stable/modules/calibration.html): decision score dan probabilitas tidak dipertukarkan.
- [HGB](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) dan [CatBoost](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier): missing handling dan training options mengikuti versi yang dibekukan.
- [FT-Transformer, penelitian dan implementasi](https://github.com/yandex-research/rtdl-revisiting-models).
- [TabPFN, Nature 2025](https://www.nature.com/articles/s41586-024-08328-6) dan [repository resmi](https://github.com/PriorLabs/TabPFN): checkpoint/hardware/lisensi harus diperiksa sebelum dipakai.
