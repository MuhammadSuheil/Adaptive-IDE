# Rencana Batch 05

## Tujuan

Menguji perbaikan yang dapat ditelusuri sebelum memperluas pencarian model. Diagnosis batch-04 menunjukkan masalah komposisi monitor, cakupan sumber pada pembagian waktu, serta seleksi accuracy yang dapat menerima model dengan recall satu kelas nol. Rangkaian ini memisahkan pengaruh split, normalisasi, dan training neural.

Target tetap klasifikasi Low, Medium, dan High dari fitur eye tracking dan HRV. Sasaran accuracy 70% akan dinilai bersama Macro-F1 dan recall tiap kelas. Tahap persiapan tidak membuktikan bahwa target tersebut dapat dicapai.

## Tiga tahap

| Tahap | Pekerjaan | Syarat melanjutkan |
|---|---|---|
| 1. Persiapan | Bekukan dataset dan Test referensi; periksa split, fold, monitor, preprocessing, serta aturan seleksi | Pemeriksaan teknis lolos dan window 25 detik siap untuk tabular serta neural |
| 2. Kontrol development | Replikasi classifier sederhana, uji perubahan split dan normalisasi secara terpisah, lalu periksa learning curve dan confusion matrix | Bukti perbaikan pada beberapa blok dan pengenalan tiga kelas sebelum memperluas model |
| 3. Training terarah | Uji satu arsitektur neural, mekanisme monitor, checkpoint, dan jadwal learning rate; perluas kandidat bila hasil development mendukung | Kandidat memenuhi kelayakan per kelas dan penentuan durasi fitting stabil |

Notebook yang sama, `exp01/EyeHeartCognitiveLoad.ipynb`, melanjutkan tiga tahap secara bertahap. Versi tahap 3 memvalidasi snapshot persiapan, menjalankan neural dan kontrol yang ditetapkan di bawah, lalu mengevaluasi satu model terkunci pada Test referensi. Run all tidak mengulang 71 fitting tahap 2. Hasil tahap 1/2 tetap dipertahankan.

## Dataset dan fitur

Ketiga file berikut wajib tersedia di Drive:

```text
MyDrive/Penelitian-Adaptive-IDE/
├── Dataset/v2_session_phase/
│   ├── window_15s_step_5s/good_acceptable.csv
│   ├── window_25s_step_5s/good_acceptable.csv
│   └── window_45s_step_5s/good_acceptable.csv
└── models/batch-05/
    └── exp01/EyeHeartCognitiveLoad.ipynb
```

Dataset yang dibekukan berisi 5.253, 4.608, dan 3.509 baris untuk window 15, 25, dan 45 detik. Notebook memeriksa SHA-256 terhadap versi yang digunakan pada batch-04. Jika berbeda, proses berhenti agar perubahan data tidak ikut mengganti cakupan Test tanpa peninjauan.

Profil default menggunakan 10 fitur core: RMSSD, SDNN, laju kedipan, mean/std EAR kiri dan kanan, mean/std ukuran geometri iris, serta mean perubahan ukuran iris. Core-head menambahkan mean/std pitch dan yaw jika lengkap dan layak secara numerik. Ukuran iris tersebut bukan pengukuran diameter pupil.

`cognitive_load_label` menjadi target langsung. Identitas, waktu, fase, difficulty, soal, dan kualitas mendukung audit serta split; kolom tersebut tidak menjadi input classifier. Protokol Low yang mencakup fase nonpengerjaan tetap dipertahankan. Pipeline tidak mengganti label mengikuti prediksi atau menghapus fase Low untuk memperbaiki skor.

## Tahap 1

### 1. Pemeriksaan sumber

Periksa keberadaan tiga file, kolom wajib, label, identitas, timestamp berzona waktu, durasi window, duplikasi ID/endpoint, nilai core finite, dan rentang fisik yang dapat diperiksa. Catat fitur core konstan atau vektor core berulang untuk ditinjau. Pemeriksaan tidak mengimputasi, mengubah nilai sensor, atau menulis ulang CSV.

Endpoint sama pada beberapa panjang window harus mempunyai label yang sama. Satu ID sesi harus merujuk ke identitas peserta yang konsisten. Fase window dapat berbeda karena rentang agregasinya berbeda; fase tidak dipaksa sama antarwindow.

### 2. Bekukan referensi

Notebook menyertakan manifest batch-04, checksum berkas sumber manifest, serta hash tiga dataset. Operator tidak perlu menyalin output batch-04 ke Drive. Setiap run menyimpan salinan `reference_manifest.json` dan `development_manifest.json` secara terpisah.

Pertahankan persis endpoint dan ID window Test batch-04. Test tidak ikut fitting, statistik preprocessing, monitor, atau pemilihan kandidat. Karena tim sudah menganalisis Test tersebut, kedudukannya adalah referensi eksploratif yang telah dilihat. Pengujian akhir independen memerlukan data atau interval yang belum digunakan untuk menentukan perubahan.

### 3. Development baru

- Gunakan union endpoint ketiga window, dengan blok 240 detik per sesi. Endpoint bersama mendapat penetapan bagian yang sama.
- Buang kandidat development yang berjarak kurang dari 45 detik dari endpoint Test pada sesi yang sama. Pertahankan endpoint Test.
- Gunakan kandidat seed 42 sampai 169. Target pembagian Train/validation adalah 80/20 sebelum purge, dengan pengacakan blok berstrata kelas mayoritas blok.
- Purge Train terhadap validation. Jarak endpoint minimal 45 detik mencakup dukungan window terpanjang tanpa history tambahan. Jika eksperimen menambah history, manifest harus dibangun ulang dengan guard sesuai dukungan fitur.
- Wajib tersedia tiga kelas pada Train dan validation setiap panjang window.
- Urutkan kandidat berdasarkan kekurangan cakupan sesi-kelas pada validation/Test terhadap Train, lalu perbedaan distribusi kelas dan fase Low, kemudian kehilangan baris serta deviasi ukuran. Seed terkecil memecahkan seri.
- Pemilihan kandidat hanya membaca metadata target, fase, identitas sumber, dan waktu. Tidak membaca nilai fitur sensor atau skor classifier.

Laporkan jumlah baris/blok setiap sesi-kelas dan bagian, termasuk sumber validation/Test yang tidak memiliki dukungan di Train. Sel pendek tidak dipaksa hadir di semua bagian dan tidak dibuang diam-diam. Label `short_cell` pada laporan berarti kurang dari tiga blok; itu indikator dukungan untuk ditinjau, bukan bukti otomatis bahwa semua sel lainnya dapat dipisah dengan baik.

### 4. Lima fold internal

Gunakan Train saja. Cari kandidat seed 42 sampai 73 untuk lima fold berdasarkan blok berstrata kelas mayoritas. Purge fitting terhadap validation fold dengan guard yang sama. Semua fitting/validation fold wajib memiliki tiga kelas pada ketiga panjang window.

Peringkat kandidat mengutamakan cakupan sumber, kemiripan distribusi kelas/fase Low, dan retensi baris. Simpan ID endpoint fitting dan validation tiap fold. Semua konfigurasi model yang dibandingkan kelak menggunakan fold yang sama.

### 5. Monitor neural

Setiap fitting fold dan fitting akhir Train menyediakan monitor yang terpisah. Pilih sekitar 15% blok fitting dengan kandidat seed 310 sampai 437. Purge fitting terhadap monitor.

Syarat keras:

- Ketiga kelas ada pada fitting dan monitor, minimal 10 window tiap kelas.
- Selisih proporsi setiap kelas antara fitting dan monitor maksimal 10 poin persentase.
- Monitor hanya berasal dari fitting pool terkait; validation akhir dan Test tidak boleh dipakai sebagai pengganti.

Di antara kandidat yang memenuhi syarat, utamakan kemiripan fase Low dan sumber sesi, kemudian jumlah baris yang dipertahankan. Simpan ID fitting/monitor dan komposisi kelas, fase, serta sumber. Semua konfigurasi pada konteks fold/window yang sama memakai ID monitor yang sama.

Jika kandidat tidak tersedia, laporkan neural belum siap. Jangan melonggarkan batas atau mengganti monitor dengan validation akhir. Batas 10 window merupakan pemeriksaan operasional; window yang overlap bukan 10 pengamatan independen.

### 6. Preprocessing dan seleksi

Siapkan tiga kontrol statistik median/IQR dari fitting saja:

| Kontrol | Pemusatan | Skala |
|---|---|---|
| Global | Median fitting global | IQR fitting global |
| Session-centered | Median fitting sesi terkait | IQR fitting global |
| Session-robust | Median fitting sesi terkait | IQR fitting sesi terkait |

Sesi yang belum ada pada fitting memakai statistik global. Jika IQR sesi nol, gunakan IQR global; IQR global nol memakai skala 1. Nilai terpusat yang negatif tidak dipotong menjadi nol. Statistik sesi ini tidak membentuk classifier per orang.

Uji bahwa perubahan nilai sensor pada held-out tidak dapat mengubah statistik fitting. Tahap ini menguji dan menyiapkan transformasi, bukan fitting classifier.

Aturan kandidat untuk tahap pemodelan: prediksi finite dan recall positif untuk tiga kelas development. Urutkan kandidat layak berdasarkan Macro-F1, accuracy, recall minimum, standar deviasi fold yang lebih kecil, dan kompleksitas yang lebih rendah. Jika tidak ada kandidat layak, laporkan belum ada model memadai. Uji tahap 1 memakai contoh metrik sintetis saja.

### 7. Artefak dan gate

Setiap eksekusi membuat run baru pada `models/batch-05/outputs/exp01/<run-id>/`:

| Berkas | Isi |
|---|---|
| `run.json` | Versi implementasi, hash dataset, konfigurasi, versi Python/NumPy/pandas |
| `reference_manifest.json` | Pembagian batch-04 yang dibekukan |
| `development_manifest.json` | Split, ID window/endpoint, dan lima fold baru |
| `partition_counts.csv` | Jumlah kelas per split, fold, dan monitor |
| `source_support.csv` | Jumlah baris/blok per sesi-kelas dan bagian |
| `monitors.json` | ID fitting/monitor dan komposisi kelas, fase, sesi |
| `readiness.json` | Kesiapan per window, audit sumber, checksum manifest, gate tahap 2 |
| `PREPARATION.md` | Ringkasan pemeriksaan yang dapat dibaca tim |
| `status.json` | Selesai/gagal; tidak ada status training model pada tahap ini |

Kesiapan tabular dan neural dilaporkan terpisah untuk setiap window. Gate tahap 2 memerlukan kesiapan keduanya pada 25 detik, ditambah uji teknis yang lolos. Kekurangan sumber tetap dilaporkan meskipun gate teknis lolos. Status siap berarti pipeline dapat digunakan untuk eksperimen development; itu belum menyatakan kualitas model atau validitas label sensor telah terbukti.

## Pengujian

Uji otomatis memeriksa input rusak, timestamp/durasi, duplikasi, label endpoint bertentangan, kelas yang tidak lengkap, batas guard, perlindungan Test, konsistensi endpoint antarwindow, monitor yang tidak mungkin, normalisasi train-only, fallback sesi baru, serta penolakan kandidat dengan recall nol/nonfinite.

Validasi notebook mencakup schema Jupyter, syntax seluruh cell, linter, pemeriksaan tipe dan keamanan pada kode definisi, serta eksekusi persiapan tiga CSV asli. Jangan menjalankan search atau fitting classifier pada tahap 1.

## Tahap 2: training kontrol

### Manifest dan perlindungan evaluasi

Gunakan snapshot persiapan `20261122T074814995365Z_fe70ac0e`: seed development 125, seed CV 50, blok 240 detik, guard 45 detik, dan history nol. Hash manifest development adalah `b7a88f9eacd511e9c64019ead7c925800fe65ab82dbcf15fc00729e962d92d62`.

Notebook menyertakan snapshot tersebut. Tidak perlu mengunggah output persiapan atau batch-04. Pemeriksaan ulang mencakup hash tiga CSV, blok endpoint, perlindungan Test, dan cakupan OOF setiap baris Train tepat satu kali.

Training hanya menggunakan 25 detik dan 10 fitur core: Train 2.724 baris, validation 870, Test 683, serta purged 331. Test tidak boleh masuk fitting, statistik normalisasi, evaluasi prediksi, pemilihan kandidat, atau learning curve. Dua CSV lainnya tetap diperlukan untuk validasi sumber dan konsistensi antarwindow.

### Model dan normalisasi

| Model | Konfigurasi tetap |
|---|---|
| Logistic Regression | C=1, lbfgs, maksimal 5.000 iterasi |
| Extra Trees | 200 trees, min_samples_leaf=5, max_features=0,8 |
| HistGradientBoosting | 200 iterations, learning rate 0,05, max_leaf_nodes=15, min_samples_leaf=20, L2=1, early stopping=False |
| Dummy | Kelas terbanyak pada fitting |

Tiga model utama memakai global, session-centered, dan session-robust. Dummy memakai global sebagai pembanding. Gunakan seed model 42 dan maksimum dua thread CPU. Tidak ada class weight, oversampling, SMOTE, search, history, atau fitur tambahan pada kontrol ini.

Setiap konfigurasi memakai lima fold yang sama. Hitung ulang statistik dari fitting masing-masing fold. Simpan warning; kandidat Logistic Regression yang belum konvergen tidak layak menjadi pemenang.

### Seleksi dan validation

Gabungkan prediksi validation fold menjadi OOF pada 2.724 baris Train. Simpan accuracy, balanced accuracy, Macro-F1, precision/recall/F1 tiap kelas, confusion matrix, distribusi prediksi, skor fitting, durasi, dan warning. Laporkan rata-rata/standar deviasi antarfold dan selisih fitting-validation.

Kandidat harus konvergen, memiliki prediksi finite, dan recall OOF positif pada tiga kelas. Urutan peringkat mengikuti tahap 1. Untuk pemecah seri kompleksitas model, gunakan Logistic Regression, HistGradientBoosting, kemudian Extra Trees. Dummy tidak masuk pemilihan.

Bekukan pemenang sebelum membaca hasil validation utama. Fit pemenang pada seluruh Train dan evaluasi pada validation. Fit juga kontrol global serta Dummy yang diperlukan, dengan pemakaian ulang pekerjaan identik. Jangan mengganti pemenang berdasarkan skor validation. Simpan rincian hasil menurut kelas, fase Low, dan dukungan sumber sesi-kelas; metadata tersebut bukan input classifier.

### Kontrol pembagian lama dan baru

Gunakan Logistic Regression dan Extra Trees dengan normalisasi global yang sama. Masing-masing memakai Train referensi atau Train development, dengan guard sesuai manifest. Bandingkan pada 217 window validation bersama: 73 Low, 60 Medium, dan 84 High. Baris tersebut tidak masuk fitting kedua pembagian.

Laporkan jumlah baris/blok dan komposisi kelas fitting. Perbandingan ini menilai perubahan pembagian, guard, dan retensi data secara bersama, bukan efek normalisasi atau bukti sebab tunggal.

### Learning curve

Untuk pemenang OOF, gunakan subset bertingkat sekitar 25%, 50%, 75%, dan 100% blok fitting pada setiap fold. Urutan blok berstrata kelas mayoritas menggunakan seed 42. Pertahankan validation fold dan hitung ulang preprocessing dari subset terkait.

Jika subset tidak memiliki minimal 10 window tiap kelas, tandai titik tidak tersedia. Simpan jumlah baris/blok aktual dan komposisi kelas. Gunakan hasil CV 100% yang sudah tersedia. Tampilkan accuracy serta Macro-F1 fitting-validation, termasuk variasi antarfold.

Jika tidak ada kandidat OOF layak, laporkan tanpa pemenang dan jangan memaksakan learning curve.

### Anggaran, output, dan resume

Anggaran maksimal 71 pekerjaan fitting unik: 50 CV, maksimal empat fitting development akhir, dua kontrol referensi, dan 15 tambahan learning curve. Jumlah dapat lebih kecil jika pemenang sama dengan kontrol global. Percobaan ulang pekerjaan gagal dapat menambah jumlah pemanggilan fitting, tetapi tidak menambah konfigurasi penelitian.

Run tahap 2 memakai prefix `stage2_` pada output exp01, terpisah dari hasil tahap 1. Checkpoint lokal menyimpan prediksi/metrik per pekerjaan; penanda selesai dengan checksum baru ditulis ke arsip setelah payload tersalin. Resume memverifikasi kecocokan data, manifest, kode, konfigurasi, dan lingkungan. Satu runtime menulis satu run.

Artefak utama: `cv_leaderboard.csv`, `fold_metrics.csv`, `cv_fold_summary.csv`, `oof_predictions.csv`, `validation_metrics.csv`, `split_controls.csv`, `common_validation_predictions.csv`, `learning_curve.csv`, `validation_breakdown.csv`, `selection.json`, `selected_model.json`, `gate.json`, `RESULTS.md`, dan grafik. Folder `jobs` menyimpan prediksi, warning, dan checkpoint; fitting akhir juga menyimpan estimator serta statistik preprocessing terpisah.

Gate perencanaan neural terarah memerlukan kandidat layak pada OOF, recall positif tiga kelas pada validation, fitting akhir konvergen, serta Macro-F1 lebih tinggi daripada Dummy pada minimal tiga fold. Gate tidak membuktikan kualitas optimal atau menjamin 70%. Test dan tahap 3 tetap tidak dieksekusi.

### Validasi sebelum dan sesudah training

Uji sintetis memeriksa larangan Test, batas guard, indeks tidak valid, prediksi tidak sejajar/nonfinite, cakupan tiga kelas, subset bertingkat, checkpoint rusak/tidak kompatibel, pekerjaan terputus, resume tanpa fitting ulang, dan kesamaan prediksi setelah model dimuat ulang. Uji wrapper empat classifier pada data kecil sebelum training penelitian.

Jalankan pemeriksaan syntax/schema notebook, lint, tipe, dan keamanan. Setelah training, cocokkan ulang metrik dengan CSV prediksi, checksum arsip, jumlah fitting, dan hash dataset. Tidak menjalankan eksperimen tambahan berdasarkan skor yang muncul.

## Tahap 3: neural terarah dan Test referensi

### Batas dan konfigurasi

Pertahankan tiga CSV, label, 10 fitur core, window 25 detik, seed split 125/CV 50, blok 240 detik, guard 45 detik, dan history nol. Gunakan snapshot monitor 25 detik dari tahap 1. Tidak ada tuning, balancing, arsitektur tambahan, atau perubahan dataset.

| Komponen | Konfigurasi |
|---|---|
| MLP | Input 10, hidden 64 dan 32, output 3 |
| Hidden layer | Linear, LayerNorm, GELU, dropout 0,2 |
| Optimizer | AdamW, learning rate tetap 0,001, weight decay 0,001 |
| Training | 120 epoch, batch 128, gradient clipping 1,0 |
| Loss | Cross-entropy dari logits, tanpa class weight atau label smoothing |
| Seed | 42, 43, 44 |
| Normalisasi | Global, session-centered, session-robust |

Lokal memakai CPU maksimal dua thread. Colab memakai CUDA otomatis jika tersedia. Aktifkan operasi deterministik dan catat lingkungan/device. Tidak ada scheduler yang berubah akibat pemilihan epoch.

### Monitor dan checkpoint

Statistik preprocessing dan optimizer hanya membaca optimizer-fitting. Monitor memilih checkpoint; validation fold/utama serta Test tidak memilih checkpoint. Periksa kembali minimal 10 window/kelas, gap proporsi maksimal 10 poin persentase, dan pemisahan interval sebelum training.

Selesaikan 120 epoch agar tiga aturan memakai lintasan yang sama:

1. Loss monitor terendah; seri memilih epoch lebih awal.
2. Macro-F1 monitor tertinggi; seri memilih recall minimum lebih tinggi, loss lebih rendah, lalu epoch lebih awal.
3. Epoch 120 sebagai kontrol durasi tetap.

Simpan salinan bobot yang tidak berubah saat training berlanjut. Catat loss optimizer, loss fitting dalam mode evaluasi, loss monitor, Macro-F1, recall, serta distribusi prediksi monitor tiap epoch. Cross-entropy evaluasi dihitung dari logits, bukan probabilitas yang dipotong.

Tiga seed tidak dipilih satu per satu: rata-ratakan probabilitasnya. Setiap normalisasi/aturan checkpoint membentuk satu kandidat ensemble, dengan sembilan kandidat neural secara keseluruhan.

### Kontrol fitting dan kandidat

Extra Trees menggunakan parameter tahap 2. Tiga kontrol matched-data memakai persis baris optimizer-fitting neural dan tiga normalisasi. Satu baseline penuh menggunakan session-robust serta seluruh fitting pool CV, seperti pemenang tahap 2. Baseline dihitung ulang agar notebook mandiri tanpa output lama.

Semua 13 kandidat menghasilkan OOF pada 2.724 baris Train yang sama. Kandidat wajib selesai lengkap, mempunyai probabilitas finite yang valid, serta recall OOF positif pada tiga kelas. Pilih berdasarkan Macro-F1, accuracy, recall minimum, standar deviasi Macro-F1 fold yang lebih kecil, lalu jumlah estimator inference yang lebih sedikit. Pemecah seri tetap: global/centered/robust, baseline penuh/matched/neural, kemudian checkpoint loss/macro/last.

### Fitting akhir dan penguncian

Untuk kandidat neural terbaik menurut OOF, jalankan tiga seed pada optimizer-fitting/monitor akhir. Optimizer memakai 2.121 window; monitor 501; sisanya terkena purge. Terapkan aturan checkpoint yang sama. Tidak menggunakan median epoch CV atau refit tersembunyi pada seluruh Train.

Sediakan kontrol akhir Extra Trees pada tiga normalisasi matched-data serta baseline penuh. Bekukan pemenang OOF sebelum membaca validation utama. Kandidat dengan recall validation nol ditandai belum memadai, bukan diganti dengan runner-up setelah melihat hasil.

Kunci identitas kandidat, preprocessing, dan checksum bobot sebelum Test. Evaluasi hanya pemenang pada 683 window Test memakai model yang sama dengan validation. Jangan menambahkan validation ke fitting atau mengganti kandidat berdasarkan Test. Jika prediksi nonfinite, hentikan dan laporkan kegagalan.

Test sudah pernah dianalisis pada rangkaian sebelumnya. Nilainya merupakan referensi eksploratif, bukan pengujian independen baru. Jika hasilnya tersimpan dan checksum cocok, resume memakai hasil tersebut tanpa evaluasi model baru.

### Anggaran dan resume

Maksimal 48 lintasan neural: 45 CV dan tiga fitting akhir. Maksimal 24 fitting tabular: 15 matched CV, lima baseline CV, tiga matched akhir, dan satu baseline akhir. Total maksimal 72 pekerjaan unik dan 5.760 epoch neural. Tidak memperluas search secara otomatis.

Checkpoint training setiap lima epoch menyimpan model, optimizer, state RNG, epoch, history, dan bobot terpilih. Dua slot menjaga checkpoint aktif tetap utuh ketika slot berikutnya sedang disalin. Simpan tiga checkpoint terpilih dan dua slot resume, bukan bobot untuk setiap epoch. State dict serta konfigurasi arsitektur menggantikan pickle objek neural yang bergantung pada lokasi kelas notebook.

Resume memerlukan data, manifest, monitor, kode, konfigurasi, versi pustaka/Python, serta device yang kompatibel. Satu runtime menulis satu run. Tidak menjanjikan hasil numerik identik lintas CPU/GPU atau versi PyTorch.

### Output dan validasi

Run baru memakai prefix `stage3_`. Simpan leaderboard, prediksi OOF/probabilitas, hasil per seed/fold, pilihan epoch, history, validation, Test referensi, rincian kesalahan, model terkunci, grafik, dan laporan. `model_lock.json` memuat checksum preprocessing/bobot kandidat yang dievaluasi.

Uji sintetis memeriksa pemisahan peran, preprocessing train-only, snapshot bobot, tiga aturan checkpoint, probabilitas ensemble, cross-entropy ekstrem, resume terinterupsi, pemuatan ulang model, dan perlindungan Test sebelum/sesudah lock. Setelah training, cocokkan metrik dengan prediksi, audit checksum serta hash data, dan periksa jumlah pekerjaan. Perbarui ringkasan tanpa mengubah hasil tahap sebelumnya.

Tahap 3 selesai ketika seluruh pekerjaan terarah, evaluasi kandidat terkunci, dan laporan tersedia. Jika skor masih rendah, laporkan bukti yang ditemukan. Jangan mengganti dataset atau membuka batch/search baru secara otomatis.

## Audit penutup: selesai

Setelah tiga tahap, audit penutup memeriksa sumber label/performa dari rekaman kanonis, deskripsi fitur menurut blok/sesi, dan enam window benar/salah dari prediksi Test tersimpan. Perhitungan sensor pada enam kasus menggunakan rumus agregasi terpisah; penerimaan paket RR tetap mengikuti protokol kausal sumber.

Gate redundansi diperiksa pada fitting fold saja: minimal 20 window per sesi-fold, kedua fitur bervariasi, dan R² hubungan affine minimal 1−10⁻¹⁰. Minimal separuh sel layak dan tiga sesi harus mendukung hubungan tersebut sebelum kontrol dijalankan. Gate lolos pada 86/106 sel; kontrol menghapus `iris_size_delta_mean` saja, memakai Extra Trees tetap dan delapan fitur kandidat per split. Dua normalisasi × lima fold menghasilkan tepat 10 fitting diagnostik, tanpa fitting final, evaluasi validation/Test baru, atau pemilihan pemenang baru.

Audit mempertahankan tiga CSV, notebook, model lock, serta hasil tahap 1–3. Bukti di `outputs/closing_audit/` mencakup ledger tertaut, perhitungan sensor, deskripsi blok, hasil kontrol, dan hash sebelum/sesudah. [SUMMARY.md](SUMMARY.md) dan [DIAGNOSIS.md](DIAGNOSIS.md) menutup rangkaian ini. Tidak ada batch/search lanjutan yang berjalan otomatis.
