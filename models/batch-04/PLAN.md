# Batch 4: Rancangan Optimasi Model

## Tujuan

Mencari model klasifikasi Low/Medium/High dari fitur eye tracking dan HRV pada tiga dataset `good_acceptable` terbaru. Batch ini menggunakan fitur inti setelah pemeriksaan kualitas dan pengecualian window lonjakan RR. Dataset dan label tidak diubah selama pemodelan.

Empat notebook mengelompokkan percobaan menurut pertanyaan yang diuji. Masing-masing mencakup training, pemilihan model, test, serta penyimpanan. Tidak ada ketergantungan pada hasil notebook lain.

Accuracy menjadi kriteria pemilihan utama; Macro-F1, balanced accuracy, recall kelas terlemah, variasi fold, dan ukuran input melengkapi penilaian. Sasaran accuracy test 75% tidak digunakan untuk menghentikan pelaporan hasil yang lebih rendah.

## Data dan evaluasi

- Input utama: 10 fitur core. Core-head menambahkan mean/std pitch dan yaw bila lengkap. Eye-only memakai delapan fitur core eye; HRV-only memakai RMSSD dan SDNN.
- Target: `cognitive_load_label` langsung. Identitas, fase, difficulty, kualitas, waktu absolut, dan soal bukan predictor classifier.
- Tidak melakukan imputasi core atau gaze gagal. Pemeriksaan rentang fisik dilakukan sebelum transformasi. Deviasi negatif setelah pemusatan tetap sah.
- Tiga panjang window tidak ditumpuk sebagai satu dataset. Endpoint yang sama memakai assignment bagian yang sama.
- Evaluasi internal mengizinkan orang yang sama pada Train, validation akhir, dan test. Tidak menggunakan LOPO.
- Blok waktu 240 detik ditargetkan ke 70/15/15. Dari 128 kandidat seed 42–169, pilih penempatan dengan biaya metadata terendah: bobot 4 untuk deviasi proporsi kelas, 2 untuk fase Low, 1 untuk sumber sesi–kelas, dan penalti 0,5 untuk proporsi purge. Kelengkapan tiga kelas wajib.
- Fase Low untuk stratifikasi dibedakan menjadi active, mixed, dan lainnya. Label sumber tidak diganti untuk menyeimbangkan bagian.
- Purge 105 detik mengendalikan dukungan window maksimum 45 detik ditambah riwayat 60 detik. Persentase baris setelah purge tidak harus tepat 70/15/15.
- Lima fold internal pada Train memakai pengelompokan blok waktu, dengan seed 42. Shortlist diperiksa kembali pada lima fold seed 137. Ini bukan pembagian satu fold per peserta.
- OOF repeat pertama dan kedua dilaporkan sebagai pengulangan evaluasi, bukan tambahan observasi independen.
- Pemilihan akhir menggunakan validation yang terpisah dari CV. Pemenang per-window dan model utama dikunci sebelum test dibuka. Tidak ada refit setelah seleksi.

Pemeriksaan split pada dataset saat penyiapan menghasilkan seed 124 dan hash `da5549e9d7fd62ce0f6cb0375f2dd486accace366f84b125a3f77b1c0b5709c7`:

| Window | Train | Validation | Test | Purged |
|---|---:|---:|---:|---:|
| 15 detik | 3.019 | 609 | 768 | 857 |
| 25 detik | 2.677 | 509 | 683 | 739 |
| 45 detik | 2.049 | 381 | 529 | 550 |

Angka/hash tersebut hanya pemeriksaan pembagian, bukan hasil training. Rebuild dengan dataset berbeda menghasilkan identitas baru dan memerlukan run baru.

## Exp01: Tabular dan representasi sensor

Bandingkan Dummy, Logistic Regression, shrinkage LDA, KNN, SVM, Random Forest, Extra Trees, HistGradientBoosting, XGBoost, dan CatBoost. Dummy merupakan pembanding mayoritas Train, bukan calon model sensor utama.

Screening menggunakan dua preset pada core dan core-head. Empat keluarga teratas memperoleh masing-masing 40 trial TPE per window. Konfigurasi identik menggunakan kembali checkpoint CV yang sama.

Ruang pencarian utama:

| Kelompok | Parameter |
|---|---|
| Linear/SVM | Regularisasi C; kernel linear/RBF dan gamma bila relevan |
| LDA/KNN | Shrinkage; jumlah tetangga, pembobotan jarak, dan metrik L1/L2 |
| Forest/Extra Trees | 300/600 pohon, ukuran leaf 1–32, kedalaman 4/8/16/tanpa batas, fraksi fitur |
| Boosting | Iterasi 200–1.000 sesuai keluarga, kompleksitas pohon, ukuran leaf/child, learning rate, dan L2 |
| Preprocessing | Standard, robust, atau log-HRV + robust; semua fitting memakai Train |
| Representasi | Core/core-head, fitur turunan, semua fitur/MI hingga delapan fitur/PCA 95% |

Fitur turunan adalah rasio RMSSD/SDNN, perbedaan relatif EAR, variasi relatif EAR, dan variasi relatif estimasi iris. Setelah tuning, dua keluarga berbeda diuji kembali dengan ablation fitur/seleksi yang terkontrol, eye-only/HRV-only, serta tanpa balancing/class weight/ROS/SMOTE. Resampling hanya pada core terskala tanpa PCA, fitur turunan, atau ringkasan riwayat.

Dua keluarga probabilistik terbaik juga membandingkan multiclass, ordinal, dan hierarchy. Ordinal memodelkan dua probabilitas kumulatif dengan koreksi crossing; hierarchy memodelkan Low versus elevated lalu Medium versus High. SVM mempertahankan keputusan kelas langsung dan tidak dianggap menghasilkan probabilitas.

## Exp02: Acuan dan riwayat statistik

Gunakan Logistic Regression, SVM RBF/linear, dan HistGradientBoosting untuk membandingkan acuan global, pemusatan median per sesi training, skala robust per sesi training, dan acuan kausal.

Tuning gabungan beranggaran 60 trial per window. Kandidat classifier terbaik—bukan Dummy—menjalani ablation riwayat 15/30/60 detik dan representasi saat ini versus tambahan median, dispersi, serta tren sensor sebelumnya.

Acuan sesi hanya memakai fitting data. Sesi tanpa acuan memakai statistik global fitting. Acuan kausal tidak membaca label atau data masa depan. Statistik riwayat memerlukan minimal dua window sebelumnya; representasi saat ini menjadi fallback bila belum cukup.

Riwayat berasal dari urutan window valid yang tersedia dalam CSV. Celah endpoint lebih dari 5 detik mereset konteks. Tidak ada interpolasi, penerusan nilai melewati celah, atau penghapusan baris hanya karena history pendek.

Kontrol satu-window dan ringkasan riwayat dikunci lewat validation, kemudian dinilai pada endpoint test yang sama dengan pemenang. Hasil menurut kecukupan riwayat ikut dilaporkan.

## Exp03: Deep learning tabular

Bandingkan MLP, residual MLP, TabM resmi, dan Transformer tabular kecil. Setiap arsitektur memperoleh 30 trial per window, ditambah baseline arsitektur.

Ruang pencarian: width 32/64/128/256, depth 1–3, dropout 0–0,6, learning rate `1e-4`–`3e-3`, weight decay `1e-6`–`1e-1`, batch 64/128/256, label smoothing 0/0,05/0,1, core/core-head, serta tanpa/balanced class weight. TabM memakai 16 anggota internal. Transformer memakai empat attention head.

AdamW, cosine schedule, gradient clipping 1, maksimum 200 epoch, dan patience 20 digunakan. Cross-entropy monitor tanpa pembobotan memilih checkpoint; monitor berasal dari bagian training dan dipurge dari fitting. Validation penilaian tidak memilih epoch.

Fitting seluruh Train memakai median epoch terbaik kandidat pada fold. Kandidat akhir membandingkan seed 42 dengan rata-rata probabilitas seed 42–46. Seed tidak dipilih menurut hasil test. Training accuracy CV neural dihitung pada bagian yang benar-benar dipakai optimizer, bukan gabungan fitting dan monitor.

## Exp04: Model temporal

Bandingkan GRU dan TCN kausal, masing-masing 30 trial per window. Lookback 15/30/60 detik menjadi parameter pencarian.

GRU memakai width 16/32/64 dan satu/dua layer. TCN memakai proyeksi kanal 16/32/64, kernel 3, dilation 1/2/4, serta satu/dua pengulangan pada tiap dilation. Pengaturan optimizer/monitor mengikuti exp03.

Urutan dibatasi sesi dan kontinuitas endpoint 5 detik. Panjang variabel didukung, termasuk satu window. GRU menggunakan panjang urutan nyata; TCN membaca posisi terakhir yang valid. Padding tidak dianggap sebagai pengukuran nol.

Kontrol MLP dan HistGradientBoosting menggunakan satu-window serta ringkasan history 15/30/60 detik. Kontrol dipertahankan dalam tahap seleksi meskipun tidak masuk delapan kandidat teratas, agar manfaat urutan dapat dibandingkan pada endpoint yang sama.

## Shortlist dan ensemble

Dua recipe teratas tiap keluarga menjalani CV repeat kedua. Maksimal delapan kandidat terbaik ditambah kontrol yang diwajibkan masuk validation akhir.

Setiap kandidat dilatih pada seluruh Train. Voting dan rata-rata probabilitas membandingkan maksimal tiga keluarga berbeda. Rata-rata probabilitas hanya memakai komponen yang benar-benar menghasilkan probabilitas. Stacking memakai Logistic Regression dengan C=0,1 dan StandardScaler pada prediksi OOF repeat pertama; label validation/test tidak dipakai melatih meta-model.

Pemilihan diurutkan menurut accuracy, Macro-F1, recall minimum, simpangan accuracy fold, jumlah fitur, dan ID untuk tie deterministik. Pemenang keseluruhan harus mengikuti aturan validation yang sama; skor test tertinggi dapat dibahas sebagai observasi, bukan mengganti pilihan resmi.

## Pelaksanaan dan validasi

Notebook membaca CSV dari Drive dan menyimpan output di folder run masing-masing. Dependency dipasang pada cell setup; tidak ada shared, skrip pendamping, requirements, atau notebook persiapan wajib. Tidak ada mode smoke/full.

Checkpoint fold memuat skor, prediksi OOF, dan riwayat neural bila ada. Model disimpan saat fitting seluruh Train dan tahap selection, sehingga seluruh model fold tidak perlu memenuhi Drive. State tuning disimpan sebelum dan sesudah trial. Trial yang terputus melanjutkan recipe yang sama dan menggunakan kembali fold lengkap.

Identitas run mencakup hash CSV/split/implementasi, anggaran, serta environment. Checksum memverifikasi artefak sebelum penanda selesai. Dataset, implementasi, atau environment berbeda tidak boleh melanjutkan run lama.

Validasi penyiapan meliputi syntax/schema notebook, split nyata tanpa training penelitian, semua keluarga model pada data sintetis, batas riwayat, fitting Train-only, resampling/ordinal/hierarchy, stacking, serialisasi native booster, bobot neural, serta resume penuh/terputus. Pencarian penelitian penuh baru dijalankan di Colab.
