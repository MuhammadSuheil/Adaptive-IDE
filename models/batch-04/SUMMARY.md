# Hasil eksperimen Batch 04

Tanggal pemeriksaan: 22 November 2026.

## Kesimpulan

Exp01, exp03, dan exp04 selesai. Ketiganya belum menghasilkan classifier Low, Medium, dan High yang mencapai accuracy test 70%. Exp02 tidak masuk evaluasi karena belum memiliki hasil lengkap. Peringatan bahwa GPU tidak digunakan bukan hasil evaluasi model dan tidak membuktikan bahwa exp02 akan memperoleh skor yang sama dengan eksperimen lain.

Hasil accuracy test tertinggi yang teramati adalah **44,80%**, dari rata-rata probabilitas lima seed TabM pada window 45 detik di exp03. Model tersebut benar pada 237 dari 529 window. Pembanding kelas mayoritas Train, yang selalu memprediksi Low pada window ini, benar pada 215 window atau 40,64%.

Dengan demikian, TabM menambah 22 prediksi benar atau 4,16 poin accuracy di atas Dummy. Macro-F1 naik dari 19,27% menjadi 41,77%, tetapi pengenalan Medium dan High masih lemah. Hasil ini belum memadai untuk sasaran tim.

Diagnosis dan kontrol tambahan tersedia di [DIAGNOSIS.md](DIAGNOSIS.md). Angka audit yang dapat ditelusuri tersedia di [diagnostic_checks.json](diagnostic_checks.json). Kontrol diagnostik tidak mengganti skor resmi atau model yang telah dipilih pada validation.

## Data dan prosedur

Ketiga run menggunakan CSV terbaru setelah kebijakan kelayakan fitur dan pengecualian window lonjakan RR. Hash CSV yang tercatat pada run cocok dengan sumber lokal saat rekap. Dataset tidak diubah selama analisis ini.

| Window dan step | Low | Medium | High | Total |
|---|---:|---:|---:|---:|
| 15 dan 5 detik | 1.687 | 1.763 | 1.803 | 5.253 |
| 25 dan 5 detik | 1.549 | 1.487 | 1.572 | 4.608 |
| 45 dan 5 detik | 1.394 | 966 | 1.149 | 3.509 |

Input utama adalah core 10 fitur dan core-head 14 fitur. Exp01 juga menguji eye-only, HRV-only, fitur turunan, dan seleksi fitur. Identitas, difficulty, fase, soal, serta timestamp absolut tidak menjadi predictor. Gaze yang tidak lolos kelayakan spasial tidak dimasukkan ke profil core/core-head.

Evaluasi memakai blok waktu 240 detik, purge konservatif 105 detik, lima fold di dalam Train, dan pengulangan CV untuk shortlist. Validation akhir memilih model sebelum test dibaca. Orang yang sama boleh hadir pada beberapa bagian; ini bukan evaluasi leave-one-participant-out. Model untuk test sama dengan bundle pilihan validation, tanpa refit setelah seleksi.

Hash split ketiga run sama: `da5549e9d7fd62ce0f6cb0375f2dd486accace366f84b125a3f77b1c0b5709c7`.

| Window | Train | Validation akhir | Test | Purged |
|---|---:|---:|---:|---:|
| 15 detik | 3.019 | 609 | 768 | 857 |
| 25 detik | 2.677 | 509 | 683 | 739 |
| 45 detik | 2.049 | 381 | 529 | 550 |

Window dalam bagian yang sama masih dapat overlap. Ribuan baris tersebut bukan ribuan pengamatan sensor yang independen. Endpoint test per-window identik di antara ketiga eksperimen, tetapi endpoint test antarwindow berbeda.

## Perlakuan yang dijalankan

| Eksperimen | Perlakuan | Baris konfigurasi pada leaderboard CV | Bentuk model pada validation akhir |
|---|---|---:|---:|
| Exp01 | Logistic Regression, LDA, KNN, SVM, forest/boosting; fitur turunan, MI/PCA, sensor ablation, class weight, ROS, SMOTE, ordinal/hierarchy, ensemble | 512 | 30 |
| Exp03 | MLP, residual MLP, TabM, Transformer; tuning regularisasi, class weight, preprocessing, ensemble seed dan keluarga model | 287 | 57 |
| Exp04 | GRU dan TCN; riwayat 15/30/60 detik; pembanding MLP dan HistGradientBoosting dengan input saat ini atau ringkasan history; ensemble | 157 | 48 |

Jumlah konfigurasi mencakup tiga window dan kontrol Dummy. Konfigurasi tersebut bukan observasi independen atau jumlah dataset baru. Trial yang dipruning tidak dihitung sebagai kandidat CV lengkap.

## Hasil pemenang setiap window

Semua model dalam tabel ini menang melalui validation akhir. Menampilkan skor test tidak mengubah penetapan pemenang.

| Eksperimen | Window | Model pilihan validation | Accuracy validation | Accuracy test | Macro-F1 test |
|---|---:|---|---:|---:|---:|
| Exp01 | 15 detik | Stacking | 41,54% | 28,78% | 28,50% |
| Exp01 | 25 detik | SVM RBF core | 42,83% | 31,63% | 31,87% |
| Exp01 | 45 detik | CatBoost core | 42,26% | 38,37% | 36,57% |
| Exp03 | 15 detik | TabM core-head | 42,69% | 36,07% | 26,50% |
| Exp03 | 25 detik | Residual MLP core | 40,67% | 33,24% | 31,99% |
| Exp03 | 45 detik | TabM core-head, rata-rata lima seed | 49,61% | 44,80% | 41,77% |
| Exp04 | 15 detik | Stacking | 40,07% | 34,11% | 33,73% |
| Exp04 | 25 detik | Hard voting | 44,20% | 34,26% | 33,72% |
| Exp04 | 45 detik | GRU core, history 60 detik, rata-rata lima seed | 52,76% | 40,08% | 26,21% |

Sumber: rekap [exp01][e1-test], [exp03][e3-test], dan [exp04][e4-test], dicocokkan dengan `selection/record.json` masing-masing run.

### Model utama tiap notebook

| Eksperimen | Model utama | Accuracy test | Balanced accuracy | Recall Low | Recall Medium | Recall High |
|---|---|---:|---:|---:|---:|---:|
| Exp01 | SVM 25 detik | 31,63% | 31,22% | 40,76% | 20,85% | 32,05% |
| Exp03 | TabM 45 detik | 44,80% | 41,86% | 66,98% | 27,87% | 30,73% |
| Exp04 | GRU 45 detik | 40,08% | 35,23% | 89,30% | 16,39% | 0,00% |

Tidak ada satu model utama gabungan yang ditetapkan melalui eksperimen baru setelah test dibaca. TabM disebut tertinggi berdasarkan hasil test yang sudah teramati, bukan berdasarkan seleksi pada test yang belum pernah dilihat.

### Model kontrol yang perlu dicatat

Pada exp04, HistGradientBoosting dengan ringkasan history 60 detik pada window 45 detik mencapai accuracy test 44,42% dan Macro-F1 42,02%. TCN dengan history 15 detik pada window 25 detik mencapai accuracy 44,07% dan Macro-F1 43,92%. Keduanya merupakan kontrol yang dipilih melalui validation untuk perannya, bukan model utama notebook.

Hasil tersebut menunjukkan bahwa model utama menurut accuracy validation belum selalu menjadi model dengan test terbaik. Hasil TCN juga tidak menetapkan bahwa neural temporal konsisten mengungguli model tabular: pada window lain, urutannya berbeda.

| Window | Accuracy Dummy pada test yang sama |
|---|---:|
| 15 detik | 32,81% |
| 25 detik | 34,26% |
| 45 detik | 40,64% |

Seluruh pemenang exp01 berada di bawah Dummy dalam accuracy. Macro-F1 mereka lebih tinggi karena mengenali lebih dari satu kelas, tetapi belum menunjukkan pemisahan kelas yang kuat. Model utama GRU exp04 juga berada sedikit di bawah Dummy dalam accuracy dan tidak memprediksi High.

## Temuan utama

1. **Gap generalisasi tabular besar.** SVM utama exp01 mencapai 93,91% Train, 42,83% validation akhir, dan 31,63% test. CatBoost 45 detik mencapai 95,22% Train tetapi 38,37% test.
2. **Accuracy-first menerima kehilangan satu kelas.** GRU utama exp04 tidak memprediksi High pada validation maupun test. TabM 15 detik tidak memprediksi Medium pada kedua bagian itu.
3. **Monitor neural perlu diperiksa.** Pada exp03, 16 dari 24 recipe finalist neural memakai durasi akhir satu epoch. Pada exp04, 12 dari 15 recipe finalist neural memakai satu epoch. Ini durasi hasil pemilihan checkpoint, bukan bukti bahwa CV hanya dijalankan satu epoch.
4. **Normalisasi sesi belum menyelesaikan masalah.** Dalam 30 fitting diagnostik terbatas, beberapa normalisasi sesi memperbaiki validation Logistic Regression, tetapi hasil tertinggi kontrol tersebut baru 46,46%. Skor kontrol ini bukan test akhir.
5. **Angka tinggi pada pembagian baris acak bisa menyesatkan.** Uji 1-nearest-neighbor menghasilkan OOF 84,17–98,39% ketika window training diacak, tetapi 41,40–49,00% pada blok purged. Hampir semua baris penilaian acak mempunyai window latihan yang overlap. Uji tersebut dilakukan di dalam Train, bukan dengan mengubah Test resmi.

Pembahasan sebab, batas kesimpulan, dan urutan pemeriksaan berikutnya ada pada [DIAGNOSIS.md](DIAGNOSIS.md). Belum ada bukti bahwa satu perubahan akan membawa model melewati 70%, dan hasil sekarang juga tidak membuktikan bahwa 70% mustahil secara teoretis.

## Integritas pemeriksaan

Pemeriksaan menghitung ulang metrik dari 26 tabel prediksi test dan mencocokkan label dengan CSV sumber. Checksum 1.457 artefak pada checkpoint fitting, selection, test, dan complete yang diperiksa cocok; tidak ada berkas yang hilang pada cakupan itu. Notebook tidak menyimpan output error. Pemeriksaan ini tidak menyatakan bahwa seluruh checkpoint trial telah diperiksa satu per satu.

Analisis tambahan tidak mengubah CSV, notebook, bobot, split, atau hasil resmi. Tim tidak perlu menunggu exp02 untuk mendokumentasikan tiga run lengkap ini. Hasil exp02 tetap tidak diketahui.

## Sumber

- [Rancangan Batch 04](PLAN.md).
- [Laporan kualitas dataset](../../analysis_sessions/dataset_fusion/output/v2_session_phase/QUALITY_REPORT.md).
- [Audit numerik dan rancangan kontrol](diagnostic_checks.json).
- [Rekap exp01][e1-test], [exp03][e3-test], [exp04][e4-test].

[e1-test]: outputs/exp01/20261121T025639248779Z_9443342741/test_summary.csv
[e3-test]: outputs/exp03/20261121T073428389875Z_8be6ef1c10/test_summary.csv
[e4-test]: outputs/exp04/20261121T090223169904Z_62687ffa6e/test_summary.csv
