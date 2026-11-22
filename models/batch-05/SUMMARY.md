# Ringkasan Batch 05

## Cakupan pelaksanaan

Tahap 1, 2, dan 3 serta audit penutup selesai. Tahap 1 memeriksa dataset dan membekukan split/monitor. Tahap 2 menjalankan kontrol tabular dan learning curve. Tahap 3 menjalankan neural terarah serta evaluasi satu model terkunci pada Test referensi. Training resmi tetap memakai window 25 detik dan 10 fitur core. Audit penutup menambahkan kontrol diagnostik core9 pada fold Train saja, tanpa mengganti hasil resmi. Bagian awal dokumen merangkum tahap 2; hasil tahap 3 dan audit penutup berada pada bagian akhir. [DIAGNOSIS.md](DIAGNOSIS.md) memuat diagnosis lengkap beserta bukti sumbernya.

Run tahap 2: [`stage2_20261122T104602264199Z_f8aacc8829`](outputs/exp01/stage2_20261122T104602264199Z_f8aacc8829/RESULTS.md). Notebook yang sama menyimpan keluaran pelaksanaannya di [exp01](exp01/EyeHeartCognitiveLoad.ipynb).

Data fitting memakai 2.724 window: 924 Low, 875 Medium, dan 925 High. Validation terdiri dari 870 window: 278 Low, 292 Medium, dan 300 High. Sebanyak 683 window Test tetap dipisahkan dan tidak dievaluasi pada tahap 2; tahap 3 mengevaluasi satu model terkunci. Pembagian menggunakan blok 240 detik dan guard 45 detik tanpa history tambahan.

## Eksperimen dan hasil CV

Tiga classifier memakai konfigurasi tetap pada tiga normalisasi. Dummy memprediksi kelas terbanyak pada fitting sebagai pembanding. Seluruh kandidat menggunakan lima fold yang sama; statistik normalisasi berasal dari fitting masing-masing fold. Tidak ada search, balancing, perubahan label, atau fitur tambahan.

| Model | Normalisasi | OOF accuracy | OOF Macro-F1 |
|---|---|---:|---:|
| Extra Trees | Session-robust | **47,03%** | **46,86%** |
| HistGradientBoosting | Session-robust | 44,49% | 44,44% |
| Logistic Regression | Session-centered | 44,64% | 43,97% |
| Logistic Regression | Session-robust | 44,27% | 43,91% |
| Extra Trees | Session-centered | 42,80% | 42,66% |
| HistGradientBoosting | Session-centered | 40,79% | 40,80% |
| HistGradientBoosting | Global | 35,43% | 35,48% |
| Extra Trees | Global | 34,03% | 34,11% |
| Logistic Regression | Global | 31,39% | 31,32% |
| Dummy | Global | 29,88% | 23,55% |

OOF menggabungkan satu prediksi held-out untuk setiap baris Train. Angka ini berbeda dari rata-rata skor lima fold karena jumlah baris dan proporsi kelas tiap fold tidak persis sama. [Leaderboard](outputs/exp01/stage2_20261122T104602264199Z_f8aacc8829/cv_leaderboard.csv) dan [ringkasan fold](outputs/exp01/stage2_20261122T104602264199Z_f8aacc8829/cv_fold_summary.csv) menyimpan kedua jenis informasi tersebut.

Normalisasi session-robust meningkatkan Macro-F1 OOF Extra Trees dari 34,11% menjadi 46,86%. Pada lima fold, Macro-F1 session-robust juga lebih tinggi daripada global. Hasil ini mendukung pengujian normalisasi per sesi, tetapi hasil validation berikut menunjukkan manfaatnya belum stabil di seluruh pembagian.

![Perbandingan OOF](outputs/exp01/stage2_20261122T104602264199Z_f8aacc8829/figures/cv_macro_f1.png)

## Kandidat terpilih dan validation

Aturan yang dibakukan memilih **Extra Trees + session-robust** dari OOF sebelum membaca validation. Semua kelas mempunyai recall positif dan fitting konvergen.

| Model pada seluruh Train | Validation accuracy | Validation Macro-F1 | Kedudukan |
|---|---:|---:|---|
| Extra Trees + session-robust | **44,83%** | **43,90%** | Kandidat terpilih melalui CV |
| Extra Trees + global | 57,01% | 55,96% | Kontrol; bukan pengganti pemenang setelah melihat validation |
| Logistic Regression + global | 39,66% | 39,66% | Kontrol |
| Dummy | 34,48% | 17,09% | Pembanding |

Nilai tertinggi pada tabel validation adalah 57,01%, tetapi performa OOF kontrol tersebut hanya 34,03% accuracy. Karena itu, hasil validation tertinggi tidak dijadikan alasan mengganti pemenang. Ketidaksesuaian peringkat CV dan validation menunjukkan sensitivitas hasil terhadap bagian development yang diperiksa; tidak ada bukti pencapaian target 70%.

Kandidat terpilih mencapai accuracy fitting 95,23%, jauh di atas validation 44,83%. Pada validation, model mengenali 84 dari 278 Low, 118 dari 292 Medium, dan 188 dari 300 High. Recall masing-masing adalah 30,22%, 40,41%, dan 62,67%. Tidak ada kelas yang hilang sepenuhnya, tetapi kesalahannya masih banyak, terutama Low.

![Confusion matrix validation](outputs/exp01/stage2_20261122T104602264199Z_f8aacc8829/figures/confusion_validation.png)

## Kontrol pembagian waktu

Kontrol berikut menggunakan 217 window validation yang sama, dengan model dan normalisasi global yang sama. Pembagian referensi memiliki 2.677 baris fitting; development baru 2.724.

| Model | Pembagian | Accuracy bersama | Macro-F1 bersama |
|---|---|---:|---:|
| Logistic Regression | Referensi | 45,16% | 44,47% |
| Logistic Regression | Development baru | 44,24% | 43,40% |
| Extra Trees | Referensi | 52,07% | 50,98% |
| Extra Trees | Development baru | 50,23% | 49,23% |

Perbaikan cakupan sumber dan pengurangan baris yang terkena purge belum menghasilkan kenaikan skor pada kontrol bersama ini. Komposisi fitting, retensi baris, dan guard ikut berubah. Hasil tersebut tidak membuktikan bahwa seluruh aspek pembagian baru lebih buruk, tetapi menolak anggapan bahwa pembagian yang lolos pemeriksaan teknis akan langsung memperbaiki klasifikasi.

## Learning curve

Gunakan kandidat terpilih dengan validation fold yang tetap. Angka berikut merupakan rata-rata fold.

| Blok fitting | Accuracy fitting | Accuracy validation fold | Macro-F1 validation fold |
|---|---:|---:|---:|
| 25% | 94,70% | 35,89% | 33,57% |
| 50% | 95,58% | 40,31% | 39,12% |
| 75% | 95,70% | 45,55% | 44,77% |
| 100% | 95,10% | 47,51% | 47,26% |

Penambahan blok fitting pada kontrol ini meningkatkan skor validation fold. Namun, jarak terhadap skor fitting tetap sekitar 48 poin persentase pada 100%. Bukti tersebut menunjukkan gap generalisasi yang besar pada kandidat ini. Kurva belum membuktikan bahwa penambahan data tertentu akan mencapai 70%, dan tidak menentukan sendiri apakah sumber keterbatasan berasal dari fitur, protokol label, atau karakteristik rekaman.

![Learning curve](outputs/exp01/stage2_20261122T104602264199Z_f8aacc8829/figures/learning_curve.png)

## Rincian kesalahan dan keputusan berikutnya

Pada validation, 42 window tidak memiliki sumber sesi-kelas yang tersedia pada Train; model benar pada 12 window atau 28,57%. Pada 828 window yang memiliki dukungan sumber, model benar pada 378 atau 45,65%. Menghapus masalah cakupan sumber saja belum menjelaskan seluruh kesalahan: performa pada kelompok yang didukung pun masih rendah.

Low berfase active_task dikenali pada 24 dari 79 window; mixed Low pada 52 dari 167. Hasil Low yang rendah pada dua konteks tersebut tidak terbatas pada satu fase nonpengerjaan. [Rincian validation](outputs/exp01/stage2_20261122T104602264199Z_f8aacc8829/validation_breakdown.csv) memuat jumlah baris dan jawaban benar tiap kelompok.

Pada akhir tahap 2, gate untuk **merencanakan** satu uji neural terarah lolos: kandidat mengenali tiga kelas pada OOF/validation dan mengungguli Dummy di lima fold. Gate tersebut merupakan syarat melanjutkan pemeriksaan pipeline, bukan penilaian bahwa model sudah baik. Pelaksanaan tahap 3 dijelaskan di bawah.

Pekerjaan berikutnya tetap terarah pada mekanisme monitor, pemilihan checkpoint, dan konsistensi durasi/jadwal fitting satu arsitektur. Jangan memperluas pencarian besar atau mengganti label berdasarkan skor tahap ini. Normalisasi per sesi perlu tetap menjadi kontrol karena manfaat CV-nya belum bertahan pada validation utama.

## Validasi pelaksanaan

Sebanyak 71 pekerjaan fitting penelitian selesai. Setiap fitting akhir memverifikasi kesamaan prediksi setelah estimator dan statistik preprocessing dimuat ulang. Pemeriksaan arsip memverifikasi 225 artefak pekerjaan. Seluruh classifier konvergen dan tidak mencatat warning pada run ini.

Dataset, label, manifest tahap 1, dan hasil batch sebelumnya tetap dipertahankan. [Status run](outputs/exp01/stage2_20261122T104602264199Z_f8aacc8829/status.json), [gate](outputs/exp01/stage2_20261122T104602264199Z_f8aacc8829/gate.json), serta [audit prediksi dan resume](outputs/exp01/stage2_20261122T104602264199Z_f8aacc8829/audit.json) menjadi rujukan pemeriksaan.

## Tahap 3: neural dan kontrol fitting

Run: [`stage3_20261122T115353575899Z_933c7b4bcb`](outputs/exp01/stage3_20261122T115353575899Z_933c7b4bcb/RESULTS.md). Sebanyak **72 pekerjaan fitting** selesai: 48 lintasan MLP dan 24 fitting Extra Trees. Training neural berjumlah 5.760 epoch, tanpa search tambahan.

MLP menggunakan hidden layer 64/32, LayerNorm, GELU, dropout 0,2, dan AdamW dengan learning rate tetap 0,001. Tiga seed menghasilkan probabilitas yang dirata-ratakan. Setiap lintasan menyelesaikan 120 epoch; checkpoint loss monitor, Macro-F1 monitor, dan epoch terakhir berasal dari lintasan yang sama.

Baris optimizer-fitting, monitor, dan held-out tetap terpisah. Kontrol matched-data menggunakan baris fitting yang sama dengan neural. Baseline penuh mempertahankan seluruh fitting pool tahap 2, sehingga biaya menyisihkan monitor terlihat pada perbandingan kontrol.

### Hasil 13 kandidat OOF

| Kandidat | OOF accuracy | OOF Macro-F1 |
|---|---:|---:|
| Extra Trees penuh, session-robust | **47,03%** | **46,86%** |
| MLP session-robust, checkpoint Macro-F1 | 45,26% | 45,08% |
| MLP session-robust, checkpoint loss | 44,64% | 44,35% |
| MLP session-robust, epoch 120 | 43,69% | 43,52% |
| MLP session-centered, checkpoint Macro-F1 | 42,84% | 42,62% |
| MLP session-centered, checkpoint loss | 42,91% | 42,60% |
| Extra Trees matched, session-robust | 42,55% | 42,28% |
| MLP session-centered, epoch 120 | 42,36% | 42,10% |
| Extra Trees matched, session-centered | 38,84% | 38,68% |
| MLP global, epoch 120 | 34,54% | 34,72% |
| MLP global, checkpoint Macro-F1 | 34,14% | 34,28% |
| Extra Trees matched, global | 32,60% | 32,59% |
| MLP global, checkpoint loss | 25,88% | 25,73% |

Baseline penuh menghasilkan prediksi OOF yang sama dengan tahap 2. Neural terbaik belum melampauinya. Pada dukungan fitting yang disamakan, pipeline MLP session-robust memperoleh Macro-F1 45,08%, lebih tinggi daripada Extra Trees matched 42,28%. Perbandingan ini juga melibatkan ensemble tiga seed neural, sehingga tidak membuktikan keunggulan intrinsik satu arsitektur.

![OOF tahap 3](outputs/exp01/stage3_20261122T115353575899Z_933c7b4bcb/figures/stage3_oof.png)

### Checkpoint dan loss monitor

| Normalisasi MLP | Loss: OOF Macro-F1 | Macro-F1: OOF Macro-F1 | Epoch 120: OOF Macro-F1 |
|---|---:|---:|---:|
| Global | 25,73% | 34,28% | 34,72% |
| Session-centered | 42,60% | 42,62% | 42,10% |
| Session-robust | 44,35% | 45,08% | 43,52% |

Tidak ada satu aturan checkpoint yang paling baik pada seluruh normalisasi. Memaksakan epoch terakhir memperbaiki hasil global terhadap checkpoint loss, tetapi menurunkan Macro-F1 session-robust dibanding checkpoint Macro-F1. Bukti tersebut tidak mendukung aturan umum bahwa training lebih lama akan menyelesaikan masalah.

Dari 45 lintasan CV, median epoch checkpoint loss adalah 5 dan median checkpoint Macro-F1 adalah 10; rentangnya masing-masing 1–16 dan 1–120. Semua lintasan tetap menyelesaikan 120 epoch. Epoch awal bukan tanda proses training berhenti atau gagal berjalan.

Kurva menunjukkan loss fitting terus turun sementara loss monitor kemudian meningkat, termasuk setelah komposisi kelas monitor diperbaiki. Normalisasi mengubah pola dan besarnya gap, tetapi belum menghilangkan pemburukan loss monitor. Hasil ini membuktikan adanya gap fitting-monitor pada konfigurasi yang diuji, bukan membuktikan satu penyebab tunggal pada sensor atau label.

![Loss fitting dan monitor](outputs/exp01/stage3_20261122T115353575899Z_933c7b4bcb/figures/training_monitor_curves.png)

### Validation dan model terkunci

Pemenang OOF tetap **Extra Trees penuh + session-robust**. Neural terbaik menurut OOF, MLP session-robust dengan checkpoint Macro-F1, memperoleh validation accuracy 34,02% dan Macro-F1 33,63%. Checkpoint epoch 120 pada normalisasi yang sama memperoleh validation accuracy 42,53%, tetapi tidak dipromosikan setelah membaca validation.

| Bagian kandidat terkunci | Accuracy | Macro-F1 |
|---|---:|---:|
| OOF | 47,03% | 46,86% |
| Validation | 44,83% | 43,90% |
| Test referensi | **37,92%** | **37,59%** |

Identitas kandidat dan checksum preprocessing/bobot disimpan dalam [model_lock.json](outputs/exp01/stage3_20261122T115353575899Z_933c7b4bcb/model_lock.json) sebelum evaluasi Test. Hanya kandidat tersebut yang dievaluasi pada Test. Test ini sudah pernah dianalisis pada rangkaian sebelumnya; hasilnya merupakan referensi eksploratif, bukan pengujian independen baru.

Test berisi 238 Low, 211 Medium, dan 234 High. Model mengenali:

| Kelas | Benar / total | Recall | F1 |
|---|---:|---:|---:|
| Low | 80 / 238 | 33,61% | 37,21% |
| Medium | 69 / 211 | 32,70% | 32,24% |
| High | 110 / 234 | 47,01% | 43,31% |

![Test referensi](outputs/exp01/stage3_20261122T115353575899Z_933c7b4bcb/figures/confusion_test_reference.png)

Pada 648 window Test dengan dukungan sumber sesi-kelas pada Train, accuracy masih 38,73%. Pada 35 window tanpa dukungan tersebut, accuracy 22,86%. Kekurangan cakupan sumber tidak menjelaskan seluruh kesalahan.

Model benar pada 259 dari 683 window. Untuk mencapai 70%, diperlukan minimal 479 jawaban benar, atau tambahan 220. Kesalahan tersebar pada Low (158), Medium (142), dan High (124). Sebagai ilustrasi hitungan, membetulkan seluruh kesalahan Low tanpa perubahan kelas lain hanya menghasilkan 61,05%. Ilustrasi tersebut bukan prediksi dampak pengambilan data tambahan.

### Kesimpulan tahap 3

Hasil masih belum memenuhi sasaran model optimal. Perbaikan checkpoint dan monitor telah diuji pada rancangan yang dibekukan, tetapi belum menghasilkan kenaikan yang melampaui baseline. Neural juga tidak menunjukkan keuntungan validation yang stabil.

Temuan yang dapat dipastikan adalah gap fitting-monitor, sensitivitas hasil terhadap normalisasi/aturan checkpoint, serta kesalahan yang tetap besar pada interval held-out dengan cakupan kelas dan sumber yang tersedia. Eksperimen ini belum membuktikan bahwa 70% mustahil dicapai atau menetapkan satu fitur/sensor sebagai penyebab tunggal. Tahap ini tidak diikuti search baru, relabel, atau perubahan dataset otomatis.

### Audit tahap 3

Pemeriksaan arsip memverifikasi 632 artefak pekerjaan. Audit independen mencocokkan 480 tabel metrik per pekerjaan dan 144 pemuatan ulang checkpoint neural. Resume run lengkap mempertahankan hash/waktu modifikasi artefak fitting serta memakai hasil Test tersimpan tanpa evaluasi ulang.

Seluruh 33 uji lolos. Uji sintetis mencakup resume terinterupsi dengan pemulihan optimizer/RNG, preprocessing train-only, snapshot bobot, serta larangan Test sebelum model terkunci. Pemeriksaan syntax/schema notebook, lint, tipe, dan keamanan juga lolos.

[Audit independen](outputs/exp01/stage3_20261122T115353575899Z_933c7b4bcb/audit.json) memuat pencocokan metrik, reproduksi baseline tahap 2, dan pemeriksaan resume run lengkap. [Status pelaksanaan](outputs/exp01/stage3_20261122T115353575899Z_933c7b4bcb/status.json) menegaskan bahwa tiga tahap telah selesai tanpa mengubah dataset dan hasil tahap sebelumnya.

## Audit penutup

Audit penutup memakai sumber kanonis dan hasil prediksi yang sudah tersimpan. Dataset, notebook, model terkunci, dan artefak tahap 1–3 tetap utuh; 1.495 hash sebelum/sesudah cocok. Audit tidak menambahkan evaluasi validation atau Test.

| Pemeriksaan | Hasil |
|---|---|
| Ledger sumber | 23 rekaman dari 22 orang pada gabungan tiga CSV; 650 attempt tertaut ke kunjungan terakhir pada episode set/puzzle yang sama |
| Ketepatan attempt pertama | Easy 58,74%; medium 30,08%; hard 3,70%. Ada perbedaan performa tugas, tetapi ini bukan pengukuran beban mental langsung |
| Rating | Ada event fase rating, tetapi angka jawaban peserta tidak tersedia dalam sumber kanonis yang diperiksa |
| Trace sensor | Enam window Test tersimpan: satu benar/salah per kelas. Seluruh 102 pemeriksaan angka cocok; enam target cocok dengan ledger |
| Konsistensi tiga CSV | Tidak ada pertentangan target-rincian fase tersimpan atau pelanggaran eksklusi timestamp ambigu Alif |
| Variasi fitur | Pada median blok 25 detik, asosiasi marginal RMSSD dengan sesi 83,66%, dengan kelas 0,38%; ini deskripsi, bukan kontribusi kausal/batas accuracy |
| Redundansi iris | Terbukti pada 86/106 sel sesi-fold fitting yang layak, sebelum dan sesudah preprocessing |
| Kontrol core9 | Sepuluh fitting Train-only. Macro-F1 global 34,11% → 34,30%; session-robust 46,86% → 46,29%. Tidak ada perbaikan yang stabil |

![Kontrol penghapusan fitur iris](outputs/closing_audit/redundancy_control.png)

Model resmi tetap **Extra Trees penuh + session-robust**, dengan accuracy OOF **47,03%**, validation **44,83%**, dan Test referensi **37,92%**. Audit penutup tidak menghasilkan pemenang baru.

Diagnosis akhir tidak mendukung satu penyebab tunggal seperti jumlah Low, salah agregasi pada contoh yang diperiksa, atau satu fitur berulang. Gap fitting-held-out tetap besar, variasi dasar antarsesi kuat, dan perubahan fitur antarkelas tidak searah pada seluruh sesi. Cleaning dan pembenahan pipeline belum membuktikan bahwa informasi sensor yang tersedia cukup untuk mencapai sasaran 70%.

Batch ditutup untuk pelaporan; tidak ada training tambahan yang perlu dijalankan untuk menyelesaikannya. Bukti sekarang tidak membuktikan 70% mustahil, tetapi juga tidak memberi dasar untuk menjanjikan pencapaiannya melalui batch pencarian berikutnya pada data yang sama. [Diagnosis lengkap](DIAGNOSIS.md) membedakan hasil terukur, keterbatasan bukti, dan kebutuhan penelitian yang berada di luar audit ini.

[Hasil audit penutup](outputs/closing_audit/audit_result.json) dan [pemeriksaan tambahan](outputs/closing_audit/supplement_checks.json) menyimpan status. Seluruh **43 uji otomatis** serta pemeriksaan syntax, lint, tipe, dan keamanan kode audit lolos.
