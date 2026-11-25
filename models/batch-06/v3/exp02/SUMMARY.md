# Hasil ablasi baseline Exp02

Run `ablation_20261124T034359Z_3ff2c1ec` selesai pada 24 November 2026 WIB. Notebook mencatat 45 fitting CV, empat fitting validation, dan keluaran laporan akhir tanpa exception. Keenam pemeriksaan metrik raw/gabungan terhadap Exp01 bernilai cocok.

Normalisasi RMSSD sendiri memperbaiki kedua protokol CV terhadap raw. Normalisasi eye sendiri menurunkan accuracy dan Macro-F1 pada kedua CV. Gabungan tetap tertinggi pada OOF peserta yang ditahan, tetapi keunggulannya terhadap RMSSD-only belum bertahan pada kontrol waktu dan validation historis. Ini membatasi klaim bahwa satu transformasi selalu terbaik.

## Cakupan bukti

Analisis memakai [notebook hasil run](../outputs/exp02/ablation_20261124T034359Z_3ff2c1ec/executed_notebook.ipynb). ZIP yang disertakan pengguna masih run Exp01 `baseline_20261123T192338Z_8d8cee5e`, bukan ZIP ablation Exp02. Tabel, laporan Markdown, dan grafik diekstrak dari output notebook. Berkas CSV prediksi per window, JSON pekerjaan, plan runtime, serta bobot Exp02 belum tersedia untuk audit terpisah.

Source cell notebook sama dengan notebook yang disiapkan. Lima belas kontras dihitung ulang dari tabel tampilan dan cocok dalam toleransi pembulatan enam desimal. Empat belas tabel metrik mempunyai accuracy yang konsisten dengan recall dan support masing-masing kelas. Enam kontrol dicocokkan dengan CSV asli Exp01. Pemeriksaan ini tidak setara dengan menghitung ulang Macro-F1 dari seluruh prediksi Exp02 atau memverifikasi checksum model.

[Audit notebook](../outputs/exp02/ablation_20261124T034359Z_3ff2c1ec/analysis/notebook_audit.json) menyimpan cakupan pemeriksaan. Tidak ada fitting, inference, penggantian split, atau perubahan data saat analisis.

## OOF tiga kelas pada peserta yang ditahan

Semua varian tetap menerima sepuluh fitur HRV dan eye. Kata only menunjukkan normalisasi yang aktif, bukan penghapusan modalitas. Dataset dan fold dibekukan pada empat peserta dengan 671 window Train: 264 Low, 180 Medium, dan 227 High.

| Normalisasi | Accuracy | Macro-F1 tiga kelas | Mean accuracy per orang | Benar / window |
|---|---:|---:|---:|---:|
| Raw | 35,77% | 33,96% | 37,14% | 240/671 |
| RMSSD-only | 38,15% | 35,95% | 39,35% | 256/671 |
| Eye-only | 31,74% | 31,18% | 31,45% | 213/671 |
| Gabungan | **41,43%** | **39,85%** | **41,93%** | 278/671 |
| Dummy | 39,34% | 18,82% | 40,33% | 264/671 |

Jumlah benar diperoleh dari accuracy/recall tampilan dan support integer yang tersedia, bukan dari tabel prediksi per window yang belum disertakan. Gabungan menambah 38 jawaban benar terhadap raw dan 22 terhadap RMSSD-only. Terhadap Dummy, selisih accuracy gabungan hanya 14 jawaban atau 2,09 poin, meskipun Macro-F1 menunjukkan pengenalan lebih dari satu kelas.

## Kontribusi transformasi bergantung pada representasi lain

| Kontras OOF peserta | Delta accuracy | Delta Macro-F1 | Delta mean accuracy orang |
|---|---:|---:|---:|
| RMSSD-only vs raw | +2,38 pp | +2,00 pp | +2,20 pp |
| Eye-only vs raw | −4,02 pp | −2,77 pp | −5,69 pp |
| Gabungan vs raw | +5,66 pp | +5,90 pp | +4,79 pp |
| Gabungan vs RMSSD-only | +3,28 pp | +3,90 pp | +2,59 pp |
| Gabungan vs eye-only | +9,69 pp | +8,67 pp | +10,49 pp |

Eye-only merugikan accuracy/Macro-F1 window ketika RMSSD masih mentah, tetapi menambah skor ketika RMSSD juga dinormalkan pada OOF peserta. Hasil menunjukkan ketergantungan antartransformasi pada perilaku Extra Trees; belum membuktikan sinergi fisiologis HRV-eye. Arah perubahan model nonlinear tidak harus sama dengan penjumlahan perubahan masing-masing representasi.

Eye-only tidak menurunkan semua metrik: mean Macro-F1 per orang naik kecil dari 20,45% menjadi 21,87%, meskipun accuracy dan Macro-F1 window turun. Mean Macro-F1 orang gabungan mencapai 31,06%. Semua metrik per orang tetap memakai tiga label, termasuk kelas yang tidak hadir pada orang tertentu.

## Kelas High tetap sulit pada orang yang tidak ikut fitting

| Normalisasi | Recall Low | Recall Medium | Recall High |
|---|---:|---:|---:|
| Raw | 50,00% | 40,56% | 15,42% |
| RMSSD-only | 50,38% | 50,00% | 14,54% |
| Eye-only | 38,64% | 38,33% | 18,50% |
| Gabungan | 50,00% | 55,00% | **20,70%** |

RMSSD-only terutama menambah pengenalan Medium: 73 menjadi 90 dari 180; High turun 35 menjadi 33 dari 227. Eye-only menurunkan pengenalan Low dari 132 menjadi 102 dari 264, meskipun High naik 35 menjadi 42. Gabungan mempunyai 132 Low, 99 Medium, dan 47 High benar. Jadi High masih salah pada 180 dari 227 window dalam evaluasi ini. Model belum menjadi pendeteksi tiga tingkat yang andal pada orang yang ditahan dari fitting.

## Hasil antarprotokol tidak memilih varian yang sama

| Normalisasi | Accuracy CV waktu | Macro-F1 CV waktu | Accuracy validation historis |
|---|---:|---:|---:|
| Raw | 35,47% | 34,26% | 52,53% |
| RMSSD-only | **39,79%** | **39,03%** | **66,67%** |
| Eye-only | 33,68% | 32,48% | 59,60% |
| Gabungan | 39,34% | 38,38% | 65,66% |

RMSSD-only memperbaiki accuracy dan Macro-F1 terhadap raw pada kedua CV. Eye-only menurunkan keduanya pada dua CV, walaupun validation naik. Gabungan unggul pada evaluasi orang, sementara RMSSD-only sedikit lebih tinggi pada kontrol waktu dan validation. Selisih validation RMSSD-only terhadap gabungan hanya satu jawaban benar, 66 vs 65 dari 99 window.

Validation tetap **43 Low, nol Medium, dan 56 High**, dari tiga orang yang juga ada di Train. Recall Medium N/A adalah pembacaan yang tepat. Angka 66,67% tidak menjadi bukti generalisasi tiga kelas atau peningkatan terhadap hasil Test batch sebelumnya. Validation ini sudah pernah diperiksa dan bukan hold-out baru.

Fitting accuracy tercatat 94,63% raw, 95,23% RMSSD-only, 93,89% eye-only, dan 94,49% gabungan. Dibanding OOF peserta 31,74–41,43%, gap fitting-held-out tetap besar. Mengganti normalisasi belum menghilangkan keterbatasan generalisasi tersebut.

## Keputusan penelitian yang didukung

Gabungan tetap kandidat untuk tujuan generalisasi orang karena paling tinggi pada OOF utama dan rata-rata accuracy per orang pada subset ini. RMSSD-only merupakan kontrol yang lebih sederhana dengan perbaikan terhadap raw pada kedua CV; ia perlu dipertahankan sebagai pembanding. Bukti dua CV tidak mendukung normalisasi eye-only sebagai pilihan utama. Ini tidak berarti modalitas eye tidak berguna: fitur eye tetap dipakai dalam semua varian.

Exp02 mereproduksi skor terbaik gabungan Exp01 sambil memperjelas kontribusi transformasinya. Belum ada pencapaian accuracy 70%. Langkah yang lebih berguna daripada langsung memperluas search adalah memeriksa kestabilan gabungan/RMSSD-only pada lebih banyak sumber dengan acuan eye awal yang layak, tetap melaporkan tiga kelas, dan memperkuat bukti independen label beban mental. Empat peserta membatasi kesimpulan populasi; window overlap bukan observasi independen.

## Catatan runtime dan artefak

Notebook mencatat dua `FutureWarning` Pandas tentang penggabungan tabel yang mempunyai kolom all-NA, pada tahap pembuatan kontras dan pemeriksaan kontrol setelah fitting. Kolom recall Medium validation memang N/A. Run mencapai laporan akhir, tanpa output exception; peringatan tersebut bukan bukti training berhenti atau sensor rusak.

Artefak yang tersedia: [laporan yang tampil di notebook](../outputs/exp02/ablation_20261124T034359Z_3ff2c1ec/notebook_RESULTS.md), [tabel OOF tampilan](../outputs/exp02/ablation_20261124T034359Z_3ff2c1ec/analysis/notebook_cv_leaderboard.csv), [kontras tampilan](../outputs/exp02/ablation_20261124T034359Z_3ff2c1ec/analysis/notebook_paired_contrasts.csv), [validation tampilan](../outputs/exp02/ablation_20261124T034359Z_3ff2c1ec/analysis/notebook_validation_metrics.csv), dan [kontrol tampilan](../outputs/exp02/ablation_20261124T034359Z_3ff2c1ec/analysis/notebook_exp01_control_checks.csv).

Untuk audit prediksi per window serta checksum pekerjaan/model, artefak yang masih diperlukan adalah ZIP folder `ablation_20261124T034359Z_3ff2c1ec` dari Drive. Hasil lama Exp01 dan notebook sumber Exp02 tetap dipertahankan.
