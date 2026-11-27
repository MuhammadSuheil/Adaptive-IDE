# Batch 07 · Exp02 · Riwayat Bersih pada IQR Personal, Ensemble SVM+ExtraTrees, dan Kalibrasi Ambang V3

**Riwayat kualitas murni (`all_data.csv`) dipertahankan karena menghilangkan seleksi global, dan audit ini membuktikan bahwa akurasi validasi `81,8%` pada `exp13` batch 06 berasal dari kurasi global `good_acceptable.csv`.** Dengan riwayat bersih, validasi `SVM RBF + personal_iqr` kembali ke `6/11 (54,5%)`. AUC LOPO keenam kombinasi naik `+0,033` s.d. `+0,092` (37 episode, 16 High), tetapi kenaikan ini lebih kecil daripada galat baku AUC pada ukuran sampel tersebut (~0,08).

> **Catatan revisi (27 November 2026, lihat [DIAGNOSIS.md](../DIAGNOSIS.md)).** Skor V3 berkorelasi ρ = 0,63 dengan durasi episode bahkan di antara episode Non-High, dan tidak menambah informasi di atas durasi (p = 0,79). Pada Test historis (3 peserta, 18 episode, 2 High), AUC `svm_rbf · center_only` adalah **0,500** dan hanya **1/8** pasangan dalam-peserta terurut benar. Tebakan selalu Non-High memberi 16/18 (88,9%). Angka Test pada README ini telah dievaluasi pada 18 kombinasi dan bersifat eksploratif.

## 1. Perbandingan Representasi & Model pada Riwayat Bersih (`fixed_default`)

| Evaluasi | Representasi | Model | Benar | Akurasi | Balanced Acc | Macro-F1 | ROC AUC | High Terdeteksi | Alarm Palsu | Pasangan Dalam-Peserta |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LOPO Train · 15p | `center_only` | `extra_trees` | 24/37 | 64,9% | 67,56% | 0,644 | **0,786** | **14/16** | 11/21 | 11/16 (68,8%) |
| LOPO Train · 15p | `center_only` | `blend_svm_et` | 24/37 | 64,9% | 66,82% | 0,648 | 0,777 | 13/16 | 10/21 | **14/16 (87,5%)** |
| LOPO Train · 15p | `center_only` | `svm_rbf` | **25/37** | **67,6%** | 69,20% | **0,675** | 0,768 | 13/16 | **9/21** | 13/16 (81,3%) |
| LOPO Train · 15p | `personal_iqr` | `svm_rbf` | **25/37** | **67,6%** | 69,20% | **0,675** | 0,783 | 13/16 | **9/21** | 13/16 (81,3%) |
| LOPO Train · 15p | `personal_iqr` | `extra_trees` | **25/37** | **67,6%** | **69,94%** | 0,674 | 0,744 | **14/16** | 10/21 | 11/16 (68,8%) |
| LOPO Train · 15p | `personal_iqr` | `blend_svm_et` | **25/37** | **67,6%** | 69,20% | **0,675** | 0,765 | 13/16 | **9/21** | 13/16 (81,3%) |
| Validasi · 3p | `center_only` | `svm_rbf` | 6/11 | 54,5% | 58,93% | 0,545 | **0,750** | **3/4** | 4/7 | **7/7 (100%)** |
| Validasi · 3p | `center_only` | `blend_svm_et` | 6/11 | 54,5% | 58,93% | 0,545 | 0,714 | **3/4** | 4/7 | **7/7 (100%)** |
| Validasi · 3p | `center_only` | `extra_trees` | 6/11 | 54,5% | 58,93% | 0,545 | 0,643 | **3/4** | 4/7 | **7/7 (100%)** |
| Validasi · 3p | `personal_iqr` | `svm_rbf` | 6/11 | 54,5% | 53,57% | 0,530 | 0,714 | 2/4 | 3/7 | **7/7 (100%)** |
| Validasi · 3p | `personal_iqr` | `blend_svm_et` | **7/11** | **63,6%** | **60,71%** | **0,607** | 0,643 | 2/4 | **2/7** | **7/7 (100%)** |
| Validasi · 3p | `personal_iqr` | `extra_trees` | **7/11** | **63,6%** | **60,71%** | **0,607** | 0,571 | 2/4 | **2/7** | 6/7 (85,7%) |

## 2. Audit Dampak Kurasi Global (`good_acceptable.csv` vs `all_data.csv`)

Eksperimen ini menuntaskan Prioritas #2 pada [EVALUASI_V3_V4.md](../../batch-06/EVALUASI_V3_V4.md) dengan membekukan parameter fitting dan membandingkan skor pada episode yang persis sama:

| Evaluasi | Representasi & Model | Skor Berubah | Keputusan Berubah | AUC Terkurasi | AUC Bersih | $\Delta$ AUC | Benar Terkurasi $\to$ Bersih |
|---|---|---:|---:|---:|---:|---:|---:|
| LOPO Train (37 ep) | `center_only · svm_rbf` (`exp11`) | 37/37 | 3/37 | 0,699 | **0,768** | **+0,068** | 26/37 $\to$ 25/37 |
| LOPO Train (37 ep) | `personal_iqr · svm_rbf` (`exp13`) | 37/37 | 2/37 | 0,690 | **0,783** | **+0,092** | 25/37 $\to$ 25/37 |
| Validasi (11 ep) | `center_only · svm_rbf` (`exp11`) | 11/11 | 0/11 | 0,607 | **0,750** | **+0,143** | 6/11 $\to$ 6/11 |
| Validasi (11 ep) | `personal_iqr · svm_rbf` (`exp13`) | 11/11 | 3/11 | 0,714 | 0,714 | 0,000 | **9/11 $\to$ 6/11** |
| Test Historis (18 ep) | `center_only · svm_rbf` (`exp11`) | 18/18 | 1/18 | 0,500 | 0,500 | 0,000 | 11/18 $\to$ 10/18 |
| Test Historis (18 ep) | `personal_iqr · svm_rbf` (`exp13`) | 18/18 | 3/18 | 0,406 | **0,594** | **+0,188** | 9/18 $\to$ 10/18 |

## 3. Kalibrasi Ambang Dalam-Fold (`Inner-LOPO`) & Keselarasan Rating V4

- **Ambang dalam-fold:** Pada `svm_rbf | center_only`, `nested_bal_acc` mempertahankan akurasi LOPO `25/37 (67,6%; mayoritas 56,8%)` dan mengurangi alarm palsu dari `9/21` menjadi `8/21`. Akurasi validasi naik dari `6/11` menjadi `7/11 (63,6%; mayoritas 63,6%; BalAcc 66,07%)`. Pada Test historis, `nested_recall75` memberi `13/18 (72,2%)`, **di bawah tebakan selalu Non-High `16/18 (88,9%)`**, dengan 1/2 High terdeteksi dan AUC 0,500. Kombinasi ini dipilih dari 18 kombinasi yang dinilai pada Test; akurasi Test kombinasi lain 50–72%.
- **Urutan dalam-peserta dan rating subjektif:** Pada LOPO, `blend_svm_et | center_only` mengurutkan 14/16 pasangan High/Non-High dalam peserta yang sama dengan benar. Pasangan ini hanya berasal dari 5 peserta, dan pada Test historis turun menjadi 1/8. Pada 21 episode LOPO yang memiliki rating V4, `blend_svm_et` mencapai AUC 0,847 untuk rating $\ge 7$ ($\rho = +0{,}600$). Hasil ini searah dengan exp04, tetapi skor V3 juga mengikuti durasi dan rating berkorelasi dengan durasi, sehingga belum menjadi bukti pengukuran beban kognitif yang independen.

## 4. Reproduksi Lokal & Artefak

Jalankan langsung di lokal tanpa Google Drive:
```powershell
python models/batch-07/exp02/build_notebook.py
```
Lalu jalankan seluruh sel [EyeHeartCognitiveLoad.ipynb](EyeHeartCognitiveLoad.ipynb).

- [Metrik lengkap](outputs/exp02_v3_20261125T152650Z_6705f25b/metrics.csv) · [Audit kurasi](outputs/exp02_v3_20261125T152650Z_6705f25b/curation_impact_audit.csv) · [Prediksi episode](outputs/exp02_v3_20261125T152650Z_6705f25b/episode_predictions.csv) · [Grafik ringkasan](outputs/exp02_v3_20261125T152650Z_6705f25b/v3_exp02_overview.png) · [ZIP hasil](outputs/exp02_v3_20261125T152650Z_6705f25b.zip)
