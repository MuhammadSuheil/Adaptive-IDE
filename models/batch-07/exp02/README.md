# Batch 07 · Exp02 · Riwayat Bersih pada IQR Personal, Ensemble SVM+ExtraTrees, dan Kalibrasi Ambang V3

**Pada sumber riwayat kualitas murni (`all_data.csv`), seluruh enam kombinasi model meningkatkan AUC LOPO dibanding sumber terkurasi global (`+0,033` s.d. `+0,092` AUC), sekaligus membuktikan bahwa lonjakan akurasi validasi `81,8%` pada `exp13` sebelumnya berasal dari kurasi global `good_acceptable.csv`.** Ketika kurasi global dihapus, akurasi validasi `SVM RBF + personal_iqr` kembali ke `6/11 (54,5%)`, sementara AUC LOPO-nya justru naik dari `0,690` menjadi **`0,783`** dan AUC Test historisnya naik dari `0,406` ke **`0,594`**.

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

- **Pengurangan alarm palsu tanpa melihat data uji:** Pada `svm_rbf | center_only`, pemilihan ambang `nested_bal_acc` di dalam fold fitting mempertahankan akurasi LOPO `25/37 (67,6%)` sambil memangkas alarm palsu dari `9/21` menjadi **`8/21`**, menaikkan akurasi validasi dari `6/11 (54,5%)` menjadi **`7/11 (63,6%, BalAcc 66,07%, 3/4 High, 3/7 FA)`**, dan pada `nested_recall75` memangkas alarm palsu Test historis dari `7/16` menjadi **`4/16` (`13/18` benar, `72,2%`)**.
- **Keunggulan `blend_svm_et` pada urutan dalam-peserta & rating subjektif:** Pada 16 pasangan High/Non-High dalam peserta yang sama di LOPO, `blend_svm_et | center_only` mengurutkan **14/16 (`87,5%`)** pasangan dengan benar (dibanding `13/16` pada SVM dan `11/16` pada Extra Trees). Pada 21 episode LOPO yang memiliki rating V4, `blend_svm_et` mencapai **AUC `0,847`** untuk memprediksi rating $\ge 7$ ($\rho = +0{,}600$), sementara `extra_trees` mencapai korelasi peringkat tertinggi ($\rho = +0{,}692$).

## 4. Reproduksi Lokal & Artefak

Jalankan langsung di lokal tanpa Google Drive:
```powershell
python models/batch-07/exp02/build_notebook.py
```
Lalu jalankan seluruh sel [EyeHeartCognitiveLoad.ipynb](EyeHeartCognitiveLoad.ipynb).

- [Metrik lengkap](outputs/exp02_v3_20261125T152650Z_6705f25b/metrics.csv) · [Audit kurasi](outputs/exp02_v3_20261125T152650Z_6705f25b/curation_impact_audit.csv) · [Prediksi episode](outputs/exp02_v3_20261125T152650Z_6705f25b/episode_predictions.csv) · [Grafik ringkasan](outputs/exp02_v3_20261125T152650Z_6705f25b/v3_exp02_overview.png) · [ZIP hasil](outputs/exp02_v3_20261125T152650Z_6705f25b.zip)
