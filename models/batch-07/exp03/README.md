# Batch 07 · Exp03 · Kalibrasi Ambang Keputusan OOF & Perbandingan Representasi pada V4 (`1–6` vs `7–10`)

**Pemilihan ambang dengan batas anggaran alarm palsu $\le 25\%$ di dalam fold peserta (`nested_fa_cap25_max_rec`) menaikkan akurasi LOPO 17 peserta dari `65,9%` menjadi `77,8%` (`98/126`, Balanced Accuracy `76,28%`) pada `win_logreg_core9_global`, serta memangkas alarm palsu hampir separuh dari `40/103` menjadi `22/103` sambil tetap menangkap `17/23` episode overload.** Pada `win_logreg_full14_global`, kebijakan yang sama mempertahankan recall overload penuh (`16/23`, identik dengan ambang `0,50`) sambil menurunkan alarm palsu dari `31/103` menjadi `22/103` (akurasi `77,0%`, Balanced Accuracy `74,10%`), dan pada 4 peserta Test historis menaikkan akurasi dari `51,9%` menjadi **`81,5%` (`22/27`, Balanced Accuracy `73,18%`, alarm palsu turun dari `12/22` ke `3/22`)**.

## 1. Perbandingan 6 Kandidat Model & Representasi (`fixed_050`)

Eksperimen ini membandingkan model per-window rata-rata episode (gaya Batch 06 V4) langsung terhadap arsitektur median episode relatif riwayat bersih `all_data.csv` (gaya Batch 07 Exp01):

| Kandidat | Representasi & Normalisasi | Dev LOPO AUC (13p, 99 ep) | Dev BalAcc | Full LOPO AUC (17p, 126 ep) | Full LOPO BalAcc | Test Historis AUC (4p, 27 ep) |
|---|---|---:|---:|---:|---:|---:|
| **`win_logreg_core9_global`** | Per-window `core9` · `global_robust` | **0,739** (0,589–0,907) | **75,31%** | 0,778 (0,644–0,895) | **74,06%** | **0,773** |
| **`win_logreg_full14_global`** | Per-window `core9+head4+base` · `global_robust` | 0,729 (0,561–0,899) | 66,36% | **0,781** (0,647–0,894) | 69,73% | 0,745 |
| `win_logreg_full14_causal` | Per-window `core9+head4+base` · `causal_session_robust` | 0,654 (0,530–0,761) | 65,12% | 0,612 (0,470–0,733) | 59,56% | 0,627 |
| `ep_et_prior14` | Median episode 14 fitur · riwayat bersih `all_data.csv` | 0,647 (0,528–0,767) | 50,00% | 0,655 (0,557–0,745) | 59,08% | 0,636 |
| `win_et_full14_global` | Per-window `ExtraTrees` · `global_robust` | 0,586 (0,408–0,815) | 56,17% | 0,583 (0,433–0,766) | 52,07% | 0,555 |
| `ep_svm_rbf_prior14` | Median episode `SVM RBF` · riwayat bersih `all_data.csv` | 0,568 (0,439–0,699) | 52,78% | 0,672 (0,586–0,748) | 59,31% | 0,764 |

> **Temuan arsitektural penting:** Pada rating subjektif V4, regresi logistik L2 per-window yang dirata-ratakan per episode (`win_logreg_core9_global` dan `win_logreg_full14_global`) secara konsisten mengungguli model median episode relatif riwayat pra-soal (`ep_svm_rbf_prior14` / `ep_et_prior14`) baik pada Dev LOPO (`0,729–0,739` vs `0,568–0,647`) maupun Full LOPO (`0,778–0,781` vs `0,655–0,672`).

## 2. Hasil 5 Kebijakan Kalibrasi Ambang Dalam-Fold (`Nested Inner-LOPO`)

Seluruh ambang di bawah dipilih **murni dari inner-LOPO peserta fitting** tanpa melihat peserta yang dievaluasi (menjawab Prioritas #1 [EVALUASI_V3_V4.md](../../batch-06/EVALUASI_V3_V4.md)):

| Evaluasi | Kandidat | Kebijakan Ambang | Rerata Ambang | Presisi Overload | Recall Overload | Alarm Palsu | Balanced Acc | Akurasi | Macro-F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Dev LOPO (13p, 99 ep)** | `win_logreg_core9_global` | `fixed_050` | 0,500 | 34,0% | **16/18 (88,9%)** | 31/81 (38,3%) | **75,31%** | 66,7% | 0,622 |
| Dev LOPO (13p, 99 ep) | `win_logreg_core9_global` | `nested_recall75_min_fa` | 0,506 | **34,1%** | 15/18 (83,3%) | 29/81 (35,8%) | 73,77% | 67,7% | **0,624** |
| Dev LOPO (13p, 99 ep) | `win_logreg_core9_global` | `nested_fa_cap25_max_rec` | 0,626 | 30,3% | 10/18 (55,6%) | **23/81 (28,4%)** | 63,58% | **68,7%** | 0,591 |
| **Full LOPO (17p, 126 ep)** | `win_logreg_core9_global` | `fixed_050` | 0,500 | 33,3% | **20/23 (87,0%)** | 40/103 (38,8%) | 74,06% | 65,9% | 0,614 |
| Full LOPO (17p, 126 ep) | `win_logreg_core9_global` | `nested_youden` | 0,545 | 38,3% | 18/23 (78,3%) | 29/103 (28,2%) | 75,05% | 73,0% | 0,664 |
| **Full LOPO (17p, 126 ep)** | `win_logreg_core9_global` | **`nested_fa_cap25_max_rec`** | 0,589 | **43,6%** | 17/23 (73,9%) | **22/103 (21,4%)** | **76,28%** | **77,8%** | **0,701** |
| Full LOPO (17p, 126 ep) | `win_logreg_full14_global` | `fixed_050` | 0,500 | 34,0% | 16/23 (69,6%) | 31/103 (30,1%) | 69,73% | 69,8% | 0,624 |
| **Full LOPO (17p, 126 ep)** | `win_logreg_full14_global` | **`nested_fa_cap25_max_rec`** | 0,581 | 42,1% | 16/23 (69,6%) | 22/103 (21,4%) | **74,10%** | **77,0%** | **0,686** |
| Full LOPO (17p, 126 ep) | `win_logreg_full14_global` | `nested_youden` | 0,615 | **48,3%** | 14/23 (60,9%) | **15/103 (14,6%)** | 73,15% | **81,0%** | **0,709** |
| **Test Historis (4p, 27 ep)** | `win_logreg_core9_global` | `fixed_050` | 0,500 | 23,5% | **4/5 (80,0%)** | 13/22 (59,1%) | 60,45% | 48,1% | 0,463 |
| **Test Historis (4p, 27 ep)** | `win_logreg_core9_global` | **`nested_fa_cap25_max_rec`** | 0,625 | **28,6%** | **4/5 (80,0%)** | **10/22 (45,5%)** | **67,27%** | **59,3%** | **0,553** |
| Test Historis (4p, 27 ep) | `win_logreg_full14_global` | `fixed_050` | 0,500 | 25,0% | **4/5 (80,0%)** | 12/22 (54,5%) | 62,73% | 51,9% | 0,494 |
| **Test Historis (4p, 27 ep)** | `win_logreg_full14_global` | **`nested_fa_cap25_max_rec`** | 0,700 | **50,0%** | 3/5 (60,0%) | **3/22 (13,6%)** | **73,18%** | **81,5%** | **0,715** |

## 3. Reproduksi Lokal & Artefak

```powershell
python models/batch-07/exp03/build_notebook.py
```
Lalu jalankan seluruh sel [EyeHeartCognitiveLoad.ipynb](EyeHeartCognitiveLoad.ipynb).

- [Metrik lengkap](outputs/exp03_v4_20261125T153235Z_d3e8d715/metrics.csv) · [Kurva trade-off ambang Dev 13p](outputs/exp03_v4_20261125T153235Z_d3e8d715/dev_oof_threshold_curve.csv) · [Ambang terpilih per fold](outputs/exp03_v4_20261125T153235Z_d3e8d715/selected_thresholds.csv) · [Grafik kalibrasi ambang](outputs/exp03_v4_20261125T153235Z_d3e8d715/v4_exp03_threshold_calibration.png) · [ZIP hasil](outputs/exp03_v4_20261125T153235Z_d3e8d715.zip)
