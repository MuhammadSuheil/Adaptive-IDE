# Batch 07 · Keandalan Riwayat Bersih V3, Kalibrasi Ambang V4, dan Evaluasi Silang Beban Kognitif

Batch 07 melanjutkan eksperimen pemodelan beban kognitif **secara penuh di lokal** (tanpa memerlukan Google Drive/Colab, namun setiap notebook tetap kompatibel dengan Colab) untuk menjawab ketiga agenda utama pada [EVALUASI_V3_V4.md](../batch-06/EVALUASI_V3_V4.md).

## Ringkasan Eksperimen & Dokumen

| Eksperimen | Fokus Penelitian | Temuan Kunci | Dokumen & Notebook |
|---|---|---|---|
| **Exp01** | Perbandingan 3 keluarga model (`SVM RBF`, `LogReg L2`, `Extra Trees`) pada V3 dengan riwayat kualitas murni | `Extra Trees` meraih AUC LOPO tertinggi (**0,786**), `SVM RBF` meraih akurasi LOPO (**67,6%**) & AUC validasi tertinggi (**0,750**). | [Notebook](exp01/EyeHeartCognitiveLoad.ipynb) · [Metrik](exp01/outputs/models_20261125T124030Z_fe00ac84/metrics.csv) |
| **Exp02** | Audit riwayat bersih vs terkurasi pada IQR personal (`exp13`), ensemble `SVM + Extra Trees`, & kalibrasi ambang dalam-fold V3 | Riwayat bersih menaikkan LOPO AUC seluruh 6 kombinasi (`+0,033` s.d. `+0,092`); lonjakan validasi `81,8%` pada `exp13` lama terbukti berasal dari kurasi global `good_acceptable.csv`. `blend_svm_et` mengurutkan **14/16 (87,5%)** pasangan dalam-peserta. | [README Exp02](exp02/README.md) · [Notebook](exp02/EyeHeartCognitiveLoad.ipynb) |
| **Exp03** | Kalibrasi ambang `nested inner-LOPO` & perbandingan 6 arsitektur model pada V4 (`1–6` vs `7–10`) | Kebijakan batas alarm palsu $\le 25\%$ (`nested_fa_cap25_max_rec`) menaikkan akurasi Full LOPO 17p menjadi **77,8%** (BalAcc **76,28%**, alarm palsu dipangkas dari `40/103` ke `22/103`) dan akurasi Test historis `win_logreg_full14_global` menjadi **81,5%** (BalAcc **73,18%**, alarm palsu `3/22`). | [README Exp03](exp03/README.md) · [Notebook](exp03/EyeHeartCognitiveLoad.ipynb) |
| **Exp04** | Evaluasi silang terpadu model V3 vs V4 pada 40 episode irisan dengan peserta *held-out* yang sama | Kesepakatan label perilaku V3 dan rating V4 mencapai **85,0% (34/40)**. `V3_SVM_RBF` unggul pada target perilaku (AUC **0,727**), `V4_LogReg_Core9` unggul pada rating $\ge 7$ (**0,774**) & konkordan keduanya (**0,788**; CI 95% seluruhnya `> 0,50`). | [README Exp04](exp04/README.md) · [Notebook](exp04/EyeHeartCognitiveLoad.ipynb) |
| **SUMMARY** | Sintesis lintas eksperimen Batch 07 & rekomendasi operasional Adaptive-IDE | Ringkasan komparatif lengkap, analisis metodologis, dan keputusan arsitektur model. | [SUMMARY.md](SUMMARY.md) |

---

## Menjalankan di Lokal (Tanpa Google Colab)

Seluruh empat notebook di `exp01`–`exp04` bersifat **mandiri** dan otomatis mendeteksi lingkungan lokal dengan membaca langsung paket dataset dari `analysis_sessions/dataset_fusion/output`:

1. **Via Jupyter / VS Code Notebook:** Buka `models/batch-07/exp0X/EyeHeartCognitiveLoad.ipynb` dan pilih **Run All** pada kernel Python lokal.
2. **Via Terminal PowerShell (dari root repositori):**
   ```powershell
   # Membangun ulang notebook mandiri setelah perubahan experiment.py
   python models/batch-07/exp01/build_notebook.py
   python models/batch-07/exp02/build_notebook.py
   python models/batch-07/exp03/build_notebook.py
   python models/batch-07/exp04/build_notebook.py
   ```
3. **Verifikasi integritas otomatis:** Setiap eksperimen memeriksa SHA-256 seluruh CSV input (`v3_behavioral_proxy`, `v4_self_reported_rating`, serta snapshot `v2_session_phase/window_15s_step_5s/all_data.csv`), memvalidasi pemisahan peserta antar-fold, dan menyimpan CSV prediksi, metrik, grafik PNG, `manifest.json`, serta arsip `.zip` di folder `outputs/` masing-masing.
