# Batch 07 · Keandalan Riwayat Bersih V3, Kalibrasi Ambang V4, dan Evaluasi Silang Beban Kognitif

Batch 07 melanjutkan eksperimen pemodelan beban kognitif **secara penuh di lokal** (tanpa memerlukan Google Drive/Colab, namun setiap notebook tetap kompatibel dengan Colab) untuk menjawab ketiga agenda utama pada [EVALUASI_V3_V4.md](../batch-06/EVALUASI_V3_V4.md).

## Ringkasan Eksperimen & Dokumen

| Eksperimen | Fokus Penelitian | Temuan Kunci | Dokumen & Notebook |
|---|---|---|---|
| **Exp01** | Perbandingan 3 keluarga model (`SVM RBF`, `LogReg L2`, `Extra Trees`) pada V3 dengan riwayat kualitas murni | AUC LOPO 0,768–0,786 (37 episode, 16 High; tebakan mayoritas 56,8%). Skor V3 belakangan terbukti mengikuti durasi episode; lihat [DIAGNOSIS §4](DIAGNOSIS.md#4-v3-skor-sensor-mengikuti-durasi-episode). | [Notebook](exp01/EyeHeartCognitiveLoad.ipynb) · [Metrik](exp01/outputs/models_20261125T124030Z_fe00ac84/metrics.csv) |
| **Exp02** | Audit riwayat bersih vs terkurasi pada IQR personal (`exp13`), ensemble `SVM + Extra Trees`, & kalibrasi ambang dalam-fold V3 | Riwayat bersih lebih benar secara metodologis (AUC LOPO +0,033 s.d. +0,092, di bawah galat baku ~0,08). Validasi 81,8% pada `exp13` lama terbukti efek kurasi global. **Test historis V3: AUC 0,500, 1/8 pasangan dalam-peserta benar**; 72,2% (13/18) di bawah tebakan mayoritas 88,9%. | [README Exp02](exp02/README.md) · [Notebook](exp02/EyeHeartCognitiveLoad.ipynb) |
| **Exp03** | Kalibrasi ambang `nested inner-LOPO` & perbandingan 6 kandidat model pada V4 (`1–6` vs `7–10`) | `win_logreg_core9_global` menang di Dev: AUC 0,739, BalAcc 0,753 (13p); Full LOPO 17p AUC 0,778 (CI 0,63–0,89). `nested_fa_cap25` memangkas alarm palsu tetapi peringkat ke-8 di Dev; akurasi 77,8%/81,5% **tidak melampaui tebakan mayoritas 81,7%/81,5%**. | [README Exp03](exp03/README.md) · [Notebook](exp03/EyeHeartCognitiveLoad.ipynb) |
| **Exp04** | Evaluasi silang terpadu model V3 vs V4 pada 40 episode irisan dengan peserta *held-out* yang sama | Kesepakatan label V3–V4 85,0% (34/40), wajar karena keduanya mengikuti durasi dan kegagalan. `V4_LogReg_Core9` satu-satunya model dengan batas bawah CI AUC > 0,50 pada keempat target (rating ≥ 7: 0,774). Ensemble dual (0,814) CI 0,43–1,00. | [README Exp04](exp04/README.md) · [Notebook](exp04/EyeHeartCognitiveLoad.ipynb) |
| **SUMMARY** | Sintesis lintas eksperimen Batch 07 & rekomendasi operasional Adaptive-IDE (direvisi 27 November 2026) | Metrik utama AUC/balanced accuracy dengan pembanding mayoritas dan konteks tanpa sensor. | [SUMMARY.md](SUMMARY.md) |
| **DIAGNOSIS** | Audit ulang klaim Batch 07 dan arah Batch 08 | Akurasi headline di bawah tebakan mayoritas; V4 memiliki sinyal nyata di atas konteks tugas; V3 mengikuti durasi; Test historis tidak informatif. | [DIAGNOSIS.md](DIAGNOSIS.md) · [Skrip audit](diagnosis/audit_batch07.py) |

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
