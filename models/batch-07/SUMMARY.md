# Ringkasan Hasil Batch 07 — Keandalan Riwayat V3, Kalibrasi Ambang V4, & Evaluasi Silang V3 vs V4

Tanggal eksekusi lokal: 25 November 2026 (`Python 3.10.11`, `NumPy 2.2.6`, `Pandas 2.3.3`, `scikit-learn 1.7.2`).

## 1. Kesimpulan Utama

Batch 07 menuntaskan seluruh agenda lanjutan dari Batch 06 melalui empat eksperimen terkontrol (`exp01`–`exp04`) yang dieksekusi penuh secara lokal dengan verifikasi hash SHA-256 dan pemisahan peserta ketat:

1. **Masalah tingginya alarm palsu pada V4 (`1–6` vs `7–10`) teratasi melalui kalibrasi ambang anggaran alarm palsu di dalam fold (`Exp03`):**
   - Pada Batch 06, ambang probabilitas tetap `0,50` memberi AUC tinggi (`0,778–0,781`) namun memicu `31–40` alarm palsu dari `103` episode non-overload (`30%–39%`).
   - Dengan aturan pra-registrasi `nested_fa_cap25_max_rec` (batas alarm palsu $\le 25\%$ pada inner-LOPO peserta fitting, lalu maksimalkan recall overload):
     - **`win_logreg_core9_global` (Full LOPO 17 peserta, 126 episode):** Akurasi melonjak dari `65,87%` menjadi **`77,78%` (`98/126`)**, Balanced Accuracy naik ke **`76,28%`**, Macro-F1 naik ke **`0,701`**, dan alarm palsu dipangkas dari `40/103` (`38,8%`) menjadi **`22/103` (`21,4%`)** sambil tetap menangkap **`17/23` (`73,9%`)** episode overload.
     - **`win_logreg_full14_global` (Full LOPO 17 peserta):** Akurasi naik dari `69,84%` ke **`76,98%`**, Balanced Accuracy naik ke **`74,10%`**, dan alarm palsu turun dari `31/103` ke **`22/103`** **tanpa kehilangan satu pun deteksi overload (`16/23`, tetap `69,6%`)**. Dengan kebijakan `nested_youden`, akurasi mencapai **`80,95%`** (Balanced Accuracy `73,15%`, Macro-F1 `0,709`, alarm palsu hanya `15/103 = 14,6%`).
     - **Pada 4 peserta Test historis (27 episode, ambang dikunci murni dari 13 peserta Dev):** `win_logreg_core9_global` naik dari akurasi `48,15%` ke **`59,26%`** (Balanced Accuracy **`67,27%`**, recall tetap `4/5 = 80%`), sedangkan `win_logreg_full14_global` melonjak dari `51,85%` ke **`81,48%` (`22/27` benar, Balanced Accuracy `73,18%`, Macro-F1 `0,715`, alarm palsu turun dari `12/22` menjadi `3/22`)**.

2. **Sumber riwayat kualitas murni (`all_data.csv`) terbukti lebih jujur dan menaikkan daya urut LOPO V3 (`Exp01` & `Exp02`):**
   - Mengganti riwayat terkurasi global `good_acceptable.csv` dengan riwayat kausal berbasis kualitas dari `all_data.csv` **menaikkan LOPO ROC AUC pada seluruh 6 kombinasi model dan representasi V3 sebesar `+0,033` hingga `+0,092`** (`Extra Trees` mencapai **AUC `0,786`**, `SVM RBF + personal_iqr` mencapai **AUC `0,783`**, `blend_svm_et` mencapai **AUC `0,777`** dan mengurutkan **`14/16` (`87,5%`)** pasangan dalam-peserta dengan benar).
   - Audit berpasangan membuktikan bahwa akurasi validasi `81,8%` (`9/11`) pada `exp13` Batch 06 merupakan **efek samping kurasi global `good_acceptable.csv`**: dengan riwayat bersih, validasi `SVM RBF + personal_iqr` kembali ke `6/11 (54,5%)`, sementara performa pada Test historis justru membaik dari AUC `0,406` menjadi **`0,594`**.
   - Kalibrasi ambang dalam-fold (`nested_bal_acc`) pada `SVM RBF | center_only` menaikkan akurasi validasi V3 ke **`7/11 (63,6%, BalAcc 66,07%)`** dan dengan `nested_recall75` memangkas alarm palsu Test historis dari `7/16` menjadi **`4/16` (`13/18` benar, `72,2%`)**.

3. **Asimetri arsitektur antara V3 dan V4 terjelaskan secara empiris (`Exp02` & `Exp03`):**
   - Pada **V3** (37 episode Train, target rubrik waktu + kesalahan relatif rekan), representasi **median episode relatif riwayat pra-soal** dengan model non-linear dangkal (`SVM RBF` / `Extra Trees`) unggul karena mencegah dominasi episode berdurasi panjang.
   - Pada **V4** (126 episode, 1.693 window, target persepsi kesulitan peserta), **regresi logistik L2 per-window dengan bobot setara-episode yang dirata-ratakan ke tingkat episode** (`AUC 0,739–0,781`) mengungguli model median episode pra-soal (`AUC 0,568–0,672`) karena menangkap kondisi fisiologis tonik dan fasik secara stabil tanpa overfit.

4. **Evaluasi silang pada 40 episode irisan membuktikan konvergensi konstruk perilaku dan persepsi (`Exp04`):**
   - Label perilaku V3 dan rating subjektif V4 bersepakat pada **`34/40` episode (`85,0%`)**, dengan korelasi kuat antara rating kesulitan dan durasi aktif ($\rho = +0{,}633$) maupun kesalahan manual ($\rho = +0{,}640$).
   - Pada peserta *held-out* yang identik, **`V3_SVM_RBF` terbaik untuk memprediksi hambatan perilaku aktual (`AUC 0,727`)**, **`V4_LogReg_Core9` terbaik untuk memprediksi overload yang dirasakan peserta (`AUC 0,774`)**, dan keduanya mencapai pemisahan kuat (**`AUC 0,766–0,788`**, bahkan **`0,814`** pada ensemble `Dual_V3SVM_V4LR` dengan peserta fitting seragam) ketika perilaku dan persepsi peserta sama-sama menunjukkan overload (`concordant_both_high`).

---

## 2. Tabel Komparasi Utama Batch 07

| Eksperimen | Dataset & Evaluasi | Konfigurasi Terbaik | ROC AUC | Balanced Acc | Akurasi | Recall Positif | Alarm Palsu |
|---|---|---|---:|---:|---:|---:|---:|
| **Exp01** | V3 · LOPO Train (15p, 37 ep) | `extra_trees \| center_only \| fixed` | **0,786** | 67,56% | 64,86% | **14/16 (87,5%)** | 11/21 (52,4%) |
| **Exp01** | V3 · LOPO Train (15p, 37 ep) | `svm_rbf \| center_only \| fixed` | 0,768 | **69,20%** | **67,57%** | 13/16 (81,3%) | **9/21 (42,9%)** |
| **Exp02** | V3 · LOPO Train (15p, 37 ep) | `svm_rbf \| personal_iqr \| nested_bal_acc` | **0,783** | 68,45% | **67,57%** | 12/16 (75,0%) | **8/21 (38,1%)** |
| **Exp02** | V3 · Validasi (3p, 11 ep) | `svm_rbf \| center_only \| nested_bal_acc` | **0,750** | **66,07%** | **63,64%** | **3/4 (75,0%)** | 3/7 (42,9%) |
| **Exp03** | V4 · Dev LOPO (13p, 99 ep) | `win_logreg_core9_global \| nested_recall75` | **0,739** | **73,77%** | **67,68%** | **15/18 (83,3%)** | 29/81 (35,8%) |
| **Exp03** | V4 · Full LOPO (17p, 126 ep) | `win_logreg_core9_global \| nested_fa_cap25` | 0,778 | **76,28%** | **77,78%** | **17/23 (73,9%)** | 22/103 (21,4%) |
| **Exp03** | V4 · Full LOPO (17p, 126 ep) | `win_logreg_full14_global \| nested_youden` | **0,781** | 73,15% | **80,95%** | 14/23 (60,9%) | **15/103 (14,6%)** |
| **Exp03** | V4 · Test Historis (4p, 27 ep) | `win_logreg_full14_global \| nested_fa_cap25` | 0,745 | **73,18%** | **81,48%** | 3/5 (60,0%) | **3/22 (13,6%)** |
| **Exp04** | Irisan V3∩V4 · LOPO (15p, 40 ep) | `V4_LogReg_Core9` (Target: Konkordan High) | **0,788** | **71,21%** | 52,50% | **7/7 (100%)** | 19/33 (57,6%) |
| **Exp04** | Irisan V3∩V4 · LOPO (15p, 40 ep) | `V3_SVM_RBF` (Target: Perilaku V3 High) | **0,727** | 62,23% | **57,50%** | 8/11 (72,7%) | **14/29 (48,3%)** |

---

## 3. Rekomendasi Operasional untuk Sistem Adaptive-IDE

1. **Detektor utama pemicu bantuan (Overload Trigger):** Gunakan **`win_logreg_core9_global`** atau **`win_logreg_full14_global`** (bila rasio baseline RMSSD & head pose tersedia) dengan ambang terkalibrasi anggaran alarm palsu **`~0,58–0,62`** (`nested_fa_cap25_max_rec`). Konfigurasi ini melampaui target akurasi & balanced accuracy **74%–78%** pada LOPO 17 peserta dengan hanya ~1 alarm palsu per 5 tugas normal (`21,4%`), sehingga intervensi IDE tidak mengganggu alur kerja pemrogram.
2. **Pemantau hambatan perilaku pasca-episode (Behavioral Struggle Monitor):** Pertahankan **`V3_SVM_RBF` (`center_only`, ambang terkalibrasi)** atau **`blend_svm_et`** dengan riwayat kualitas murni (`all_data.csv` window `good/acceptable` pra-soal) sebagai sinyal pendamping untuk mengurutkan tingkat kesulitan pengerjaan di dalam sesi pengguna yang sama (`87,5%` ketepatan urutan dalam-peserta pada LOPO, `100%` pada validasi).
3. **Penggunaan sinyal ganda (Dual-Trigger):** Intervensi berintensitas rendah (misalnya petunjuk halus UI) dapat dipicu oleh skor V4 yang melewati ambang `0,59`, sementara intervensi aktif (bantuan langkah/dekomposisi kode) dipicu saat **skor V4 dan skor V3 sama-sama tinggi**, karena kondisi konkordan tersebut memiliki pemisahan fisiologis terkuat (`AUC 0,788–0,814`).
