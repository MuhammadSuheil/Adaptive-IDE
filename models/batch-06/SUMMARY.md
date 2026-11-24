# Ringkasan Hasil Batch 06 (`v3_behavioral_proxy` & `v4_self_reported_rating`)

Tanggal eksekusi: 24 November 2026.

## 1. Kesimpulan Utama

Batch 06 menyelesaikan eksperimen terkontrol dan bebas kebocoran pada kedua paket dataset baru (`v3` dan `v4`) dengan total **200+ konfigurasi kandidat** lintas profil fitur (`core10`, `core9`, `core9_head4`, `core9_head4_base`), normalisasi (`global_robust`, `causal_session_robust`, `unsupervised_session_robust`), pembobotan, dan keluarga model (`LogisticRegression`/`Ridge`, `ExtraTrees`, `HistGradientBoosting`, dan `PyTorchMLP` 3-seed).

**Empat temuan empiris terpenting:**

1. **Klasifikasi 10-kelas, 3-kelas, dan regresi penuh 1–10 lintas-peserta tetap jauh di bawah 70%:**
   - **V4 Kontrol 10-kelas (5-fold `GroupKFold` seluruh 17 peserta):** Klasifikasi langsung 10 rating (`1–10`) menghasilkan Window Accuracy hanya **14,0%–14,1%** (Balanced Acc **12,6%–12,9%**, Macro-F1 **12,0%–13,4%**, Episode Acc **11,1%–13,5%** dengan `LogisticRegression`; `ExtraTrees` lebih rendah di **6,7%–8,7%**). Peluang acak 10 kelas adalah 10%, dan beberapa kelas hanya didukung 3–4 episode total (rating 8 = 3 episode/3 orang; rating 9 = 4 episode/4 orang), sehingga pemecahan ke 10 kelas tidak meningkatkan performa.
   - **V3 (3-kelas behavioral proxy, Test 3 peserta / 18 episode):** Model per-window terkunci (`LogisticRegression | core9 | causal_session_robust | class_balanced`) mencapai Window Macro-F1 **33,50%** (Acc **35,50%**, BalAcc **34,05%**) dan Episode Macro-F1 **30,13%** (Acc **33,33%**, BalAcc **36,11%**). Meski mengalahkan Dummy secara tegas pada Macro-F1 (22,67% window / 6,67% episode) dan berhasil mengenali ketiga kelas tanpa collapse, performanya masih dekat peluang acak (33,3%).
   - **V4 Track A (Regresi 1–10, Held-Out Test 4 peserta / 27 episode):** Model terkunci (`Ridge | core9_head4_base | causal_session_robust`) menghasilkan Window MAE **2,768** ($R^2 = -0{,}326$) dan Episode MAE **2,528** ($R^2 = -0{,}136$, Spearman $\rho = -0{,}196$), tidak melampaui Dummy Mean pada window (MAE **2,527**) walau sedikit menurunkan Episode MAE (vs Dummy **2,969**).
   - **V4 Track B (3-kelas `1–3` / `4–6` / `7–10`, Held-Out Test 4 peserta / 27 episode):** Model terkunci (`LogisticRegression | core10 | causal_session_robust | episode_equalized_clipped`) mencapai Window Macro-F1 **35,49%** (Acc **36,16%**, BalAcc **35,13%**) dan Episode Macro-F1 **34,53%** (Acc **37,04%**, BalAcc **35,95%**) — konsisten lebih baik daripada Dummy (Macro-F1 19,23% / 10,42%), tetapi belum mendekati target 70%.

2. **Sinyal fisiologis lintas-peserta NYATA muncul pada pemisahan biner ekstrem `v4` (`1–3` Low vs `7–10` High) bila memakai fitur tambahan postur kepala + rasio baseline HRV:**
   - **V4 Track C1 (`LogisticRegression | core9_head4_base | global_robust`):**
     - **Dev 5-Fold OOF (13 peserta, 70 episode):** Window ROC-AUC **0,700** (BalAcc **64,69%**) | **Episode ROC-AUC 0,739 (BalAcc 72,54%, Acc 70,00%)**.
     - **Held-Out Test (4 peserta baru, 19 episode):** Window ROC-AUC **0,562** (BalAcc **53,96%**) | **Episode ROC-AUC 0,714 (BalAcc 62,14%, Acc 63,16%)**.
   - Temuan ini krusial: ketika kelas tengah yang kabur (`4–6`) dilepas dan model linear teregulasi diberi **postur kepala (`head_pitch/yaw`) + `rmssd_baseline_ratio`** lalu diagregasi ke tingkat episode, sinyal pembeda soal sangat mudah vs sangat sulit bertahan melampaui 0,71 AUC pada peserta yang belum pernah dilihat!

3. **Model linear sederhana mengalahkan pohon & neural; pelatihan personal per-peserta gagal karena jumlah episode terlalu kecil:**
   - Pada seluruh track V3 dan V4, **model linear teregulasi (`LogisticRegression` / `Ridge`)** secara konsisten terpilih sebagai pemenang per-window karena `ExtraTrees`, `HistGradientBoosting`, dan `PyTorchMLP` cepat overfit pada ukuran sampel independen yang kecil (37–99 episode).
   - **V4 Track C2 (Diagnostik Within-Participant Leave-One-Episode-Out pada 13 peserta):** Melatih model hanya pada episode milik 1 peserta menghasilkan LOEO Episode MAE **1,939** (`Ridge`) dan **2,063** (`ExtraTrees`), **lebih buruk daripada Dummy rata-rata peserta sendiri (1,671)** dengan median Spearman negatif (`-0,116` dan `-0,237`). Penyebabnya: 3–13 episode per orang terlalu sedikit untuk fit model personal dari nol; pooling lintas-peserta dengan normalisasi jauh lebih stabil.

4. **Peringatan metodologis penting: jebakan kebocoran durasi pada `episode_summary` & pembobotan ekstrem V3:**
   - Cabang diagnostik `episode_summary` (yang menghitung `IQR` dan `slope` lintas window per episode) sempat meraih 69,5%–80,4% Macro-F1 pada Validation V3 namun anjlok ke 31,7% pada Test. Audit menemukan bahwa episode pendek 1–2 window (mayoritas kelas Low) selalu menghasilkan `IQR = 0` dan `slope = 0`, sehingga **membocorkan durasi episode secara implisit ke label V3 yang berbasis durasi**. Oleh karena itu cabang tersebut dikeluarkan dari seleksi pemenang utama dan seluruh pemenang terkunci adalah **model murni per-window**.
   - Pada V3, `train_sample_weights.csv` memiliki rasio bobot 214,3× (Kish ESS kelas Low hanya 13,9); pemangkasan bobot (`v3_clipped_weights`) dan `class_balanced` secara konsisten lebih stabil daripada bobot penuh tanpa pangkas.

---

## 2. Tabel Komparasi Lintas Eksperimen Batch 06

| Eksperimen & Track | Model Terkunci (Per-Window) | Dev / Val (Episode) | Held-Out Test (Window) | Held-Out Test (Episode) | Dummy Test (Episode) |
|---|---|---:|---:|---:|---:|
| **Exp01 — V3** (3-kelas proxy) | `LogReg \| core9 \| causal_sess \| class_bal` | mF1 **64,29%** (Val, 11 ep)<br>mF1 **47,18%** (OOF) | Acc 35,50%<br>BalAcc 34,05%<br>mF1 **33,50%** | Acc 33,33%<br>BalAcc 36,11%<br>mF1 **30,13%** | Acc 11,11%<br>mF1 6,67% |
| **Exp02 — V4 Track A** (Regresi 1–10) | `Ridge \| core9_head4_base \| causal_sess` | MAE 2,330<br>$R^2 = -0{,}081$ | MAE 2,768<br>$R^2 = -0{,}326$ | MAE **2,528**<br>$R^2 = -0{,}136$<br>$\rho = -0{,}196$ | MAE 2,969<br>$R^2 = -0{,}293$ |
| **Exp02 — V4 Track B** (3-kelas `1-3/4-6/7-10`) | `LogReg \| core10 \| causal_sess \| ep_clip` | Acc 43,43%<br>mF1 **41,75%** | Acc 36,16%<br>BalAcc 35,13%<br>mF1 **35,49%** | Acc 37,04%<br>BalAcc 35,95%<br>mF1 **34,53%** | Acc 18,52%<br>mF1 10,42% |
| **Exp02 — V4 Track C1** (Biner `1-3` vs `7-10`) | `LogReg \| core9_head4_base \| global` | **AUC 0,739**<br>BalAcc **72,54%**<br>Acc 70,00% | AUC 0,562<br>BalAcc 53,96%<br>mF1 53,80% | **AUC 0,714**<br>BalAcc **62,14%**<br>Acc **63,16%** | AUC 0,500<br>BalAcc 50,00% |

---

## 3. Hasil Deep-Dive Exp03: Deteksi Biner Overload (`v4`) Tanpa Data Baru

Eksekusi penuh [EyeHeartCognitiveLoad_v4_binary.ipynb](exp03_v4_binary_overload/EyeHeartCognitiveLoad_v4_binary.ipynb) (150 evaluasi LOPO 17 peserta + cluster-bootstrap 95% CI + audit perancu + audit zona netral Medium `4–6`) menghasilkan temuan utama:

| Skema Biner (LOPO 17 Peserta) | Model Operasional Terbaik | Win AUC | Ep AUC (95% CI) | Ep BalAcc | Ep Acc | MWU $p$ |
|---|---|---:|---|---:|---:|---:|
| **`full_1_6_vs_7_10`** (100% data, 126 ep) | `LogReg_L2 \| core9_head4_base \| global_robust` | 0,739 | **0,781 (0,647–0,894)** | **69,73%** | **69,84%** | $1{,}3\times 10^{-5}$ |
| **`full_1_6_vs_7_10`** (nested C + threshold) | `LogReg_NestedC \| core9_head4_base \| global_robust` | 0,739 | **0,773 (0,637–0,889)** | **70,26%** | **81,75%** | $2{,}2\times 10^{-5}$ |
| **`full_1_6_vs_7_10`** (hanya 9 fitur inti) | `LogReg_L2 \| core9 \| global_robust` | 0,691 | **0,778 (0,644–0,895)** | **74,06%** | 65,87% | $1{,}6\times 10^{-5}$ |
| **`extreme_1_3_vs_7_10`** (drop `4–6`, 89 ep) | `LogReg_L2 \| core9_head4_base \| global_robust` | 0,743 | **0,777 (0,610–0,924)** | 68,21% | 69,66% | $4{,}1\times 10^{-5}$ |
| **`full_1_5_vs_6_10`** (100% data, 126 ep) | `ExtraTrees_Shallow \| core9 \| causal_session_robust` | 0,583 | 0,643 | 60,84% | 56,35% | 0,0070 |

**Implikasi langsung untuk Adaptive-IDE:**
1. **Tidak perlu membuang Medium (`4–6`):** `full_1_6_vs_7_10` mempertahankan 100% data dengan menempatkan Low (`1–3`) + Medium (`4–6`) sebagai zona **Normal/Netral** dan `7–10` sebagai **Overload**, sekaligus memberi Episode AUC tertinggi (**0,781**) dan CI 95% paling sempit (`0,647–0,894`). Audit zona netral mengonfirmasi bahwa episode Medium (`4–6`) secara fisiologis memang berada pada rezim non-overload (median $p = 0{,}278$ vs High `7–10` median $p = 0{,}746$).
2. **Lolos audit perancu:** Pada `full_1_6_vs_7_10`, sinyal bertahan setelah menghapus baris missing baseline (Ep AUC **0,802**), tetap signifikan setelah mengontrol waktu sesi (**partial $r = +0{,}387, p = 7{,}8\times 10^{-6}$**), serta memisahkan overload di dalam setiap tingkat kesulitan protokol (`easy` AUC 0,904, `medium` 0,798, `hard` 0,727).
3. **Catatan jujur tentang normalisasi:** `global_robust` (Ep AUC 0,781) jauh lebih tinggi daripada `unsupervised_session_robust` (~0,42–0,48), sementara `causal_session_robust` tetap signifikan (~0,67–0,73 pada `extreme_1_3_vs_7_10`). Artinya model `global_robust` memanfaatkan kombinasi perbedaan antar-peserta dan perubahan dalam-sesi, serta perlu kalibrasi ambang yang hati-hati pada pengguna baru agar spesifisitas tetap tinggi.

---

## 4. Bukti & Artefak Terverifikasi

- [Exp01 V3 Notebook](exp01_v3_behavioral_proxy/EyeHeartCognitiveLoad_v3.ipynb) & [Hasil V3](outputs/exp01_v3_behavioral_proxy/RESULTS.md) *(sudah dieksekusi)*
- [Exp02 V4 Notebook](exp02_v4_self_reported_rating/EyeHeartCognitiveLoad_v4.ipynb) & [Hasil V4](outputs/exp02_v4_self_reported_rating/RESULTS.md) *(sudah dieksekusi)*
- [Exp03 V4 Binary Overload Notebook](exp03_v4_binary_overload/EyeHeartCognitiveLoad_v4_binary.ipynb) & [Hasil Exp03](outputs/exp03_v4_binary_overload/RESULTS.md) *(sudah dieksekusi)*
- Seluruh CSV leaderboard, prediksi Test, diagnostik episode, JSON metadata/checksum, serta grafik PNG tersimpan di `models/batch-06/outputs/`.
