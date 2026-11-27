# Batch 07 · Exp04 · Evaluasi Silang & Penyetaraan Pertanyaan Beban Kognitif V3 vs V4

**Pada 40 episode irisan yang memiliki label perilaku V3 sekaligus rating peserta V4 (15 peserta), kedua indikator bersepakat pada 34/40 episode (`85,0%`).** Pada peserta *held-out* yang sama, `V4_LogReg_Core9` adalah satu-satunya model yang batas bawah CI 95% AUC-nya berada di atas 0,50 pada keempat definisi target (rating $\ge 7$: AUC `0,774`, CI 0,56–0,95). `V3_SVM_RBF` mencapai AUC `0,727` pada target perilaku V3, tetapi CI-nya (0,47–0,88) mencakup 0,50.

> **Catatan revisi (27 November 2026, lihat [DIAGNOSIS.md](../DIAGNOSIS.md)).**
> - Kesepakatan 85% wajar karena kedua label mengikuti durasi dan kegagalan pengerjaan: rating berkorelasi ρ = 0,63 dengan durasi dan 0,64 dengan jumlah kesalahan, dan rating ditanyakan tepat setelah peserta selesai atau timeout. Ini bukan validasi konstruk beban kognitif yang independen.
> - Dengan 7–13 episode positif, CI AUC sebagian besar model sangat lebar. Ensemble `Dual_V3SVM_V4LR` (0,814 pada target konkordan) mempunyai CI 0,43–1,00 sehingga belum didukung statistik.
> - Akurasi model pada tabel `metrics.csv` (45–65%) berada di bawah tebakan selalu negatif (67,5–82,5%, tergantung target). Gunakan AUC dan balanced accuracy.

## 1. Konkordansi Label Perilaku V3 dan Rating V4 pada 40 Episode Irisan

Dari 66 episode layak V3 dan 126 episode layak V4, terdapat **40 episode dari 15 peserta** yang memenuhi syarat kualitas sensor kedua paket:

| Sebaran Bersama | V4 Non-Overload (`1–6`) | V4 Overload (`7–10`) | Total V3 |
|---|---:|---:|---:|
| **V3 Non-High (`Low/Medium`)** | **27** | 2 | 29 |
| **V3 High (waktu lambat & $\ge 3$ error)** | 4 | **7** | 11 |
| **Total V4** | 31 | 9 | **40** |

- **Kesepakatan biner:** `34/40` episode (`85,0%`).
- **Korelasi perilaku vs rating (`1–10`):** Durasi aktif memiliki Spearman $\rho = +0{,}633$, dan jumlah kesalahan manual memiliki Spearman $\rho = +0{,}640$ terhadap rating kesulitan yang dilaporkan peserta.

## 2. Perbandingan Model pada Peserta Held-Out yang Sama (`Native_HeldOut_LOPO`, 40 Episode)

Setiap peserta $p$ dari 15 peserta disisihkan sepenuhnya dari kedua cabang pelatihan (`V3` dilatih pada peserta `v3_Train - {p}` dengan rekonstruksi kuantil referensi tanpa $p$; `V4` dilatih pada peserta `v4_all - {p}`):

| Model | AUC Perilaku V3 High (`11/40`) | AUC Rating V4 $\ge 7$ (`9/40`) | AUC Konkordan Keduanya (`7/40`) | AUC Gabungan Salah Satu (`13/40`) | Spearman vs Rating | Spearman vs Durasi |
|---|---:|---:|---:|---:|---:|---:|
| `V3_SVM_RBF` | **0,727** (0,474–0,882) | 0,720 (0,368–0,939) | 0,766 (0,393–0,961) | 0,707 | +0,372 | **+0,428** |
| `V3_Blend_SVM_ET` | 0,708 (0,468–0,889) | 0,720 (0,374–0,963) | 0,775 (0,408–0,980) | 0,684 | +0,346 | +0,423 |
| `V3_ExtraTrees` | 0,699 (**0,503–0,909**) | 0,674 (0,383–0,929) | 0,745 (0,411–0,963) | 0,658 | +0,302 | +0,347 |
| `V4_LogReg_Core9` | 0,712 (**0,520–0,910**) | **0,774 (0,555–0,955)** | **0,788 (0,606–0,947)** | **0,721 (0,513–0,908)** | +0,364 | +0,200 |
| `V4_LogReg_Full14` | 0,633 (0,229–0,818) | 0,746 (0,374–0,969) | 0,740 (0,297–0,947) | 0,658 | +0,357 | +0,112 |
| `Dual_V3SVM_V4LR` | 0,696 (0,286–0,879) | 0,767 (0,300–1,000) | 0,784 (0,348–1,000) | 0,704 | **+0,430** | +0,280 |

Pada rezim kontrol peserta identik (`Matched_Participants_LOPO`, di mana kedua cabang sama-sama dilatih pada seluruh `all_people - {p}`), ensemble lintas-konstruk **`Dual_V3SVM_V4LR` mencapai AUC `0,814`** pada target konkordan (`concordant_both_high`).

## 3. Interpretasi untuk Adaptive-IDE

1. **Skor V4 adalah sinyal sensor yang paling tangguh lintas definisi target.** `V4_LogReg_Core9` tetap di atas peluang acak pada target perilaku V3 (AUC `0,712`, CI 0,52–0,91), rating subjektif (`0,774`), konkordan (`0,788`), dan gabungan (`0,721`). Skor ini kurang terikat pada durasi (Spearman `+0,200`) dibanding skor V3 (`+0,428`).
2. **Skor V3 lebih banyak mengikuti durasi pengerjaan daripada rating.** Spearman terhadap durasi (`+0,428`) lebih tinggi daripada terhadap rating (`+0,372`). Ini konsisten dengan temuan bahwa model V3 belum dapat dibedakan dari detektor durasi.
3. **Episode konkordan (V3 High ∩ rating 7–10) memang paling mudah dipisahkan**, tetapi hanya berjumlah 7 episode. Skema *dual-trigger* yang memadukan V3 dan V4 belum layak dipakai sebelum diuji pada protokol terkunci dengan sampel lebih besar.

## 4. Reproduksi Lokal & Artefak

```powershell
python models/batch-07/exp04/build_notebook.py
```
Lalu jalankan seluruh sel [EyeHeartCognitiveLoad.ipynb](EyeHeartCognitiveLoad.ipynb).

- [Tabel konkordansi 40 episode](outputs/exp04_unified_20261125T152940Z_ebddb53d/shared_episodes_concordance.csv) · [Metrik lengkap](outputs/exp04_unified_20261125T152940Z_ebddb53d/metrics.csv) · [Prediksi per episode](outputs/exp04_unified_20261125T152940Z_ebddb53d/episode_predictions.csv) · [Grafik perbandingan silang](outputs/exp04_unified_20261125T152940Z_ebddb53d/exp04_v3_vs_v4_cross_evaluation.png) · [ZIP hasil](outputs/exp04_unified_20261125T152940Z_ebddb53d.zip)
