# Batch 07 · Exp04 · Evaluasi Silang & Penyetaraan Pertanyaan Beban Kognitif V3 vs V4

**Pada 40 episode irisan yang memiliki label perilaku V3 sekaligus rating peserta V4 (15 peserta), kedua indikator bersepakat pada 34/40 episode (`85,0%`).** Saat dinilai pada peserta *held-out* yang persis sama, **`V3_SVM_RBF` mencapai AUC tertinggi pada target perilaku (`0,727`)**, sedangkan **`V4_LogReg_Core9` mencapai AUC tertinggi pada target rating subjektif $\ge 7$ (`0,774`) dan konkordan keduanya (`0,788`)**, serta menjadi satu-satunya model yang seluruh interval kepercayaan bootstrap peserta 95%-nya berada di atas peluang acak (`> 0,50`) pada keempat definisi target.

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

## 3. Interpretasi Ilmiah untuk Adaptive-IDE

1. **Kedua model menangkap sinyal fisiologis yang saling melengkapi, bukan bertentangan:** Skor sensor `V3_SVM_RBF` lebih kuat mengikuti hambatan pengerjaan aktual ($\rho = +0{,}428$ terhadap durasi aktif; AUC `0,727` pada V3 High), sedangkan `V4_LogReg_Core9` lebih tajam menangkap persepsi overload subjektif (AUC `0,774` pada V4 $\ge 7$, menangkap `9/9` overload pada ambang `0,50`) sekaligus tetap kompetitif pada target perilaku (AUC `0,712`).
2. **Episode konkordan (`V3 High` $\cap$ `V4 Rating 7–10`) paling konsisten terpisahkan:** Ketika perilaku objektif dan laporan subjektif sama-sama menunjukkan beban tinggi (`7/40` episode), baik model V3 (`0,766–0,775`), model V4 (`0,788`), maupun gabungan keduanya (`0,784–0,814`) mencapai pemisahan paling tinggi.

## 4. Reproduksi Lokal & Artefak

```powershell
python models/batch-07/exp04/build_notebook.py
```
Lalu jalankan seluruh sel [EyeHeartCognitiveLoad.ipynb](EyeHeartCognitiveLoad.ipynb).

- [Tabel konkordansi 40 episode](outputs/exp04_unified_20261125T152940Z_ebddb53d/shared_episodes_concordance.csv) · [Metrik lengkap](outputs/exp04_unified_20261125T152940Z_ebddb53d/metrics.csv) · [Prediksi per episode](outputs/exp04_unified_20261125T152940Z_ebddb53d/episode_predictions.csv) · [Grafik perbandingan silang](outputs/exp04_unified_20261125T152940Z_ebddb53d/exp04_v3_vs_v4_cross_evaluation.png) · [ZIP hasil](outputs/exp04_unified_20261125T152940Z_ebddb53d.zip)
