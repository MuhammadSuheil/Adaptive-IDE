# Batch 08 · Catatan Ronde Pemodelan

User mengizinkan iterasi bebas sampai hasil melampaui Batch 07. Agar skor akhir tetap dapat dipertanggungjawabkan, setiap ronde dicatat di sini. Seleksi konfigurasi, kalibrasi, dan ambang selalu terjadi di inner-LOPO. Perubahan *pool* kandidat antar-ronde adalah keputusan peneliti yang dibuat setelah melihat hasil ronde sebelumnya. Keputusan semacam ini tidak tercakup oleh nested LOPO, sehingga dilaporkan terbuka.

Kriteria lulus (disepakati sebelum run): Track A sensor-only pada prediksi outer nested-LOPO 17 peserta / 126 episode harus mencapai **AUC > 0,778 dan balanced accuracy > 0,741** (referensi Batch 07 `core9 · global · C=0,5 · fixed_050`, direproduksi di exp01: AUC 0,77797).

## Ronde 0 · Reproduksi referensi (exp01)

- Run: `exp01_reference/outputs/reference_20261127T023815Z_356d8138`
- Referensi Batch 07 direproduksi: AUC **0,77797** (Batch 07: 0,77797), balanced accuracy 0,7406.
- Pembanding tanpa sensor pada episode yang sama:
  - logistik waktu aktif berjalan (nested): AUC 0,818, balanced accuracy 0,811
  - durasi final: AUC 0,816
  - menit dalam sesi: AUC 0,819
  - set soal: AUC 0,776
  - tebakan mayoritas: akurasi 81,7%

## Ronde 1 · Pool penuh 648 konfigurasi

- Run: `exp02_nested_main/outputs/nested_round1_20261127T023945Z_f09686af`
- Pool: 3 set fitur × 2 normalisasi × 3 target × 2 prior auxiliary × 6 C × 3 agregasi = 648 konfigurasi per track.

| Model | AUC | Balanced acc. | AUC dalam peserta |
|---|---:|---:|---:|
| Track A (sensor) | **0,799** | **0,760** | 0,716 |
| Track B (sensor + waktu berjalan) | 0,835 | 0,780 | 0,828 |
| Konteks saja | 0,818 | 0,811 | 0,909 |
| Batch 07 referensi | 0,778 | 0,741 | 0,645 |

- **Verdict:** Track A lulus kriteria.
- **Masalah:** Track B belum mengalahkan konteks saja pada balanced accuracy (0,780 vs 0,811), padahal uji likelihood-ratio menunjukkan sensor menambah informasi di atas konteks (p = 0,0002).
- **Pola seleksi inner** (dari `trackA_selection_log.csv`):
  - normalisasi `hybrid` dan agregasi `top50` terpilih di **17/17 fold**
  - target `zrating` dan fitur `core9_head4` **tidak pernah** terpilih
  - pilihan fitur, target, auxiliary, dan C bervariasi antar-fold
- **Diagnosis:** pool 648 konfigurasi menambah variansi seleksi pada inner-OOF yang hanya berisi sekitar 110 episode. Smoke test dengan satu konfigurasi `core9 · hybrid · bin · C=0,5 · top50` memberi AUC 0,825 (diagnostik, bukan hasil resmi).
- **Sensitivitas ambang overload (Track A):** rating ≥ 6 AUC 0,675, ≥ 7 AUC 0,799, ≥ 8 AUC 0,782. Ambang ≥ 7 dipertahankan.

## Ronde 2 · Pool diperkecil + Track B2 (stacking)

- **Perubahan:**
  1. Pool diperkecil menjadi 16 konfigurasi: `core9`/`core9_head4_base` × `hybrid` × `bin`/`rating` × auxiliary ya/tidak × C 0,1/0,5 × `top50`. Dasarnya pola seleksi inner ronde 1.
  2. Track B2 ditambahkan: regresi logistik dua variabel pada skor sensor Track A dan skor konteks. Stacker, kalibrasi, dan ambang dipelajari dari skor inner-LOPO.
- **Alasan:** mengurangi noise seleksi dan memperbaiki cara menggabungkan sensor dengan konteks untuk Track B.
- **Catatan kejujuran:** penyempitan pool memakai informasi dari ronde 1 (yang melihat seluruh 17 peserta). Uji permutasi pipeline ronde 2 tidak mencakup langkah penyempitan ini.
- Run: `exp02_nested_main/outputs/nested_round2_20261127T030014Z_f2c15e8a`

| Model | AUC | Balanced acc. | vs Batch 07 (bootstrap berpasangan) |
|---|---:|---:|---|
| Track A (seleksi 16 konfigurasi) | 0,802 | 0,760 | ΔAUC +0,024 (CI −0,07 s.d. +0,12) |
| Track B (sensor + waktu, satu model) | 0,867 | 0,806 | — |
| **Track B2 (stacking)** | **0,887** | **0,855** | ΔAUC +0,109 (p = 0,026); ΔBalAcc +0,114 (CI +0,005 s.d. +0,22) |

- **Verdict:** Track A tetap lulus. Track B2 mengalahkan konteks saja (0,818 / 0,811) pada kedua metrik.
- **Masalah tersisa:** keunggulan Track A atas Batch 07 masih kecil, dan seleksi tunggal per fold masih bervariasi (6 konfigurasi berbeda terpilih).

## Ronde 3 · Ensemble seluruh pool (ronde final)

- **Perubahan:**
  - Track A-ens: rata-rata skor z dari seluruh 16 konfigurasi. Statistik z, kalibrasi, dan ambang diambil dari inner-OOF fold tersebut, jadi tidak ada seleksi konfigurasi.
  - Track B-ens: perlakuan yang sama untuk Track B.
  - Track B3: stacking Track A-ens + konteks.
- **Penetapan sebelum run:** model utama sensor-only adalah **A-ens**, dan model deploy adalah **B3**. Penetapan ini dicatat di kode notebook exp02 (`PRIMARY_SENSOR`, `PRIMARY_DEPLOY`) sebelum hasil ronde 3 terlihat.
- **Alasan:** merata-ratakan konfigurasi adalah cara baku menghilangkan variansi seleksi tanpa memilih setelah melihat hasil.
- Run resmi: `exp02_nested_main/outputs/nested_round3_20261127T031955Z_8703732e` (eksekusi notebook). Run skrip sebelumnya menghasilkan prediksi identik (selisih < 1e-12) dan sudah dihapus saat pembersihan.

| Model | AUC (CI 95%) | Balanced acc. (CI 95%) | Akurasi (mayoritas 81,7%) | AUC dalam peserta |
|---|---:|---:|---:|---:|
| **Track A-ens (sensor)** | **0,826 (0,69–0,93)** | **0,768 (0,63–0,88)** | 78,6% | 0,712 |
| Track A (seleksi) | 0,802 (0,66–0,90) | 0,760 (0,66–0,85) | 74,6% | 0,702 |
| **Track B3 (deploy)** | **0,890 (0,77–0,97)** | **0,840 (0,75–0,91)** | 84,9% | 0,885 |
| Track B2 | 0,887 (0,78–0,96) | 0,855 (0,75–0,93) | 87,3% | 0,925 |
| Track B-ens | 0,905 (0,81–0,96) | 0,843 (0,75–0,91) | 82,5% | 0,906 |
| Konteks saja | 0,818 | 0,811 | 80,2% | 0,909 |
| Batch 07 referensi | 0,778 | 0,741 | 65,9% | 0,645 |

- **Verdict:** kedua model utama lulus. A-ens melampaui Batch 07 (AUC +0,048, BalAcc +0,027). B3 melampaui Batch 07 (ΔAUC +0,112, p = 0,029) dan konteks saja (ΔAUC +0,072, p = 0,058).
- **Iterasi dihentikan.** Kriteria sudah terpenuhi, dan menambah ronde lagi pada 17 peserta yang sama akan memperbesar risiko overfit ke kohort ini.
- **Uji permutasi pipeline A-ens** (200 permutasi rating antar-episode; seluruh pipeline nested ronde 3 diulang): AUC asli 0,826 tidak pernah tercapai. **p = 0,005** untuk AUC maupun balanced accuracy. AUC null rata-rata 0,465, persentil-95 0,638, maksimum 0,736. Hasil: `permutation_result.json` dan `permutation_null_trackA_ens.csv` di folder run resmi ronde 3 (dihitung pada run skrip yang prediksinya identik).
