# Exp11 · kapan fitur sensor sebaiknya dirangkum?

**Pertahankan median fitur per episode sebagai acuan v3.** Pada 37 episode evaluasi bergantian, model yang dilatih per window dan dirata-ratakan skornya memiliki AUC 0,515. Model median episode mencapai 0,699 pada endpoint yang sama. Keduanya menangkap 14 dari 16 High, tetapi model per window juga memprediksi High pada 18 dari 21 Non-High, dibanding 9 pada model median.

## Perbandingan terkontrol

- **A, median fitur.** Satu vektor median untuk tiap episode, lalu satu skor SVM. Resep ini mereproduksi model gabungan exp09. Selisih maksimum dari 48 skor exp09 kurang dari 10⁻¹².
- **B, rata-rata skor.** Setiap window dilatih dengan label episode asal. Satu episode dinilai dengan rata-rata aritmetika skor keputusan seluruh window layaknya. Bobot satu window adalah bobot fitting episode pada A dibagi jumlah window episode itu. Jumlah bobotnya diperiksa **per episode di tiap fold**. Episode panjang tidak menjadi lebih berpengaruh hanya karena jumlah window bertambah.

Keduanya menggunakan 14 fitur Eye, HRV, dan posisi kepala, referensi sesi sebelum episode, fallback median fitting, serta scaler yang sama dari median episode fitting. Kernel RBF memakai C=1 dan satu nilai gamma numerik per fold, dihitung seperti `gamma="scale"` pada A dan dipakai kembali pada B. Batas klasifikasi adalah skor di atas nol. Kedua resep menggunakan split peserta, rekonstruksi referensi label v3 per fold, dan daftar episode evaluasi yang sama seperti exp09. Rating v4 baru digabung setelah semua prediksi selesai. Test v3 tidak dinilai.

## Hasil pada label perilaku v3

| Evaluasi | Model | Benar | AUC episode | High terdeteksi | Non-High disebut High | Pasangan terurut benar dalam peserta |
|---|---|---:|---:|---:|---:|---:|
| LOPO Train, 15 peserta | A · median fitur | 26/37 | 0,699 | 14/16 | 9/21 | 12/16 |
| LOPO Train, 15 peserta | B · rata-rata skor | 17/37 | 0,515 | 14/16 | 18/21 | 6/16 |
| Validation, 3 peserta | A · median fitur | 6/11 | 0,607 | 3/4 | 4/7 | 7/7 |
| Validation, 3 peserta | B · rata-rata skor | 6/11 | 0,571 | 3/4 | 4/7 | 5/7 |

Lima peserta menyumbang 16 pasangan High/Non-High pada LOPO Train. Dua peserta menyumbang tujuh pasangan pada validation. Beberapa pasangan berbagi episode. Kedua resep merangkum episode sesudah selesai dan mengabaikan urutan kronologis window. Hasil ini menunjukkan bahwa fitting per window dengan label episode dan merata-ratakan skornya belum memperbaiki generalisasi. Hasil ini belum menjawab apakah lintasan perubahan sensor dari awal ke akhir episode membawa informasi tambahan.

## Kesesuaian dengan rating kesulitan v4

| Evaluasi | Episode dengan rating / eligible | Rating ≥7 | Spearman A / B | AUC rating ≥7 A / B |
|---|---:|---:|---:|---:|
| LOPO Train | 21/37, 10 orang | 7 | 0,525 / 0,400 | 0,827 / 0,684 |
| Validation | 10/11, 3 orang | 1 | 0,324 / 0,474 | 0,778 / 0,889 |

Pada validation, kenaikan AUC rating B setara dengan mengurutkan satu pasangan tambahan dengan benar dari sembilan pasangan yang membandingkan episode berating tinggi dengan episode lainnya. Bukti itu terlalu terbatas untuk mengganti keputusan pada v3. Label perilaku dan rating kesulitan sama-sama memberi informasi tentang tugas. Klaim mengenai beban kognitif yang dialami peserta tetap memerlukan ukuran usaha mental yang independen.

## Reproduksi dan artefak

Buka [notebook exp11](EyeHeartCognitiveLoad.ipynb) di Colab, gunakan CPU, lalu pilih **Run all**. Di bawah `DATASET_ROOT = PROJECT_ROOT / 'Dataset'` sediakan paket v3 dan kedua CSV v2 15 detik seperti exp09, serta `v4_self_reported_rating/episode_ratings.csv` untuk evaluasi rating. ZIP disimpan di `models/batch-06/v3/exp11/outputs/`. Run lokal memakai sel komputasi notebook yang sama.

[ZIP hasil lokal](outputs/paired_20261125T083045Z_4d048232.zip) memuat skor episode kedua model, skor window B, nilai gamma dan total bobot per fold, metrik, kesesuaian dalam peserta, kesesuaian rating, dan manifest input. [Tabel metrik](outputs/paired_20261125T083045Z_4d048232/metrics.csv) · [Tabel rating](outputs/paired_20261125T083045Z_4d048232/rating_alignment.csv)
