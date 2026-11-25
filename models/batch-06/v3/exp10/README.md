# Exp10 · apa yang bisa diambil dari v4 exp02 dan exp03?

**V4 berhasil mengurutkan episode dengan rating subjektif tinggi.** Resep linier yang memberi hasil terbaik pada v4 lebih lemah dari kontrol exp09 untuk mengurutkan label perilaku v3 pada peserta Train yang ditahan bergantian. Pemakaian rating v4 sebagai pembanding terpisah untuk skor sensor v3 memberi temuan yang lebih menjanjikan.

## Membaca hasil tim v4

| Target dan unit evaluasi | Hasil episode | Interpretasi |
|---|---|---|
| Exp02, prediksi rating 1–10 | Test 27 episode, MAE 2,528 dibanding dummy 2,969; R² −0,136 dan Spearman −0,196 | Galat absolut membaik atas dummy yang dipakai, tetapi urutan rating pada empat peserta Test masih lemah. |
| Exp02, tiga kelas | Test accuracy 37,0%, balanced accuracy 36,0% | Angka sekitar 70% dari eksperimen biner tidak berlaku untuk tiga kelas. |
| Exp02, ekstrem 1–3 vs 7–10 | Episode AUC 0,739 di Dev dan 0,714 di Test, balanced accuracy Test 62,1%. Test memiliki 19 episode yang memenuhi syarat. | Rating 4–6 dikeluarkan; di seluruh cohort cakupan menjadi 89/126 episode. |
| Exp03, ekstrem 1–3 vs 7–10 | AUC terbaik LOPO 0,784, AUC model terkunci pada empat peserta Test 0,629 | Pemilihan model dan kelompok peserta memengaruhi kesimpulan. |
| Exp03, seluruh rating, 1–6 vs 7–10 | LOPO 17 peserta: AUC 0,781 (CI bootstrap peserta 0,647–0,894), balanced accuracy 69,7%, accuracy 69,8%. Model terkunci pada Test: AUC 0,773, balanced accuracy 60,5%, accuracy 48,1%. | Bukti terbaik ada pada urutan skor. Cutoff 0,5 pada Test mendeteksi 4/5 High dengan 13 positif palsu dari 22 Non-High. |

Tiga hash CSV v4 cocok dengan manifest tim. Untuk skema 1–6 vs 7–10, fitting ulang Logistic Regression L2 dari CSV mereproduksi AUC 0,781342 (126 episode, 23 High) dan 0,772727 (27 episode Test, 5 High). Kandidat terbaik leaderboard memakai core9 + head + baseline. Pemilihan terpisah di 13 peserta Dev mengunci **core9** untuk pengujian empat peserta. LOPO seluruh 17 orang sekaligus dipakai untuk membandingkan kandidat sehingga skor kandidat terbaik perlu dibaca sebagai hasil development. Menebak 1–6 pada semua episode akan mencapai accuracy 81,7% pada cohort penuh dan balanced accuracy 50%. [Hasil exp02](../../outputs/exp02_v4_self_reported_rating/RESULTS.md) · [Hasil exp03](../../outputs/exp03_v4_binary_overload/RESULTS.md)

Pada resep L2 global yang sama, core9 sendiri mencapai AUC 0,778; menambahkan head menghasilkan 0,691, lalu menambahkan rasio RMSSD menghasilkan 0,781. Selisih AUC penuh terhadap core9 hanya 0,003. Klasifikasi 1–5 vs 6–10 juga lebih sulit (AUC terbaik LOPO 0,643). Temuan tersebut mendukung perbandingan fitur yang terkontrol dan pelaporan definisi High secara eksplisit. [Leaderboard per resep](../../outputs/exp03_v4_binary_overload/lopo_all_tasks_leaderboard.csv)

## Uji transfer pada v3

Saya memindahkan fitting **per window**, bobot per kelas dan episode dari fungsi tim (clip awal 5 dan normalisasi ulang per kelas), imputasi median Train, penskalaan median/IQR Train dengan clipping ±10, Logistic Regression L2 (C=0,5), dan rata-rata probabilitas per episode. Core9 terdiri dari dua fitur HRV dan tujuh fitur eye, dengan kolom perubahan rata-rata iris dikeluarkan. Resep penuh menambahkan empat fitur head pose dan rasio RMSSD terhadap baseline. Ambang episode tetap 0,5. Fitur identitas, durasi tugas dan kesalahan manual tidak masuk classifier.

Target v3 tetap High vs Non-High (Low + Medium). Untuk setiap fold, referensi dan label dibangun ulang di dalam peserta fitting sesuai exp09. Hanya window dari episode berlabel di fold tersebut yang ikut fitting. Seluruh kandidat memprediksi episode held-out yang sama. Pembandingan dengan exp09 mengevaluasi resep lengkap karena cara fitting, pembobotan dan representasinya juga berbeda. Test v3 tidak dipakai.

| Model | AUC LOPO Train, 37 episode | Benar LOPO | AUC validation, 11 episode | Benar validation | High benar / positif palsu di validation |
|---|---:|---:|---:|---:|---:|
| SVC RBF relatif, kontrol exp09 | **0,699** | **26/37** | 0,607 | 6/11 | 3/4 High · 4/7 Non-High |
| L2 global, core9 | 0,381 | 14/37 | 0,679 | 8/11 | 1/4 · 0/7 |
| L2 global, core9 + head | 0,321 | 12/37 | 0,750 | 9/11 | 3/4 · 1/7 |
| L2 global, core9 + head + baseline (transfer utama) | **0,283** | **11/37** | **0,750** | **9/11** | **3/4 · 1/7** |
| L2, core9 + head relatif ke riwayat sesi | 0,268 | 14/37 | 0,607 | 7/11 | 1/4 · 1/7 |

Pada transfer utama, LOPO hanya mendeteksi 5/16 High dan memberi 15 positif palsu pada 21 Non-High. Dalam peserta yang sama, skor resep ini mengurutkan 6/16 pasangan Train dengan benar; model exp09 mengurutkan 12/16. Hasil validation datang dari tiga orang yang sudah pernah dinilai. Dengan dan tanpa rasio baseline, seluruh hitungan keputusan validation sama. Setelah hasil awal diketahui, saya menguji satu variasi tambahan pada core9 + head: mengurangi tiap fitur dengan referensi sesi sebelum episode, lalu tetap memakai penskalaan global, fitting dan agregasi yang sama. Baris terakhir menunjukkan hasilnya. Bukti lintas peserta belum mendukung mengganti model exp09 dengan resep linier ini. [Metrik lengkap](outputs/transfer_20261124T174738Z_aca0b5f8/metrics.csv)

## Penggunaan rating untuk menilai skor v3

Saya mencocokkan prediksi v3 yang sudah dihasilkan pada peserta held-out dengan rating peserta di `episode_ratings.csv` melalui `episode_id` dan `participant_id`. Rating hanya dipakai sesudah fitting. Analisis ini eksploratif dan cakupannya terbatas pada episode yang memiliki rating.

| Cohort v3 | Episode dengan rating | Skor vs rating, Spearman (exp09 / transfer utama) | AUC untuk rating ≥7 (exp09 / transfer utama) |
|---|---|---:|---:|
| LOPO Train | 21/37, 10 orang, 7 rating ≥7 | **0,525 / −0,244** | **0,827 / 0,459** |
| Validation | 10/11, 3 orang, 1 rating ≥7 | 0,324 / 0,292 | 0,778 / 0,667 |

Di 21 episode Train yang cocok, lima dari tujuh episode berlabel High v3 juga diberi rating ≥7, dan dua dari 14 Non-High mendapat rating ≥7. Ini bukti awal kesesuaian dua indikator yang berbeda. Pertanyaan v4 meminta *kesulitan soal* yang dirasakan. Kesimpulan tentang usaha mental memerlukan ukuran yang sesuai dengan konstruk tersebut. Validation hanya punya satu rating tinggi, dan episode tanpa rating mungkin berbeda secara sistematis. [Kode pemeriksaan rating](rating_check.py)

## Reproduksi

Jalankan [notebook v3 exp10](EyeHeartCognitiveLoad.ipynb) pada CPU dengan `DATASET_ROOT = PROJECT_ROOT / 'Dataset'` dan input exp09 (paket v3 serta kedua CSV v2 15 detik versi repo). Run menyimpan ZIP ke `models/batch-06/v3/exp10/outputs/`. Fitting v3 dapat dilakukan tanpa paket v4. Untuk mengulang audit rating di repo, jalankan `python models/batch-06/v3/exp10/rating_check.py <episode_predictions.csv> models/batch-06/dataset/v4_self_reported_rating/episode_ratings.csv`. [ZIP run lengkap](outputs/transfer_20261124T174738Z_aca0b5f8.zip)
