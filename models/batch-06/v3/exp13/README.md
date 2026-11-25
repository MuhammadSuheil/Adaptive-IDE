# Exp13 · normalisasi menurut variasi sensor peserta sebelum soal

Dengan normalisasi personal, akurasi validasi naik dari 6/11 menjadi **9/11 (81,8%)** pada tiga peserta. Pada evaluasi bergantian 15 peserta Train, akurasi turun dari **26/37 ke 25/37**. Jumlah alarm palsu LOPO tetap sembilan dan satu High tambahan terlewat. Pertahankan exp11 sebagai acuan v3. Angka validasi exp13 menggambarkan hasil pada 11 episode yang sudah digunakan untuk mengembangkan hipotesis ini.

## Hasil per episode

| Evaluasi | Normalisasi | Benar | High terdeteksi | Non-High disebut High | AUC episode |
|---|---|---:|---:|---:|---:|
| LOPO Train, 15 peserta | Exp11 | 26/37 (70,3%) | 14/16 | 9/21 | 0,699 |
| LOPO Train, 15 peserta | Exp13 | 25/37 (67,6%) | 13/16 | 9/21 | 0,690 |
| Validasi, 3 peserta | Exp11 | 6/11 (54,5%) | 3/4 | 4/7 | 0,607 |
| Validasi, 3 peserta | Exp13 | 9/11 (81,8%) | 4/4 | 2/7 | 0,714 |

Ketiga koreksi validasi terjadi pada tiga peserta berbeda. Satu episode Medium dan satu Low terbebas dari alarm palsu, sementara satu High yang terlewat kini terdeteksi. Dua alarm palsu yang tersisa berasal dari peserta yang sama. Pada LOPO terjadi tiga perubahan juga. Satu alarm palsu Medium terkoreksi, satu Medium lain mendapat alarm palsu baru, dan satu High yang semula tepat menjadi terlewat.

Semua 11 episode validasi dan 28/37 episode LOPO memiliki riwayat cukup untuk menghitung variasi semua 14 fitur. Jika hanya menghitung 28 episode LOPO dengan riwayat cukup, jumlah alarm palsu tetap 9/13 Non-High dan High terdeteksi turun dari 14/15 ke 13/15. Pada sembilan episode lain, kedua model menghasilkan keputusan yang sama.

Kedua model mengurutkan 12/16 pasangan High dan Non-High di dalam peserta dengan benar pada LOPO dan 7/7 pada validasi. Pada 21 episode LOPO yang bisa dicocokkan dengan rating v4, AUC untuk rating ≥7 bergeser dari 0,827 ke 0,786. Pada validasi, AUC rating sama-sama 0,778 dengan hanya satu rating ≥7 dari sepuluh yang tersedia. Target utama v3 tetap label perilaku. [Metrik](outputs/iqr_20261125T091155Z_5a63ee6f/metrics.csv) · [Pemeriksaan rating](outputs/iqr_20261125T091155Z_5a63ee6f/rating_alignment.csv)

## Cara menghitung variasi

Kedua model memakai median 14 fitur mata, HRV, dan posisi kepala dalam episode dengan window 15 detik, step 5 detik. Referensi pusat mengikuti exp11. Untuk tiap peserta dan sesi, ambil maksimal 12 window layak, saling lepas, dengan waktu selesai paling lambat saat episode dimulai. Kurangi median riwayat dari sensor episode jika tersedia minimal tiga window. Jika kurang, gunakan median episode fitting. Riwayat dapat mencakup aktivitas sebelumnya, sehingga referensi ini menggambarkan variasi dalam sesi.

Exp13 menghitung IQR tiap fitur (persentil ke-75 dikurangi persentil ke-25) dari minimal empat window riwayat tersebut. IQR pembanding dihitung per fitur dari episode fitting, dengan median per peserta terlebih dahulu lalu median antarpeserta. Faktor pengali perubahan sensor adalah `2 / (1 + IQR_personal / IQR_pembanding)`. Jika kedua IQR sama, faktor bernilai satu. Jika riwayat pendek atau IQR pembanding kosong atau nol, faktor juga satu. Pencampuran dengan IQR pembanding membatasi faktor hingga dua bila riwayat terlihat sangat tenang. Batas empat window dan formula ditetapkan sebelum evaluasi.

Label perilaku, bobot, scaler, dan SVM RBF dihitung pada fold fitting sesuai exp11. Kedua model memakai C=1, gamma=`scale` dan ambang skor nol. Semua 48 keputusan kontrol cocok dengan exp11, dengan selisih skor maksimum 1,3 × 10⁻¹⁵. Test v3 tidak dievaluasi.

## Reproduksi di Colab

Jalankan [notebook exp13](EyeHeartCognitiveLoad.ipynb) pada CPU dengan `DATASET_ROOT = PROJECT_ROOT / 'Dataset'`. Sediakan paket `v3_behavioral_proxy`, kedua file `all_data.csv` dan `good_acceptable.csv` dalam `v2_session_phase/window_15s_step_5s/`, serta `v4_self_reported_rating/episode_ratings.csv`. Run lokal mengeksekusi sel komputasi yang sama dengan input dari `analysis_sessions/dataset_fusion/output`.

[ZIP hasil lokal](outputs/iqr_20261125T091155Z_5a63ee6f.zip) mencakup prediksi per episode, jumlah dan batas waktu riwayat, skala pembanding tiap fold, metrik klasifikasi dan pengurutan, perbandingan rating, versi pustaka, dan hash input.
