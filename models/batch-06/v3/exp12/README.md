# Exp12 · apakah 45 detik memperbaiki HRV untuk klasifikasi v3?

**Pertahankan model median episode exp11 dengan window 15 detik.** Pada 11 episode validasi, mengganti HRV dengan 45 detik menemukan satu High yang sebelumnya terlewat dan menambah satu alarm palsu. Pada 37 episode LOPO, model acuan masih menghasilkan prediksi benar terbanyak.

## Hasil pada peserta yang disisihkan

| Cara menghitung HRV | LOPO benar | LOPO AUC | Validasi benar | Validasi AUC | High ditemukan saat validasi | Alarm palsu saat validasi |
|---|---:|---:|---:|---:|---:|---:|
| 15 detik, semua window layak (exp11 A) | 26/37 | 0,699 | 6/11 | 0,607 | 3/4 | 4/7 |
| 15 detik, waktu akhir berpasangan | 22/37 | 0,646 | 6/11 | 0,679 | 3/4 | 4/7 |
| 45 detik, waktu akhir berpasangan | 24/37 | 0,673 | 6/11 | 0,679 | 4/4 | 5/7 |

Baris kedua dan ketiga mengisolasi pengaruh panjang window HRV pada waktu akhir yang sama. RMSSD dan SDNN berasal dari CSV sensor tiap durasi. Window harus lolos kualitas dan seluruhnya berada dalam interval aktif yang sama pada satu episode. Pada episode tanpa pasangan, kedua model memakai median HRV 15 detik biasa. Semua resep menggunakan median mata dan posisi kepala dari seluruh window 15 detik serta referensi sesi 15 detik sebelum tugas. Setiap resep melatih kembali scaler dan SVM dengan label dan bobot dari fold fitting. Kode memisahkan peserta, ambang tetap nol, dan evaluasi mencakup LOPO di dalam Train serta validasi beku. Test v3 tidak dinilai.

Pada 360 pasangan waktu akhir dari episode evaluasi, median jumlah pasangan interval RR naik dari 18 ke 58. Cakupan 45 detik berbeda menurut kelas. Pada LOPO, 15/16 High dan 6/21 Non-High memiliki setidaknya satu pasangan. Pada validasi, angkanya 4/4 dan 2/7. Menghapus semua episode pendek akan menggeser komposisi kelas. Bahkan pada enam episode validasi yang memiliki pasangan, model 45 detik menyebut semuanya High. Dua dari enam itu sebenarnya Non-High.

## Audit alarm dan rating

Pada exp11 A, empat alarm palsu validasi terdiri dari dua Low dan dua Medium. Tiga berasal dari satu peserta. Rating aktual keempatnya adalah 1, 1, 2, dan 4. Dengan ambang rating ≥7 seperti v4, semuanya juga Non-High menurut peserta. Satu High perilaku yang luput memiliki rating 4. Dari tiga High perilaku validasi yang memiliki rating, hanya satu mencapai 7. Dua ukuran ini menjawab pertanyaan berbeda. Pada LOPO, 21 episode memiliki rating, dan skor model acuan mengurutkan rating ≥7 dengan AUC 0,827. Di validasi hanya satu episode dengan rating ≥7.

Pada 126 episode v4 yang masuk evaluasi tim, 55 berasal dari set easy dan tiga di antaranya diberi rating ≥7 (7, 10, 10). Peserta bisa merasa soal easy sulit. Pertahankan rating yang tercatat untuk analisis pengalaman peserta. [Laporan v4](../../outputs/exp03_v4_binary_overload/RESULTS.md)

## Reproduksi

Jalankan [notebook Colab](EyeHeartCognitiveLoad.ipynb) dengan CPU. Di bawah `DATASET_ROOT = PROJECT_ROOT / 'Dataset'`, gunakan input [exp11](../exp11/README.md) beserta `v2_session_phase/window_45s_step_5s/all_data.csv`. Run lokal mengeksekusi sel komputasi notebook dengan paket di `analysis_sessions/dataset_fusion/output`. Kontrol 15 detik mereproduksi semua 48 skor model A hingga selisih maksimum 1,3 × 10⁻¹⁵.

[ZIP hasil](outputs/hrv_20261125T084848Z_c8a7b984.zip) · [Metrik lengkap](outputs/hrv_20261125T084848Z_c8a7b984/metrics.csv) · [Prediksi, label, rating, dan cakupan per episode](outputs/hrv_20261125T084848Z_c8a7b984/episode_predictions.csv)
