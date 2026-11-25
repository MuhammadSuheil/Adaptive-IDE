# Batch 07 · Perbandingan model pada label perilaku v3

**Extra Trees memperoleh AUC LOPO tertinggi, 0,786, diikuti SVM RBF dengan 0,768.** Pada ambang tetap, SVM benar pada satu episode LOPO lebih banyak dan menghasilkan 9 alarm palsu dibanding 11 pada Extra Trees. SVM juga memperoleh AUC validasi lebih tinggi. Extra Trees layak diteruskan sebagai kandidat pengurutan, dan SVM tetap menjadi acuan untuk keputusan High pada ambang saat ini.

## Exp01 · Hasil 25 November 2026

| Evaluasi | Model | Akurasi | Balanced accuracy | Macro-F1 | ROC AUC | High terdeteksi | Alarm palsu |
|---|---|---:|---:|---:|---:|---:|---:|
| LOPO Train · 15 peserta | SVM RBF | **25/37 (67,6%)** | **69,20%** | **0,675** | 0,768 | 13/16 | **9/21** |
| LOPO Train · 15 peserta | Regresi logistik L2 | 23/37 (62,2%) | 63,69% | 0,621 | 0,616 | 12/16 | 10/21 |
| LOPO Train · 15 peserta | Extra Trees | 24/37 (64,9%) | 67,56% | 0,644 | **0,786** | **14/16** | 11/21 |
| Validasi · 3 peserta | SVM RBF | **6/11 (54,5%)** | **58,93%** | **0,545** | **0,750** | **3/4** | 4/7 |
| Validasi · 3 peserta | Regresi logistik L2 | 5/11 (45,5%) | 46,43% | 0,450 | 0,714 | 2/4 | 4/7 |
| Validasi · 3 peserta | Extra Trees | **6/11 (54,5%)** | **58,93%** | **0,545** | 0,643 | **3/4** | 4/7 |

Pada LOPO, Extra Trees mendeteksi satu High tambahan dan menambah dua alarm palsu dibanding SVM. Selisih AUC kedua model setara dengan enam dari 336 pasangan High dan Non-High. Pasangan tersebut berbagi episode dan berasal dari sampel kecil. Validasi tidak menguatkan keunggulan AUC Extra Trees. Pada split validasi, tebakan selalu Non-High mencapai 7/11 benar dengan recall High nol. Test Batch 06 tetap merupakan hasil historis.

Pada episode dan label evaluasi yang sama, **SVM dengan riwayat berbasis kualitas** memperoleh AUC 0,768 di LOPO dan 0,750 di validasi. SVM Batch 06 dengan riwayat terkurasi memperoleh 0,699 dan 0,607. Akurasi LOPO bergeser dari 26/37 ke 25/37. Hasil ini mengukur dampak perubahan sumber riwayat untuk SVM, sementara perbandingan tiga model di atas memakai sumber riwayat yang sama. [Metrik SVM Batch 06](../batch-06/v3/exp11/README.md)

## Desain perbandingan

- Targetnya High versus Non-High. Ketiga model menerima median 14 fitur mata, HRV, dan posisi kepala per episode. Label dibentuk ulang dari peserta fitting pada setiap fold. Bobot kelas, peserta, dan episode serta scaler dihitung dari fold fitting.
- Riwayat memakai semua window berstatus good atau acceptable dengan fitur lengkap dari CSV v2 `all_data.csv`. Referensi median memakai 3–12 window yang tidak saling tumpang tindih dari sesi peserta yang sama, dengan window terakhir selesai sebelum atau tepat saat soal dimulai. Jika riwayat kurang, gunakan median episode fitting. Cara ini menghilangkan ketergantungan pada seleksi window yang menyeimbangkan kelas dan peserta di seluruh dataset v2.
- SVM RBF memakai C=1 dan gamma=`scale`. Regresi logistik L2 memakai C=1. Extra Trees memakai 300 pohon, kedalaman maksimum 3, minimal 3 episode per daun, dan seed 42. Ambang keputusan tetap 0 pada margin SVM dan 0,50 pada probabilitas dua model lain. Peringkat utama memakai AUC LOPO, lalu balanced accuracy jika AUC seri.
- Satu peserta Train tidak menyumbang episode evaluasi yang layak dalam LOPO. Semua kandidat menilai episode dan label yang sama dengan acuan Batch 06. Split validasi resmi tetap dipisahkan dari fitting.

## Reproduksi dan hasil

Buka [notebook Exp01](exp01/EyeHeartCognitiveLoad.ipynb) di Google Colab, pilih CPU, dan jalankan semua sel. Mount Drive menggunakan `PROJECT_ROOT` di sel pertama. Tempatkan input berikut di bawah `DATASET_ROOT = PROJECT_ROOT / 'Dataset'`.

```text
Dataset/
  v3_behavioral_proxy/
    episode_labels.csv
    participant_splits.csv
    window_15s_step_5s/
      good_acceptable.csv
      window_assignments.csv
      train_sample_weights.csv
  v2_session_phase/
    window_15s_step_5s/
      all_data.csv
```

Paket v3 tersedia di [dataset Batch 06](../batch-06/dataset/DATASET_README.md). Gunakan [snapshot `all_data.csv` v2 dari repo](../../analysis_sessions/dataset_fusion/output/v2_session_phase/window_15s_step_5s/all_data.csv). Notebook memeriksa hash snapshot tersebut. Hasil tersimpan di `PROJECT_ROOT / 'models/batch-07/exp01/outputs'`.

Run lokal memakai sel komputasi notebook dengan dataset dari `analysis_sessions/dataset_fusion/output`. Arsip hasil memuat skor per episode, metrik, peringkat LOPO, koefisien regresi logistik per fold, cakupan riwayat, dan manifest input.

[ZIP hasil lokal](exp01/outputs/models_20261125T124030Z_fe00ac84.zip) · [Metrik lengkap](exp01/outputs/models_20261125T124030Z_fe00ac84/metrics.csv) · [Prediksi per episode](exp01/outputs/models_20261125T124030Z_fe00ac84/episode_predictions.csv)
