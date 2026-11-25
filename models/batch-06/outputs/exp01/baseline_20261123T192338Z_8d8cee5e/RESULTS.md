# Hasil eksperimen baseline Batch 06

Window 25 detik, 10 fitur core, model umum Logistic Regression dan Extra Trees. Pembanding raw dan baseline-relative memakai baris serta fold identik. Test tidak dievaluasi. Hasil ini merupakan evaluasi development eksploratif.

Protokol split: `batch05_train_only_historical_reference`.

## Ketersediaan acuan awal

Sesi dalam CSV: 22; baseline RMSSD layak: 19; baseline RMSSD dan acuan eye awal layak: 4.

`hrv_rmssd` mengganti RMSSD dengan RMSSD / baseline - 1; sembilan fitur lain tetap. `hrv_eye` juga menormalkan statistik eye terhadap window pre_task murni 25 detik. SDNN tetap mentah karena CSV tidak menyimpan baseline SDNN independen. Pre_task adalah acuan sensor sebelum tugas, bukan bukti kondisi istirahat fisiologis.

Batas awal tugas dan akhir baseline berasal dari audit sumber kanonis yang disertakan. Nilai baseline RMSSD dan coverage harus cocok dengan CSV Drive. Sumber baru yang belum diaudit dikeluarkan dan dilaporkan. Notebook tidak mengklaim telah membaca ulang sumber mentah di Drive.

## Perbandingan peserta yang ditahan dari fitting

| Subset | Model | Representasi | Accuracy OOF | Macro-F1 OOF | Recall Low/Medium/High |
|---|---|---|---:|---:|---|
| hrv_rmssd | logistic_regression | raw | 37.73% | 35.59% | 60.26%/39.59%/12.80% |
| hrv_rmssd | logistic_regression | baseline_relative | 39.41% | 38.15% | 60.61%/34.93%/21.74% |
| hrv_rmssd | extra_trees | raw | 35.66% | 34.03% | 54.87%/34.93%/16.55% |
| hrv_rmssd | extra_trees | baseline_relative | 37.33% | 35.10% | 61.43%/32.21%/17.27% |
| hrv_rmssd | dummy | raw | 26.12% | 14.73% | 74.33%/0.00%/0.85% |
| hrv_eye | logistic_regression | raw | 34.43% | 32.56% | 50.38%/35.56%/14.98% |
| hrv_eye | logistic_regression | baseline_relative | 31.59% | 26.78% | 54.17%/34.44%/3.08% |
| hrv_eye | extra_trees | raw | 35.77% | 33.96% | 50.00%/40.56%/15.42% |
| hrv_eye | extra_trees | baseline_relative | 41.43% | 39.85% | 50.00%/55.00%/20.70% |
| hrv_eye | dummy | raw | 39.34% | 18.82% | 100.00%/0.00%/0.00% |

## Perubahan berpasangan

| Subset | Model | Delta accuracy | Delta Macro-F1 | Fold accuracy naik/total |
|---|---|---:|---:|---:|
| hrv_eye | extra_trees | +5.66 pp | +5.90 pp | 2/4 |
| hrv_eye | logistic_regression | -2.83 pp | -5.78 pp | 1/4 |
| hrv_rmssd | extra_trees | +1.67 pp | +1.07 pp | 13/19 |
| hrv_rmssd | logistic_regression | +1.67 pp | +2.56 pp | 11/19 |

OOF menggabungkan prediksi held-out setiap window Train tepat sekali. Fold seorang peserta dapat tidak mempunyai ketiga kelas; Macro-F1 per fold memakai tiga kelas tetap. Window overlap bukan sampel independen. Selisih deskriptif bukan bukti signifikansi statistik.

## Validation kandidat terkunci

Model dipilih menurut OOF peserta sebelum validation. Tidak ada pemilihan ulang menurut validation.

| Subset | Representasi | Model | Baris | Accuracy | Macro-F1 |
|---|---|---|---:|---:|---:|
| hrv_eye | baseline_relative | extra_trees | 99 | 65.66% | 44.58% |
| hrv_eye | raw | extra_trees | 99 | 52.53% | 33.46% |
| hrv_rmssd | baseline_relative | logistic_regression | 721 | 39.39% | 39.24% |
| hrv_rmssd | raw | logistic_regression | 721 | 36.34% | 36.11% |

## Interpretasi

Manfaat baseline dinilai dari perbandingan raw-relative pada subset dan protokol yang sama. Skor subset kecil tidak dibandingkan langsung dengan seluruh data batch sebelumnya. Subset eye dengan hanya beberapa orang merupakan kontrol eksploratif dengan ketidakpastian besar. Validation historis masih berisi sesi yang dikenal, sedangkan OOF peserta menahan seluruh rekaman orang tersebut. Dataset baru memakai split berbasis orang untuk Train, validation, dan Test.

Baseline RMSSD recorder umumnya lebih panjang daripada window tugas 25 detik; rasio ini menguji representasi relatif, bukan kesetaraan pengukuran HRV dua durasi. Belum ada rating beban mental independen. Perbaikan skor tidak memvalidasi label difficulty secara fisiologis.

## Status subset

```json
{
  "hrv_rmssd": {
    "status": "complete",
    "cv_fits": 120,
    "train_rows": 2454,
    "train_participants": 19
  },
  "hrv_eye": {
    "status": "complete",
    "cv_fits": 45,
    "train_rows": 671,
    "train_participants": 4
  }
}
```
