# Hasil Eksperimen Cognitive Load

Target memakai kesulitan tugas sebagai proxy cognitive load. Identitas peserta dipisahkan pada evaluasi.

## Perbandingan eksploratif

| Eksperimen | Metode | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| exp02 | catboost | 0.3819 | 0.2961 | 0.2921 |
| exp02 | extra_trees | 0.4263 | 0.3191 | 0.3045 |
| exp02 | logistic | 0.3862 | 0.3323 | 0.3260 |
| exp02 | random_forest | 0.4128 | 0.3110 | 0.3001 |
| exp02 | svm | 0.3793 | 0.3184 | 0.3175 |
| exp02 | xgboost | 0.3570 | 0.2771 | 0.2768 |
| exp03 | eye | 0.3426 | 0.2631 | 0.2578 |
| exp03 | hrv | 0.4067 | 0.3066 | 0.2959 |
| exp03 | no_gaze_state | 0.3200 | 0.2573 | 0.2579 |
| exp03 | no_iris | 0.3596 | 0.2891 | 0.2896 |
| exp03 | original | 0.3570 | 0.2774 | 0.2786 |
| exp04 | logistic_absolute | 0.4032 | 0.3409 | 0.3389 |
| exp04 | logistic_clean | 0.3840 | 0.3342 | 0.3294 |
| exp04 | logistic_clean_robust | 0.3875 | 0.3359 | 0.3311 |
| exp04 | logistic_ear | 0.3357 | 0.3016 | 0.2939 |
| exp04 | logistic_log | 0.3862 | 0.3295 | 0.3270 |
| exp04 | logistic_original | 0.3840 | 0.3342 | 0.3294 |
| exp04 | logistic_ratio | 0.3819 | 0.3226 | 0.3206 |
| exp04 | xgboost_absolute | 0.3649 | 0.2864 | 0.2847 |
| exp04 | xgboost_clean | 0.3570 | 0.2774 | 0.2786 |
| exp04 | xgboost_ear | 0.3684 | 0.2864 | 0.2866 |
| exp04 | xgboost_log | 0.3570 | 0.2774 | 0.2786 |
| exp04 | xgboost_original | 0.3570 | 0.2774 | 0.2786 |
| exp04 | xgboost_ratio | 0.3535 | 0.2729 | 0.2690 |
| exp05 | logistic_balanced | 0.3840 | 0.3342 | 0.3294 |
| exp05 | logistic_control | 0.3840 | 0.3342 | 0.3294 |
| exp05 | xgboost_balanced | 0.3522 | 0.2745 | 0.2776 |
| exp05 | xgboost_control | 0.3570 | 0.2774 | 0.2786 |
| exp06 | logistic_balanced | 0.3304 | 0.3150 | 0.3042 |
| exp06 | logistic_control | 0.3840 | 0.3342 | 0.3294 |
| exp06 | xgboost_balanced | 0.3278 | 0.2620 | 0.2664 |
| exp06 | xgboost_control | 0.3570 | 0.2774 | 0.2786 |
| exp07 | logistic_balanced | 0.3374 | 0.3186 | 0.3085 |
| exp07 | logistic_control | 0.3840 | 0.3342 | 0.3294 |
| exp07 | xgboost_balanced | 0.3339 | 0.2694 | 0.2713 |
| exp07 | xgboost_control | 0.3570 | 0.2774 | 0.2786 |
| exp08 | logistic_selection | 0.3348 | 0.2925 | 0.2896 |
| exp08 | xgboost_selection | 0.3422 | 0.2657 | 0.2656 |
| exp09 | logistic_history0 | 0.3840 | 0.3342 | 0.3294 |
| exp09 | logistic_history3 | 0.3784 | 0.3358 | 0.3292 |
| exp09 | logistic_history5 | 0.3714 | 0.3305 | 0.3241 |
| exp09 | xgboost_history0 | 0.3570 | 0.2774 | 0.2786 |
| exp09 | xgboost_history3 | 0.3531 | 0.2747 | 0.2754 |
| exp09 | xgboost_history5 | 0.3461 | 0.2731 | 0.2748 |
| exp10 | logistic_direct | 0.3840 | 0.3342 | 0.3294 |
| exp10 | logistic_hierarchy | 0.3806 | 0.3287 | 0.3249 |
| exp10 | xgboost_direct | 0.3570 | 0.2774 | 0.2786 |
| exp10 | xgboost_hierarchy | 0.3444 | 0.2898 | 0.2849 |
| exp11 | logistic_early | 0.3840 | 0.3342 | 0.3294 |
| exp11 | logistic_late | 0.4176 | 0.3293 | 0.3268 |
| exp11 | xgboost_early | 0.3570 | 0.2774 | 0.2786 |
| exp11 | xgboost_late | 0.3758 | 0.2827 | 0.2781 |
| exp12 | nested_selection | 0.3291 | 0.2711 | 0.2695 |

## Nested selection

Macro-F1 prosedur pemilihan: 0.2695. Kandidat dipilih dari inner validation setiap outer fold, bukan peringkat outer test.

## Kelas dan peserta

- Low: precision 0.0723, recall 0.0924, F1 0.0811, support 249.
- Medium: precision 0.3627, recall 0.4081, F1 0.3841, support 968.
- High: precision 0.3799, recall 0.3129, F1 0.3432, support 1077.
Accuracy antar peserta: 0.0069 sampai 0.6195. Rincian kelas per peserta terdapat di class_metrics.csv.

## Ketahanan dan batas interpretasi

- seed_sensitivity.csv memuat seed 42, 17, dan 2026; tidak memilih seed terbaik dari data uji.
- bootstrap_intervals.csv memakai resampling peserta, bukan window independen. Interval bersifat deskriptif pada jumlah peserta yang terbatas.
- nonoverlap_test_metrics.json hanya menipiskan prediksi test; tidak menghitung ulang HRV atau membuktikan efek overlap secara kausal.
- learning_curve.csv memakai model referensi tetap dan jumlah peserta training.
- Peringkat eksperimen merupakan eksplorasi pada dataset yang sama, bukan validasi eksternal independen.
- final_model.joblib merupakan refit seluruh data. Skor training model akhir bukan skor generalisasi baru.
