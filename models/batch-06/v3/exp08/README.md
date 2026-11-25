# Exp08 · apa yang berubah bila ambang alarm High dinaikkan?

Pada 11 episode validation, ambang yang dipilih di Train menghindari dua alarm palsu dan melewatkan satu High tambahan. Dua alarm palsu yang tersisa memiliki skor lebih tinggi daripada keempat episode High. Pada cohort ini, menaikkan ambang lagi sampai salah satu dari kedua alarm tersebut hilang juga akan menghilangkan semua deteksi High.

## Model dan evaluasi

Eksperimen memakai model biner v3 dari exp07 dengan 14 fitur relatif terhadap riwayat sebelum tugas, meliputi eye, HRV, dan posisi kepala. Input model adalah median fitur seluruh window layak dalam satu episode, sehingga hasil berlaku untuk prediksi sesudah episode selesai. Pembobotan kelas, peserta dan episode, fallback referensi, StandardScaler, serta SVC RBF (C=1, gamma=scale) mengikuti exp07. Skor adalah nilai keputusan SVC dengan arah positif menuju High. Ambang standar adalah nol. Kontrol mereproduksi semua 37 prediksi LOPO Train dan 11 prediksi validation exp07. [Exp07](../exp07/README.md)

Sebelum fitting, batas recall High untuk pemilihan ambang ditetapkan **75%** sebagai asumsi eksperimen. Kandidat ambang adalah 0, 0,1, 0,2, 0,3, 0,4, 0,5, 0,75, dan 1,0. Dari kandidat yang memenuhi batas recall, pilih jumlah alarm palsu terkecil, lalu jumlah High terdeteksi terbesar, lalu ambang terkecil. Jika tidak ada yang memenuhi batas, gunakan nol. Model mengeluarkan alarm bila skor **lebih besar** dari ambang.

Untuk tiap peserta outer-LOPO Train, jalankan LOPO lagi di dalam peserta fitting untuk memilih ambang. Setiap fitting membangun ulang referensi waktu, label, bobot, dan scaler dari peserta yang tersedia baginya. Untuk validation, pilih satu ambang dari prediksi LOPO Train lengkap, kemudian latih model dengan seluruh Train. Validation yang sama telah dilihat pada exp07 dan menjadi sumber pertanyaan diagnostik. Test v3 tetap di luar evaluasi model.

## Hasil

| Evaluasi | Aturan | Benar | Alarm palsu / Non-High | High terdeteksi / seluruh High | Precision High |
|---|---|---:|---:|---:|---:|
| LOPO Train, 15 peserta dengan endpoint | Standar | 26/37 (70,3%) | 9/21 | 14/16 (87,5%) | 60,9% |
| LOPO Train yang sama | Ambang dipilih dalam tiap fold | 26/37 (70,3%) | 8/21 | 13/16 (81,3%) | 61,9% |
| Validation, 3 peserta | Standar | 6/11 (54,5%) | 4/7 | 3/4 (75,0%) | 42,9% |
| Validation yang sama | Ambang 0,5 | 7/11 (63,6%) | 2/7 | 2/4 (50,0%) | 50,0% |

Pada prediksi LOPO yang dipakai untuk memilih ambang final, 0,5 mempertahankan 12/16 High dan menghasilkan 6 alarm palsu. Itu skor pemilihan. Baris LOPO dengan pemilihan di dalam tiap fold pada tabel memberikan estimasi terpisah atas prosedurnya. Pada validation, satu High berskor 0,255 ikut kehilangan alarm bersama dua Non-High berskor 0,044 dan 0,403. Bila IDE membutuhkan recall High 75% pada peserta baru, bukti saat ini belum mendukung mengganti ambang standar menjadi 0,5.

## Tiga alarm palsu pada satu peserta

Peserta A menyumbang tiga dari empat alarm palsu pada validation exp07. Satu episode Low lain miliknya dikenali dengan benar. Ketiga episode di bawah memiliki cukup riwayat sensor asli pada sesi peserta. Ujung setiap referensi diperiksa terhadap awal episode. Referensi berasal dari aktivitas sebelumnya dan dapat berbeda dari kondisi istirahat.

| Label asli | Window riwayat non-overlap | Skor menuju High | Hasil sesudah ambang dinaikkan |
|---|---:|---:|---|
| Medium | 6 | 0,403 | Alarm hilang |
| Low | 12 | 0,884 | Masih alarm |
| Medium | 12 | 1,166 | Masih alarm |

Sebagai pembanding, skor tertinggi di antara seluruh High validation adalah 0,859. Dua kesalahan yang bertahan berada di atas skor tersebut. Ini membatasi apa yang dapat diperbaiki dengan satu ambang global pada model ini.

Untuk melihat perbedaan sensor, median selisih RMSSD dan SDNN episode terhadap referensi sesi pada tiga alarm palsu A masing-masing **−1,57 ms dan −1,43 ms**. Pada tiga episode Non-High validation yang dikenali benar oleh model standar, mediannya **+2,10 ms dan +7,99 ms**. Median selisih EAR kiri rata-rata adalah −0,0005 pada tiga alarm palsu dan +0,0262 pada tiga pembanding. Selisih rata-rata yaw kepala adalah +3,75 dan +10,41 pada skala fitur sumber. Masing-masing kelompok hanya mencakup tiga episode.

Untuk menguji apakah prediksinya bergantung pada satu modalitas, model tambahan dilatih pada Train dan dinilai pada validation yang sama dengan satu kelompok fitur dikeluarkan bergantian. Ketiga refit tetap memprediksi High untuk **ketiga** alarm palsu A pada ambang standar. Untuk seluruh validation, mengeluarkan HRV menghasilkan 5 alarm palsu dan 4/4 High terdeteksi, mengeluarkan eye menghasilkan 4 alarm palsu dan 4/4 High terdeteksi, dan mengeluarkan posisi kepala menghasilkan 3 alarm palsu dan 2/4 High terdeteksi. Diagnosis ini belum mengisolasi satu sensor penyebab. [Audit nilai tiap fitur](outputs/alarms_20261124T114747Z_b041fd42/validation_features.csv) · [Prediksi ablasi](outputs/alarms_20261124T114747Z_b041fd42/validation_ablation.csv)

## Reproduksi di Colab

Buka [notebook exp08](EyeHeartCognitiveLoad.ipynb), gunakan CPU, dan pilih **Run all**. Di bawah `DATASET_ROOT = PROJECT_ROOT / 'Dataset'`, sediakan paket `v3_behavioral_proxy/` serta `v2_session_phase/window_15s_step_5s/{good_acceptable.csv, all_data.csv}` versi repo yang digunakan exp07. Hash kedua CSV v2 diperiksa. Folder output exp07 tidak diperlukan untuk menjalankan exp08.

Run menyimpan satu ZIP ke `models/batch-06/exp08/outputs/`. Arsip berisi prediksi per episode, skor dan pemilihan ambang per fold, keanggotaan inner-fold, audit referensi dan sensor, hasil ablasi, manifest, dan metrik. Audit lokal memverifikasi pemisahan peserta, menghitung ulang confusion matrix, membandingkan kontrol dengan hasil exp07, serta mencocokkan isi ZIP dengan hasil run. [ZIP run lokal](outputs/alarms_20261124T114747Z_b041fd42.zip) · [Metrik](outputs/alarms_20261124T114747Z_b041fd42/metrics.csv)
