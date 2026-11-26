# Batch 00: Treatment Awal pada Sembilan Dataset V2

## Ringkasan

Tim menjalankan sembilan notebook dengan treatment pemodelan batch-00 pada dataset `v2_session_phase`: window 15, 25, dan 45 detik, masing-masing dalam view all_data, good_acceptable, dan good_only. Step seluruh varian adalah 5 detik. Tujuannya memeriksa hasil metode awal setelah pengolahan window dan aturan label diperbarui.

Hasil tiga kelas tertinggi yang tercetak adalah sekitar **58% accuracy dan macro-F1 0,57** pada `25s / good_only`, memakai XGBoost dengan rolling. Untuk fokus good_acceptable, konfigurasi tertinggi mencapai sekitar **55% dan 0,53** pada window 45 detik, memakai XGBoost dengan sample weight. Keduanya masih menyisakan kesalahan pada ketiga kelas; Medium mempunyai recall 50% dan 42% pada dua konfigurasi tersebut.

Penilaian: hasil ini memberi baseline eksplorasi yang berguna, tetapi belum cukup kuat untuk menyatakan model Low/Medium/High mampu mentransfer ke rekaman lain. Skema blok waktu masih membagi window dari orang yang sama antara training dan test, dan sebagian interval sensor beririsan di kedua sisi. Uji binary mencapai sekitar 69%, sedangkan novelty detection tetap kesulitan mengenali Low. Skor kedua jenis uji tersebut tidak mewakili classifier tiga kelas.

## Sumber hasil dan pemeriksaan arsip

- Angka berasal dari output tersimpan pada sembilan notebook yang ditautkan di tabel, bukan training ulang saat penulisan laporan.
- Sembilan notebook memuat output seluruh bagian model, tanpa output exception. Execution count bagian model mengikuti urutan 9 sampai 18; tidak ada bagian model yang kosong.
- Path pada output pemuatan, jumlah row, dan support kelas pada classification report cocok dengan manifest dataset v2 lokal.
- Deklarasi model, hyperparameter, splitter, SMOTE, dan detector cocok dengan notebook [dataset lama](../old-dataset/EyeHeartCognitiveLoad.ipynb). Penyesuaian v2 meliputi path, target langsung dari CSV, konversi timestamp ke detik, serta penanganan NaN.
- Output mencatat scikit-learn 1.6.1, XGBoost 3.4.1, dan imbalanced-learn 0.14.2. Notebook tidak menyimpan hash CSV run, manifest fold, atau vektor prediksi test terpisah.
- Notebook mencetak metrik ke dua angka desimal. Persentase dan F1 pada laporan ini mengikuti pembulatan tersebut; selisih satu poin tidak diperlakukan sebagai bukti peningkatan yang signifikan.

## Data dan target

Target model adalah `cognitive_load_label` yang sudah ada pada CSV. Active easy/practice/demo, pre-task setelah kalibrasi, feedback, rating, jeda, dan post-task berlabel Low; active medium berlabel Medium dan active hard berlabel High. Rating/feedback setelah medium atau hard juga Low. Seluruh rentang window harus mempunyai target yang sama. Window boleh berisi beberapa fase dengan target Low yang sama.

Difficulty, fase, peserta, set, dan soal tetap metadata. Model menerima subset fitur sensor, bukan difficulty atau identitas sebagai predictor. Jumlah berikut cocok dengan [laporan kualitas dataset v2](../../../analysis_sessions/dataset_fusion/output/v2_session_phase/QUALITY_REPORT.md).

| Window | Dataset/notebook | Low | Medium | High | Total | Orang |
|---|---|---:|---:|---:|---:|---:|
| 15s | [all_data](window_15s_step_5s/EyeHeartCognitiveLoad_AllData.ipynb) | 1447 | 1805 | 1964 | 5216 | 18 |
| 15s | [good_acceptable](window_15s_step_5s/EyeHeartCognitiveLoad_GoodAcceptable.ipynb) | 732 | 1105 | 1174 | 3011 | 13 |
| 15s | [good_only](window_15s_step_5s/EyeHeartCognitiveLoad_GoodOnly.ipynb) | 202 | 271 | 338 | 811 | 5 |
| 25s | [all_data](window_25s_step_5s/EyeHeartCognitiveLoad_AllData.ipynb) | 1281 | 1477 | 1696 | 4454 | 18 |
| 25s | [good_acceptable](window_25s_step_5s/EyeHeartCognitiveLoad_GoodAcceptable.ipynb) | 666 | 968 | 1077 | 2711 | 13 |
| 25s | [good_only](window_25s_step_5s/EyeHeartCognitiveLoad_GoodOnly.ipynb) | 185 | 234 | 313 | 732 | 5 |
| 45s | [all_data](window_45s_step_5s/EyeHeartCognitiveLoad_AllData.ipynb) | 1105 | 966 | 1250 | 3321 | 18 |
| 45s | [good_acceptable](window_45s_step_5s/EyeHeartCognitiveLoad_GoodAcceptable.ipynb) | 614 | 650 | 854 | 2118 | 13 |
| 45s | [good_only](window_45s_step_5s/EyeHeartCognitiveLoad_GoodOnly.ipynb) | 168 | 146 | 240 | 554 | 4 |

Ini pemrosesan ulang rekaman yang tersedia, bukan penambahan peserta atau rekaman Low baru. Perubahan jumlah Low juga mengikuti aturan fase v2. good_only 45 detik mencakup empat orang, sedangkan dua varian good_only lainnya mencakup lima. Baris, durasi yang lolos, dan komposisi kelas berbeda antarvarian; perbandingan skor window tidak memakai test yang identik.

## Apa yang dilakukan

Setiap notebook menjalankan enam konfigurasi tiga kelas, satu konfigurasi binary, dan tiga detector novelty. Perlakuan utamanya mengikuti batch-00:

| Kode | Perlakuan | Evaluasi |
|---|---|---|
| RF-T | Random Forest, 100 tree, max_depth 5, balanced_subsample; imputasi median sebelum split seperti sumber | TimeSeriesSplit, 5 fold berdasarkan urutan baris CSV |
| XGB-T | XGBoost, 100 tree, max_depth 4, learning_rate 0,05; penanganan NaN bawaan | TimeSeriesSplit, 5 fold berdasarkan urutan baris CSV |
| XGB-W | XGBoost dengan sample weight balanced dari training fold | GroupKFold, 5 fold berdasarkan blok waktu 120 detik |
| RF-R | Random Forest, EAR gabungan, rolling tiga row, dan delta RMSSD; imputasi di training fold | GroupKFold blok waktu |
| XGB-R | XGBoost, fitur rolling/EAR gabungan, sample weight, subsample dan colsample_bytree 0,8 | GroupKFold blok waktu |
| XGB-S | Fitur rolling/EAR gabungan, imputasi median, SMOTE auto dengan k=5 di training fold, lalu XGBoost tanpa sample weight | GroupKFold blok waktu |
| Binary XGB | XGBoost dengan fitur rolling dan sample weight; Low dikeluarkan | GroupKFold blok waktu; target Medium versus High |
| IF-05 | Isolation Forest 100 tree, contamination 0,05 | Training: 80% awal Elevated; test: sisanya dan seluruh Low |
| IF-25 | Isolation Forest 200 tree, contamination 0,25, StandardScaler | Test novelty yang sama |
| OC-SVM | One-Class SVM RBF, nu 0,25, gamma scale, StandardScaler | Test novelty yang sama |

Novelty memakai median imputer yang di-fit pada Elevated training untuk mengatasi fitur sensor kosong. Median imputer pada salinan v2 juga mempertahankan jumlah kolom bila seluruh nilai suatu fitur kosong. Hyperparameter model dan daftar fitur mengikuti sumber.

Blok 120 detik adalah unit split, bukan panjang window HRV. Rolling tiga row bekerja pada fitur hasil window, tanpa menghitung ulang RR. Pada grid tanpa celah, tiga endpoint berturut-turut berjarak total 10 detik; filtering dapat membuat jarak antar-row lebih panjang.

## Hasil klasifikasi tiga kelas: validasi blok waktu

Isi sel adalah **accuracy rata-rata fold / macro-F1 dari classification report gabungan**. Setiap model blok waktu menghasilkan satu prediksi test per row. Angka accuracy pada report gabungan dapat berbeda dari rata-rata fold bila ukuran fold berbeda; keduanya juga telah dibulatkan pada output.

| Window | Dataset | XGB-W | RF-R | XGB-R | XGB-S |
|---|---|---|---|---|---|
| 15s | [all_data](window_15s_step_5s/EyeHeartCognitiveLoad_AllData.ipynb) | 47% / 0,47 | 45% / 0,45 | 46% / 0,46 | 46% / 0,46 |
| 15s | [good_acceptable](window_15s_step_5s/EyeHeartCognitiveLoad_GoodAcceptable.ipynb) | 46% / 0,45 | 41% / 0,40 | 44% / 0,43 | 45% / 0,45 |
| 15s | [good_only](window_15s_step_5s/EyeHeartCognitiveLoad_GoodOnly.ipynb) | 49% / 0,47 | 50% / 0,47 | 51% / 0,49 | 50% / 0,48 |
| 25s | [all_data](window_25s_step_5s/EyeHeartCognitiveLoad_AllData.ipynb) | 49% / 0,48 | 48% / 0,48 | 50% / 0,49 | 51% / 0,50 |
| 25s | [good_acceptable](window_25s_step_5s/EyeHeartCognitiveLoad_GoodAcceptable.ipynb) | 51% / 0,50 | 48% / 0,47 | 50% / 0,49 | 50% / 0,49 |
| 25s | [good_only](window_25s_step_5s/EyeHeartCognitiveLoad_GoodOnly.ipynb) | 53% / 0,52 | 54% / 0,53 | 58% / 0,57 | 57% / 0,56 |
| 45s | [all_data](window_45s_step_5s/EyeHeartCognitiveLoad_AllData.ipynb) | 54% / 0,54 | 51% / 0,50 | 56% / 0,55 | 55% / 0,54 |
| 45s | [good_acceptable](window_45s_step_5s/EyeHeartCognitiveLoad_GoodAcceptable.ipynb) | 55% / 0,53 | 50% / 0,49 | 54% / 0,53 | 54% / 0,53 |
| 45s | [good_only](window_45s_step_5s/EyeHeartCognitiveLoad_GoodOnly.ipynb) | 53% / 0,50 | 49% / 0,46 | 51% / 0,49 | 53% / 0,51 |

Temuan pada konfigurasi yang tersimpan:

- good_acceptable: macro-F1 tertinggi naik dari 0,45 pada 15 detik menjadi 0,50 pada 25 detik dan 0,53 pada 45 detik. Row test dan persentase overlap lintas split ikut berubah, sehingga pola ini belum mengisolasi manfaat fisiologis window lebih panjang.
- good_only: XGB-R 25 detik menghasilkan 58% / 0,57. Pada 45 detik, hasil tertinggi berdasarkan macro-F1 adalah XGB-S 53% / 0,51. Window lebih panjang tidak memberi kenaikan yang konsisten pada semua view.
- all_data: hasil tertinggi per varian adalah XGB-W 15 detik 47% / 0,47, XGB-S 25 detik 51% / 0,50, dan XGB-R 45 detik 56% / 0,55.
- Konfigurasi SMOTE tidak memberi peningkatan yang konsisten. Pada good_only 25 detik, XGB-S mencapai 57% / 0,56, di bawah XGB-R 58% / 0,57. Pada good_acceptable 45 detik, keduanya mencapai macro-F1 0,53. Perbandingan ini mencakup paket preprocessing/weighting masing-masing, bukan isolasi efek SMOTE saja.

### Hasil per kelas pada tiga konfigurasi acuan

| Konfigurasi | Kelas | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| 25s / good_only / XGB-R | Low | 0,56 | 0,54 | 0,55 |
| 25s / good_only / XGB-R | Medium | 0,47 | 0,50 | 0,48 |
| 25s / good_only / XGB-R | High | 0,68 | 0,66 | 0,67 |
| 45s / good_acceptable / XGB-W | Low | 0,51 | 0,52 | 0,51 |
| 45s / good_acceptable / XGB-W | Medium | 0,46 | 0,42 | 0,44 |
| 45s / good_acceptable / XGB-W | High | 0,64 | 0,67 | 0,65 |
| 45s / all_data / XGB-R | Low | 0,60 | 0,56 | 0,58 |
| 45s / all_data / XGB-R | Medium | 0,46 | 0,44 | 0,45 |
| 45s / all_data / XGB-R | High | 0,59 | 0,65 | 0,62 |

Medium tetap sulit dikenali: recall hanya 42–50% pada tiga konfigurasi acuan. Pada 45s good_acceptable, F1 Medium 0,44 dibanding High 0,65. Menambah atau memperbesar porsi Low belum menjawab pemisahan Medium dan High.

Sebagai referensi deskriptif, selalu menebak High pada seluruh 25s good_only menghasilkan 42,8% accuracy, sedangkan XGB-R tercetak 58%. Pada 45s good_acceptable, proporsi High 40,3% dibanding accuracy XGB-W 55%. Ini menunjukkan informasi prediktif di atas tebakan kelas konstan tersebut, tetapi kesalahan keseluruhan tetap sekitar 42–45%. Angka proporsi ini bukan hasil melatih Dummy pada setiap fold.

## Hasil TimeSeriesSplit

Angka berikut adalah accuracy rata-rata lima fold. Classification report pada kedua bagian ini hanya mencakup **fold terakhir**, sehingga macro-F1 report tersebut tidak dijadikan macro-F1 keseluruhan.

| Window | Dataset | RF-T | XGB-T |
|---|---|---:|---:|
| 15s | [all_data](window_15s_step_5s/EyeHeartCognitiveLoad_AllData.ipynb) | 34% | 37% |
| 15s | [good_acceptable](window_15s_step_5s/EyeHeartCognitiveLoad_GoodAcceptable.ipynb) | 30% | 33% |
| 15s | [good_only](window_15s_step_5s/EyeHeartCognitiveLoad_GoodOnly.ipynb) | 18% | 21% |
| 25s | [all_data](window_25s_step_5s/EyeHeartCognitiveLoad_AllData.ipynb) | 35% | 35% |
| 25s | [good_acceptable](window_25s_step_5s/EyeHeartCognitiveLoad_GoodAcceptable.ipynb) | 31% | 31% |
| 25s | [good_only](window_25s_step_5s/EyeHeartCognitiveLoad_GoodOnly.ipynb) | 26% | 20% |
| 45s | [all_data](window_45s_step_5s/EyeHeartCognitiveLoad_AllData.ipynb) | 35% | 33% |
| 45s | [good_acceptable](window_45s_step_5s/EyeHeartCognitiveLoad_GoodAcceptable.ipynb) | 30% | 34% |
| 45s | [good_only](window_45s_step_5s/EyeHeartCognitiveLoad_GoodOnly.ipynb) | 25% | 32% |

Accuracy rata-rata berada pada rentang 18–37%. Contoh 45s good_acceptable: RF-T mencetak rata-rata 30%, tetapi report fold terakhir menunjukkan accuracy 24% dan macro-F1 0,21. Perbedaan ini menegaskan bahwa skor satu fold tidak boleh dipakai sebagai ringkasan seluruh evaluasi. TimeSeriesSplit mengikuti urutan row CSV, bukan pemisahan peserta atau bukti urutan global lintas rekaman.

## Hasil binary Medium versus High

Low tidak masuk training atau test pada bagian ini. Isi accuracy berasal dari rata-rata fold, sedangkan macro-F1 berasal dari report gabungan.

| Window | Dataset | Row Medium/High | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| 15s | [all_data](window_15s_step_5s/EyeHeartCognitiveLoad_AllData.ipynb) | 3769 | 56% | 0,56 |
| 15s | [good_acceptable](window_15s_step_5s/EyeHeartCognitiveLoad_GoodAcceptable.ipynb) | 2279 | 58% | 0,58 |
| 15s | [good_only](window_15s_step_5s/EyeHeartCognitiveLoad_GoodOnly.ipynb) | 609 | 66% | 0,66 |
| 25s | [all_data](window_25s_step_5s/EyeHeartCognitiveLoad_AllData.ipynb) | 3173 | 61% | 0,60 |
| 25s | [good_acceptable](window_25s_step_5s/EyeHeartCognitiveLoad_GoodAcceptable.ipynb) | 2045 | 61% | 0,61 |
| 25s | [good_only](window_25s_step_5s/EyeHeartCognitiveLoad_GoodOnly.ipynb) | 547 | 69% | 0,69 |
| 45s | [all_data](window_45s_step_5s/EyeHeartCognitiveLoad_AllData.ipynb) | 2216 | 65% | 0,64 |
| 45s | [good_acceptable](window_45s_step_5s/EyeHeartCognitiveLoad_GoodAcceptable.ipynb) | 1504 | 58% | 0,58 |
| 45s | [good_only](window_45s_step_5s/EyeHeartCognitiveLoad_GoodOnly.ipynb) | 386 | 69% | 0,67 |

Hasil tertinggi mencapai sekitar 69% pada good_only 25 dan 45 detik. Ini belum merupakan model tiga kelas. Pada good_only 45 detik, High mencakup 240 dari 386 row binary, sehingga tebakan konstan High sudah menghasilkan 62,2%. Pada good_acceptable 45 detik, hasil binary 58% hanya sedikit di atas proporsi High 56,8%. Medium/High masih memerlukan perbaikan, bahkan setelah Low dikeluarkan.

## Hasil novelty detection

Detector mempelajari gabungan Medium/High sebagai Elevated/inlier. Test berisi 20% akhir Elevated dan seluruh Low. Isi sel adalah **accuracy / macro-F1 / recall Low**. Kolom terakhir adalah accuracy referensi bila seluruh row test ditebak Low, dihitung dari support report.

| Window | Dataset | IF-05 | IF-25 | OC-SVM | Selalu Low |
|---|---|---|---|---|---:|
| 15s | [all_data](window_15s_step_5s/EyeHeartCognitiveLoad_AllData.ipynb) | 41% / 0,38 / 13% | 43% / 0,43 / 35% | 44% / 0,44 / 34% | 65,7% |
| 15s | [good_acceptable](window_15s_step_5s/EyeHeartCognitiveLoad_GoodAcceptable.ipynb) | 37% / 0,35 / 16% | 42% / 0,42 / 38% | 45% / 0,45 / 36% | 61,6% |
| 15s | [good_only](window_15s_step_5s/EyeHeartCognitiveLoad_GoodOnly.ipynb) | 40% / 0,32 / 5% | 49% / 0,47 / 26% | 49% / 0,49 / 30% | 62,3% |
| 25s | [all_data](window_25s_step_5s/EyeHeartCognitiveLoad_AllData.ipynb) | 41% / 0,38 / 15% | 42% / 0,42 / 36% | 41% / 0,41 / 33% | 66,9% |
| 25s | [good_acceptable](window_25s_step_5s/EyeHeartCognitiveLoad_GoodAcceptable.ipynb) | 39% / 0,36 / 14% | 40% / 0,40 / 38% | 42% / 0,42 / 37% | 62,0% |
| 25s | [good_only](window_25s_step_5s/EyeHeartCognitiveLoad_GoodOnly.ipynb) | 39% / 0,30 / 3% | 47% / 0,45 / 22% | 45% / 0,44 / 25% | 62,7% |
| 45s | [all_data](window_45s_step_5s/EyeHeartCognitiveLoad_AllData.ipynb) | 36% / 0,33 / 12% | 42% / 0,41 / 37% | 41% / 0,41 / 36% | 71,3% |
| 45s | [good_acceptable](window_45s_step_5s/EyeHeartCognitiveLoad_GoodAcceptable.ipynb) | 33% / 0,33 / 18% | 44% / 0,43 / 44% | 45% / 0,44 / 45% | 67,1% |
| 45s | [good_only](window_45s_step_5s/EyeHeartCognitiveLoad_GoodOnly.ipynb) | 35% / 0,29 / 5% | 49% / 0,49 / 33% | 57% / 0,56 / 48% | 68,3% |

IF-05 hanya mengenali sekitar 3–18% Low pada sembilan dataset. OC-SVM tertinggi pada 45s good_only mencapai accuracy 57%, macro-F1 0,56, dan recall Low 48%. Pada test yang sama, tebakan konstan Low menghasilkan 68,3% accuracy. Ketiga detector pada sembilan dataset berada di bawah accuracy referensi konstan Low; referensi konstan itu sendiri mempunyai recall Elevated nol, sehingga tetap bukan solusi dua kelas.

Novelty dan binary berjalan sebagai uji terpisah. Notebook belum menguji alur end-to-end yang memutuskan Low/Elevated lalu Medium/High pada input yang sama. Accuracy binary dan novelty tidak dapat digabung menjadi klaim accuracy tiga kelas.

## Pemeriksaan pemisahan data

Audit membaca CSV lokal tanpa menjalankan ulang model. Audit merekonstruksi `GroupKFold(n_splits=5)` menggunakan grup `floor((window_start_epoch - min_epoch) / 120)` seperti source notebook. Satu row test dihitung beririsan bila intervalnya mempunyai overlap berdurasi positif dengan setidaknya satu interval training **dalam session_id yang sama**. Audit menghitung setiap row test satu kali pada fold tempatnya menjadi test.

| Window | Dataset | Row test beririsan dengan training | Persentase |
|---|---|---:|---:|
| 15s | all_data | 666 / 5216 | 12,8% |
| 15s | good_acceptable | 339 / 3011 | 11,3% |
| 15s | good_only | 98 / 811 | 12,1% |
| 25s | all_data | 1045 / 4454 | 23,5% |
| 25s | good_acceptable | 635 / 2711 | 23,4% |
| 25s | good_only | 160 / 732 | 21,9% |
| 45s | all_data | 1543 / 3321 | 46,5% |
| 45s | good_acceptable | 927 / 2118 | 43,8% |
| 45s | good_only | 212 / 554 | 38,3% |

Pada rekonstruksi tersebut, row test pada kesembilan kombinasi berasal dari peserta yang juga mempunyai row training pada fold yang sama. Pemisahan blok waktu tidak menjadi pemisahan orang. Window 45 detik juga mempunyai proporsi overlap lintas split lebih besar daripada 15 detik. Karena itu, kenaikan skor pada 45 detik dapat beriringan dengan perubahan representasi, row yang lolos, dan kedekatan train/test. Audit ini tidak menghitung berapa poin skor yang berasal dari masing-masing faktor.

Angka overlap adalah hasil rekonstruksi kode dan CSV lokal, bukan pembacaan manifest fold tersimpan. Audit memakai scikit-learn 1.9.1; run notebook mencatat 1.6.1. Notebook tidak menyimpan indeks fold/prediksi yang dapat dipakai untuk mengonfirmasi keanggotaan tiap split dari run Colab.

Batas lain yang mengikuti treatment awal:

- RF-T mempelajari median imputasi sebelum split, sehingga distribusi fitur test ikut memengaruhi preprocessing.
- Rolling memakai urutan global tanpa reset eksplisit per sesi/peserta dan tanpa purge terhadap riwayat yang melintasi split.
- SMOTE bekerja pada fitur hasil imputasi tanpa scaling sebelumnya. Bagian novelty baru memakai scaling pada dua detector terakhir.
- Perbandingan view mengubah jumlah row dan peserta. Label good/acceptable mengacu pada aturan kelengkapan/clipping sensor, bukan jaminan bahwa view dengan nama lebih baik akan memberi skor lebih tinggi.
- Tidak ada interval ketidakpastian atau pengulangan seed tersimpan yang membuktikan selisih kecil antarkonfigurasi. Laporan ini tidak menetapkan pemenang model untuk penggunaan akhir.

## Perbandingan deskriptif dengan dataset lama

Pada good_acceptable 25 detik, dataset lama mempunyai 2.294 row dengan Low 249; v2 mempunyai 2.711 row dengan Low 666. Medium dan High tetap 968 dan 1.077. Ini konteks terdekat untuk membaca [summary lama](../old-dataset/SUMMARY.md), tetapi target Low sekarang mencakup fase tambahan dan sebagian penanganan kompatibilitas berubah.

| Perlakuan blok waktu | Lama: accuracy / macro-F1 | V2 25s good_acceptable: accuracy / macro-F1 |
|---|---|---|
| XGB-W | 54% / 0,46 | 51% / 0,50 |
| RF-R | 48% / 0,41 | 48% / 0,47 |
| XGB-R | 53% / 0,45 | 50% / 0,49 |
| XGB-S | 53% / 0,45 | 50% / 0,49 |

Pada XGB-W, F1 Low berubah dari sekitar 0,18 menjadi 0,44, tetapi accuracy berubah dari 54% menjadi 51%. Macro-F1 lebih tinggi menunjukkan hasil antar-kelas yang lebih seimbang pada output tersebut. Perbedaan definisi target, isi test, dan arsip environment membatasi penafsiran sebab-akibat. Perbandingan ini tidak mengukur efek tambahan rekaman Low baru.

## Kesimpulan

Pengulangan treatment batch-00 pada dataset v2 belum menghasilkan classifier Low/Medium/High yang kuat. Hasil blok waktu terbaik berada pada 58% / 0,57, dengan keterbatasan split dan overlap yang terukur. Fokus good_acceptable mencapai 55% / 0,53; Medium masih sering tidak dikenali. Binary 69% dan hasil novelty tidak menutup kekurangan klasifikasi tiga kelas.

Untuk pengujian berikutnya, pertahankan 25s good_only dan 45s good_acceptable sebagai kandidat eksplorasi, lalu nilai ulang dengan split dan purge yang sesuai tujuan penggunaan. Bandingkan window pada endpoint/cohort test yang sebanding, laporkan precision/recall tiap kelas, dan uji kontribusi fitur/sensor serta penanganan imbalance secara terpisah. Pengambilan Low tambahan tetap dapat dilanjutkan, tetapi hasil ini belum menunjukkan bahwa jumlah Low merupakan satu-satunya penyebab kesalahan.

Dokumen ini hanya merangkum run dan audit. Tidak ada notebook, CSV sumber, parameter model, atau hasil training yang diubah saat penulisan.
