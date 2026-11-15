# Diagnosis Batch-01 pada Dataset Baru

## Kesimpulan

Hasil baru belum cukup kuat untuk menjadi classifier Low/Medium/High yang andal. Namun, **tidak benar bahwa seluruh hasilnya lebih buruk daripada batch-01 dataset lama**. Macro-F1 prosedur seleksi nested meningkat, sedangkan accuracy pada window yang sama hampir tidak berubah. Dibandingkan angka terbaik batch-00, skornya memang lebih rendah, tetapi cara pengujiannya berbeda dan tidak dapat diperlakukan sebagai pertandingan model pada kondisi yang sama.

Masalah yang ditemukan bukan hanya jumlah Low. Uji diagnostik menunjukkan pengaruh pemisahan rekaman, ketergantungan fitur history pada konteks soal/fase, lemahnya pemisahan Medium–High, representasi baseline HRV yang tidak selalu membantu, serta overfitting. Menambah Low tetap relevan, tetapi tidak menyelesaikan seluruh masalah tersebut dengan sendirinya.

Dokumen ini melengkapi [ringkasan hasil](SUMMARY.md), bukan mengganti hasil eksperimen. Dataset, notebook, model resmi, dan arsip lama tidak diubah. Temuan numerik berasal dari artefak tersimpan dan pengujian tambahan di [diagnostics](diagnostics/diagnose.py). Pengujian tambahan bukan batch eksperimen resmi baru dan tidak dimasukkan ke leaderboard.

## 1. Apakah hasil baru lebih buruk?

### Dibandingkan batch-01 dataset lama

Berikut hasil exp12, yaitu evaluasi prosedur pemilihan model melalui inner validation, bukan memilih skor test tertinggi dari seluruh eksperimen:

| Dataset dan window | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| Lama, 25 detik | 32,91% | 27,11% | 0,2695 |
| Baru, 15 detik | 31,35% | 32,25% | 0,3136 |
| Baru, 25 detik | 32,87% | 33,30% | 0,3281 |
| Baru, 45 detik | 36,26% | 35,57% | 0,3546 |

Pada 25 detik, accuracy turun hanya sekitar 0,05 poin persentase, sedangkan macro-F1 naik 0,0587. Jadi hasil baru tidak tepat disebut lebih buruk secara menyeluruh. Macro-F1 memberi bobot yang sama kepada ketiga kelas; accuracy dipengaruhi proporsi kelas yang berubah.

Beberapa konfigurasi tetap menurun. Logistic Regression tanpa rasio baseline, misalnya, turun dari accuracy 40,32% menjadi 35,74%, tetapi macro-F1 justru naik dari 0,3389 menjadi 0,3551. XGBoost history-5 pada 25 detik meningkat dari macro-F1 0,2748 menjadi 0,3676. Angka terakhir merupakan hasil konfigurasi eksploratif, bukan evaluasi seleksi nested.

Sumber: [55 pasangan konfigurasi pada window yang sama](diagnostics/results/old_new_same_window.csv), [perbandingan eksperimen baru](experiment_comparison.csv), dan [prediksi pada bagian test yang sama](diagnostics/results/common_test_comparison.csv).

Perubahan ini bukan uji penambahan Low murni: target/komposisi dataset berubah, 14 window bersama memiliki nilai HRV hasil replay yang diperbarui, dan runtime lama direkonstruksi karena source aslinya sudah dihapus. [Audit rekonstruksi](RECONSTRUCTION_AUDIT.json) mendukung kesamaan rancangan melalui kontrol regresi, bukan klaim source identik.

### Dibandingkan batch-00

| Rangkaian | Hasil tiga kelas yang dirujuk | Cara evaluasi |
|---|---|---|
| Batch-00, dataset lama | Sekitar 54% accuracy / 0,46 macro-F1 | Blok waktu 120 detik, lima fold |
| Batch-00, dataset baru good_acceptable 45 detik | Sekitar 55% / 0,53 | Blok waktu 120 detik, lima fold |
| Batch-01, dataset baru 45 detik, exp12 | 36,26% / 0,3546 | Seluruh rekaman seseorang dipisahkan dari training; seleksi nested |

Secara angka, batch-01 lebih rendah. Akan tetapi, batch-00 mengizinkan bagian rekaman orang yang sama muncul di train dan test, sedangkan batch-01 tidak. Fitur, parameter, serta prosedur seleksinya juga berbeda. Skor sekitar 58% pada batch-00 baru berasal dari good_only 25 detik dengan lima orang, bukan good_acceptable dengan 13 orang. Accuracy sekitar 61% pada batch-00 lama adalah percobaan dua kelas, bukan Low/Medium/High.

Karena itu, kesimpulannya bukan “dataset baru rusak” atau “batch-00 pasti menghasilkan model yang lebih baik”. Perbedaan evaluasi memang menghasilkan perbedaan besar; bagian berikut mengujinya dengan model dan fitur yang dibuat tetap.

Sumber: [batch-00 lama](../../batch-00/old-dataset/SUMMARY.md) dan [batch-00 baru](../../batch-00/new-dataset/SUMMARY.md).

## 2. Cakupan pemeriksaan

Pemeriksaan menelusuri CSV sumber, pemetaan target, preprocessing, history, manifest fold, hasil pencarian, prediksi OOF, dan model outer fold yang tersimpan.

- Seluruh 36 kombinasi resmi selesai dan lolos pemeriksaan checksum output.
- Label pada ketiga good_acceptable sesuai pemetaan fase v2. Tidak ditemukan pelanggaran pemetaan pada window yang masuk training.
- Nama peserta, session ID, nama soal, dan difficulty tidak masuk langsung sebagai kolom predictor. Namun, metadata soal/fase memengaruhi pembentukan history; dampaknya dibahas di bagian 5.
- Pengujian tambahan memakai Logistic Regression dan XGBoost dengan konfigurasi tetap, seed 42, dan 300 tree untuk XGBoost. Ada 182 fit untuk kontras split/history/komposisi training, ditambah 39 fit diagnostik penanda reset. Tidak ada pencarian hyperparameter tambahan.
- Pembacaan model hierarchy dan pengukuran train/test menggunakan model resmi yang sudah tersimpan, bukan training ulang model resmi.
- Uji overlap memakai perpotongan berdurasi positif pada rekaman yang sama. Window yang hanya menyentuh batas tidak dihitung overlap.

[Manifest audit](diagnostics/results/audit_manifest.json) mencatat environment, hash script utama, dan verifikasi bahwa dataset, output resmi, serta arsip lama tidak berubah. Script [supplement.py](diagnostics/supplement.py) menghasilkan ringkasan dan grafik dari bukti tersebut. Empat unit test diagnostik menguji batas interval, komposisi fase, aturan reset, dan penghitungan kandidat yang hilang di antara window.

## 3. Temuan utama dan tingkat kepastiannya

| Temuan | Bukti | Kesimpulan yang dapat diambil |
|---|---|---|
| Cara split sangat memengaruhi skor | Model dan fitur tetap; split blok, blok dengan purge, dan pemisahan rekaman menghasilkan selisih besar | Sebagian selisih terhadap batch-00 berasal dari kondisi pengujian yang lebih mudah, bukan otomatis kualitas model yang lebih baik |
| History membawa informasi konteks aktivitas | Reset/panjang history tanpa sensor sudah memprediksi label; mengubah aturan reset mengubah hasil XGBoost | History saat ini belum menjadi representasi sensor yang bebas konteks CodeArrange |
| Low bukan satu-satunya kelas yang sulit | Tahap Medium–High pada model hierarchy tetap lemah tanpa melewati keputusan Low | Kekurangan Low bukan penjelasan tunggal untuk hasil tiga kelas |
| Rasio baseline tidak selalu membantu | Menghapusnya meningkatkan macro-F1 pada kedua model dan ketiga panjang window | Representasi baseline perlu diuji ulang, bukan dipertahankan sebagai fitur wajib |
| Model kompleks sangat cocok dengan training tetapi lemah di test | XGBoost history-3 memiliki accuracy training sekitar 94–98%, test sekitar 34% | Ada overfitting/transfer antarrekaman yang lemah; menambah kapasitas bukan prioritas pertama |
| Tambahan baris Low tidak sama dengan tambahan rekaman easy | Tambahan 417 baris pada 25 detik mayoritas merupakan window campuran fase | Jumlah baris naik, tetapi keragaman rekaman dan cakupan pengerjaan easy belum otomatis bertambah |

Penyebab fisiologis, kondisi perangkat, dan ketepatan label terhadap beban yang benar-benar dirasakan tidak dapat dipisahkan hanya dari pengujian ini. Bukti di atas cukup untuk menetapkan prioritas teknis tanpa mengklaim satu penyebab menjelaskan seluruh kesalahan.

## 4. Perbedaan split menjelaskan sebagian besar kebingungan perbandingan skor

Uji ini memakai dataset dan representasi clean yang sama, tanpa history, balancing, atau tuning. Yang diubah adalah pemisahan train/test:

1. **Rekaman terpisah:** 13 fold LOPO resmi.
2. **Blok waktu:** 13 fold berdasarkan blok absolut 120 detik. Jumlah fold disamakan agar proporsi training lebih dekat dengan LOPO.
3. **Blok waktu dengan purge:** test sama seperti nomor 2, tetapi semua window training yang overlap dengan interval test pada rekaman yang sama dibuang sebelum fitting.

| Model | Window | Rekaman terpisah: macro-F1 | Blok + purge | Blok tanpa purge |
|---|---:|---:|---:|---:|
| Logistic Regression | 25 | 0,3074 | 0,4272 | 0,4452 |
| XGBoost | 25 | 0,2956 | 0,5258 | 0,5821 |
| Logistic Regression | 45 | 0,3147 | 0,4208 | 0,4592 |
| XGBoost | 45 | 0,2802 | 0,4828 | 0,6270 |

![Kontras split dengan model dan fitur tetap](diagnostics/figures/split_contrast.png)

Tanpa purge, 25,53% window test pada 25 detik dan 42,68% pada 45 detik berbagi sebagian interval mentah dengan training. Setelah purge, overlap menjadi nol, tetapi 8–11 orang pada setiap fold 25 detik dan 6–10 orang pada setiap fold 45 detik masih memiliki data di kedua sisi. Skor tetap jauh di atas LOPO.

Ini mendukung dua penjelasan: overlap membuat test lebih mudah, dan bagian lain dari rekaman yang sama membantu model. Selisih setelah purge tidak boleh seluruhnya disebut efek identitas biologis; kondisi kamera, baseline, urutan tugas, dan lingkungan rekaman ikut sama. Jumlah training juga berkurang setelah purge.

Audit terpisah terhadap rumus split batch-00 lima fold menemukan overlap pada 19,96% test 25 detik dan 39,47% test 45 detik di environment audit. Angka ini bukan rekonstruksi pasti manifest Colab lama: notebook batch-00 tidak mengarsipkan manifest dan versi scikit-learn berbeda. Kontras 13 fold di atas merupakan pengujian terkontrol baru, bukan klaim reproduksi angka batch-00.

Model yang dituju tetap satu classifier sensor, bukan model perorangan. Metadata dipakai untuk menghindari evaluasi pada rekaman yang terlalu mirip dengan training. Ini sejalan dengan penggunaan grouped cross-validation untuk data berkelompok dalam [dokumentasi scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).

Sumber: [skor probe](diagnostics/results/probe_scores.csv), [ledger setiap fold](diagnostics/results/probe_fold_integrity.csv), dan [ringkasan overlap](diagnostics/results/split_summary.csv).

## 5. Temuan baru: fitur history belum bebas dari konteks soal/fase

### Jalur pembentukan fitur

Pada [shared/features.py](shared/features.py), history reset ketika puzzle tidak diketahui, fase bernilai mixed, puzzle berubah, sesi berubah, atau jarak antarwindow melebihi 7,5 detik. Setelah reset, history sama dengan nilai saat ini dan delta bernilai nol untuk fitur yang tersedia.

Metadata tersebut memang tidak ditempelkan langsung sebagai angka dalam matriks predictor. Namun, aturan transformasinya meninggalkan pola pada fitur turunan. Jadi “tidak ada kolom soal/fase dalam X” tidak sama dengan “fitur sepenuhnya bebas dari konteks soal/fase”. Pernyataan sensor-only dalam ringkasan hasil perlu dibaca dengan pembedaan ini.

Pada window 25 detik, seluruh 354 Low yang mencampur pengerjaan dan nonpengerjaan mengalami reset; panjang history selalu satu. Sebaliknya, reset terjadi pada 20,45% Medium dan 13,93% High. Perbedaan ini mengikuti cara aktivitas dan label tersusun dalam dataset.

### Pengujian tanpa nilai sensor

Logistic Regression hanya diberi dua penanda: apakah history reset dan panjang history 1–3. Tidak ada nilai HRV, eye tracking, nama peserta, difficulty, atau label sebagai predictor. Evaluasinya tetap 13 fold LOPO.

| Window | Accuracy | Macro-F1 |
|---|---:|---:|
| 15 detik | 48,42% | 0,4458 |
| 25 detik | 52,23% | 0,4751 |
| 45 detik | 58,64% | 0,4761 |

Ini **bukan model CL yang direkomendasikan** dan tidak masuk hasil resmi. Ini kontrol negatif yang menunjukkan kuatnya petunjuk dari struktur aktivitas. Ia tidak membuktikan bahwa seluruh performa model sensor berasal dari petunjuk tersebut.

### Mengubah hanya aturan reset

Probe berikut mempertahankan model, sensor, panjang history tiga, dan LOPO. Konteks soal/fase diganti menjadi satu aliran; reset hanya mengikuti sesi dan jarak waktu. Atribut cache frame dibersihkan agar fitur benar-benar dihitung ulang.

| Model | Window | Tanpa history | History aliran | History soal/fase resmi |
|---|---:|---:|---:|---:|
| Logistic Regression | 25 | 0,3074 | 0,3080 | 0,3128 |
| XGBoost | 25 | 0,2956 | 0,3038 | 0,3627 |
| Logistic Regression | 45 | 0,3147 | 0,3084 | 0,3091 |
| XGBoost | 45 | 0,2802 | 0,2890 | 0,3710 |

![Kontras aturan history](diagnostics/figures/history_contrast.png)

Kenaikan besar pada XGBoost history soal/fase tidak bertahan pada history aliran. Kedua versi juga merata-ratakan rentang data yang berbeda akibat reset, sehingga selisihnya bukan ukuran pasti besarnya kebocoran. Namun, manfaat history resmi belum terbukti sebagai manfaat dinamika sensor yang dapat dipakai tanpa CodeArrange. Model refit akhir 25 dan 45 detik memilih history-3; ketergantungan ini relevan bagi pemakaian model, bukan hanya varian eksperimen tambahan.

### Temuan lanjutan: membuat history setelah filter label juga meninggalkan batas buatan

History sekarang dibentuk dari good_acceptable yang sudah mengeluarkan window campuran label. Bahkan history aliran pada probe di atas masih melihat gap akibat filtering tersebut.

Di antara window yang dipertahankan dalam sesi yang sama, ditemukan kandidat yang hilang berikut:

| Window | Kandidat hilang karena campuran label | Karena kualitas/admission |
|---|---:|---:|
| 15 detik | 1.017 | 726 |
| 25 detik | 1.479 | 501 |
| 45 detik | 2.131 | 339 |

Hitungan ini hanya kandidat yang terletak di antara dua window retained dengan gap >7,5 detik, bukan seluruh exclusion dataset. Satu gap dapat mencakup kedua alasan. Campuran label tidak sama dengan sensor putus, tetapi keduanya dapat memicu reset yang sama bila history dibuat setelah penyaringan target.

Implikasi untuk batch-02: bangun riwayat kausal pada aliran fitur sensor sebelum mask label/admission untuk evaluasi. Jangan menggunakan label, batas soal, atau keberadaan label valid untuk menentukan history ketika deployment tidak memiliki informasi tersebut. Sensor yang benar-benar tidak layak tetap memerlukan aturan fallback/expiry; jangan memasukkan nilai invalid hanya untuk menutup gap.

Sumber: [reset menurut kelompok](diagnostics/results/history_reset.csv), [probe tanpa sensor](diagnostics/results/reset_only_probe.csv), [probe history](diagnostics/results/probe_scores.csv), serta [kandidat di dalam gap](diagnostics/results/history_gap_candidates.csv).

## 6. Imbalance Low masih ada, tetapi bukan penjelasan tunggal

### Bertambahnya baris belum sama dengan bertambahnya rekaman

| Window | Low | Medium | High | Porsi Low |
|---|---:|---:|---:|---:|
| 15 detik | 732 | 1.105 | 1.174 | 24,31% |
| 25 detik | 666 | 968 | 1.077 | 24,57% |
| 45 detik | 614 | 650 | 854 | 28,99% |

Imbalance jauh lebih ringan daripada dataset lama, yang hanya memiliki 249 Low dari 2.294 window atau 10,85%. Pada 45 detik, perbandingan kelas terbesar dan terkecil sekitar 1,39 kali, tetapi hasil tetap lemah. Karena itu, imbalance memang nyata, tetapi belum terbukti sebagai penyebab utama seluruh kesalahan.

Pada 25 detik, 2.294 window lama seluruhnya tetap ada dengan label yang sama. Tambahan 417 window berasal dari rekaman yang sama, bukan peserta atau sesi baru: 356 memiliki study_phase mixed, 38 feedback, 12 pretask, 5 rest, 3 active_task, 2 posttask, dan 1 interrupted. Definisi Low v2 tetap dipertahankan. Temuan ini membedakan perluasan cakupan fase dari pengambilan easy baru.

### Low mencakup beberapa komposisi fase

Pengelompokan berikut membaca durasi positif dalam window_phase_mix, bukan hanya nama fase teratas. Semua baris tetap berlabel Low sesuai protokol.

| Window | Pengerjaan saja | Pengerjaan + nonpengerjaan | Nonpengerjaan saja |
|---|---:|---:|---:|
| 15 detik | 340 | 261 | 131 |
| 25 detik | 252 | 354 | 60 |
| 45 detik | 146 | 453 | 15 |

Pada exp12 25 detik, recall Low masing-masing 26,98%, 43,79%, dan 26,67%. Jadi tambahan fase Low belum membuat seluruh bentuk Low dikenali dengan baik. Dua window berfase mixed pada 25 detik ternyata seluruh durasinya masih active_task lintas soal; inilah alasan jumlah mixed pada tabel sebelumnya tidak identik dengan kelompok campuran aktivitas di sini.

Ini bukan instruksi membatalkan label Low untuk rating, feedback, atau istirahat. Yang diperlukan adalah memastikan data latihan mencakup variasi Low yang dituju, lalu melaporkan kemampuannya per kelompok tanpa mengubah target classifier tiga kelas.

### Uji terkontrol penambahan 417 Low

Untuk memisahkan perubahan isi training dari perubahan environment/replay, model tetap dilatih dalam runtime baru dengan nilai fitur v2 yang sama. Kondisi A memakai hanya ID training yang ada pada v1; kondisi B memakai seluruh training v2. Test, fold, model, dan preprocessing recipe sama. Tabel ini bukan eksperimen batch-02 dan tidak menjadi kewajiban perbandingan sebelum–sesudah pada batch mendatang.

| Model | Training | Macro-F1 pada 2.294 test bersama | Recall pada 417 Low tambahan |
|---|---|---:|---:|
| Logistic Regression | ID lama, fitur v2 | 0,3297 | 33,81% |
| Logistic Regression | Seluruh v2 | 0,2652 | 47,00% |
| XGBoost | ID lama, fitur v2 | 0,2783 | 9,35% |
| XGBoost | Seluruh v2 | 0,2659 | 34,53% |

Tambahan Low membantu mengenali Low tambahan, tetapi mengubah keputusan pada bagian lama dan menurunkan macro-F1 bagian tersebut untuk dua kontrol ini. Pada seluruh test v2, Logistic turun dari 0,3665 menjadi 0,3074, sedangkan XGBoost naik dari 0,2756 menjadi 0,2956. Jadi dampaknya bergantung model dan bagian data yang diukur; “jumlah lebih seimbang” tidak otomatis berarti semua metrik membaik. Pengujian ini mengisolasi paket penambahan baris beserta dampaknya pada preprocessing training, bukan memastikan satu alasan fisiologis mengapa keputusan bergeser.

Di eksperimen resmi 25 detik, weighting dan random oversampling tidak meningkatkan macro-F1 kedua kontrol. SMOTE menaikkan Logistic sekitar 0,0020 dan XGBoost sekitar 0,0112. Kenaikannya terbatas, bukan penyelesaian masalah pemisahan kelas.

### Medium–High tetap sulit walaupun keputusan Low dilewati

Tahap kedua hierarchy exp10 diuji langsung pada test yang benar-benar Medium atau High, tanpa melewati keputusan tahap Low/Elevated:

| Window | Logistic: accuracy / macro-F1 biner | XGBoost: accuracy / macro-F1 biner | Accuracy selalu High |
|---|---|---|---:|
| 15 detik | 52,79% / 0,5272 | 45,33% / 0,4533 | 51,51% |
| 25 detik | 52,32% / 0,5226 | 45,97% / 0,4592 | 52,67% |
| 45 detik | 55,85% / 0,5577 | 39,96% / 0,3993 | 56,78% |

Low tidak menentukan keputusan tahap ini, tetapi Medium–High belum terpisah dengan kuat. Ini bukan klaim bahwa seluruh model biner pasti gagal: yang diuji adalah tahap kedua model resmi yang ada, dengan preprocessing tersimpan, bukan pencarian biner baru. Temuannya cukup untuk menolak dugaan bahwa semua kesalahan semata-mata berasal dari Low yang sedikit.

Sumber: [komposisi/error Low](diagnostics/results/low_phase_errors.csv), [kontras dataset](diagnostics/results/dataset_contrast.json), [uji training tetap](diagnostics/results/probe_scores.csv), dan [tahap kedua hierarchy](diagnostics/results/binary_second_stage.csv).

## 7. Representasi sensor dan baseline masih bermasalah

### Rasio baseline konsisten tidak membantu pada kontrol yang diperiksa

Varian absolute exp04 berbeda dari clean dengan menghapus rmssd_baseline_ratio; RMSSD dan SDNN tetap tersedia. Pada dua model dan tiga panjang window, penghapusan rasio meningkatkan macro-F1:

| Model | Window | Dengan rasio | Tanpa rasio | Selisih |
|---|---:|---:|---:|---:|
| Logistic Regression | 15 | 0,3307 | 0,3571 | +0,0264 |
| Logistic Regression | 25 | 0,3074 | 0,3551 | +0,0477 |
| Logistic Regression | 45 | 0,3147 | 0,3594 | +0,0447 |
| XGBoost | 15 | 0,3012 | 0,3101 | +0,0090 |
| XGBoost | 25 | 0,2950 | 0,3090 | +0,0141 |
| XGBoost | 45 | 0,2797 | 0,3180 | +0,0383 |

Pola enam dari enam ini tetap berlaku ketika metrik dibatasi pada test yang rasio baselinenya tersedia. Jadi hasilnya tidak hanya berasal dari test satu rekaman dengan rasio kosong. Namun, ini pembatasan bagian test dari model yang sama, bukan retraining setelah membuang rekaman tertentu; penyebab tepat ketidakbergunaan rasio belum terbukti.

Baseline recorder berstatus ready untuk seluruh 13 rekaman. Satu rekaman, Tio, memiliki coverage baseline 61,28%, di bawah syarat builder 80%, sehingga baseline_usable false dan rasio dibiarkan kosong. Perbedaan ready dan usable memang harus dipertahankan, bukan mengisi rasio kosong dengan nol atau menganggapnya baseline valid.

### Perbedaan antarrekaman lebih kuat daripada perbedaan rata-rata kelas pada banyak fitur

Sebagai pemeriksaan deskriptif, diambil subset window tanpa overlap dalam setiap rekaman. Pada 25 detik, R² penjelasan oleh grup rekaman sekitar 0,706 untuk RMSSD, 0,871 untuk EAR kiri rata-rata, dan 0,919 untuk iris rata-rata. R² pengelompokan berdasarkan tiga label pada fitur yang sama sekitar 0,014, 0,001, dan 0,017.

Ini bukan uji signifikansi atau pembuktian bahwa sensor tidak mengandung CL: jumlah grup berbeda, pola nonlinier belum diukur, dan nonoverlap tidak menghilangkan seluruh korelasi waktu. Dengan satu rekaman per orang pada view ini, faktor tubuh, kamera, pencahayaan, dan posisi tidak dapat dipisahkan. Temuan ini mendukung pengujian fitur yang relatif/dinamis serta pengurangan mean geometris, bukan memasukkan ID peserta ke model.

### Kualitas gaze tidak sama dengan ketersediaan nilai eye

Validasi gaze asli gagal pada 12 dari 13 orang; hanya Fabian lolos. Ini sudah menjadi masalah pada data lama dan tidak diperbaiki oleh memperluas fase Low. Good/acceptable menilai syarat yang ditentukan builder, bukan sertifikat bahwa seluruh koordinat gaze akurat.

Ablation exp03 menunjukkan HRV-only mengungguli fusion referensi pada ketiga window; menghapus iris juga membantu XGBoost referensi pada ketiganya. Sebaliknya, menghapus seluruh gaze/eye-state tidak membantu. Jadi jangan otomatis membuang semua eye atau 12 orang. Pisahkan kontribusi EAR/blink, iris, head pose, gaze, serta eye-state dengan kontrol yang sama dan pemeriksaan kualitas yang lebih spesifik.

Sumber: [ablation rasio](diagnostics/results/baseline_ratio_ablation.csv), [variasi fitur](diagnostics/results/feature_variance.csv), [admission sensor](diagnostics/results/sensor_admission.csv), dan hasil exp03 pada [SUMMARY](SUMMARY.md#exp03-ablation-sensor).

## 8. Overfitting dan seleksi kandidat belum terselesaikan

Berikut rata-rata accuracy per outer fold dari model history-3 resmi yang sama. Nilai training adalah resubstitution, bukan validasi:

| Model | Window | Accuracy training | Accuracy test |
|---|---:|---:|---:|
| Logistic Regression | 25 | 59,30% | 32,46% |
| XGBoost | 25 | 94,48% | 34,26% |
| Logistic Regression | 45 | 64,91% | 33,47% |
| XGBoost | 45 | 98,33% | 34,15% |

Ini rata-rata fold, berbeda dari pooled accuracy pada tabel hasil utama. Prediksi test hasil pembacaan model dicocokkan persis dengan prediksi resmi yang tersimpan.

XGBoost dapat menyesuaikan training dengan sangat baik, tetapi penyesuaian itu tidak berpindah dengan baik ke rekaman test. Penyebabnya dapat mencakup kapasitas, petunjuk rekaman/konteks, dan lemahnya fitur yang stabil lintas kondisi. Menambah model yang lebih kompleks sebelum memperbaiki representasi tidak didukung hasil ini.

Pemilihan exp12 juga belum stabil: median jarak skor inner kandidat pertama–kedua hanya sekitar 0,0057 pada 15 detik, 0,0093 pada 25 detik, dan 0,0046 pada 45 detik. Banyak kandidat berdekatan. Pada 45 detik, rata-rata skor inner pemenang 0,3848, sedangkan rata-rata macro-F1 outer per fold 0,2396, atau 0,2537 bila hanya kelas yang hadir diperhitungkan. Kedua sisi memakai komposisi dan agregasi berbeda, sehingga selisihnya bukan estimasi bias yang persis. Ia tetap menunjukkan pilihan inner belum menghasilkan transfer yang konsisten.

Jangan menjadikan skor outer tertinggi dari 154 kandidat sebagai estimasi final tanpa seleksi nested. Bank batch-02 sebaiknya lebih kecil, berisi hipotesis yang jelas, termasuk Logistic tanpa rasio dan history generik sebagai kontrol. Prinsip pemisahan tuning dan evaluasi dirangkum dalam [dokumentasi nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

Sumber: [train/test setiap fold](diagnostics/results/train_test_gap.csv), [rata-rata fold](diagnostics/results/train_test_gap_mean.csv), dan [transfer seleksi](diagnostics/results/selection_transfer.csv).

## 9. Window panjang belum terbukti menjadi solusi utama

Window 45 detik memberikan skor nested teratas saat ini, tetapi juga mengubah bagian data yang lolos. Dari 614 Low, sebanyak 453 atau 73,78% mencampur pengerjaan dengan nonpengerjaan; Low pengerjaan saja tinggal 146. Pada 15 detik, Low pengerjaan saja masih 340.

Untuk menggambarkan kepadatan temporal, pemilihan greedy interval yang tidak overlap menghasilkan:

| Window | Seluruh row | Subset nonoverlap | Low / Medium / High pada subset |
|---|---:|---:|---|
| 15 detik | 3.011 | 1.144 | 277 / 420 / 447 |
| 25 detik | 2.711 | 651 | 157 / 237 / 257 |
| 45 detik | 2.118 | 321 | 81 / 110 / 130 |

Subset tersebut bukan estimasi jumlah sampel independen: korelasi fisiologis masih dapat bertahan meskipun interval tidak overlap. Ia menunjukkan bahwa ribuan row dari step 5 detik tidak setara dengan ribuan potongan rekaman baru.

Batch-02 tetap perlu menilai window berdasarkan coverage, kualitas HRV, keterwakilan fase, serta hasil pada endpoint yang sama. Jangan memilih 45 detik hanya dari satu tabel skor atau memperbanyak row dengan step lebih rapat lalu menyebutnya pengambilan data tambahan.

Sumber: [jumlah nonoverlap](diagnostics/results/nonoverlap_counts.csv), [komposisi Low](diagnostics/results/low_phase_errors.csv), dan [perbandingan endpoint yang sama](paired_endpoint_metrics.csv).

## 10. Apa yang tidak ditemukan atau belum dapat dipastikan

- Tidak ditemukan label NaN/di luar Low–Medium–High pada input training resmi, atau pemetaan difficulty lama yang menggantikan cognitive_load_label v2.
- Tidak ditemukan penggunaan langsung identitas atau difficulty sebagai kolom predictor. Temuan konteks history adalah jalur tidak langsung melalui transformasi, bukan bukti kolom ID terselip dalam matriks fitur.
- Tidak ada bukti perubahan 14 window HRV merupakan kerusakan acak. Semua 14 cocok dengan endpoint yang sudah didokumentasikan sebagai live_timer_before_endpoint_packet pada replay. Fitur eye dan label seluruh 2.294 window bersama tetap sama. [Pencocokan per window](diagnostics/results/rr_replay_matches.csv) dan [laporan replay](../../../analysis_sessions/dataset_fusion/output/v2_session_phase/replay_validation.json) tersedia.
- Belum dapat ditentukan berapa persen kesalahan disebabkan imbalance, gaze, variasi kondisi, atau definisi target. Ablation/probe hanya mengisolasi kontras yang diuji, bukan seluruh sebab sekaligus.
- Belum ada data tambahan Low dari pengambilan baru dalam hasil ini. Keuntungan dari peserta/sesi baru belum diuji.
- Data ini belum mengisolasi pengaruh difficulty dari urutan sesi, kelelahan, atau perubahan kondisi perekaman. Menyebut ketiganya sebagai penyebab pasti tanpa rekaman pembanding tidak dibenarkan.
- Target tetap pemetaan protokol v2 yang disepakati. Pemeriksaan ini tidak mengubah fase rating/feedback/rest menjadi target lain dan tidak mengklaim telah memverifikasi CL fisiologis setiap window dengan pengukuran rujukan independen.

## 11. Prioritas batch-02

Batch-02 tetap mengevaluasi dataset terbaru secara mandiri. Tidak perlu menjadikannya kewajiban mengulang atau mengalahkan skor arsip batch-00/batch-01. Temuan diagnosis dipakai untuk menyusun kontrol dan menghindari masalah yang sama.

| Prioritas | Pekerjaan | Hasil yang harus tersedia sebelum melanjutkan |
|---|---|---|
| 1 | Terima rekaman Low baru, satukan identitas lintas sesi, periksa baseline/gaze, bangun versi dataset yang dibekukan | Provenance, distribusi label dan fase, coverage sensor, durasi unik, admission, hash input; tidak hanya target jumlah row |
| 2 | Pisahkan aliran fitur sensor dari mask target dan evaluasi; buat history tanpa puzzle/fase | Uji bahwa perubahan metadata/label mask tidak mengubah fitur pada aliran sensor yang sama; reset hanya dari kontrak stream dan validitas sensor |
| 3 | Bekukan fold kelompok dan baseline sederhana pada dataset terbaru | Kontrol Dummy, Logistic clean, Logistic tanpa rasio, serta XGBoost berkapasitas terbatas; preprocessing training-only; hasil ketiga kelas |
| 4 | Audit baseline dan uji representasi sensor | Kontrol tanpa rasio, fitur relatif/dinamis yang tersedia sebelum prediksi, ablation EAR/blink/iris/head pose/gaze/state; fallback baseline invalid jelas |
| 5 | Bandingkan window 15/25/45 dan history kausal pada kontrak yang sama | Skor keseluruhan dan endpoint yang sama, nonoverlap sensitivity, cakupan kelompok Low dan kesalahan Medium–High |
| 6 | Uji balancing setelah baseline representasi tersedia | Tanpa balancing, weight, random oversampling, SMOTE di training saja; ukur precision/recall/F1 Low dan perubahan dua kelas lain |
| 7 | Seleksi bank model kecil lalu evaluasi nested dan refit | Kandidat dipilih dari inner validation, bukan outer; kurva training/validation dan ketidakpastian berbasis kelompok; input deployment benar-benar sensor-only |

Rencana [batch-02](../../batch-02/PLAN.md) sudah mencakup sebagian besar arahnya, tetapi perlu diperbarui sebelum implementasi dalam empat hal:

1. Kaitkan konteks diagnosis dengan arsip old-dataset dan diagnosis baru ini; beberapa tautan masih menunjuk lokasi sebelum pemindahan folder.
2. Tambahkan uji invariant metadata dan history-sebelum-mask, bukan hanya mengganti reset soal menjadi reset sesi/gap pada CSV yang sudah difilter.
3. Jadikan ablation rasio baseline, bank sederhana, dan gap training/test sebagai kontrol utama, bukan opsi tambahan setelah pencarian besar.
4. Pertahankan pemetaan fase Low, tetapi laporkan recall Low pengerjaan saja, campuran, dan nonpengerjaan secara terpisah; sertakan pemeriksaan Medium–High agar keberhasilan tidak hanya dilihat dari Low yang bertambah.

Tidak ada perubahan PLAN.md atau pipeline produksi pada pekerjaan diagnosis ini. Rincian implementasi dan training batch-02 dibuat setelah rekaman baru siap. Prioritasnya bukan menambah jumlah eksperimen, melainkan menguji fitur yang dapat digunakan tanpa konteks CodeArrange dan memastikan perbaikannya berlaku pada rekaman yang tidak dipakai untuk fitting.

## Reproduksi audit

Dari root repository, gunakan environment dependency batch-01 baru:

```powershell
& "models/.venv-batch01-replay/Scripts/python.exe" -m unittest discover -s models/batch-01/new-dataset/diagnostics -p "test_*.py"
& "models/.venv-batch01-replay/Scripts/python.exe" models/batch-01/new-dataset/diagnostics/diagnose.py
& "models/.venv-batch01-replay/Scripts/python.exe" models/batch-01/new-dataset/diagnostics/supplement.py
& "models/.venv-batch01-replay/Scripts/python.exe" models/batch-01/new-dataset/diagnostics/validate_evidence.py
```

Environment lokal bukan artefak yang harus dipush. Dependency resmi tersedia di [requirements.txt](requirements.txt); seluruh CSV diagnostik dan dua grafik disimpan agar temuan dapat dibaca tanpa menjalankan ulang fit. Menjalankan audit memerlukan model lokal tersimpan; file model besar tetap mengikuti aturan ignore proyek. Audit tidak mengunduh atau melatih ulang seluruh 36 eksperimen resmi.

Validasi akhir: 26 test runtime resmi dan empat test diagnostik lulus; critical lint lulus. Validator menghitung ulang 14 kontras prediksi, memeriksa tautan lokal, dan memverifikasi kembali integritas 36 run resmi serta input yang dilindungi.
