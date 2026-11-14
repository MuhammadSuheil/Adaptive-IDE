# Batch 00: Eksperimen Awal 

## Ringkasan

Batch ini menyimpan notebook awal yang dibuat oleh Pak Hadi , yang sebelumnya berada di `models/exp0`. Pak Hadi mencoba Random Forest, XGBoost, pembobotan kelas, fitur rolling, SMOTE, serta novelty detection untuk data eye tracking dan HRV.

Output tersimpan menunjukkan accuracy sekitar 54% untuk salah satu konfigurasi tiga kelas dengan validasi blok waktu. Low masih sulit dikenali. Percobaan dua kelas Medium/High mencapai sekitar 61%, tetapi tidak mengevaluasi kemampuan mengenali Low.

Angka di bawah berasal dari output yang sudah tersimpan di [notebook asli](EyeHeartCognitiveLoad.ipynb), bukan hasil menjalankan ulang notebook. Notebook membulatkan skor ke dua angka desimal. Tidak ditemukan folder hasil CSV/model terpisah untuk batch ini pada saat penataan arsip.

## Data dan target

Tujuan pemodelan adalah klasifikasi CL per window dari fitur eye tracking dan HRV, bukan pengenalan peserta atau soal. Identitas dan konteks CodeArrange tetap berguna sebagai metadata audit/evaluasi. Pelurusan protokol serta peran kolom terdapat pada [konteks penelitian](../README.md#konteks-penelitian-dan-peran-kolom).

- Dataset yang dirujuk: `good_acceptable.csv`.
- Output pemuatan mencatat 2.294 window.
- Label mengikuti kesulitan tugas: `easy` menjadi Low, `medium` menjadi Medium, dan `hard` menjadi High.
- Distribusi tersimpan: Low 249, Medium 968, High 1.077.
- Kelompok fitur meliputi RMSSD/SDNN, rasio baseline HRV, blink/EAR, iris, head pose, gaze, dan eye state. Subset fitur berubah antarpercobaan.

Label tersebut merupakan proxy cognitive load dari kesulitan tugas. Notebook tidak menyimpan hash dataset atau manifest evaluasi seperti batch berikutnya, sehingga kesamaan nama file dan jumlah baris belum membuktikan kesamaan versi data.

Batas waktu set pada protokol adalah easy 5 menit, medium 12 menit, dan hard 15 menit. Sebagian peserta sengaja tidak diberi easy karena keterbatasan waktu pelaksanaan. Karena itu, durasi berbeda dan sebagian kelas yang tidak tersedia tidak otomatis menunjukkan galat perekaman. Angka kelas yang tidak seimbang tetap menjadi karakteristik data yang perlu diperhitungkan, bukan alasan mengubah hasil arsip atau menghapus peserta tersebut.

## Apa yang dilakukan

1. Membaca CSV, membuat label, dan memeriksa distribusi kelas.
2. Melatih Random Forest dan XGBoost menggunakan `TimeSeriesSplit` lima fold berdasarkan urutan baris.
3. Menggunakan `GroupKFold` lima fold berdasarkan blok waktu 120 detik. Pembobotan kelas dihitung dari training fold.
4. Mencoba gabungan EAR kiri/kanan, rolling tiga window, dan delta RMSSD.
5. Mencoba klasifikasi Medium versus High dengan mengeluarkan Low.
6. Mencoba SMOTE pada training fold setelah imputasi median.
7. Mencoba Isolation Forest dan One-Class SVM untuk membedakan Low dari gabungan Medium/High, yang disebut Elevated.

Blok 120 detik adalah unit pemisahan evaluasi. Angka itu bukan panjang window fisiologis yang digunakan untuk menghitung ulang HRV dari RR mentah. Rolling tiga window juga bekerja pada fitur yang sudah tersedia.

## Hasil klasifikasi tiga kelas

| Percobaan | Validasi | Accuracy tersimpan | Macro-F1 tersimpan |
|---|---|---:|---:|
| Random Forest awal, dengan class weight | TimeSeriesSplit, lima fold | Rata-rata sekitar 31% | Tidak diringkas untuk gabungan fold |
| XGBoost awal, penanganan NaN bawaan | TimeSeriesSplit, lima fold | Rata-rata sekitar 34% | Tidak diringkas untuk gabungan fold |
| XGBoost dengan sample weight | GroupKFold blok waktu | Sekitar 54% | 0,46 |
| Random Forest dengan EAR gabungan dan rolling | GroupKFold blok waktu | Sekitar 48% | 0,41 |
| XGBoost dengan rolling dan sampling baris/fitur | GroupKFold blok waktu | Sekitar 53% | 0,45 |
| XGBoost dengan SMOTE | GroupKFold blok waktu | Sekitar 53% | 0,45 |

Untuk dua percobaan TimeSeriesSplit awal, classification report hanya merangkum fold terakhir. Angkanya tidak boleh diperlakukan sebagai macro-F1 rata-rata lima fold. Pada percobaan blok waktu, classification report menggabungkan prediksi test kelima fold.

Pada XGBoost dengan sample weight yang mencatat sekitar 54% accuracy, precision Low sekitar 0,15, recall Low 0,22, dan F1 Low 0,18. Jadi skor keseluruhan tersebut belum berarti ketiga kelas dikenali dengan baik. SMOTE juga belum memberikan peningkatan yang jelas pada skor yang dicetak.

## Hasil dua kelas dan novelty detection

| Percobaan | Target dan data uji | Accuracy tersimpan | Macro-F1 tersimpan |
|---|---|---:|---:|
| Binary XGBoost | Medium versus High; 2.045 window | Sekitar 61% | 0,61 |
| Isolation Forest, contamination 0,05 | Low versus Elevated; 658 window test | Sekitar 50% | 0,36 |
| Isolation Forest, contamination 0,25 dan scaling | Low versus Elevated; test yang sama | Sekitar 39% | 0,38 |
| One-Class SVM, nu 0,25 dan scaling | Low versus Elevated; test yang sama | Sekitar 43% | 0,42 |

Novelty detection menggunakan 80% awal data Elevated untuk training. Data test berisi 20% sisanya dan seluruh 249 window Low. Recall Low yang dicetak masing-masing sekitar 5%, 31%, dan 35% pada tiga percobaan novelty tersebut.

Percobaan Medium/High dan novelty detection berjalan terpisah. Notebook tidak menyajikan evaluasi end-to-end yang menggabungkan keduanya menjadi satu sistem prediksi Low/Medium/High. Karena target dan data uji berbeda, accuracy 61% tidak dapat dibandingkan langsung dengan accuracy klasifikasi tiga kelas.

## Batas interpretasi

- Pemisahan memakai urutan baris atau blok waktu, bukan pemisahan seluruh rekaman berdasarkan peserta. Peserta yang sama tidak dijamin terpisah antara training dan test.
- Pada Random Forest awal, imputasi median di-fit sebelum pemisahan fold. Ini memakai informasi distribusi fitur di luar training fold. Beberapa percobaan berikutnya sudah melakukan imputasi di dalam training fold.
- Kode rolling memakai urutan waktu global tanpa reset eksplisit per peserta, sesi, atau soal. Hal ini perlu diperiksa sebelum menafsirkan fitur tersebut sebagai riwayat peserta yang sama.
- Pemisahan blok waktu tidak disertai purge/gap eksplisit untuk window yang overlap pada batas training/test.
- SMOTE berjalan pada fitur hasil imputasi tanpa scaling sebelumnya. Scaling baru muncul pada bagian novelty detection.
- Output cell tersimpan merupakan catatan eksekusi lama. Sebagian path pada source dan output berbeda; arsip ini tidak membuktikan bahwa seluruh cell pernah dijalankan ulang berurutan dengan source dan environment yang sekarang.


## Kesimpulan

Eksperimen awal menunjukkan kesulitan mengenali Low dan mendorong pemeriksaan ulang evaluasi, preprocessing, fitur temporal, serta balancing. Belum ada hasil yang membuktikan klasifikasi tiga kelas yang kuat untuk peserta baru.

Rangkaian lanjutan terdapat di [batch-01](../batch-01/SUMMARY.md). Batch tersebut memakai LOPO, sehingga skor yang lebih rendah tidak dapat langsung disimpulkan sebagai penurunan kualitas model dibanding skor validasi blok waktu di sini.
