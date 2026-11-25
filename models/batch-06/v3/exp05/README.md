# Exp05 · evaluasi dua target per episode

**Belum ada model yang layak dipilih untuk penggunaan pada peserta baru.** Pada v3, kandidat terbaik exp03 benar pada 4 dari 18 episode Test. Exp05 mengubah unit fitting menjadi episode dan masih hanya mengenali 5 dari 11 episode validation. Pada v4, kandidat sensor di exp04 kalah dari tebakan median untuk jarak rating. Regresi median exp05 menyamai tebakan itu pada setiap episode.

## Pemeriksaan run Colab exp03 dan exp04

Hash delapan CSV pada kedua manifest cocok dengan paket lokal. Accuracy dan Macro-F1 dihitung ulang dari prediksi setiap kandidat pada tingkat window dan episode, beserta MAE dan proporsi dalam ±1 tingkat untuk v4. ID, label, dan peserta pada prediksi cocok dengan dataset. Sel pemodelan pada kedua notebook kiriman cocok dengan notebook repo. Notebook kiriman v3 mencetak ID run akhir `v3_20261124T102005Z_9365a1d3`, sedangkan ZIP yang diterima ber-ID `v3_20261124T101649Z_a3b8945d`. Skor episode yang ditampilkan cocok. Angka di bawah berasal dari ZIP. ID notebook dan ZIP v4 cocok.

| Evaluasi pada episode | Model | Accuracy tepat | Macro-F1 | MAE rating | Dalam ±1 |
|---|---|---:|---:|---:|---:|
| v3 validation, 11 episode | Extra Trees core | 45,45% | 0,324 | — | — |
| v3 Test, 18 episode | Extra Trees core | 22,22% | 0,178 | — | — |
| v3 Test, 18 episode | Dummy High | 11,11% | 0,067 | — | — |
| v4 LOPO, 126 episode | Extra Trees core | 15,08% | 0,091 | 3,183 | 29,37% |
| v4 LOPO, 126 episode | Logistic ordinal | 5,56% | 0,062 | 2,675 | 30,16% |
| v4 LOPO, 126 episode | Dummy modus | 24,60% | 0,040 | 2,992 | 35,71% |
| v4 LOPO, 126 episode | Dummy median | 15,87% | 0,032 | 2,317 | 38,89% |

**V3.** Extra Trees mengenali 3/12 Low, 0/4 Medium, dan 1/2 High pada tiga peserta Test. Model memprediksi High pada 13 dari 18 episode. Pada tingkat window, accuracy Test 42,86% bahkan di bawah Dummy High 51,52%, meski Macro-F1 lebih besar (0,325 vs 0,227). Dua episode High menyumbang 119 dari 231 window Test. Itulah sebabnya accuracy window dan accuracy episode memberi gambaran yang berbeda. [ZIP Colab exp03](../outputs/exp03/v3_20261124T101649Z_a3b8945d.zip)

**V4.** Extra Trees mempunyai exact accuracy tertinggi di antara model sensor. Model mengenali 19 dari 126 episode, sementara tebakan modus fitting benar pada 31 episode. Enam rating memiliki recall nol pada Extra Trees (2, 4, 6, 7, 8, dan 9). Logistic ordinal mempunyai MAE terendah di antara kandidat sensor, tetapi mengalahkan MAE Dummy median hanya pada 4 dari 17 peserta. Lebih banyak kelas yang terprediksi belum menghasilkan ketepatan atau jarak rating yang memadai. [ZIP Colab exp04](../outputs/exp04/v4_20261124T101813Z_7392b217.zip)

## Apa yang dicoba pada exp05

Satu baris fitting mewakili satu episode. Kedua representasi menggunakan sepuluh fitur sensor yang sama. Varian `first` mengambil window layak pertama dalam episode menurut timestamp. Prediksi ini bisa dibuat begitu window tersebut tersedia dan awal episode diketahui. Varian `median` mengambil median tiap fitur dari seluruh window layak dan baru dapat digunakan setelah episode selesai. Model hanya menerima nilai sensor. Jumlah window, durasi, ID peserta, soal, kesalahan, dan label kasus lain dikeluarkan dari fitur.

- **V3** memakai 37 episode Train dan 11 episode validation resmi. Bobot per episode adalah jumlah bobot window Train resmi, dengan rata-rata dinormalisasi menjadi satu. Kedua representasi memakai Logistic Regression dengan C = 0,1 dan StandardScaler berbobot dari Train. Kontrol modus dihitung dari frekuensi episode Train. Test v3 sudah dibaca pada exp03. Perbandingan lanjutan ini memakai validation saja.
- **V4** memakai satu suara per episode dan 17 fold LOPO yang sama dengan exp04. Logistic Regression nominal menargetkan kelas tepat. Regresi kuantil 0,5 dengan alpha 0,10 menargetkan median kondisional untuk mengurangi galat absolut. Estimasi kontinu disimpan, lalu dibatasi ke 1–10 dan dibulatkan ke tingkat terdekat (setengah dibulatkan ke atas). Modus dan median bawah episode fitting menjadi baseline. StandardScaler dan model selalu dipasang pada peserta fitting tiap fold.

## Hasil run lokal exp05

Sel komputasi [notebook gabungan](EyeHeartCognitiveLoad.ipynb) dijalankan pada CSV repo. Runtime lokal menggunakan scikit-learn 1.8.0 dan dua run Colab awal menggunakan 1.6.1. Pembandingan memakai ID episode dan peserta yang sama. ZIP diperiksa, termasuk keutuhan arsip, pemisahan peserta, cakupan prediksi serta penghitungan ulang metrik. [ZIP hasil lokal](outputs/episodes_20261124T103258Z_6f309dfe.zip)

| Kasus | Model | Benar | Accuracy | Macro-F1 | MAE | Dalam ±1 |
|---|---|---:|---:|---:|---:|---:|
| v3 validation | Dummy modus | 4/11 | 36,36% | 0,178 | — | — |
| v3 validation | Logistic, window pertama | 5/11 | 45,45% | 0,357 | — | — |
| v3 validation | Logistic, median episode | 5/11 | 45,45% | 0,357 | — | — |
| v4 LOPO | Dummy modus | 31/126 | 24,60% | 0,040 | 2,992 | 35,71% |
| v4 LOPO | Dummy median | 20/126 | 15,87% | 0,032 | 2,317 | 38,89% |
| v4 LOPO | Logistic, window pertama | 15/126 | 11,90% | 0,056 | 3,302 | 23,02% |
| v4 LOPO | Logistic, median episode | 17/126 | 13,49% | 0,067 | 3,056 | 29,37% |
| v4 LOPO | Regresi kuantil, window pertama | 20/126 | 15,87% | 0,032 | 2,317 | 38,89% |
| v4 LOPO | Regresi kuantil, median episode | 20/126 | 15,87% | 0,032 | 2,317 | 38,89% |

Pada **v3**, kedua representasi memberi prediksi identik di 11 episode validation. Model baru mengenali 4/4 Low, 1/3 Medium, dan 0/4 High. Dibanding Extra Trees exp03 pada episode yang sama, empat kesalahan lama terkoreksi dan empat jawaban lama yang benar menjadi salah. Kenaikan Macro-F1 sebesar 0,033 pada tiga peserta validation belum menjadi alasan mengganti model.

Pada **v4**, hasil pembulatan kedua regresi kuantil identik dengan Dummy median pada 126/126 episode. Model nominal median episode hanya tepat pada 17 episode. Dataset ini belum memperlihatkan keuntungan dari dua representasi tersebut untuk exact accuracy atau MAE terhadap baseline masing-masing. Angka ini berlaku untuk resep yang dicoba pada cohort ini.

Untuk pengambilan data berikutnya, prioritas v3 adalah menambah **peserta dan episode independen** dengan keterwakilan ketiga kelas pada evaluasi baru. Untuk v4, pertahankan rating asli 1–10, tambah dukungan rating yang langka (rating 8 hanya tiga episode dan rating 9 empat), dan uji pada peserta baru dengan modus dan median sebagai pembanding. Jika sistem ditujukan untuk prediksi langsung, tentukan juga bagaimana awal episode tersedia dan nilai prediksi dari window pertama secara terpisah.

## Menjalankan kedua kasus di Google Colab

1. Buka [EyeHeartCognitiveLoad.ipynb](EyeHeartCognitiveLoad.ipynb) pada runtime CPU. Kedua kasus berjalan dari satu notebook.
2. Taruh folder `v3_behavioral_proxy/` dan `v4_self_reported_rating/` langsung di `MyDrive/Penelitian-Adaptive-IDE/Dataset/`. Sel konfigurasi menggunakan `DATASET_ROOT = PROJECT_ROOT / 'Dataset'`.
3. Pilih **Run all**. Ambil satu ZIP dari `MyDrive/Penelitian-Adaptive-IDE/models/batch-06/exp05/outputs/`. Path ZIP juga dicetak di sel terakhir. Arsip memuat prediksi dan fold per kasus, serta metrik gabungan, matriks konfusi, metrik peserta, dan manifest. Kembalikan ZIP ke folder `exp05/outputs/` di repo.
