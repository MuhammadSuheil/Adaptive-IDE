# Exp06 · apakah rekaman yang sudah ada cukup untuk mendekati 70%?

## Verifikasi ekspor Colab exp06

[ZIP Colab 105533Z](outputs/history_20261124T105533Z_59b4e276.zip) memuat sembilan berkas. Pemeriksaan integritas ZIP berhasil. Jumlah benar, accuracy, Macro-F1, MAE dan proporsi dalam ±1 cocok dengan penghitungan ulang dari prediksi episode untuk semua resep yang berlaku.

Delapan CSV v3/v4 sama dengan run lokal di bawah. Kedua CSV v2 pada Drive cocok dengan revisi Git `d9abee16d7a7`. Riwayat `good_acceptable.csv` pada revisi itu mempunyai 5.253 window, sedangkan salinan repo saat ini mempunyai 4.299. Jumlah episode v4 dengan sedikitnya tiga window riwayat berubah dari 108 ke 110. Terhadap run lokal sebelumnya, dua prediksi v3 dan 88 prediksi v4 berbeda, semuanya pada resep yang memakai riwayat relatif. Menjalankan ulang notebook lokal dengan kedua CSV v2 yang persis sesuai hash Colab mereproduksi **99/99 prediksi v3 dan 1.764/1.764 prediksi v4**. Angka tersebut menghitung semua pasangan model dan episode.

| Endpoint Colab | Hasil | Implikasi |
|---|---|---|
| V3 validation, SVC core + head relatif | 6/11 benar, Macro-F1 0,522. Recall Low 1/4, Medium 2/3, High 3/4 | Eye-only juga benar 6/11, tetapi hanya dua episode dikenali dengan benar oleh keduanya. Eye-only melewatkan ketiga Medium. |
| V4 LOPO, exact accuracy | SVC core mentah 28/126 (22,22%), modus Train 31/126 (24,60%) | Modus masih mengungguli setiap model sensor untuk rating tepat. |
| V4 LOPO, MAE | SVR relatif core 2,325, SVR relatif core + head 2,333, median Train 2,317 | Keuntungan MAE dari run lokal tidak bertahan pada riwayat Drive ini. SVR core + head memberi rating 3 atau 4 pada 103/126 episode. |

Untuk perbandingan berikut, bekukan hash v2 bersama v3/v4. Untuk memakai paket repo saat ini, salin ulang **kedua** CSV v2 di bagian reproduksi. Perbedaan kesalahan antar-model v3 layak diuji melalui prediksi out-of-fold berbasis peserta di dalam Train dengan rekonstruksi referensi dan label per fold. Memilih model per episode setelah melihat label validation akan melebihkan estimasi kinerja gabungan.

## Hasil dengan riwayat v2 dari repo saat ini

Pada klasifikasi peserta baru, **belum ada bukti 70% accuracy per episode** untuk kedua target. Seluruh angka di sini memakai label asli. Pada v3, kontrol eye mencapai 6/11 benar pada validation tetapi melewatkan semua Medium. Pada v4, klasifikasi sensor terbaik di exp06 benar pada 28/126 episode, sedangkan modus rating Train benar pada 31/126. Perbaikan yang tampak pada v4 adalah berkurangnya rata-rata jarak kesalahan rating.

## Apakah run terbaru menurun?

Semua **33 prediksi kelas v3** (3 model × 11 episode) dan **756 prediksi kelas v4** (6 model × 126 episode) pada ekspor Colab terbaru identik dengan exp05 lokal jika dipasangkan menurut model, fold, episode, peserta, dan label aktual. Hash delapan CSV pada manifest juga cocok dengan folder dataset Batch 06 dan bundle sumber di dataset_fusion. Kedua run ini mereproduksi hasil yang sama. [ZIP Colab yang diperiksa](../exp05/outputs/episodes_20261124T103659Z_f4d1eb89.zip) · [Run pembanding exp05](../exp05/README.md)

Pada v4, angka **17** menyatakan jumlah peserta yang ditahan bergantian dalam LOPO. Setiap fold melatih model pada 16 orang lalu menilai semua episode satu orang. Untuk Logistic Regression atas median fitur episode, **14 dari 17 orang** memperoleh exact accuracy di bawah 20%. Jumlah episode per orang berkisar 2–13, sehingga satu jawaban salah mengubah persentase per orang sebesar 7,7–50 poin. Rata-rata accuracy dengan bobot sama per orang adalah 10,13%. Jika setiap episode diberi bobot sama, hasilnya 17/126 atau 13,49%. Semua kelas 1–10 tersedia di fitting pada setiap fold. Pembanding modus mencapai 24,60% secara keseluruhan.

## Telaah paket terbaru

| Dari 358 episode tercatat | V3 · proxy perilaku | V4 · rating kesulitan |
|---|---:|---:|
| Mempunyai target | 82 | 164 |
| Mempunyai target dan minimal satu window fusion layak | 66 | 126 |
| Berlabel, hanya memiliki window berstatus poor di assignments | 8 | 24 |
| Berlabel, tanpa window di assignments | 8 | 14 |

V3 menyisakan 163 episode dengan referensi kurang, 112 dengan bukti perilaku tidak cukup atau tidak sejalan, dan satu dengan jejak peristiwa tidak valid. Pada v4 ada 194 episode tanpa rating. Periksa angka ini sebelum menganggap 358 episode sebagai contoh training berlabel. Pada v4, 126 episode mencakup 31 contoh rating 1, tiga contoh rating 8, dan empat contoh rating 9. Skor 70% pada cohort ini memerlukan **89/126** rating tepat. Pada 11 episode validation v3 diperlukan **8/11**.

Untuk v4, saya menghitung proporsi variasi median fitur episode yang berada di antara rata-rata kelompok, yaitu jumlah kuadrat antarkelompok dibagi jumlah kuadrat total. Untuk EAR kiri, 93% variasi berada di antara peserta dan 9% di antara rating. Untuk RMSSD angkanya 80% dan 7%. Kedua pengelompokan dihitung sendiri-sendiri dan tidak memberi estimasi kausal atau batas atas accuracy. Hasil ini menjadi alasan untuk menguji referensi dalam sesi orang yang sama.

## Eksperimen memakai data lama

Kedua kasus menggunakan satu baris per episode. Evaluasi v3 memakai 37 episode Train dan 11 episode validation dari split resmi. Evaluasi v4 memakai 17 fold LOPO dan 126 episode asli. Prediksi dihasilkan sesudah episode selesai. ID, soal, jumlah window, waktu pengerjaan, kesalahan manual, dan rating peserta yang ditahan tidak menjadi fitur.

1. **Riwayat sensor sebelum episode.** Bandingkan sepuluh fitur core dengan core ditambah empat fitur head pose. Untuk masing-masing, uji median episode mentah dan selisih terhadap median riwayat sesi. Riwayat berasal dari maksimal 12 window sensor layak yang tidak saling tumpang tindih dan sudah berakhir saat episode dimulai. Pencarian memakai peserta dan sesi yang sama. Minimal tiga window tersedia untuk 11/11 episode validation v3 dan 108/126 episode v4. Sisanya memakai median episode fitting sebagai referensi. Pembanding memakai episode evaluasi yang sama. Model klasifikasi memakai SVC RBF dengan C=1. Model jarak rating memakai SVR RBF dengan C=2 dan epsilon=0,5, lalu membulatkan rating ke 1–10. StandardScaler memakai fitting tiap fold.
2. **Pulihkan satu modalitas dari rekaman yang sudah ada.** Pada episode yang semula tanpa window fusion layak, ambil fitur eye lengkap bila seluruh alasan gagal ada pada HRV. Ambil HRV lengkap bila satu-satunya alasan gagal adalah rasio ketersediaan iris. Cocokkan window, peserta, sesi, bukti label, dan split. Ini memberi tambahan **7 episode eye dan 1 HRV pada Train v3**, serta **19 episode eye dan 5 HRV** yang bisa masuk fitting v4 saat pesertanya tidak ditahan. Uji SVC per modalitas dengan dan tanpa tambahan episode. Prediksi dinilai pada cohort asli yang identik. Bobot v3 dihitung kembali menurut kelas, peserta dalam kelas, dan episode dalam peserta. Pada kontrol asli hasilnya diverifikasi terhadap bobot yang diterbitkan tim. V4 memberi satu suara per episode.

## Hasil run lokal

| V3 · validation, 11 episode | Benar | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Modus Train | 4 | 36,36% | 0,178 |
| Exp05 Logistic median episode | 5 | 45,45% | 0,357 |
| SVC core + head relatif | 5 | 45,45% | 0,429 |
| Eye asli → diperluas | 6 → 4 | 54,55% → 36,36% | 0,432 → 0,178 |
| HRV asli → diperluas | 4 → 5 | 36,36% → 45,45% | 0,262 → 0,444 |

Eye asli mengenali 3/4 Low, 0/3 Medium, dan 3/4 High. Core + head relatif mengenali 1/4 Low, 1/3 Medium, dan 3/4 High. Perubahan satu episode saja menggeser accuracy 9,09 poin. Validation ini telah dilihat di beberapa eksperimen. Test v3 tidak dipakai untuk memilih model di exp06.

| V4 · LOPO, 126 episode | Tepat | Accuracy | MAE dalam tingkat rating | Dalam ±1 |
|---|---:|---:|---:|---:|
| Modus Train | 31 | 24,60% | 2,992 | 35,71% |
| Median Train | 20 | 15,87% | 2,317 | 38,89% |
| SVC core mentah | 28 | 22,22% | 2,659 | 38,10% |
| SVC core + head relatif | 20 | 15,87% | 3,119 | 26,98% |
| SVR core + head relatif | 15 | 11,90% | **2,238** | 40,48% |
| Eye asli → diperluas | 21 → 14 | 16,67% → 11,11% | 2,706 → 2,849 | 37,30% → 31,75% |
| HRV asli → diperluas | 25 → 25 | 19,84% → 19,84% | 2,651 → 2,524 | 40,48% → 40,48% |

Selisih MAE SVR terhadap median adalah 0,079 tingkat. Sepuluh peserta membaik, dua seri, dan lima memburuk. Interval persentil deskriptif untuk perbaikan MAE adalah −0,12 sampai +0,30 tingkat (10.000 bootstrap berpasangan dengan resampling peserta, seed 42, MAE dihitung kembali per episode). Keuntungan ini masih belum pasti. [Semua metrik](outputs/history_20261124T104956Z_d9e9194c/metrics.csv) · [ZIP prediksi, referensi waktu, dan window yang dipulihkan](outputs/history_20261124T104956Z_d9e9194c.zip)

## Prioritas pengembangan tanpa perekaman ulang

- **V3.** HRV yang diperluas layak diperiksa ketahanannya karena Macro-F1 naik, tetapi kenaikan accuracy hanya satu episode. Tahap berikut yang paling informatif adalah evaluasi grup di dalam Train dengan membangun ulang referensi, label, dan bobot dari inner-Train pada tiap fold sesuai kontrak tim. Ini memungkinkan pemeriksaan kestabilan recall Medium dan High pada lebih banyak peserta tanpa memilih resep dari tiga orang validation yang sama.
- **V4.** Pertahankan exact accuracy, MAE, dan baseline modus/median berdampingan. Bila keputusan UI dapat memakai urutan kesulitan, kandidat SVR bisa diteliti untuk galat jarak dengan kriteria manfaat yang ditentukan sebelum perbandingan berikutnya. Untuk memperluas data berlabel, audit alignment dan kualitas stream mentah pada **14 episode yang punya rating tetapi tanpa assignment**. Delapan berdurasi aktif kurang dari 15 detik. Prioritaskan enam sisanya untuk memeriksa cakupan sensor dan kontinuitas interval. Rekonstruksi hanya window yang lolos syarat kualitas dan memang berada pada episode itu. Rating yang tidak pernah diberikan tidak dapat dipulihkan dari sensor sebagai ground truth.
- **Target 70%.** Catat kriteria sebagai accuracy tepat per episode pada peserta yang ditahan, disertai Macro-F1 dan recall per kelas. Mengizinkan selisih ±1 atau mengelompokkan rating menjadi tiga tingkat mengubah pertanyaannya. Pada v3, High sendiri sudah mencakup 69,24% window keseluruhan. Split acak atas window yang overlap juga dapat memasukkan rekaman dari episode yang sama di kedua sisi evaluasi. Kedua hal itu dapat memberi angka yang menyesatkan untuk penggunaan pada peserta baru.

## Reproduksi di Google Colab

Buka [satu notebook untuk kedua kasus](EyeHeartCognitiveLoad.ipynb) dan pilih **Run all** pada runtime CPU. Sel pertama memakai `DATASET_ROOT = PROJECT_ROOT / 'Dataset'`. Taruh kedua paket v3 dan v4 langsung di folder itu, bersama `v2_session_phase/window_15s_step_5s/good_acceptable.csv` dan `all_data.csv` dari [bundle v2 yang sudah tersedia](../../../analysis_sessions/dataset_fusion/output/v2_session_phase/window_15s_step_5s). Kedua CSV v2 memasok riwayat sensor dan alasan kegagalan mutu. Sel terakhir membuat satu ZIP di `models/batch-06/exp06/outputs/`. Run lokal memakai sel notebook yang sama, dengan root yang mengarah ke bundle sumber di repo.
