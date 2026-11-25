# Eksperimen relabeling Batch 06

| Notebook | Pertanyaan utama | Unit evaluasi utama |
|---|---|---|
| [Exp03 · proxy perilaku v3](../exp03/EyeHeartCognitiveLoad.ipynb) | Apakah kombinasi HRV dan eye membantu mengenali Low, Medium, High pada peserta lain? | Macro-F1 tiga kelas per episode pada validation, lalu Test untuk model terkunci |
| [Exp04 · rating peserta v4](../exp04/EyeHeartCognitiveLoad.ipynb) | Apakah memodelkan urutan rating mengurangi jarak antara prediksi dan rating peserta? | MAE per episode dari prediksi leave-one-participant-out (LOPO) |

## Menjalankan di Colab dan mengembalikan hasil

1. Salin folder [dataset](../dataset/DATASET_README.md) beserta susunan subfoldernya ke `MyDrive/Penelitian-Adaptive-IDE/models/batch-06/dataset/`. Tiap notebook hanya memerlukan paket CSV untuk targetnya sendiri.
2. Buka notebook pilihan di Google Colab dengan runtime Python CPU. Jika lokasi Drive berbeda, sesuaikan `DATASET_ROOT` dan `OUTPUT_ROOT` di sel pertama. Pilih **Run all**. Kedua notebook mandiri dan dapat berjalan paralel di runtime terpisah.
3. Ambil ZIP dari path yang dicetak sel terakhir. Secara default, ZIP ada di `MyDrive/Penelitian-Adaptive-IDE/models/batch-06/outputs/exp03/` atau `exp04/`. Aktifkan `DOWNLOAD_ZIP` sebelum menjalankan sel terakhir jika ingin mengunduhnya lewat browser. Setiap run membuat direktori dan arsip baru.
4. Salin ZIP ke `models/batch-06/outputs/exp03/` atau `exp04/` di repo sesuai nomor eksperimen. ZIP sudah memuat prediksi, metrik, dan provenance yang diperlukan untuk analisis.

## Dasar pemilihan eksperimen

Pada batch-05, Extra Trees dengan normalisasi session-robust mendapat Macro-F1 OOF 46,86%, HistGradientBoosting 44,44%, dan Logistic Regression 43,91% pada normalisasi yang sama. Macro-F1 validation Extra Trees session-robust turun menjadi 43,90%. Kami mengambil keluarga Extra Trees sebagai kandidat nonlinear dan Logistic Regression sebagai pembanding linear. Kedua notebook menggunakan fitur dari window yang sedang dinilai, dengan RobustScaler untuk model linear yang dipelajari dari fitting. Pembagian berbasis peserta memungkinkan evaluasi pada orang baru. [Ringkasan batch-05](../../batch-05/SUMMARY.md)

Di batch-06 exp02, kombinasi normalisasi RMSSD dan eye mendapat accuracy OOF peserta 41,43%, sedangkan eye-only 31,74%, pada subset empat orang. Exp03 menguji pertanyaan berbeda melalui ablasi **fitur** HRV dan eye pada seluruh cohort v3 yang layak. Di batch-04, beberapa model neural berhenti pada epoch 1–2 dan GRU terpilih melewatkan satu kelas. Dua target baru ini diprioritaskan untuk perbandingan tabular dengan kontrol sederhana dan pelaporan recall. Skor batch terdahulu memakai label protokol dan tidak dapat dijadikan pembanding langsung. [Hasil exp02](../exp02/SUMMARY.md) · [Diagnosis batch-04](../../batch-04/DIAGNOSIS.md)

## Exp03 · tiga kelas proxy perilaku

Gunakan kelima CSV v3. Notebook memeriksa kesesuaian window, peserta, episode, target, split, dan bobot melalui ID. Hanya sepuluh fitur core masuk model gabungan. Seluruh fitting memakai 744 window Train. Empat kandidat memakai bobot Train resmi. Dummy memprediksi kelas terbanyak pada window Train sebagai acuan untuk ketimpangan kelas.

| Kandidat | Fitur | Konfigurasi |
|---|---|---|
| Logistic Regression | 10 core | RobustScaler, C = 1 |
| Extra Trees | 10 core | 200 pohon, minimal 5 baris per daun, max_features = 0,8 |
| Extra Trees HRV | RMSSD dan SDNN | Parameter Extra Trees yang sama |
| Extra Trees eye | Delapan fitur kedipan, EAR dan iris | Parameter Extra Trees yang sama |

Label dan bobot yang dipublikasikan mengikuti referensi Train penuh. Karena itu exp03 memakai split tetap, dengan 218 window validation dan 231 window Test. Semua kandidat dinilai pada validation tanpa bobot. Peringkat memakai Macro-F1 episode, lalu accuracy episode, lalu urutan kandidat pada tabel. Pilihan dicatat di `selection.json` sebelum evaluasi Test. Model dipasang ulang pada Train yang sama dengan konfigurasi dan seed tetap. Test hanya menilai model terpilih dan Dummy. Gunakan hasil validation untuk perbandingan ablasi dan Test untuk estimasi akhir kandidat terpilih. Label v3 menyatakan proxy perilaku yang dibangun tim.

## Exp04 · sepuluh kelas rating subjektif

Gunakan ketiga CSV v4. Satu dari 17 peserta ditahan bergantian. Dalam tiap fold, semua episode peserta lain menjadi fitting. Bobot tiap window sebanding dengan kebalikan jumlah window pada episodenya, dinormalisasi agar rata-rata bobot fitting satu. Dengan demikian, satu episode panjang dan satu episode pendek ikut menentukan fitting dengan total bobot yang sama. Seluruh resep ditetapkan sebelum LOPO.

| Perbandingan | Cara membaca |
|---|---|
| Dummy mode dan median | Prediksi konstan dari distribusi rating episode fitting. Mode menjadi acuan exact accuracy dan median menjadi acuan MAE. |
| Logistic Regression dan Extra Trees | Klasifikasi nominal sepuluh kelas pada fitur core, dengan rating berprobabilitas terbesar sebagai prediksi. Parameter sama dengan exp03. |
| Logistic Regression dengan median | Probabilitas identik dengan Logistic Regression nominal. Memilih median distribusi mengisolasi dampak aturan keputusan pada MAE. |
| Logistic Regression ordinal | Sembilan classifier memperkirakan peluang rating melebihi tiap ambang 1–9. Minimum kumulatif memastikan peluang tidak meningkat pada ambang yang lebih tinggi. Selisih antarpeluang membentuk distribusi sepuluh kelas. Prediksi mengambil median. Bandingkan terhadap model nominal yang juga memakai median. |

Rating 8 berasal dari tiga episode dan rating 9 dari empat episode. Baca MAE dalam satuan tingkat rating, exact accuracy, proporsi prediksi dalam ±1 tingkat, dan recall beserta support tiap kelas. LOPO memberi satu prediksi di luar fitting per window per kandidat. Angka gabungan membandingkan resep tetap. Memilih pemenang sesudah membaca tabel memerlukan evaluasi baru untuk mengukur performa model pilihan itu secara independen.

## Membaca isi ZIP

| Berkas | Isi |
|---|---|
| `manifest.json`, `input_support.csv` | Hash CSV input, versi paket, seed, resep fitting, daftar kelas, jumlah window, episode dan peserta per kelompok/kelas teramati |
| `participant_splits.csv`, `selection.json` (v3) atau `folds.csv` (v4) | Split peserta yang terwakili, keputusan seleksi v3, atau peserta held-out tiap fold v4 |
| `window_predictions.csv`, `episode_predictions.csv` | ID, fold, kelas aktual, kelas prediksi dan probabilitas untuk setiap kelas |
| `metrics.csv`, `participant_metrics.csv` | Hasil gabungan dan per peserta, masing-masing pada tingkat window dan episode |
| `confusion.csv` | Confusion matrix, support dan recall, termasuk baris kelas dengan support nol |

Prediksi episode memakai rata-rata probabilitas seluruh window dalam episode itu, lalu menerapkan aturan keputusan kandidat. Ini adalah ringkasan setelah episode selesai. Untuk prediksi sensor saat aktivitas berlangsung, gunakan tabel window. Window berdurasi 15 detik bergeser 5 detik dan saling bertumpang tindih. Tiap episode menyumbang satu observasi ke metrik episode. Macro-F1 selalu memakai seluruh kelas target, dengan F1 nol bila tidak terdefinisi. Recall pada kelas tanpa observasi dikosongkan. Periksa jumlah peserta pada tabel dukungan sebelum menafsirkan perubahan skor.

Untuk membangun ulang kedua notebook setelah mengubah [sumber eksperimen](experiment.py), jalankan `python models/batch-06/relabeling/build_notebooks.py` dari root repo.
