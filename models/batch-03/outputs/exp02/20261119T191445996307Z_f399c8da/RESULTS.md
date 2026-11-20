# Hasil run batch-03

Model utama dipilih berdasarkan validation: window 45 detik.

- Accuracy test: 45.86%
- Macro-F1 test: 45.82%
- Balanced accuracy test: 46.39%
- Jumlah test: 338
- Target accuracy 75%: belum tercapai

Model evaluasi: final_w45_a3126a45dd04aa33dfe1/model.pkl. Model ini di-fit pada training + validation, tidak pada test. Window lain merupakan pembanding yang telah ditetapkan sebelum evaluasi.

Split memakai blok waktu campuran dengan purge; data orang yang sama dapat muncul di training dan test. Hasil ini merupakan evaluasi internal dataset saat ini.
