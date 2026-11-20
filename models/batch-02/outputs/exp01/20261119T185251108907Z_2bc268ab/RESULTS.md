# Hasil run batch-03

Model utama dipilih berdasarkan validation: window 45 detik.

- Accuracy test: 35.21%
- Macro-F1 test: 33.37%
- Balanced accuracy test: 34.83%
- Jumlah test: 338
- Target accuracy 75%: belum tercapai

Model evaluasi: final_w45_668251358c8ef1843a70/model.pkl. Model ini di-fit pada training + validation, tidak pada test. Window lain merupakan pembanding yang telah ditetapkan sebelum evaluasi.

Split memakai blok waktu campuran dengan purge; data orang yang sama dapat muncul di training dan test. Hasil ini merupakan evaluasi internal dataset saat ini.
