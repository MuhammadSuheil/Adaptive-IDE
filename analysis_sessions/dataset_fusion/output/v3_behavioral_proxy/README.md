# Dataset v3: behavioral proxy

Dataset ini eksploratif, bukan label CL terukur. Baca [QUALITY_REPORT.md](QUALITY_REPORT.md) untuk rubrik, cakupan, kekurangan kelas, dan cara menggunakan pembagian peserta.

Input utama adalah window_15s_step_5s/good_acceptable.csv; 25/45 detik adalah analisis sekunder. Alternatif Train adalah full pool dengan train_sample_weights.csv atau train_balanced.csv, bukan keduanya sekaligus. Validation/Test tidak diseimbangkan. participant_splits.csv dan label_reference.json tidak boleh diabaikan saat pemodelan. V1/v2 tidak berubah.
