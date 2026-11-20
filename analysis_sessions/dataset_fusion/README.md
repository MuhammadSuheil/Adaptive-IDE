# Dataset Fusion Eye Tracking dan HRV

Builder merekonstruksi fitur eye tracking dan HRV pada interval yang sama, lalu memberi target Low, Medium, atau High dari ledger CodeArrange. Perekaman mentah tetap read-only. Tidak ada training model dalam proses ini.

## Dataset aktif

Tiga good_acceptable.csv pada output/v2_session_phase menjadi dataset utama: window_15s_step_5s, window_25s_step_5s, dan window_45s_step_5s. V1 pada output/v1_active_task tetap arsip historis.

Ketiga panjang window merupakan alternatif dari rekaman yang sama; jangan menumpuknya menjadi satu dataset. Step 5 detik menghasilkan overlap, bukan observasi independen baru. Jumlah dan durasi unik kelas dibaca dari hasil build terbaru.

## Kelayakan fitur inti

Kebijakan feature_aware_quality_v1 menilai input inti tanpa menggugurkan EAR/HRV hanya karena posisi gaze tidak akurat.

| View | Isi |
|---|---|
| good_only.csv | Window inti good |
| good_acceptable.csv | Window inti good atau acceptable |
| all_data.csv | Seluruh window berlabel dengan waktu tidak ambigu, termasuk inti poor; pengukuran spasial asal tetap tersedia |

Kolom quality_tier, good_failed_criteria, dan acceptable_failed_criteria yang sudah ada kini menilai fitur inti. curation_selected dan curation_reason mencatat admission. Tidak ada kolom core_quality_tier, core_usable, head_pose_usable, atau spatial_usable.

Semua window yang memenuhi aturan diterima tanpa kuota kelas, penilaian berdasarkan nama, atau skor model. Window yang dahulu ditolak hanya karena gaze dapat diterima setelah syarat inti dipenuhi. Riwayat Mon/Alif tetap tersimpan pada history tetapi tidak lagi mengunci jumlah kelas.

| Kriteria | Good | Acceptable |
|---|---:|---:|
| Coverage temporal eye dengan bin 100 ms | ≥95% | ≥80% |
| Face dan EAR valid | ≥95% | ≥80% |
| Iris valid nonblinking | ≥90% | ≥70% |
| Coverage RR | ≥80% | ≥80% |
| Pasangan RR | ≥10 | ≥10 |
| Umur RR terakhir | ≤3 detik | ≤3 detik |

Seluruh predictor inti harus tersedia, finite, dan memenuhi batas nilainya. Missing tidak diganti nol. Perubahan iris negatif tetap sah. Rasio baseline HRV yang tidak tersedia tidak menggugurkan core bila RMSSD/SDNN window layak.

## Predictor dan profil input

feature_columns.json menetapkan target cognitive_load_label, group participant_id, dan default 10 predictor:

    rmssd_ms, sdnn_ms, blink_rate_per_minute,
    ear_left_mean, ear_left_std, ear_right_mean, ear_right_std,
    iris_size_estimate_mean, iris_size_estimate_std, iris_size_delta_mean

Pengukuran lain tetap disimpan. iris_size_delta_std dan blink_count tidak masuk default karena redundansi. Iris adalah estimasi geometris landmark, bukan diameter pupil langsung.

| Profil | Predictor | Syarat baris |
|---|---:|---|
| core | 10 | Seluruh baris good_acceptable |
| core_head | 14 | Core dan empat statistik head pose finite |
| validated_fusion | 22 | Core/head, empat statistik gaze, dan empat rasio eye-state; seluruh aturan spasial terpenuhi |

Fusion memakai rasio fixation, saccade, transition, dan unclassified; other tetap untuk audit. Head pose diperiksa terpisah dari validasi gaze.

Pilih kolom dari profiles dan baris dengan seluruh kolom profil finite. Untuk validated_fusion, terapkan juga validated_fusion_filter: session ID terverifikasi, hasil validasi asli lolos, coverage gaze minimal sesuai kontrak, dan clipping maksimal sesuai kontrak. Gunakan good_acceptable sebagai titik awal pemodelan, bukan all_data tanpa penyaringan.

Validasi terakhir diperiksa terhadap ambang tersimpan dan prediksi lima titik uji. accepted_manual tidak mengubah hasil gagal menjadi lolos. Pada CSV curated, seluruh statistik gaze dan rasio eye-state yang tidak layak dikosongkan. Nilai asal tetap di all_data. Jangan mengimputasi gaze gagal seolah menggantikan pengukuran tervalidasi.

Identitas, waktu, fase, soal, difficulty, dan kualitas bukan predictor. Normalisasi, seleksi fitur yang dipelajari, dan resampling dihitung pada Train saat pemodelan. Notebook serta hasil historis tetap merujuk snapshot/hash saat eksperimen tersebut dijalankan.

## Target dan rekonstruksi sensor

- Active easy/practice/demo diberi Low; active medium diberi Medium; active hard diberi High.
- Pre-task setelah kedua kalibrasi, feedback, rating, rest, serta post-task/interrupted yang terlacak diberi Low sesuai protokol v2.
- Baseline/kalibrasi dan waktu yang tidak terlacak tidak diberi Low.
- Seluruh bagian window harus mempunyai satu target. Easy + rating dapat tetap Low; medium + rating dikeluarkan. Target bukan majority vote dan tidak dibentuk ulang dari difficulty.

HRV dihitung dari RR mentah pada masing-masing panjang window, bukan rata-rata RMSSD cache. Endpoint (start, end] memakai paket yang sudah tersedia pada endpoint. Pasangan RMSSD tidak menyeberangi RR ditolak atau segmen terputus. Baseline dan RR sebelum task tidak masuk task replay.

Inventory memilih satu salinan acquisition dan menggabungkan grup orang pada rekaman berulang. Ihsan tetap berbeda dari Ikhsan. bangsur_easy_only dikecualikan karena tidak memiliki ledger. Timestamp ambigu Alif tetap dipin pada hash sumber dan window yang menyentuhnya dikeluarkan; waktu mentah tidak digeser.

## Menjalankan builder

Migrasi eksplisit atau build baru dengan kebijakan fitur inti:

    python analysis_sessions/dataset_fusion/build_dataset_v2.py --admission-policy feature_aware_quality_v1 --replace

Jika output belum ada, hilangkan --replace. Rebuild berikutnya mewarisi kebijakan manifest aktif:

    python analysis_sessions/dataset_fusion/build_dataset_v2.py --replace
    python -m unittest discover -s analysis_sessions/dataset_fusion -p "test_*.py" -v

Opsi --input, --output, --config, dan --subjects tersedia. Build baru tanpa --admission-policy mempertahankan default legacy window_quality_only; verifikasi bundle historis tetap didukung. Opsi penggantian satu-banding-satu tidak boleh dicampur dengan kebijakan feature-aware.

## Publikasi dan pembaruan

Sumber diverifikasi sebelum dan sesudah build. Ketiga window dibuat pada staging dan diverifikasi bersama sebelum publikasi. Kegagalan pemasangan memulihkan bundle sebelumnya; lokasi rollback dipertahankan jika pemulihan gagal. Sumber mentah, root output, arsip v1, symlink, dan junction dilindungi.

Manifest mencatat kebijakan, sumber, kontrak fitur, audit validasi gaze, pengecualian timestamp, jumlah profil, dan checksum. replay_validation.json memuat pembandingan cache HRV serta sampel perhitungan independen. history menjelaskan kebijakan sebelum migrasi; aturan dan jumlah aktual dibaca dari manifest aktif.

Tambahkan rekaman lengkap ke sessions/saved_session, lalu rebuild. Jangan menggabungkan CSV dengan concat manual. Baca QUALITY_REPORT.md dan DATASET_CONDITION_REPORT.md setelah build untuk jumlah kelas, kontribusi rekaman, durasi unik, dan cakupan profil tambahan.
