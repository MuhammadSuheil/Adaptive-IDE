# Dataset Fusion Eye Tracking dan HRV

Builder merekonstruksi fitur eye tracking dan HRV pada interval yang sama, lalu memberi target Low, Medium, atau High dari ledger CodeArrange. Perekaman mentah tetap read-only. Tidak ada training model dalam proses ini.

## Dataset aktif

Tiga good_acceptable.csv pada output/v2_session_phase menjadi dataset utama: window_15s_step_5s, window_25s_step_5s, dan window_45s_step_5s. V1 pada output/v1_active_task tetap arsip historis.

Ketiga panjang window merupakan alternatif dari rekaman yang sama; jangan menumpuknya menjadi satu dataset. Step 5 detik menghasilkan overlap, bukan observasi independen baru. Jumlah dan durasi unik kelas dibaca dari hasil build terbaru.

## Kelayakan fitur inti

Kebijakan feature_aware_quality_v1 menilai input inti tanpa menggugurkan EAR/HRV hanya karena posisi gaze tidak akurat.

| View | Isi |
|---|---|
| good_only.csv | Subset tier good dari baris good_acceptable terpilih; tidak diseimbangkan ulang |
| good_acceptable.csv | Window inti good/acceptable terpilih dengan jumlah Low, Medium, High sama |
| all_data.csv | Seluruh window berlabel dengan waktu tidak ambigu, termasuk inti poor; pengukuran spasial asal tetap tersedia |

Kolom quality_tier, good_failed_criteria, dan acceptable_failed_criteria yang sudah ada kini menilai fitur inti. curation_selected dan curation_reason mencatat admission. Tidak ada kolom core_quality_tier, core_usable, head_pose_usable, atau spatial_usable.

Kelayakan sensor dan seleksi kelas adalah langkah terpisah. Seluruh kandidat berlabel dengan waktu tidak ambigu disimpan dalam all_data, termasuk poor dan kandidat layak yang tidak terpilih. Dataset utama diseleksi menurut balanced_primary_v1 tanpa membaca skor model atau daftar penilaian nama. Riwayat Mon/Alif tetap tersimpan pada history tetapi tidak mengunci jumlah kelas.

Basis terdiri dari window yang hanya memuat active-task dan/atau pre-task. Window dengan bagian feedback/rating/rest/post-task adalah opsional. Target per kelas adalah `T = min(jumlah basis terbesar antarkelas, jumlah total kandidat layak terkecil antarkelas)`. Basis didahulukan; opsional hanya melengkapi kekurangan. Basis mayoritas yang tidak dipilih tetap tersedia di all_data. Seleksi mendahulukan good lalu acceptable, dengan kuota merata sesuai kapasitas antarpeserta/rekaman dan timestamp tersebar. Ketiga panjang window diseleksi secara independen.

Aturan tambahan `exclude_local_rr_jump_v1` memeriksa RR task secara kausal dengan timing, kontak, rentang, dan recovery yang sama. Window curated dikeluarkan bila memuat RR diterima dengan selisih lebih dari 250 ms terhadap median referensi lokal (maksimum 11 RR; minimum 5 pada riwayat normal). Beat harus berada dalam `(start, end]` dan sudah tersedia pada endpoint. Kandidat adalah ketidakpastian lokal, bukan diagnosis artefak.

RR dan fitur sensor tidak dikoreksi, diinterpolasi, atau dipotong. Window terdampak tetap tersedia di `all_data.csv`, tetapi `quality_tier` menjadi `poor` dan alasan `rr_local_median_jump` disimpan di kolom metadata yang sudah ada. Aturan tidak membaca label dan tidak mengubah `FILTER_ENABLED` pada sistem perekaman. Manifest memuat bukti raw_id, waktu, median, hash sumber, serta jumlah eksklusi tiap kelas/window.

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
- Pre-task setelah kedua kalibrasi diberi Low.
- Feedback/rating mengikuti difficulty tugas yang menghasilkan fase tersebut, bukan otomatis Low.
- Rest/post-task mengikuti difficulty tugas terakhir. Keduanya hanya menjadi kandidat hingga 45 detik sejak pengerjaan terakhir berhenti; pergantian fase tidak mengulang batas tersebut. Feedback/rating tidak dikenai batas khusus ini.
- Interrupted, baseline/kalibrasi, dan konteks yang tidak terlacak dikeluarkan.
- Seluruh bagian window harus mempunyai satu target dan memenuhi batas waktunya. Medium + rating medium dapat masuk; easy + rating medium dikeluarkan. Window yang melewati batas grace juga dikeluarkan seluruhnya, bukan majority vote.

Aturan label memakai session_phase_context_v3. Batas 45 detik adalah batas operasional, bukan hasil pengukuran waktu pemulihan fisiologis. Konteks set/soal dan referensi waktunya disimpan dalam label_context pada JSON window_phase_mix, tanpa menambah kolom classifier. Manifest memuat ledger sumber untuk rekonstruksi dan verifikasi.

HRV dihitung dari RR mentah pada masing-masing panjang window, bukan rata-rata RMSSD cache. Endpoint (start, end] memakai paket yang sudah tersedia pada endpoint. Pasangan RMSSD tidak menyeberangi RR ditolak atau segmen terputus. Baseline dan RR sebelum task tidak masuk task replay.

Inventory memilih satu salinan acquisition dan menggabungkan grup orang pada rekaman berulang. Ihsan tetap berbeda dari Ikhsan. bangsur_easy_only dikecualikan karena tidak memiliki ledger. Timestamp ambigu Alif tetap dipin pada hash sumber dan window yang menyentuhnya dikeluarkan; waktu mentah tidak digeser.

## Menjalankan builder

Migrasi eksplisit atau build baru dengan kebijakan fitur inti, label konteks, dan seleksi seimbang:

    python -m analysis_sessions.dataset_fusion.build_dataset_v2 --admission-policy feature_aware_quality_v1 --phase-policy session_phase_context_v3 --replace

Jika output belum ada, hilangkan --replace. Rebuild berikutnya mewarisi kebijakan manifest aktif:

    python analysis_sessions/dataset_fusion/build_dataset_v2.py --replace
    python -m unittest discover -s analysis_sessions/dataset_fusion -p "test_*.py" -v

Opsi --input, --output, --config, dan --subjects tersedia. Build baru tanpa --admission-policy mempertahankan default legacy window_quality_only; verifikasi bundle historis tetap didukung. Opsi penggantian satu-banding-satu tidak boleh dicampur dengan kebijakan feature-aware.

Untuk menerapkan ulang pemeriksaan RR pada bundle feature-aware yang sudah terverifikasi tanpa menghitung ulang agregasi eye/HRV:

    python -m analysis_sessions.dataset_fusion.v2_rr_quality --refresh analysis_sessions/dataset_fusion/output/v2_session_phase

Perintah ini hanya memakai agregat lama bila hash seluruh sumber, konfigurasi kualitas, dan kode perhitungan sensor masih cocok. Jika berubah, proses berhenti dan harus menggunakan full builder. Hasil aturan RR sama dengan full build; tidak ada penggabungan CSV manual. Seluruh predictor dan label interval lama diperiksa agar tidak berubah sebelum publikasi.

## Publikasi dan pembaruan

Sumber diverifikasi sebelum dan sesudah build. Ketiga window dibuat pada staging dan diverifikasi bersama sebelum publikasi. Kegagalan pemasangan memulihkan bundle sebelumnya; lokasi rollback dipertahankan jika pemulihan gagal. Sumber mentah, root output, arsip v1, symlink, dan junction dilindungi.

Manifest mencatat kebijakan, sumber, kontrak fitur, audit validasi gaze, pengecualian timestamp, jumlah profil, dan checksum. replay_validation.json memuat pembandingan cache HRV serta sampel perhitungan independen. history menjelaskan kebijakan sebelum migrasi; aturan dan jumlah aktual dibaca dari manifest aktif.

`pre_model_audit` pada manifest dan laporan dataset memuat kelengkapan core, ID/vektor duplikat, fitur konstan, baseline opsional yang hilang, komposisi fase Low, dan variasi antarsesi/kelas. Nilai nol kedipan yang sah tidak dihapus. Pergeseran nilai dasar antarsesi serta pemisahan Train/validation/test tidak diselesaikan dengan normalisasi global pada CSV.

Tambahkan rekaman lengkap ke sessions/saved_session, lalu rebuild. Jangan menggabungkan CSV dengan concat manual. Baca QUALITY_REPORT.md dan DATASET_CONDITION_REPORT.md setelah build untuk jumlah kelas, kontribusi rekaman, durasi unik, dan cakupan profil tambahan.

## Dataset v3: proxy perilaku

`output/v3_behavioral_proxy` merupakan eksperimen label terpisah. V3 tidak mengganti v2 dan tidak memakai easy/medium/hard atau fitur sensor sebagai penentu target. Low/Medium/High mengikuti kesesuaian waktu aktif relatif terhadap referensi soal yang sama dan pola kesalahan manual. Label ini bukan pengukuran beban mental yang sudah tervalidasi.

    python -m analysis_sessions.dataset_fusion.build_dataset_v3
    python -m analysis_sessions.dataset_fusion.build_dataset_v3 --verify analysis_sessions/dataset_fusion/output/v3_behavioral_proxy

Rebuild eksplisit setelah sumber v2 diperbarui:

    python -m analysis_sessions.dataset_fusion.build_dataset_v3 --replace

Keluaran memuat sembilan CSV (tiga panjang window dan tiga tingkat kualitas), tabel episode, referensi waktu, pembagian peserta tetap, pemetaan window, dan laporan. Sumber v2 serta rekaman mentah tidak diubah. Rubrik, batas referensi, dan pembagian peserta tidak dituning menurut skor model atau keseimbangan kelas.

Baca `output/v3_behavioral_proxy/QUALITY_REPORT.md` sebelum memakai CSV. Referensi hanya berasal dari Train; `participant_splits.csv` dan `window_assignments.csv` harus dipakai ketika memodelkan v3. Mengubah pembagian peserta memerlukan pelabelan ulang. Status `readiness` pada manifest membedakan keberadaan tiga kelas secara gabungan dari ketersediaan kelas pada masing-masing bagian evaluasi. Jangan langsung mengganti path notebook lama yang membuat split sendiri.

Kebijakan `behavioral_concordance_v2` mempertahankan rubrik label dan menambahkan alias soal identik berbukti hash, pengecualian paparan ulang dari referensi, serta pencarian pembagian peserta berbasis cakupan. Varian utama adalah 15 detik; dukungan peserta window 25/45 detik dilaporkan terpisah. Bukti alias dibekukan pada `v3_item_aliases.json` dan disalin ke bundle, bukan membaca versi HEAD saat rebuild.

Setiap folder window menyediakan `train_sample_weights.csv` untuk full Train dari good_acceptable atau `train_balanced.csv` sebagai alternatif Train seimbang. Bobot menyetarakan kelas, peserta dalam kelas, dan episode dalam peserta. Keduanya bukan fitur classifier dan tidak diterapkan sekaligus. Validation/Test tetap utuh. Dalam inner cross-validation, referensi, label, bobot, dan sampling dihitung ulang dari inner-Train. `reference_sensitivity.csv`, `partition_search.json`, dan riwayat perubahan mendokumentasikan hasil audit; builder tidak menjalankan training.
