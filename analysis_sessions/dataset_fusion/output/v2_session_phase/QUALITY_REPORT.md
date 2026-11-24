# Dataset fusion v2 berdasarkan kelayakan fitur

Kebijakan aktif: `feature_aware_quality_v1`. Kebijakan sebelum build ini: `feature_aware_quality_v1`.

## Data dan target

Window 15, 25, dan 45 detik dihitung dari RR mentah dan frame eye pada interval absolut yang sama, dengan step 5 detik dan endpoint (start, end]. Tidak ada interpolasi, normalisasi statistik seluruh dataset, augmentasi, atau pemilihan menurut skor model.

Target adalah cognitive_load_label menurut session_phase_context_v3. Active easy/practice/demo dan pre-task setelah kalibrasi diberi Low; active medium diberi Medium dan active hard diberi High. Feedback/rating mengikuti difficulty tugas terkait. Rest dan post-task mengikuti difficulty tugas terakhir, bukan otomatis Low atau otomatis High. Fase-fase tersebut menjadi kandidat opsional, bukan otomatis masuk dataset utama. Interrupted, baseline/kalibrasi, dan konteks yang tidak dapat ditelusuri dikeluarkan. Seluruh bagian window harus mempunyai satu target; medium + rating medium boleh masuk, tetapi easy + rating medium tidak.

## Distribusi

| Window | Dataset | Low | Medium | High | Total | Orang |
|---|---|---:|---:|---:|---:|---:|
| 15s | good_only | 820 | 1207 | 1180 | 3207 | 20 |
| 15s | good_acceptable | 1433 | 1433 | 1433 | 4299 | 22 |
| 15s | all_data | 1897 | 3199 | 3136 | 8232 | 22 |
| 25s | good_only | 796 | 1025 | 1038 | 2859 | 20 |
| 25s | good_acceptable | 1386 | 1386 | 1386 | 4158 | 21 |
| 25s | all_data | 1830 | 3151 | 3098 | 8079 | 22 |
| 45s | good_only | 726 | 813 | 790 | 2329 | 19 |
| 45s | good_acceptable | 1149 | 1149 | 1149 | 3447 | 21 |
| 45s | all_data | 1716 | 3055 | 3022 | 7793 | 22 |

Kelayakan sensor tetap feature_aware_quality_v1. Seleksi dataset utama memakai balanced_primary_v1: all_data menyimpan seluruh kandidat berlabel termasuk poor dan kandidat tidak terpilih; good_acceptable menyimpan good/acceptable yang terpilih dengan jumlah kelas sama; good_only merupakan subset good dari good_acceptable dan tidak diseimbangkan ulang. Tidak ada pembatasan berdasarkan daftar nama.

## Input utama dan profil tambahan

Default 10 fitur: `rmssd_ms`, `sdnn_ms`, `blink_rate_per_minute`, `ear_left_mean`, `ear_left_std`, `ear_right_mean`, `ear_right_std`, `iris_size_estimate_mean`, `iris_size_estimate_std`, `iris_size_delta_mean`.

iris_size_delta_std dan blink_count tetap tersimpan tetapi tidak menjadi predictor default karena redundansi. Rasio baseline HRV tidak diwajibkan untuk core. Ukuran iris adalah estimasi geometris landmark, bukan diameter pupil langsung.

| Window | Core 10 fitur | Core head 14 fitur | Validated fusion 22 fitur |
|---|---:|---:|---:|
| 15s | 4299 | 4299 | 888 |
| 25s | 4158 | 4158 | 858 |
| 45s | 3447 | 3447 | 624 |

core_head menambahkan mean/std pitch dan yaw; gunakan hanya baris dengan keempat nilai finite. validated_fusion menambahkan empat statistik gaze dan rasio fixation, saccade, transition, unclassified. Gunakan seluruh aturan filter pada feature_columns.json, termasuk verified_session_ids, validasi asli, coverage, clipping, dan kelengkapan fitur. Kelulusan validasi kalibrasi tidak menjamin tidak ada drift selama tugas.

## Aturan kualitas inti

| Kriteria | Good | Acceptable |
|---|---:|---:|
| eye_temporal_coverage | ≥95% | ≥80% |
| face_ear_ratio | ≥95% | ≥80% |
| iris_available_ratio | ≥90% | ≥70% |
| Coverage RR | ≥80% | ≥80% |
| Pasangan RR | ≥10 | ≥10 |
| Umur RR | ≤3 detik | ≤3 detik |

Seluruh fitur core harus finite dan memenuhi aturan nilai. Nilai kedipan tidak diketahui tidak diisi nol. Standar deviasi dan hitungan tidak boleh negatif; iris_size_delta_mean bertanda negatif tetap sah. Kolom quality_tier, good_failed_criteria, dan acceptable_failed_criteria sekarang menilai core. curation_selected dan curation_reason yang sudah ada menjelaskan admission; tidak ada kolom penanda baru.

## Perlakuan fitur spasial

Hasil validasi terakhir diperiksa terhadap konfigurasi tersimpan dan prediksi pada lima titik uji. accepted_manual tidak mengubah failed menjadi passed. Validasi hilang, tidak lengkap, atau tidak konsisten tidak digunakan untuk profil fusion. Ambang tidak dilonggarkan.

Dalam CSV curated, empat statistik gaze dan seluruh rasio eye-state dibuat missing jika tidak memenuhi validasi sumber dan kualitas gaze window. Nilai asal tetap ada pada all_data.csv. Masking bukan koreksi koordinat dan tidak membuktikan bahwa EAR, iris, atau HRV dari rekaman yang sama salah.

## Integritas dan provenance

Inventory memilih satu salinan acquisition dan mempertahankan grup orang lintas rekaman. Ihsan berbeda dari Ikhsan. bangsur_easy_only tetap dikeluarkan karena tidak ada task_events. Kebijakan timestamp Alif tetap dipin pada hash sumber; window yang menyentuh timestamp ambigu dikeluarkan tanpa mengubah waktu mentah.

RR diproses sesuai urutan paket dan waktu ketersediaan. Pasangan RMSSD tidak melintasi segmen terputus. replay_validation.json memuat pembandingan cache dan perhitungan independen HRV/eye. Hash seluruh sumber diperiksa sebelum/sesudah build. Nilai sensor interval yang sama diperbandingkan saat migrasi; perubahan label dicatat terpisah.

Riwayat kebijakan dan pasangan penggantian sebelumnya disimpan di history; riwayat itu tidak lagi mengunci jumlah kelas. good_only ⊆ good_acceptable ⊆ all_data menurut window_id. Nilai spasial pada curated dapat berbeda dari all_data hanya karena masking. Bundle dipublikasikan bersama dengan verifikasi dan rollback.

## Pemodelan berikutnya

Baca features/profiles pada feature_columns.json, bukan semua kolom numerik. Metadata identitas, fase, difficulty, waktu, dan kualitas tidak menjadi predictor. Normalisasi antarsesi, pembagian Train/validation/test, seleksi model, dan regularisasi tetap pekerjaan pemodelan. Notebook dan hasil historis tidak diperbarui otomatis oleh perubahan CSV.


## Pemeriksaan lonjakan RR sebelum pemodelan

Kebijakan exclude_local_rr_jump_v1 mengeluarkan window curated yang memuat RR diterima dengan deviasi lebih dari 250 ms terhadap median referensi RRPreprocessor. Riwayat normal memakai maksimum 11 RR dan minimum 5; setelah gap, referensi mengikuti kelompok recovery pada waktu rilis. Hanya beat dalam (start, end] yang sudah tersedia pada endpoint yang diperhitungkan.

Aturan ini menandai ketidakpastian lokal, bukan diagnosis bahwa setiap kandidat pasti artefak. Seluruh rekaman diperiksa dengan aturan yang sama tanpa membaca label atau skor model. Nilai RR, RMSSD, SDNN, fitur eye, dan label tidak diganti. Window terdampak tetap di all_data, dengan quality_tier poor dan alasan rr_local_median_jump pada metadata yang sudah ada. Tidak ada kolom flag baru, interpolasi, winsorization, atau aktivasi filter live.

| Window | Window terdampak di all_data | Core layak yang dikeluarkan | Low | Medium | High |
|---|---:|---:|---:|---:|---:|
| window_15s_step_5s | 69 | 43 | 11 | 9 | 23 |
| window_25s_step_5s | 113 | 75 | 16 | 23 | 36 |
| window_45s_step_5s | 184 | 147 | 30 | 43 | 74 |

| Rekaman | RR task | RR diterima pemeriksaan dasar | Kandidat lonjakan |
|---|---:|---:|---:|
| Alif | 3474 | 3460 | 28 |
| Andrean | 2905 | 2905 | 1 |
| Andreas | 2675 | 2675 | 0 |
| Athaya | 2623 | 2623 | 0 |
| Baithar | 2800 | 2800 | 0 |
| bangsur_easy_medium | 1472 | 1472 | 1 |
| Fabian | 2475 | 2475 | 0 |
| faqih | 2587 | 2587 | 2 |
| ihsan | 2016 | 1956 | 0 |
| ikhsan | 2561 | 2554 | 1 |
| irfan | 2990 | 2990 | 0 |
| kaka | 3163 | 3147 | 1 |
| Mon | 4347 | 4347 | 0 |
| Rafi | 2958 | 2958 | 1 |
| raina | 3069 | 3069 | 0 |
| Ramadhan | 556 | 556 | 0 |
| Ramadhan-2 | 2774 | 2774 | 2 |
| sam | 1578 | 1576 | 1 |
| Savitri | 4440 | 4440 | 0 |
| Shakilla | 1056 | 1052 | 0 |
| Siti | 2594 | 2578 | 0 |
| Sullivan | 741 | 705 | 0 |
| Tio | 3807 | 3803 | 2 |

manifest.json menyimpan konfigurasi, raw_id, waktu beat/ketersediaan, median referensi, serta hash rr_raw.csv untuk audit. Pemeriksaan independen saat publikasi mencocokkan penanda window, alasan kualitas, jumlah yang dikeluarkan, dan proyeksi CSV tersaring. Rasio baseline historis tetap informasi audit dan bukan input core. Normalisasi antarsesi serta pembagian Train/validation/test tidak dilakukan di CSV.


## Pemeriksaan dataset sebelum pemodelan

| Window | Missing core | Infinity core | ID window duplikat | Vektor core duplikat | Fitur core konstan | Rasio baseline tidak tersedia |
|---|---:|---:|---:|---:|---|---:|
| window_15s_step_5s | 0 | 0 | 0 | 0 | tidak ada | 412 |
| window_25s_step_5s | 0 | 0 | 0 | 0 | tidak ada | 405 |
| window_45s_step_5s | 0 | 0 | 0 | 0 | tidak ada | 253 |

Fitur inti tidak diimputasi. Ketidaktersediaan rasio baseline tidak diganti dengan nol dan tidak menggugurkan core; rasio itu bukan predictor default. Vektor sama atau korelasi tinggi tidak otomatis membuktikan duplikasi akuisisi. Duplikasi sumber dan ID interval diperiksa terpisah.

| Window | Low active_task | Seluruh Low | Persentase Low active_task |
|---|---:|---:|---:|
| window_15s_step_5s | 707 | 1433 | 49.34% |
| window_25s_step_5s | 470 | 1386 | 33.91% |
| window_45s_step_5s | 229 | 1149 | 19.93% |

Label fase tetap mengikuti protokol v2. Komposisi ini perlu dipertimbangkan saat menyusun development dan evaluasi; penyeimbangan memilih baris layak tanpa mengganti labelnya; kandidat tidak terpilih tetap ada di all_data.

| Window | Fitur | Variasi antar-rerata sesi / total | Variasi antar-rerata kelas / total |
|---|---|---:|---:|
| window_15s_step_5s | rmssd_ms | 69.75% | 0.09% |
| window_15s_step_5s | ear_left_mean | 85.39% | 2.71% |
| window_15s_step_5s | iris_size_estimate_mean | 87.06% | 0.17% |
| window_25s_step_5s | rmssd_ms | 76.05% | 0.05% |
| window_25s_step_5s | ear_left_mean | 87.39% | 2.57% |
| window_25s_step_5s | iris_size_estimate_mean | 89.24% | 0.53% |
| window_45s_step_5s | rmssd_ms | 79.11% | 0.31% |
| window_45s_step_5s | ear_left_mean | 89.32% | 1.68% |
| window_45s_step_5s | iris_size_estimate_mean | 91.97% | 1.43% |

Rasio variasi adalah eta-squared deskriptif satu faktor. Kedua kolom tidak dijumlahkan dan tidak membuktikan penyebab biologis, kerusakan sensor, atau ketiadaan informasi nonlinear. Level dasar antarsesi tidak dikoreksi menggunakan seluruh CSV. Statistik preprocessing harus berasal dari Train atau baseline yang tersedia sebelum prediksi. Identitas tetap metadata, bukan predictor.

## Pembaruan sumber pada build ini

Rekaman tanpa penanda aktivitas tidak diberi label Low secara manual. Eksklusi yang dikonfigurasi memakai session ID sehingga perubahan nama folder tidak mengubah keputusan.

| Folder yang tidak dipakai | Session ID | Alasan |
|---|---|---|
| bangsur_easy_only | 20261116_151757_P01_coding_task | excluded_session: missing_task_events_operator_not_connected: operator confirmed that bangsur_easy_only was not connected to CodeArrange; activity labels are unavailable. |

Sumber yang dikecualikan tetap disimpan dan di-hash. Tidak ada event atau ID soal yang dibuat untuk melengkapinya.

### Perubahan good_acceptable

| Window | Low sebelum | Medium sebelum | High sebelum | Tambahan Low | Tambahan Medium | Tambahan High | Total sesudah | Rasio terbesar/terkecil |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 15s | 1433 | 1433 | 1433 | 0 | 0 | 0 | 4299 | 1.000 |
| 25s | 1386 | 1386 | 1386 | 0 | 0 | 0 | 4158 | 1.000 |
| 45s | 1149 | 1149 | 1149 | 0 | 0 | 0 | 3447 | 1.000 |

Rasio tersebut menjelaskan proporsi, bukan ambang kecukupan statistik. Delta adalah perubahan bersih setelah koreksi label, batas fase, dan seleksi kelas; bukan jumlah rekaman baru. Kandidat layak yang tidak dipilih tetap berada di all_data.


## Batas fase nonpengerjaan

Rest/post-task hanya menjadi kandidat dalam 45 detik sejak active-task terakhir berhenti. Batas tidak dimulai ulang ketika feedback berubah menjadi rating, rest, atau post-task. Feedback/rating mengikuti durasi fase yang tercatat dan tidak dikenai batas khusus ini. Nilai 45 detik adalah batas operasional, bukan hasil pengukuran waktu pemulihan fisiologis. Window yang melewati batas dikeluarkan seluruhnya.

window_phase_mix menyimpan konteks sumber, waktu awal set/soal, akhir pengerjaan, dan batas grace dalam label_context. manifest.json menyimpan ledger dan readiness yang dipakai untuk merekonstruksi ulang segmen. Tidak ada timestamp, event, atau nilai sensor mentah yang diubah.


## Seleksi dataset utama

Basis adalah window yang hanya memuat active-task dan/atau pre-task. Setiap window yang memuat feedback/rating/rest/post-task dihitung sebagai opsional, termasuk campurannya dengan pengerjaan. T = min(jumlah basis terbesar antarkelas, jumlah total kandidat layak terkecil antarkelas). Setiap kelas memilih basis dahulu, kemudian opsional hanya untuk mencapai T. Basis mayoritas dapat diseleksi sebagian. Jika suatu kelas tidak tersedia, build dihentikan, bukan melonggarkan kualitas.

Seleksi mendahulukan tier good kemudian acceptable. Dalam setiap tier, kuota dibagi merata sesuai kapasitas antarpeserta lalu antarsesi, dengan sisa kuota berurutan leksikografis. Tiap rekaman diambil pada indeks tengah dari interval timestamp yang dibagi merata. Tidak ada pemilihan menurut skor model atau perubahan label.

| Window | Kelas | Basis layak | Opsional layak | Basis dipilih | Opsional dipilih | Layak tidak dipilih |
|---|---|---:|---:|---:|---:|---:|
| 15s | Low | 747 | 686 | 747 | 686 | 0 |
| 15s | Medium | 1763 | 774 | 1433 | 0 | 1104 |
| 15s | High | 1803 | 744 | 1433 | 0 | 1114 |
| 25s | Low | 498 | 888 | 498 | 888 | 0 |
| 25s | Medium | 1487 | 1080 | 1386 | 0 | 1181 |
| 25s | High | 1572 | 988 | 1386 | 0 | 1174 |
| 45s | Low | 241 | 1047 | 241 | 908 | 139 |
| 45s | Medium | 966 | 1528 | 966 | 183 | 1345 |
| 45s | High | 1149 | 1354 | 1149 | 0 | 1354 |

## Durasi unik dan fase terpilih

| Window | Kelas | Baris | Durasi unik (menit) |
|---|---|---:|---:|
| 15s | Low | 1433 | 136.58 |
| 15s | Medium | 1433 | 166.00 |
| 15s | High | 1433 | 167.92 |
| 25s | Low | 1386 | 138.92 |
| 25s | Medium | 1386 | 179.33 |
| 25s | High | 1386 | 175.25 |
| 45s | Low | 1149 | 135.42 |
| 45s | Medium | 1149 | 203.58 |
| 45s | High | 1149 | 159.83 |

Jumlah row yang sama bukan jumlah pengamatan independen atau durasi unik yang harus sama. Window tetap overlap. Rincian jumlah/durasi unik setiap versi ada di class_distribution.csv.

| Window | Kelas | Fase window | Baris terpilih |
|---|---|---|---:|
| 15s | High | active_task | 1433 |
| 15s | Low | active_task | 707 |
| 15s | Low | feedback | 12 |
| 15s | Low | mixed | 671 |
| 15s | Low | pre_task | 40 |
| 15s | Low | rest | 3 |
| 15s | Medium | active_task | 1433 |
| 25s | High | active_task | 1386 |
| 25s | Low | active_task | 470 |
| 25s | Low | feedback | 8 |
| 25s | Low | mixed | 880 |
| 25s | Low | pre_task | 28 |
| 25s | Medium | active_task | 1386 |
| 45s | High | active_task | 1149 |
| 45s | Low | active_task | 229 |
| 45s | Low | feedback | 2 |
| 45s | Low | mixed | 906 |
| 45s | Low | pre_task | 12 |
| 45s | Medium | active_task | 966 |
| 45s | Medium | mixed | 183 |

## Window yang dikeluarkan

| Window | Alasan utama | Baris |
|---|---|---:|
| 15s | ambiguous_eye_timestamp | 7 |
| 15s | interrupted | 73 |
| 15s | missing_task_context | 158 |
| 15s | outside_optional_grace | 104 |
| 25s | ambiguous_eye_timestamp | 11 |
| 25s | interrupted | 73 |
| 25s | missing_task_context | 247 |
| 25s | outside_optional_grace | 118 |
| 45s | ambiguous_eye_timestamp | 19 |
| 45s | interrupted | 73 |
| 45s | missing_task_context | 405 |
| 45s | outside_optional_grace | 146 |

excluded_windows.csv mencatat penolakan label/waktu dan timestamp ambigu. all_data menyimpan baris berlabel yang gagal kualitas atau tidak dipilih; curation_reason membedakan keduanya. curation_summary.json dan manifest menyimpan ringkasan seleksi. Riwayat label/seleksi sebelum migrasi disimpan ringkas di history. Hasil eksperimen lama tetap merujuk dataset lama dan tidak berubah.

## Rebuild

`python -m analysis_sessions.dataset_fusion.build_dataset_v2 --replace`

Manifest aktif mewariskan kebijakan label dan seleksi ini. Rebuild menghitung ulang dari rekaman mentah, bukan concat CSV atau hanya mengganti label lama. Hash sumber dan bundle diperiksa sebelum publikasi; penggantian memakai staging dan rollback.
