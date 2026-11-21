# Dataset fusion v2 berdasarkan kelayakan fitur

Kebijakan aktif: `feature_aware_quality_v1`. Kebijakan sebelum build ini: `feature_aware_quality_v1`.

## Data dan target

Window 15, 25, dan 45 detik dihitung dari RR mentah dan frame eye pada interval absolut yang sama, dengan step 5 detik dan endpoint (start, end]. Tidak ada interpolasi, normalisasi statistik seluruh dataset, augmentasi, atau pemilihan menurut skor model.

Target tetap cognitive_load_label. Active easy/practice/demo dan fase pre-task setelah kalibrasi, feedback, rating, rest, post-task/interrupted yang terlacak diberi Low. Active medium diberi Medium, active hard diberi High. Baseline/kalibrasi atau waktu tidak terlacak tidak diberi Low. Seluruh bagian window harus mempunyai satu target; campuran medium dan rating tetap dikeluarkan.

## Distribusi

| Window | Dataset | Low | Medium | High | Total | Orang |
|---|---|---:|---:|---:|---:|---:|
| 15s | good_only | 974 | 1207 | 1180 | 3361 | 20 |
| 15s | good_acceptable | 1687 | 1763 | 1803 | 5253 | 22 |
| 15s | all_data | 2389 | 2153 | 2138 | 6680 | 22 |
| 25s | good_only | 891 | 1025 | 1038 | 2954 | 20 |
| 25s | good_acceptable | 1549 | 1487 | 1572 | 4608 | 21 |
| 25s | all_data | 2176 | 1737 | 1817 | 5730 | 22 |
| 45s | good_only | 790 | 630 | 790 | 2210 | 19 |
| 45s | good_acceptable | 1394 | 966 | 1149 | 3509 | 21 |
| 45s | all_data | 1957 | 1096 | 1308 | 4361 | 22 |

Semua window yang lolos aturan inti dipertahankan. Tidak ada kuota kelas, penggantian satu-banding-satu, atau pembatasan berdasarkan nama. Pemilihan berbeda pada setiap panjang window karena dukungan waktu dan kualitasnya berbeda.

## Input utama dan profil tambahan

Default 10 fitur: `rmssd_ms`, `sdnn_ms`, `blink_rate_per_minute`, `ear_left_mean`, `ear_left_std`, `ear_right_mean`, `ear_right_std`, `iris_size_estimate_mean`, `iris_size_estimate_std`, `iris_size_delta_mean`.

iris_size_delta_std dan blink_count tetap tersimpan tetapi tidak menjadi predictor default karena redundansi. Rasio baseline HRV tidak diwajibkan untuk core. Ukuran iris adalah estimasi geometris landmark, bukan diameter pupil langsung.

| Window | Core 10 fitur | Core head 14 fitur | Validated fusion 22 fitur |
|---|---:|---:|---:|
| 15s | 5253 | 5253 | 1048 |
| 25s | 4608 | 4608 | 909 |
| 45s | 3509 | 3509 | 712 |

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

RR diproses sesuai urutan paket dan waktu ketersediaan. Pasangan RMSSD tidak melintasi segmen terputus. replay_validation.json memuat pembandingan cache dan perhitungan independen HRV/eye. Hash seluruh sumber diperiksa sebelum/sesudah build. Nilai sensor dan target interval lama diperbandingkan saat migrasi.

Riwayat kebijakan dan pasangan penggantian sebelumnya disimpan di history; riwayat itu tidak lagi mengunci jumlah kelas. good_only ⊆ good_acceptable ⊆ all_data menurut window_id. Nilai spasial pada curated dapat berbeda dari all_data hanya karena masking. Bundle dipublikasikan bersama dengan verifikasi dan rollback.

## Pemodelan berikutnya

Baca features/profiles pada feature_columns.json, bukan semua kolom numerik. Metadata identitas, fase, difficulty, waktu, dan kualitas tidak menjadi predictor. Normalisasi antarsesi, pembagian Train/validation/test, seleksi model, dan regularisasi tetap pekerjaan pemodelan. Notebook dan hasil historis tidak diperbarui otomatis oleh perubahan CSV.


## Pemeriksaan lonjakan RR sebelum pemodelan

Kebijakan exclude_local_rr_jump_v1 mengeluarkan window curated yang memuat RR diterima dengan deviasi lebih dari 250 ms terhadap median referensi RRPreprocessor. Riwayat normal memakai maksimum 11 RR dan minimum 5; setelah gap, referensi mengikuti kelompok recovery pada waktu rilis. Hanya beat dalam (start, end] yang sudah tersedia pada endpoint yang diperhitungkan.

Aturan ini menandai ketidakpastian lokal, bukan diagnosis bahwa setiap kandidat pasti artefak. Seluruh rekaman diperiksa dengan aturan yang sama tanpa membaca label atau skor model. Nilai RR, RMSSD, SDNN, fitur eye, dan label tidak diganti. Window terdampak tetap di all_data, dengan quality_tier poor dan alasan rr_local_median_jump pada metadata yang sudah ada. Tidak ada kolom flag baru, interpolasi, winsorization, atau aktivasi filter live.

| Window | Window terdampak di all_data | Core layak yang dikeluarkan | Low | Medium | High |
|---|---:|---:|---:|---:|---:|
| window_15s_step_5s | 52 | 27 | 13 | 5 | 9 |
| window_25s_step_5s | 65 | 40 | 22 | 9 | 9 |
| window_45s_step_5s | 79 | 64 | 41 | 14 | 9 |

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
| window_15s_step_5s | 0 | 0 | 0 | 0 | tidak ada | 468 |
| window_25s_step_5s | 0 | 0 | 0 | 0 | tidak ada | 410 |
| window_45s_step_5s | 0 | 0 | 0 | 0 | tidak ada | 304 |

Fitur inti tidak diimputasi. Ketidaktersediaan rasio baseline tidak diganti dengan nol dan tidak menggugurkan core; rasio itu bukan predictor default. Vektor sama atau korelasi tinggi tidak otomatis membuktikan duplikasi akuisisi. Duplikasi sumber dan ID interval diperiksa terpisah.

| Window | Low active_task | Seluruh Low | Persentase Low active_task |
|---|---:|---:|---:|
| window_15s_step_5s | 707 | 1687 | 41.91% |
| window_25s_step_5s | 470 | 1549 | 30.34% |
| window_45s_step_5s | 229 | 1394 | 16.43% |

Label fase tetap mengikuti protokol v2. Komposisi ini perlu dipertimbangkan saat menyusun development dan evaluasi; dataset tidak diseimbangkan dengan mengganti label atau membuang fase yang sah.

| Window | Fitur | Variasi antar-rerata sesi / total | Variasi antar-rerata kelas / total |
|---|---|---:|---:|
| window_15s_step_5s | rmssd_ms | 67.89% | 0.07% |
| window_15s_step_5s | ear_left_mean | 81.91% | 2.30% |
| window_15s_step_5s | iris_size_estimate_mean | 86.66% | 0.38% |
| window_25s_step_5s | rmssd_ms | 73.94% | 0.01% |
| window_25s_step_5s | ear_left_mean | 85.71% | 2.41% |
| window_25s_step_5s | iris_size_estimate_mean | 89.00% | 0.55% |
| window_45s_step_5s | rmssd_ms | 78.99% | 0.12% |
| window_45s_step_5s | ear_left_mean | 88.31% | 2.56% |
| window_45s_step_5s | iris_size_estimate_mean | 91.56% | 1.27% |

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
| 15s | 1700 | 1768 | 1812 | -13 | -5 | -9 | 5253 | 1.069 |
| 25s | 1571 | 1496 | 1581 | -22 | -9 | -9 | 4608 | 1.057 |
| 45s | 1435 | 980 | 1158 | -41 | -14 | -9 | 3509 | 1.443 |

Rasio tersebut menjelaskan proporsi, bukan ambang kecukupan statistik. Builder mempertahankan semua window yang lolos, tanpa kuota kelas atau pemotongan manual.
