# Dataset fusion v2 berdasarkan kelayakan fitur

Kebijakan aktif: `feature_aware_quality_v1`. Kebijakan sebelum build ini: `conservative_quality_replacement_v2`.

## Data dan target

Window 15, 25, dan 45 detik dihitung dari RR mentah dan frame eye pada interval absolut yang sama, dengan step 5 detik dan endpoint (start, end]. Tidak ada interpolasi, normalisasi statistik seluruh dataset, augmentasi, atau pemilihan menurut skor model.

Target tetap cognitive_load_label. Active easy/practice/demo dan fase pre-task setelah kalibrasi, feedback, rating, rest, post-task/interrupted yang terlacak diberi Low. Active medium diberi Medium, active hard diberi High. Baseline/kalibrasi atau waktu tidak terlacak tidak diberi Low. Seluruh bagian window harus mempunyai satu target; campuran medium dan rating tetap dikeluarkan.

## Distribusi

| Window | Dataset | Low | Medium | High | Total | Orang |
|---|---|---:|---:|---:|---:|---:|
| 15s | good_only | 974 | 1210 | 1187 | 3371 | 20 |
| 15s | good_acceptable | 1700 | 1768 | 1812 | 5280 | 22 |
| 15s | all_data | 2389 | 2153 | 2138 | 6680 | 22 |
| 25s | good_only | 896 | 1030 | 1043 | 2969 | 20 |
| 25s | good_acceptable | 1571 | 1496 | 1581 | 4648 | 21 |
| 25s | all_data | 2176 | 1737 | 1817 | 5730 | 22 |
| 45s | good_only | 798 | 639 | 793 | 2230 | 19 |
| 45s | good_acceptable | 1435 | 980 | 1158 | 3573 | 21 |
| 45s | all_data | 1957 | 1096 | 1308 | 4361 | 22 |

Semua window yang lolos aturan inti dipertahankan. Tidak ada kuota kelas, penggantian satu-banding-satu, atau pembatasan berdasarkan nama. Pemilihan berbeda pada setiap panjang window karena dukungan waktu dan kualitasnya berbeda.

## Input utama dan profil tambahan

Default 10 fitur: `rmssd_ms`, `sdnn_ms`, `blink_rate_per_minute`, `ear_left_mean`, `ear_left_std`, `ear_right_mean`, `ear_right_std`, `iris_size_estimate_mean`, `iris_size_estimate_std`, `iris_size_delta_mean`.

iris_size_delta_std dan blink_count tetap tersimpan tetapi tidak menjadi predictor default karena redundansi. Rasio baseline HRV tidak diwajibkan untuk core. Ukuran iris adalah estimasi geometris landmark, bukan diameter pupil langsung.

| Window | Core 10 fitur | Core head 14 fitur | Validated fusion 22 fitur |
|---|---:|---:|---:|
| 15s | 5280 | 5280 | 1064 |
| 25s | 4648 | 4648 | 930 |
| 45s | 3573 | 3573 | 748 |

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

## Pembaruan sumber pada build ini

Rekaman tanpa penanda aktivitas tidak diberi label Low secara manual. Eksklusi yang dikonfigurasi memakai session ID sehingga perubahan nama folder tidak mengubah keputusan.

| Folder yang tidak dipakai | Session ID | Alasan |
|---|---|---|
| bangsur_easy_only | 20261116_151757_P01_coding_task | excluded_session: missing_task_events_operator_not_connected: operator confirmed that bangsur_easy_only was not connected to CodeArrange; activity labels are unavailable. |

Sumber yang dikecualikan tetap disimpan dan di-hash. Tidak ada event atau ID soal yang dibuat untuk melengkapinya.

### Perubahan good_acceptable

| Window | Low sebelum | Medium sebelum | High sebelum | Tambahan Low | Tambahan Medium | Tambahan High | Total sesudah | Rasio terbesar/terkecil |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 15s | 1062 | 1266 | 1251 | 638 | 502 | 561 | 5280 | 1.066 |
| 25s | 984 | 1093 | 1137 | 587 | 403 | 444 | 4648 | 1.057 |
| 45s | 908 | 728 | 891 | 527 | 252 | 267 | 3573 | 1.464 |

Rasio tersebut menjelaskan proporsi, bukan ambang kecukupan statistik. Builder mempertahankan semua window yang lolos, tanpa kuota kelas atau pemotongan manual.
