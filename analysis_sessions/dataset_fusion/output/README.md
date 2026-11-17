# Pilihan Dataset Cognitive Load

Empat CSV berikut menjadi rujukan utama tim: tiga kandidat v2 untuk pekerjaan berikutnya dan satu referensi historis v1. Ketiga window v2 berasal dari sumber yang sama dan banyak intervalnya overlap; jangan menumpuk ketiganya sebagai dataset gabungan.

| Dataset | Window / step | Low | Medium | High | Total row | Orang | Peran |
|---|---|---:|---:|---:|---:|---:|---|
| [v1 good_acceptable](v1_active_task/good_acceptable.csv) | 25 / 5 detik | 249 | 968 | 1077 | 2294 | 13 | Referensi dataset lama |
| [v2 good_acceptable 15s](v2_session_phase/window_15s_step_5s/good_acceptable.csv) | 15 / 5 detik | 1062 | 1266 | 1251 | 3579 | 21 | Kandidat dataset aktif |
| [v2 good_acceptable 25s](v2_session_phase/window_25s_step_5s/good_acceptable.csv) | 25 / 5 detik | 984 | 1093 | 1137 | 3214 | 21 | Kandidat dataset aktif |
| [v2 good_acceptable 45s](v2_session_phase/window_45s_step_5s/good_acceptable.csv) | 45 / 5 detik | 908 | 728 | 891 | 2527 | 20 | Kandidat dataset aktif |

## Perbedaan v1 dan v2

| Bagian | v1_active_task | v2_session_phase |
|---|---|---|
| Cakupan target | Pengerjaan aktif easy/medium/hard | Pengerjaan aktif dan fase sesi sesuai protokol v2 |
| Low | Active easy | Active easy/practice/demo, pre-task setelah kalibrasi, feedback, rating, rest, post-task/interrupted yang terlacak |
| Medium / High | Active medium / hard | Active medium / hard |
| Target di CSV | difficulty_level, dipetakan easy→Low, medium→Medium, hard→High | cognitive_load_label; difficulty tetap metadata asal |
| Window | 25 detik, fitur HRV dari cache sumber | HRV dihitung ulang dari RR mentah untuk 15/25/45 detik; eye mengikuti batas yang sama |
| Admission curated | Tier sensor dan daftar review tim lama | Tier sensor dan penggantian kualitas satu-banding-satu yang tercatat |
| Status | Arsip tetap, tidak ditimpa | Bundle aktif, dapat diperbarui dengan --replace |

V2 hanya menerima window dengan satu target sepanjang interval. Easy + rating dapat tetap Low; medium + rating atau hard + rest dikeluarkan. V2 juga menerapkan aturan endpoint RR yang konsisten, sehingga perubahan terhadap v1 tidak hanya berupa tambahan baris Low.

Window panjang mencakup rentang waktu lebih besar, tetapi lebih mudah melintasi perubahan target. Karena itu jumlah row dan bagian rekaman yang lolos berbeda antarwindow. Step 5 detik menambah overlap; jumlah row bukan jumlah pengamatan independen.

## View lain yang tetap tersedia

Pada bundle penggantian, good_only memuat good terpilih dan good_acceptable memuat good/acceptable terpilih. Status curation_selected dan curation_reason menjelaskan keputusan terpisah dari tier sensor, dan all_data memuat seluruh window berlabel termasuk tier poor. Gunakan good_only untuk pembanding kualitas dan all_data untuk audit cakupan; jangan menganggap seluruh all_data layak menjadi input sensor tanpa memeriksa kualitasnya.

Good/acceptable menilai kelengkapan/clipping, bukan jaminan akurasi gaze. Baca [laporan v2](v2_session_phase/QUALITY_REPORT.md) untuk proses, fitur, kondisi sensor, dan panduan menambah rekaman.

## Pembaruan dan pemodelan

Builder memperbarui tabel ini dari jumlah hasil build setelah mengganti bundle v2. Simpan metadata identitas untuk deduplikasi/grouping, bukan sebagai predictor. Gunakan allowlist fitur dan target sesuai versi dataset. Model lama tetap merujuk snapshot/hash saat training; menimpa v2 tidak memperbarui hasil eksperimen yang sudah selesai.


## Sumber pada pembaruan terakhir

Builder menerima 23 rekaman dari 22 orang. 1 rekaman belum ada pada build sebelumnya. Penggantian selektif mempertahankan jumlah kelas awal; tidak seluruh kandidat tambahan dimasukkan ke curated. Tahap baru tidak mengubah hasil tahap sebelumnya. Window bertimestamp ambigu dikeluarkan sebelum pembentukan view.

- bangsur_easy_only tidak digunakan: excluded_session: missing_task_events_operator_not_connected: operator confirmed that bangsur_easy_only was not connected to CodeArrange; activity labels are unavailable.

Kontribusi per rekaman, perubahan jumlah kelas, durasi unik, dan batas kualitas dijelaskan pada [QUALITY_REPORT v2](v2_session_phase/QUALITY_REPORT.md). Pembaruan dataset tidak mengesahkan release pemodelan atau menjalankan training.
