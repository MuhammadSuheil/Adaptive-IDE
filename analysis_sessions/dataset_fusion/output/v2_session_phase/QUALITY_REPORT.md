# Dataset fusion v2: session phase

## Label dan window

`cognitive_load_label` adalah satu-satunya target. Active easy/practice/demo, pre-task setelah kedua kalibrasi selesai, feedback, rating, jeda antarset, dan post-task diberi Low. Rating/feedback setelah medium/hard juga Low. Active medium diberi Medium; active hard diberi High. Interrupted hanya menunjuk interval setelah event set_ended yang tercatat, tanpa event akhir penelitian berikutnya. Baseline/kalibrasi dan interval tanpa ledger/readiness yang dapat ditelusuri tidak diberi Low.

Setiap bagian interval harus menghasilkan target yang sama. Campuran easy + rating tetap Low; medium + rating tidak masuk dataset. Aturan ini bukan majority vote. Window mixed-phase dapat dipertahankan; `window_phase_mix` dan `window_label_mix` menyimpan rinciannya. Difficulty/set/soal sumber tidak ditimpa agar sama dengan target.

## Rekonstruksi sensor

Window 15/25/45 detik dihitung ulang dari rr_raw.csv dan eye_tracking.csv dengan step 5 detik, memakai origin task dan grid yang sama per sesi. Tidak ada interpolasi, averaging RMSSD tersimpan, atau perubahan timestamp mentah. RR sebelum task sensor tidak masuk replay. Paket yang tiba setelah endpoint tidak dapat memperbaiki window yang telah lewat. Paket yang tiba tepat di endpoint diproses sebelum window diterbitkan. Recovery menunggu waktu rilisnya; pasangan RMSSD tidak melintasi RR terbuang atau segmen terputus.

Syarat HRV tetap coverage 80%, minimal 10 pasangan, umur data maksimal 3 detik. Aturan eye dan admission peserta identik dengan config v1. good_only memerlukan proposed_good dan tier good; good_acceptable memerlukan proposed_good/questionable dan tier good/acceptable; all_data memuat semua window berlabel termasuk sensor poor. Peserta baru memerlukan review dan pembaruan daftar admission, bukan otomatis diterima.

Baseline recorder ready dipisahkan dari baseline usable builder. Rasio hanya tersedia untuk baseline ready dengan RMSSD positif dan coverage minimal 80%. Penolakan tercatat pada baseline_rejection_reason; nilai hilang tetap kosong, bukan nol.

## Validasi cache HRV 25 detik

Diperiksa 6689 window: 6659 sesuai replay utama; 30 sesuai urutan timer live sebelum paket tepat di endpoint; tidak ada selisih yang tidak terjelaskan. Semua rincian ada di replay_validation.json. Dataset baru memakai aturan endpoint yang konsisten, tidak menyalin angka cache agar tampak identik.

## Distribusi

| Window | Dataset | Low | Medium | High | Total | Peserta |
|---|---|---:|---:|---:|---:|---:|
| window_15s_step_5s | good_only | 202 | 271 | 338 | 811 | 5 |
| window_15s_step_5s | good_acceptable | 732 | 1105 | 1174 | 3011 | 13 |
| window_15s_step_5s | all_data | 1447 | 1805 | 1964 | 5216 | 18 |
| window_25s_step_5s | good_only | 185 | 234 | 313 | 732 | 5 |
| window_25s_step_5s | good_acceptable | 666 | 968 | 1077 | 2711 | 13 |
| window_25s_step_5s | all_data | 1281 | 1477 | 1696 | 4454 | 18 |
| window_45s_step_5s | good_only | 168 | 146 | 240 | 554 | 4 |
| window_45s_step_5s | good_acceptable | 614 | 650 | 854 | 2118 | 13 |
| window_45s_step_5s | all_data | 1105 | 966 | 1250 | 3321 | 18 |

## Penyaringan window

| Window | Kandidat | Target campuran | Kalibrasi/tidak terlacak | Berlabel, HRV tidak layak | Gagal admission peserta |
|---|---:|---:|---:|---:|---:|
| 15s | 6727 | 1511 | 0 | 565 | 1424 |
| 25s | 6689 | 2235 | 0 | 390 | 1187 |
| 45s | 6613 | 3292 | 0 | 258 | 839 |

Kolom HRV/admission di atas merupakan diagnostik pada window berlabel dan dapat saling overlap. Tier poor juga dapat berasal dari kualitas eye; rincian kriteria tersimpan di CSV. all_data tidak membuang tier poor. CSV curated menerapkan quality dan admission bersama-sama.

## Komposisi fase pada Low good_acceptable

| Window | Fase sumber | Window Low |
|---|---|---:|
| 15s | active_task | 340 |
| 15s | feedback | 87 |
| 15s | interrupted | 3 |
| 15s | mixed | 271 |
| 15s | post_task | 4 |
| 15s | pre_task | 19 |
| 15s | rest | 8 |
| 25s | active_task | 252 |
| 25s | feedback | 38 |
| 25s | interrupted | 1 |
| 25s | mixed | 356 |
| 25s | post_task | 2 |
| 25s | pre_task | 12 |
| 25s | rest | 5 |
| 45s | active_task | 146 |
| 45s | feedback | 4 |
| 45s | mixed | 454 |
| 45s | pre_task | 5 |
| 45s | rest | 5 |

mixed berarti beberapa fase dengan target Low yang sama. Detail per segmen ada pada window_phase_mix; jumlah window bukan durasi unik dan tidak menyatakan semua Low berasal dari pengerjaan easy.

## Pemodelan berikutnya

Gunakan 26 predictor sensor pada feature_columns.json, bukan semua kolom numerik. Target/difficulty/fase, nama peserta, ID soal, waktu, dan kualitas adalah metadata atau label, bukan X. Identitas tetap dipakai untuk grouping/deduplikasi. Imputasi, scaling, seleksi fitur, class weight dan oversampling dilakukan pada training fold, tidak pada CSV sumber. Belum ada training model atau rekaman Low tambahan pada build ini. Batch-01 tetap arsip; batch-02 memakai evaluasinya sendiri.

Window overlap bukan observasi independen. Durasi unik dilaporkan per kelas/rekaman pada class_distribution.csv. Mutu sensor dan komposisi kelas dapat berubah antarwindow; jumlah row terbesar bukan bukti window terbaik. Endpoint pembanding dan manifest evaluasi batch-02 dibuat setelah data tambahan diterima.
