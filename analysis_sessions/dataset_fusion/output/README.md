# Pilihan Dataset Cognitive Load

Tiga good_acceptable v2 berikut menjadi dataset utama. V1 disimpan sebagai referensi historis. Ketiga window v2 berasal dari rekaman yang sama; jangan menumpuknya sebagai satu dataset.

| Dataset | Window / step | Low | Medium | High | Total | Orang |
|---|---|---:|---:|---:|---:|---:|
| [v1 historis](v1_active_task/good_acceptable.csv) | 25 / 5 detik | 249 | 968 | 1077 | 2294 | 13 |
| [v2 utama 15s](v2_session_phase/window_15s_step_5s/good_acceptable.csv) | 15 / 5 detik | 1687 | 1763 | 1803 | 5253 | 22 |
| [v2 utama 25s](v2_session_phase/window_25s_step_5s/good_acceptable.csv) | 25 / 5 detik | 1549 | 1487 | 1572 | 4608 | 21 |
| [v2 utama 45s](v2_session_phase/window_45s_step_5s/good_acceptable.csv) | 45 / 5 detik | 1394 | 966 | 1149 | 3509 | 21 |

## Perbedaan dataset

| Bagian | V1 historis | V2 aktif |
|---|---|---|
| Target | difficulty_level dari pengerjaan aktif | cognitive_load_label menurut fase v2 |
| Low | Active easy | Easy/practice/demo serta fase sesi Low yang terlacak |
| Medium dan High | Active medium dan hard | Tetap active medium dan hard |
| HRV | Fitur cache 25 detik | Replay RR mentah pada masing-masing panjang window |
| Kelayakan curated | Aturan historis | Fitur inti HRV, blink, EAR, dan iris |
| Input default | Kontrak historis | 10 predictor core pada feature_columns.json |
| Gaze | Pengukuran historis | Kosong di curated bila tidak lolos validasi dan kualitas window |

V2 memakai kebijakan feature_aware_quality_v1. good_only berisi core good; good_acceptable berisi core good/acceptable. all_data menyimpan pengukuran berlabel untuk audit, termasuk core poor dan gaze yang tidak layak. quality_tier sekarang menyatakan kualitas inti, bukan akurasi gaze. Window dengan lonjakan RR lokal dikeluarkan dari curated tanpa mengganti nilai sensor; bukti dan aturan exclude_local_rr_jump_v1 tersimpan pada manifest dan laporan v2.

Baca [QUALITY_REPORT v2](v2_session_phase/QUALITY_REPORT.md) dan [kondisi dataset](v2_session_phase/DATASET_CONDITION_REPORT.md) untuk aturan, profil, serta jumlah yang dapat dipakai.

## Pemodelan

Gunakan features pada feature_columns.json untuk input default. core_head dan validated_fusion merupakan profil tambahan dengan persyaratan baris tersendiri. Identitas, fase, waktu, soal, dan kualitas bukan predictor. Jangan mengisi gaze gagal dengan nol atau memakai semua kolom numerik.

Statistik normalisasi, seleksi fitur yang dipelajari, imputasi bila diperlukan pada eksperimen lain, dan resampling dikerjakan pada Train, bukan pada seluruh CSV. Window overlap bukan observasi independen. Model, notebook, dan laporan historis tetap merujuk hash dataset saat eksperimen tersebut dijalankan.
