# Pilihan Dataset Cognitive Load

Tiga good_acceptable v2 berikut menjadi dataset utama. V1 disimpan sebagai referensi historis. Ketiga window v2 berasal dari rekaman yang sama; jangan menumpuknya sebagai satu dataset.

| Dataset | Window / step | Low | Medium | High | Total | Orang |
|---|---|---:|---:|---:|---:|---:|
| [v1 historis](v1_active_task/good_acceptable.csv) | 25 / 5 detik | 249 | 968 | 1077 | 2294 | 13 |
| [v2 utama 15s](v2_session_phase/window_15s_step_5s/good_acceptable.csv) | 15 / 5 detik | 1433 | 1433 | 1433 | 4299 | 22 |
| [v2 utama 25s](v2_session_phase/window_25s_step_5s/good_acceptable.csv) | 25 / 5 detik | 1386 | 1386 | 1386 | 4158 | 21 |
| [v2 utama 45s](v2_session_phase/window_45s_step_5s/good_acceptable.csv) | 45 / 5 detik | 1149 | 1149 | 1149 | 3447 | 21 |

## Perbedaan dataset

| Bagian | V1 historis | V2 aktif |
|---|---|---|
| Target | difficulty_level dari pengerjaan aktif | cognitive_load_label menurut fase v2 |
| Low | Active easy | Active easy/practice/demo dan pre-task; fase opsional mengikuti tugas easy |
| Medium dan High | Active medium dan hard | Active medium/hard serta fase opsional dengan konteks difficulty yang sama |
| HRV | Fitur cache 25 detik | Replay RR mentah pada masing-masing panjang window |
| Kelayakan curated | Aturan historis | Fitur inti HRV, blink, EAR, dan iris |
| Input default | Kontrak historis | 10 predictor core pada feature_columns.json |
| Gaze | Pengukuran historis | Kosong di curated bila tidak lolos validasi dan kualitas window |

V2 memakai kebijakan feature_aware_quality_v1. good_acceptable diseleksi seimbang menurut balanced_primary_v1; good_only adalah subset good dari baris terpilih. all_data menyimpan pengukuran berlabel untuk audit, termasuk core poor dan gaze yang tidak layak. quality_tier sekarang menyatakan kualitas inti, bukan akurasi gaze. Window dengan lonjakan RR lokal dikeluarkan dari curated tanpa mengganti nilai sensor; bukti dan aturan exclude_local_rr_jump_v1 tersimpan pada manifest dan laporan v2.

Baca [QUALITY_REPORT v2](v2_session_phase/QUALITY_REPORT.md) dan [kondisi dataset](v2_session_phase/DATASET_CONDITION_REPORT.md) untuk aturan, profil, serta jumlah yang dapat dipakai.

## Pemodelan

Gunakan features pada feature_columns.json untuk input default. core_head dan validated_fusion merupakan profil tambahan dengan persyaratan baris tersendiri. Identitas, fase, waktu, soal, dan kualitas bukan predictor. Jangan mengisi gaze gagal dengan nol atau memakai semua kolom numerik.

Statistik normalisasi, seleksi fitur yang dipelajari, imputasi bila diperlukan pada eksperimen lain, dan resampling dikerjakan pada Train, bukan pada seluruh CSV. Window overlap bukan observasi independen. Model, notebook, dan laporan historis tetap merujuk hash dataset saat eksperimen tersebut dijalankan.


## Kebijakan fase dan seleksi

Label memakai session_phase_context_v3. Feedback/rating mengikuti tugas terkait; rest/post-task mengikuti tugas terakhir dan hanya menjadi kandidat hingga 45 detik sejak pengerjaan berhenti. Peralihan fase tidak mengulang batas tersebut. Interrupted, kalibrasi, konteks tidak terlacak, dan window lintas label dikeluarkan.

Active-task/pre-task didahulukan; fase opsional dipakai hanya untuk melengkapi kelas yang kurang. Seleksi dilakukan terpisah per panjang window. Baris mayoritas yang tidak dipilih tetap tersimpan di all_data. all_data tidak dipaksa seimbang; good_only tidak diseimbangkan ulang. Jangan menumpuk ketiga panjang window.

Rebuild: `python -m analysis_sessions.dataset_fusion.build_dataset_v2 --replace`. Kebijakan label, kualitas, dan seleksi diwarisi dari manifest aktif.

## V3 eksploratif: label berdasarkan perilaku

[Dataset v3](v3_behavioral_proxy/README.md) adalah alternatif eksperimen, bukan pengganti otomatis tiga dataset utama v2. V3 memakai kesesuaian waktu pengerjaan aktif dan pola kesalahan, tanpa menjadikan sensor atau difficulty sebagai pembuat label. Low/Medium/High pada v3 merupakan proxy perilaku, bukan CL yang sudah terukur dan tervalidasi.

| Good acceptable v3 | Low | Medium | High | Total |
|---|---:|---:|---:|---:|
| [Window 15 detik](v3_behavioral_proxy/window_15s_step_5s/good_acceptable.csv) | 130 | 237 | 826 | 1.193 |
| [Window 25 detik](v3_behavioral_proxy/window_25s_step_5s/good_acceptable.csv) | 82 | 178 | 643 | 903 |
| [Window 45 detik](v3_behavioral_proxy/window_45s_step_5s/good_acceptable.csv) | 36 | 99 | 340 | 475 |

Ketiga kelas kini tersedia pada Train, validation, dan Test untuk semua panjang window. Pembagian dipilih melalui pencarian cakupan terbatas tanpa training classifier, dengan referensi selalu dihitung dari Train. Varian utama adalah 15 detik: minimal tiga peserta per kelas pada Train dan dua pada validation/Test terpenuhi. Window 25/45 detik tetap sekunder karena dukungan peserta pada beberapa kelas masih tipis, meskipun tidak ada kelas yang hilang.

CSV utama v3 mempertahankan distribusi alami; episode High panjang menghasilkan lebih banyak window. Penanganan Train tersedia sebagai alternatif bobot hierarkis pada seluruh Train atau `train_balanced.csv` dengan jumlah kelas sama. Validation/Test tidak diseimbangkan. Baca [QUALITY_REPORT v3](v3_behavioral_proxy/QUALITY_REPORT.md) untuk 82 episode berlabel dari 358 episode, perbaikan identitas soal/referensi pengulangan, sensitivitas label, serta keterbatasan sumber. Gunakan pembagian dan referensi tersimpan; mengganti pembagian memerlukan pelabelan ulang. Data v2 dan nilai sensor tidak diubah.

## V4: rating kesulitan yang diberikan peserta

[Good acceptable v4 (15 detik)](v4_self_reported_rating/window_15s_step_5s/good_acceptable.csv) memuat 1.693 window dari 126 episode dan 17 peserta. Target `difficulty_rating` memakai angka 1–10 dari log CodeArrange yang cocok dengan episode, bukan label CL atau rating tetap tim. Semua angka terwakili, tetapi rating 8 hanya berasal dari tiga episode dan tiga peserta.

Baca [QUALITY_REPORT v4](v4_self_reported_rating/QUALITY_REPORT.md) untuk asal rating, distribusi, kualitas sensor, dan batas cakupan. Paket input Batch 06 ada di [models/batch-06/dataset](../../../models/batch-06/dataset/README.md). V4 tidak menimpa v2 atau v3 dan tidak memakai pembagian maupun bobot v3.
