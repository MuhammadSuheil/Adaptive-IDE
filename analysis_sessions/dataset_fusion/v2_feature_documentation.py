"""Documentation derived from the feature-aware sensor bundle, without model scores."""

import pandas as pd

from . import v2_features as features
from .v2_documentation import update_sections


def output_index(archive_csv, counts):
    old = pd.read_csv(archive_csv)
    target = old.difficulty_level.map({'easy': 'Low', 'medium': 'Medium', 'hard': 'High'})
    lines = ['# Pilihan Dataset Cognitive Load', '',
             ('Tiga good_acceptable v2 berikut menjadi dataset utama. V1 disimpan sebagai referensi historis. '
             'Ketiga window v2 berasal dari rekaman yang sama; jangan menumpuknya sebagai satu dataset.'), '',
             '| Dataset | Window / step | Low | Medium | High | Total | Orang |',
             '|---|---|---:|---:|---:|---:|---:|',
             (f'| [v1 historis](v1_active_task/good_acceptable.csv) | 25 / 5 detik | {int(target.eq("Low").sum())} | '
             f'{int(target.eq("Medium").sum())} | {int(target.eq("High").sum())} | {len(old)} | {old.participant_id.nunique()} |')]
    for s in (15, 25, 45):
        c = counts[f'window_{s}s_step_5s']['good_acceptable']
        lines.append(f'| [v2 utama {s}s](v2_session_phase/window_{s}s_step_5s/good_acceptable.csv) | {s} / 5 detik | '
                     f'{c["Low"]} | {c["Medium"]} | {c["High"]} | {c["windows"]} | {c["participants"]} |')
    lines += ['', '## Perbedaan dataset', '',
              '| Bagian | V1 historis | V2 aktif |', '|---|---|---|',
              '| Target | difficulty_level dari pengerjaan aktif | cognitive_load_label menurut fase v2 |',
              '| Low | Active easy | Easy/practice/demo serta fase sesi Low yang terlacak |',
              '| Medium dan High | Active medium dan hard | Tetap active medium dan hard |',
              '| HRV | Fitur cache 25 detik | Replay RR mentah pada masing-masing panjang window |',
              '| Kelayakan curated | Aturan historis | Fitur inti HRV, blink, EAR, dan iris |',
              '| Input default | Kontrak historis | 10 predictor core pada feature_columns.json |',
              '| Gaze | Pengukuran historis | Kosong di curated bila tidak lolos validasi dan kualitas window |', '',
              ('V2 memakai kebijakan feature_aware_quality_v1. good_only berisi core good; good_acceptable berisi core good/acceptable. '
              'all_data menyimpan pengukuran berlabel untuk audit, termasuk core poor dan gaze yang tidak layak. '
              'quality_tier sekarang menyatakan kualitas inti, bukan akurasi gaze.'), '',
              ('Baca [QUALITY_REPORT v2](v2_session_phase/QUALITY_REPORT.md) dan '
              '[kondisi dataset](v2_session_phase/DATASET_CONDITION_REPORT.md) untuk aturan, profil, serta jumlah yang dapat dipakai.'), '',
              '## Pemodelan', '',
              ('Gunakan features pada feature_columns.json untuk input default. core_head dan validated_fusion merupakan profil tambahan '
              'dengan persyaratan baris tersendiri. Identitas, fase, waktu, soal, dan kualitas bukan predictor. '
              'Jangan mengisi gaze gagal dengan nol atau memakai semua kolom numerik.'), '',
              ('Statistik normalisasi, seleksi fitur yang dipelajari, imputasi bila diperlukan pada eksperimen lain, dan resampling '
              'dikerjakan pada Train, bukan pada seluruh CSV. Window overlap bukan observasi independen. '
              'Model, notebook, dan laporan historis tetap merujuk hash dataset saat eksperimen tersebut dijalankan.'), '']
    return '\n'.join(lines)


def write_reports(destination, results, counts, profile_counts, checks, update, config, previous_policy, operator_guide):
    lines = ['# Dataset fusion v2 berdasarkan kelayakan fitur', '',
             f'Kebijakan aktif: `{features.POLICY}`. Kebijakan sebelum build ini: `{previous_policy}`.', '',
             '## Data dan target', '',
             ('Window 15, 25, dan 45 detik dihitung dari RR mentah dan frame eye pada interval absolut yang sama, '
             'dengan step 5 detik dan endpoint (start, end]. Tidak ada interpolasi, normalisasi statistik seluruh dataset, '
             'augmentasi, atau pemilihan menurut skor model.'), '',
             ('Target tetap cognitive_load_label. Active easy/practice/demo dan fase pre-task setelah kalibrasi, feedback, rating, '
             'rest, post-task/interrupted yang terlacak diberi Low. Active medium diberi Medium, active hard diberi High. '
             'Baseline/kalibrasi atau waktu tidak terlacak tidak diberi Low. Seluruh bagian window harus mempunyai satu target; '
             'campuran medium dan rating tetap dikeluarkan.'), '',
             '## Distribusi', '', '| Window | Dataset | Low | Medium | High | Total | Orang |',
             '|---|---|---:|---:|---:|---:|---:|']
    for s in (15, 25, 45):
        for name, c in counts[f'window_{s}s_step_5s'].items():
            lines.append(f'| {s}s | {name} | {c["Low"]} | {c["Medium"]} | {c["High"]} | {c["windows"]} | {c["participants"]} |')
    lines += ['', ('Semua window yang lolos aturan inti dipertahankan. Tidak ada kuota kelas, penggantian satu-banding-satu, '
              'atau pembatasan berdasarkan nama. Pemilihan berbeda pada setiap panjang window karena dukungan waktu dan kualitasnya berbeda.'), '',
              '## Input utama dan profil tambahan', '',
              'Default 10 fitur: '+', '.join(f'`{c}`' for c in features.CORE)+'.', '',
              ('iris_size_delta_std dan blink_count tetap tersimpan tetapi tidak menjadi predictor default karena redundansi. '
              'Rasio baseline HRV tidak diwajibkan untuk core. Ukuran iris adalah estimasi geometris landmark, bukan diameter pupil langsung.'), '',
              '| Window | Core 10 fitur | Core head 14 fitur | Validated fusion 22 fitur |', '|---|---:|---:|---:|']
    for s in (15, 25, 45):
        c = profile_counts[f'window_{s}s_step_5s']
        lines.append(f'| {s}s | {c["core"]} | {c["core_head"]} | {c["validated_fusion"]} |')
    lines += ['', ('core_head menambahkan mean/std pitch dan yaw; gunakan hanya baris dengan keempat nilai finite. '
              'validated_fusion menambahkan empat statistik gaze dan rasio fixation, saccade, transition, unclassified. '
              'Gunakan seluruh aturan filter pada feature_columns.json, termasuk verified_session_ids, validasi asli, coverage, clipping, '
              'dan kelengkapan fitur. Kelulusan validasi kalibrasi tidak menjamin tidak ada drift selama tugas.'), '',
              '## Aturan kualitas inti', '', '| Kriteria | Good | Acceptable |', '|---|---:|---:|']
    for rule, label in features.CORE_METRICS.items():
        lines.append(f'| {label} | ≥{config["quality"]["good"][rule]*100:g}% | ≥{config["quality"]["acceptable"][rule]*100:g}% |')
    lines += ['| Coverage RR | ≥80% | ≥80% |', '| Pasangan RR | ≥10 | ≥10 |', '| Umur RR | ≤3 detik | ≤3 detik |', '',
              ('Seluruh fitur core harus finite dan memenuhi aturan nilai. Nilai kedipan tidak diketahui tidak diisi nol. '
              'Standar deviasi dan hitungan tidak boleh negatif; iris_size_delta_mean bertanda negatif tetap sah. '
              'Kolom quality_tier, good_failed_criteria, dan acceptable_failed_criteria sekarang menilai core. '
              'curation_selected dan curation_reason yang sudah ada menjelaskan admission; tidak ada kolom penanda baru.'), '',
              '## Perlakuan fitur spasial', '',
              ('Hasil validasi terakhir diperiksa terhadap konfigurasi tersimpan dan prediksi pada lima titik uji. '
              'accepted_manual tidak mengubah failed menjadi passed. Validasi hilang, tidak lengkap, atau tidak konsisten tidak digunakan '
              'untuk profil fusion. Ambang tidak dilonggarkan.'), '',
              ('Dalam CSV curated, empat statistik gaze dan seluruh rasio eye-state dibuat missing jika tidak memenuhi validasi sumber '
              'dan kualitas gaze window. Nilai asal tetap ada pada all_data.csv. Masking bukan koreksi koordinat dan tidak membuktikan '
              'bahwa EAR, iris, atau HRV dari rekaman yang sama salah.'), '',
              '## Integritas dan provenance', '',
              ('Inventory memilih satu salinan acquisition dan mempertahankan grup orang lintas rekaman. Ihsan berbeda dari Ikhsan. '
              'bangsur_easy_only tetap dikeluarkan karena tidak ada task_events. Kebijakan timestamp Alif tetap dipin pada hash sumber; '
              'window yang menyentuh timestamp ambigu dikeluarkan tanpa mengubah waktu mentah.'), '',
              ('RR diproses sesuai urutan paket dan waktu ketersediaan. Pasangan RMSSD tidak melintasi segmen terputus. '
              'replay_validation.json memuat pembandingan cache dan perhitungan independen HRV/eye. '
              'Hash seluruh sumber diperiksa sebelum/sesudah build. Nilai sensor dan target interval lama diperbandingkan saat migrasi.'), '',
              ('Riwayat kebijakan dan pasangan penggantian sebelumnya disimpan di history; riwayat itu tidak lagi mengunci jumlah kelas. '
              'good_only ⊆ good_acceptable ⊆ all_data menurut window_id. Nilai spasial pada curated dapat berbeda dari all_data hanya '
              'karena masking. Bundle dipublikasikan bersama dengan verifikasi dan rollback.'), '',
              '## Pemodelan berikutnya', '',
              ('Baca features/profiles pada feature_columns.json, bukan semua kolom numerik. Metadata identitas, fase, difficulty, waktu, '
              'dan kualitas tidak menjadi predictor. Normalisasi antarsesi, pembagian Train/validation/test, seleksi model, '
              'dan regularisasi tetap pekerjaan pemodelan. Notebook dan hasil historis tidak diperbarui otomatis oleh perubahan CSV.'), '']
    lines += update_sections(update)
    (destination/'QUALITY_REPORT.md').write_text('\n'.join(lines).rstrip()+'\n', encoding='utf-8')
    condition = ['# Kondisi dataset berdasarkan kelayakan fitur', '',
                 ('Dataset utama sudah disaring untuk 10 fitur core. Kualitas posisi gaze diperiksa terpisah; '
                 'jumlah baris yang besar tidak memastikan pemisahan Low/Medium/High akan kuat.'), '',
                 '| Window | Low | Medium | High | Rasio terbesar/terkecil | Gaze tidak layak pada curated |',
                 '|---|---:|---:|---:|---:|---:|']
    for s, (rows, _, _) in results.items():
        frame = pd.DataFrame(rows)
        selected = frame[frame.curation_selected]
        c = counts[f'window_{s}s_step_5s']['good_acceptable']
        ratio = max(c[k] for k in ('Low', 'Medium', 'High'))/min(c[k] for k in ('Low', 'Medium', 'High')) if min(c[k] for k in ('Low', 'Medium', 'High')) else None
        ratio_text = f'{ratio:.3f}' if ratio else 'tidak terdefinisi'
        bad = int((~features.spatial_mask(selected, config, checks)).sum())
        condition.append(f'| {s}s | {c["Low"]} | {c["Medium"]} | {c["High"]} | {ratio_text} | {bad} |')
    condition += ['', '## Kelayakan kelompok fitur', '',
                  ('Kegagalan gaze masih ada pada rekaman asal. Nilainya tidak dianggap telah diperbaiki; penggunaan spasialnya dibatasi '
                  'dan nilai curated dimasking. Core tidak memakai gaze, head pose, atau eye-state. '
                  'Seluruh predictor core pada curated tersedia tanpa imputasi. Perbedaan level sensor antarsesi tetap ada dan harus '
                  'ditangani lewat preprocessing Train atau baseline saat pemodelan.'), '',
                  '## Validasi sumber gaze', '', '| Rekaman | Lolos asli | Terverifikasi untuk profil spasial | Alasan |',
                  '|---|---|---|---|']
    for report in results[25][2]:
        check = checks[report['session_id']]
        condition.append(f'| {report["session_folder"]} | {report["gaze_validation_passed"]} | {check["verified"]} | {check["reason"]} |')
    condition += ['', '## Distribusi sumber dan fase', '',
                  ('class_distribution.csv memuat jumlah dan durasi unik per kelas. counts_by_session_label_phase_tier.csv '
                  'memuat distribusi setiap rekaman/fase. Rating, feedback, dan rest yang diberi Low tetap mengikuti protokol v2; '
                  'fase tidak dimasukkan sebagai predictor. Window overlap tidak dihitung sebagai pengamatan independen.'), '',
                  '## Pembaruan berikutnya', '',
                  ('Tambahkan rekaman lengkap ke sessions/saved_session, pastikan ledger dan kalibrasi tersedia, lalu jalankan builder '
                  'dengan --replace. Kebijakan feature-aware diwarisi dari manifest aktif. Jangan menggabungkan CSV dengan concat manual. '
                  'Timestamp baru yang ambigu harus diaudit secara eksplisit, bukan disesuaikan agar unik. '
                  'Periksa jumlah kelas, kontribusi sumber, dan profil spasial setelah build sebelum memulai eksperimen.'), '']
    if operator_guide:
        condition += [operator_guide]
    (destination/'DATASET_CONDITION_REPORT.md').write_text('\n'.join(condition).rstrip()+'\n', encoding='utf-8')
