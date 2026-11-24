"""Documentation derived from the feature-aware sensor bundle, without model scores."""

import pandas as pd

from . import v2_features as features
from .v2_documentation import update_sections


def output_index(archive_csv, counts, *, label_policy=None):
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
              'quality_tier sekarang menyatakan kualitas inti, bukan akurasi gaze. Window dengan lonjakan RR lokal '
              'dikeluarkan dari curated tanpa mengganti nilai sensor; bukti dan aturan exclude_local_rr_jump_v1 '
              'tersimpan pada manifest dan laporan v2.'), '',
              ('Baca [QUALITY_REPORT v2](v2_session_phase/QUALITY_REPORT.md) dan '
              '[kondisi dataset](v2_session_phase/DATASET_CONDITION_REPORT.md) untuk aturan, profil, serta jumlah yang dapat dipakai.'), '',
              '## Pemodelan', '',
              ('Gunakan features pada feature_columns.json untuk input default. core_head dan validated_fusion merupakan profil tambahan '
              'dengan persyaratan baris tersendiri. Identitas, fase, waktu, soal, dan kualitas bukan predictor. '
              'Jangan mengisi gaze gagal dengan nol atau memakai semua kolom numerik.'), '',
              ('Statistik normalisasi, seleksi fitur yang dipelajari, imputasi bila diperlukan pada eksperimen lain, dan resampling '
              'dikerjakan pada Train, bukan pada seluruh CSV. Window overlap bukan observasi independen. '
              'Model, notebook, dan laporan historis tetap merujuk hash dataset saat eksperimen tersebut dijalankan.'), '']
    if label_policy:
        lines = [line.replace('Easy/practice/demo serta fase sesi Low yang terlacak',
                    'Active easy/practice/demo dan pre-task; fase opsional mengikuti tugas easy')
                 .replace('Tetap active medium dan hard', 'Active medium/hard serta fase opsional dengan konteks difficulty yang sama')
                 .replace('good_only berisi core good; good_acceptable berisi core good/acceptable.',
                    'good_acceptable diseleksi seimbang menurut balanced_primary_v1; good_only adalah subset good dari baris terpilih.')
                 for line in lines]
        lines += ['', '## Kebijakan fase dan seleksi', '',
                  'Label memakai session_phase_context_v3. Feedback/rating mengikuti tugas terkait; rest/post-task mengikuti '
                  'tugas terakhir dan hanya menjadi kandidat hingga 45 detik sejak pengerjaan berhenti. Peralihan fase tidak '
                  'mengulang batas tersebut. Interrupted, kalibrasi, konteks tidak terlacak, dan window lintas label dikeluarkan.', '',
                  'Active-task/pre-task didahulukan; fase opsional dipakai hanya untuk melengkapi kelas yang kurang. '
                  'Seleksi dilakukan terpisah per panjang window. Baris mayoritas yang tidak dipilih tetap tersimpan di all_data. '
                  'all_data tidak dipaksa seimbang; good_only tidak diseimbangkan ulang. Jangan menumpuk ketiga panjang window.', '',
                  'Rebuild: `python -m analysis_sessions.dataset_fusion.build_dataset_v2 --replace`. '
                  'Kebijakan label, kualitas, dan seleksi diwarisi dari manifest aktif.', '']
    return '\n'.join(lines)


def write_reports(destination, results, counts, profile_counts, checks, update, config, previous_policy, operator_guide,
                  rr_registry=None, pre_model_audit=None, label_policy=None, curation_summary=None):
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
    if rr_registry:
        lines += rr_quality_section(rr_registry)
    if pre_model_audit:
        lines += pre_model_section(pre_model_audit)
    lines += update_sections(update)
    if label_policy:
        lines = phase_report_text(lines)
        lines += selection_section(results, curation_summary, counts)
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
    if rr_registry:
        condition += rr_quality_section(rr_registry)
    if pre_model_audit:
        condition += pre_model_section(pre_model_audit)
    if label_policy:
        condition = phase_report_text(condition)
        condition += selection_section(results, curation_summary, counts)
    (destination/'DATASET_CONDITION_REPORT.md').write_text('\n'.join(condition).rstrip()+'\n', encoding='utf-8')


def phase_report_text(lines):
    """Keep historical report generation intact; replace the active protocol prose."""
    result = []
    for line in lines:
        if line.startswith('Target tetap cognitive_load_label.'):
            line = ('Target adalah cognitive_load_label menurut session_phase_context_v3. Active easy/practice/demo dan pre-task '
                    'setelah kalibrasi diberi Low; active medium diberi Medium dan active hard diberi High. Feedback/rating '
                    'mengikuti difficulty tugas terkait. Rest dan post-task mengikuti difficulty tugas terakhir, bukan otomatis Low '
                    'atau otomatis High. Fase-fase tersebut menjadi kandidat opsional, bukan otomatis masuk dataset utama. '
                    'Interrupted, baseline/kalibrasi, dan konteks yang tidak dapat ditelusuri dikeluarkan. Seluruh bagian window '
                    'harus mempunyai satu target; medium + rating medium boleh masuk, tetapi easy + rating medium tidak.')
        elif line.startswith('Semua window yang lolos aturan inti dipertahankan.'):
            line = ('Kelayakan sensor tetap feature_aware_quality_v1. Seleksi dataset utama memakai balanced_primary_v1: '
                    'all_data menyimpan seluruh kandidat berlabel termasuk poor dan kandidat tidak terpilih; good_acceptable '
                    'menyimpan good/acceptable yang terpilih dengan jumlah kelas sama; good_only merupakan subset good '
                    'dari good_acceptable dan tidak diseimbangkan ulang. Tidak ada pembatasan berdasarkan daftar nama.')
        line = line.replace('Nilai sensor dan target interval lama diperbandingkan saat migrasi.',
                            'Nilai sensor interval yang sama diperbandingkan saat migrasi; perubahan label dicatat terpisah.')
        line = line.replace('Rating, feedback, dan rest yang diberi Low tetap mengikuti protokol v2;',
                            'Feedback, rating, rest, dan post-task mengikuti difficulty tugas yang ditelusuri dari event;')
        line = line.replace('dataset tidak diseimbangkan dengan mengganti label atau membuang fase yang sah.',
                            'penyeimbangan memilih baris layak tanpa mengganti labelnya; kandidat tidak terpilih tetap ada di all_data.')
        result.append(line)
    result += ['', '## Batas fase nonpengerjaan', '',
               'Rest/post-task hanya menjadi kandidat dalam 45 detik sejak active-task terakhir berhenti. Batas tidak dimulai '
               'ulang ketika feedback berubah menjadi rating, rest, atau post-task. Feedback/rating mengikuti durasi fase '
               'yang tercatat dan tidak dikenai batas khusus ini. Nilai 45 detik adalah batas operasional, bukan hasil '
               'pengukuran waktu pemulihan fisiologis. Window yang melewati batas dikeluarkan seluruhnya.', '',
               'window_phase_mix menyimpan konteks sumber, waktu awal set/soal, akhir pengerjaan, dan batas grace dalam '
               'label_context. manifest.json menyimpan ledger dan readiness yang dipakai untuk merekonstruksi ulang '
               'segmen. Tidak ada timestamp, event, atau nilai sensor mentah yang diubah.', '']
    return result


def selection_section(results, summary, counts):
    from .build_dataset_v2 import unique_duration

    lines = ['', '## Seleksi dataset utama', '',
             'Basis adalah window yang hanya memuat active-task dan/atau pre-task. Setiap window yang memuat '
             'feedback/rating/rest/post-task dihitung sebagai opsional, termasuk campurannya dengan pengerjaan. '
             'T = min(jumlah basis terbesar antarkelas, jumlah total kandidat layak terkecil antarkelas). '
             'Setiap kelas memilih basis dahulu, kemudian opsional hanya untuk mencapai T. Basis mayoritas dapat '
             'diseleksi sebagian. Jika suatu kelas tidak tersedia, build dihentikan, bukan melonggarkan kualitas.', '',
             'Seleksi mendahulukan tier good kemudian acceptable. Dalam setiap tier, kuota dibagi merata sesuai kapasitas '
             'antarpeserta lalu antarsesi, dengan sisa kuota berurutan leksikografis. Tiap rekaman diambil pada indeks '
             'tengah dari interval timestamp yang dibagi merata. Tidak ada pemilihan menurut skor model atau perubahan label.', '',
             '| Window | Kelas | Basis layak | Opsional layak | Basis dipilih | Opsional dipilih | Layak tidak dipilih |',
             '|---|---|---:|---:|---:|---:|---:|']
    for seconds in (15, 25, 45):
        for label, entry in summary[f'window_{seconds}s_step_5s']['classes'].items():
            lines.append(f'| {seconds}s | {label} | {entry["base_available"]} | {entry["optional_available"]} | '
                         f'{entry["base_selected"]} | {entry["optional_selected"]} | {entry["quality_eligible_not_selected"]} |')
    lines += ['', '## Durasi unik dan fase terpilih', '',
              '| Window | Kelas | Baris | Durasi unik (menit) |', '|---|---|---:|---:|']
    for seconds, (rows, _, _) in results.items():
        selected = pd.DataFrame(rows).loc[lambda f: f.curation_selected]
        for label, part in selected.groupby('cognitive_load_label', sort=False):
            lines.append(f'| {seconds}s | {label} | {len(part)} | {unique_duration(part)/60:.2f} |')
    lines += ['', 'Jumlah row yang sama bukan jumlah pengamatan independen atau durasi unik yang harus sama. '
              'Window tetap overlap. Rincian jumlah/durasi unik setiap versi ada di class_distribution.csv.', '',
              '| Window | Kelas | Fase window | Baris terpilih |', '|---|---|---|---:|']
    for seconds, (rows, excluded, _) in results.items():
        selected = pd.DataFrame(rows).loc[lambda f: f.curation_selected]
        for (label, phase), n in selected.groupby(['cognitive_load_label', 'study_phase']).size().items():
            lines.append(f'| {seconds}s | {label} | {phase} | {n} |')
    lines += ['', '## Window yang dikeluarkan', '', '| Window | Alasan utama | Baris |', '|---|---|---:|']
    for seconds, (_, excluded, _) in results.items():
        failures = pd.DataFrame(excluded)
        if not failures.empty:
            for reason, n in failures.groupby('exclusion_reason').size().items():
                lines.append(f'| {seconds}s | {reason} | {n} |')
    lines += ['', 'excluded_windows.csv mencatat penolakan label/waktu dan timestamp ambigu. all_data menyimpan '
              'baris berlabel yang gagal kualitas atau tidak dipilih; curation_reason membedakan keduanya. '
              'curation_summary.json dan manifest menyimpan ringkasan seleksi. Riwayat label/seleksi sebelum migrasi '
              'disimpan ringkas di history. Hasil eksperimen lama tetap merujuk dataset lama dan tidak berubah.', '',
              '## Rebuild', '', '`python -m analysis_sessions.dataset_fusion.build_dataset_v2 --replace`', '',
              'Manifest aktif mewariskan kebijakan label dan seleksi ini. Rebuild menghitung ulang dari rekaman mentah, '
              'bukan concat CSV atau hanya mengganti label lama. Hash sumber dan bundle diperiksa sebelum publikasi; '
              'penggantian memakai staging dan rollback.', '']
    return lines


def rr_quality_section(registry):
    lines = ['', '## Pemeriksaan lonjakan RR sebelum pemodelan', '',
             ('Kebijakan exclude_local_rr_jump_v1 mengeluarkan window curated yang memuat RR diterima '
              'dengan deviasi lebih dari 250 ms terhadap median referensi RRPreprocessor. Riwayat normal memakai '
              'maksimum 11 RR dan minimum 5; setelah gap, referensi mengikuti kelompok recovery pada waktu rilis. '
              'Hanya beat dalam (start, end] yang sudah tersedia pada endpoint yang diperhitungkan.'), '',
             ('Aturan ini menandai ketidakpastian lokal, bukan diagnosis bahwa setiap kandidat pasti artefak. '
              'Seluruh rekaman diperiksa dengan aturan yang sama tanpa membaca label atau skor model. '
              'Nilai RR, RMSSD, SDNN, fitur eye, dan label tidak diganti. Window terdampak tetap di all_data, '
              'dengan quality_tier poor dan alasan rr_local_median_jump pada metadata yang sudah ada. '
              'Tidak ada kolom flag baru, interpolasi, winsorization, atau aktivasi filter live.'), '',
             '| Window | Window terdampak di all_data | Core layak yang dikeluarkan | Low | Medium | High |',
             '|---|---:|---:|---:|---:|---:|']
    for key, value in registry['windows'].items():
        c = value['excluded_by_class']
        lines.append(f'| {key} | {value["all_data_affected"]} | {value["core_eligible_excluded"]} | '
                     f'{c["Low"]} | {c["Medium"]} | {c["High"]} |')
    lines += ['', '| Rekaman | RR task | RR diterima pemeriksaan dasar | Kandidat lonjakan |',
              '|---|---:|---:|---:|']
    for audit in registry['sessions'].values():
        lines.append(f'| {audit["session_folder"]} | {audit["task_rr_count"]} | {audit["accepted_rr_count"]} | '
                     f'{len(audit["candidates"])} |')
    lines += ['', ('manifest.json menyimpan konfigurasi, raw_id, waktu beat/ketersediaan, median referensi, '
              'serta hash rr_raw.csv untuk audit. Pemeriksaan independen saat publikasi mencocokkan penanda '
              'window, alasan kualitas, jumlah yang dikeluarkan, dan proyeksi CSV tersaring. '
              'Rasio baseline historis tetap informasi audit dan bukan input core. '
              'Normalisasi antarsesi serta pembagian Train/validation/test tidak dilakukan di CSV.'), '']
    return lines


def pre_model_section(audits):
    lines = ['', '## Pemeriksaan dataset sebelum pemodelan', '',
             '| Window | Missing core | Infinity core | ID window duplikat | Vektor core duplikat | Fitur core konstan | Rasio baseline tidak tersedia |',
             '|---|---:|---:|---:|---:|---|---:|']
    for key, audit in audits.items():
        constant = ', '.join(audit['constant_core_features']) or 'tidak ada'
        lines.append(f'| {key} | {audit["missing_core_values"]} | {audit["infinite_core_values"]} | '
                     f'{audit["duplicate_window_ids"]} | {audit["duplicate_core_vectors"]} | {constant} | '
                     f'{audit["missing_optional_baseline_ratio"]} |')
    lines += ['', ('Fitur inti tidak diimputasi. Ketidaktersediaan rasio baseline tidak diganti dengan nol dan '
              'tidak menggugurkan core; rasio itu bukan predictor default. Vektor sama atau korelasi tinggi tidak otomatis '
              'membuktikan duplikasi akuisisi. Duplikasi sumber dan ID interval diperiksa terpisah.'), '',
              '| Window | Low active_task | Seluruh Low | Persentase Low active_task |', '|---|---:|---:|---:|']
    for key, audit in audits.items():
        percent = 100*audit['low_active_windows']/audit['low_windows'] if audit['low_windows'] else 0
        lines.append(f'| {key} | {audit["low_active_windows"]} | {audit["low_windows"]} | {percent:.2f}% |')
    lines += ['', ('Label fase tetap mengikuti protokol v2. Komposisi ini perlu dipertimbangkan saat menyusun '
              'development dan evaluasi; dataset tidak diseimbangkan dengan mengganti label atau membuang fase yang sah.'), '',
              '| Window | Fitur | Variasi antar-rerata sesi / total | Variasi antar-rerata kelas / total |',
              '|---|---|---:|---:|']
    for key, audit in audits.items():
        for name, variation in audit['variation'].items():
            session = f'{100*variation["session"]:.2f}%' if variation['session'] is not None else 'tidak terdefinisi'
            label = f'{100*variation["label"]:.2f}%' if variation['label'] is not None else 'tidak terdefinisi'
            lines.append(f'| {key} | {name} | {session} | {label} |')
    lines += ['', ('Rasio variasi adalah eta-squared deskriptif satu faktor. Kedua kolom tidak dijumlahkan dan '
              'tidak membuktikan penyebab biologis, kerusakan sensor, atau ketiadaan informasi nonlinear. '
              'Level dasar antarsesi tidak dikoreksi menggunakan seluruh CSV. Statistik preprocessing harus '
              'berasal dari Train atau baseline yang tersedia sebelum prediksi. Identitas tetap metadata, bukan predictor.'), '']
    return lines
