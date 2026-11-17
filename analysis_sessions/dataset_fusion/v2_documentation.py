"""Generate team-facing dataset documentation from the published counts."""

import pandas as pd


def build_update_summary(results, counts, inventory, previous=None, *, policy='window_quality_only'):
    from .build_dataset_v2 import unique_duration

    reports = results[25][2]
    previous = previous or {}
    old_ids = set(previous.get('selected_session_ids', []))
    old_folders = set(previous.get('selected_folders', []))
    introduced = [r for r in reports if (r['session_id'] not in old_ids if old_ids
                  else r['session_folder'] not in old_folders)]
    summary = {
        'previous_build_present': bool(previous),
        'accepted_recordings': len(reports),
        'accepted_people': len({r['participant_id'].casefold() for r in reports}),
        'additional_session_ids': sorted(r['session_id'] for r in introduced),
        'excluded_sessions': [r for r in inventory if r['reason'].startswith('excluded_session:')],
        'windows': {}, 'introduced_recordings': [],
    }
    for seconds in (15, 25, 45):
        name = f'window_{seconds}s_step_5s'
        after = counts[name]['good_acceptable']
        before = previous.get('datasets', {}).get(name, {}).get('good_acceptable')
        summary['windows'][str(seconds)] = {
            'before': before, 'after': after,
            'added': {k: after[k] - (before[k] if before else 0) for k in ('Low', 'Medium', 'High', 'windows')},
            'largest_smallest_ratio': max(after[k] for k in ('Low', 'Medium', 'High')) /
                min(after[k] for k in ('Low', 'Medium', 'High')) if min(after[k] for k in ('Low', 'Medium', 'High')) else None,
        }
    for report in introduced:
        record = {k: report[k] for k in ('session_id', 'session_folder', 'participant_id')}
        record['windows'] = {}
        for seconds, (rows, _excluded, _reports) in results.items():
            frame = pd.DataFrame(rows)
            mask = frame.quality_tier.isin(('good', 'acceptable')) if len(frame) else pd.Series(dtype=bool)
            if policy != 'window_quality_only' and len(frame):
                from .v2_replacement import boolean
                mask &= boolean(frame.curation_selected)
            selected = frame.loc[frame.session_id.eq(report['session_id']) & mask] if len(frame) else frame
            classes = selected.cognitive_load_label.value_counts().to_dict() if len(selected) else {}
            record['windows'][str(seconds)] = {
                'Low': int(classes.get('Low', 0)), 'Medium': int(classes.get('Medium', 0)),
                'High': int(classes.get('High', 0)), 'windows': len(selected),
                'unique_minutes': {label: round(unique_duration(part)/60, 4)
                    for label, part in selected.groupby('cognitive_load_label')} if len(selected) else {},
            }
        summary['introduced_recordings'].append(record)
    return summary


def update_sections(summary):
    lines = ['## Pembaruan sumber pada build ini', '',
             ('Rekaman tanpa penanda aktivitas tidak diberi label Low secara manual. '
             'Eksklusi yang dikonfigurasi memakai session ID sehingga perubahan nama folder tidak mengubah keputusan.'), '']
    if summary['excluded_sessions']:
        lines += ['| Folder yang tidak dipakai | Session ID | Alasan |', '|---|---|---|']
        lines += [f'| {r["folder"]} | {r["session_id"]} | {r["reason"]} |' for r in summary['excluded_sessions']]
        lines += ['', 'Sumber yang dikecualikan tetap disimpan dan di-hash. Tidak ada event atau ID soal yang dibuat untuk melengkapinya.', '']
    if summary['previous_build_present']:
        lines += ['### Perubahan good_acceptable', '',
                  '| Window | Low sebelum | Medium sebelum | High sebelum | Tambahan Low | Tambahan Medium | Tambahan High | Total sesudah | Rasio terbesar/terkecil |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for seconds, values in summary['windows'].items():
            before, after, added = values['before'], values['after'], values['added']
            ratio = values['largest_smallest_ratio']
            ratio_text = f'{ratio:.3f}' if ratio is not None else 'tidak terdefinisi'
            lines.append(f'| {seconds}s | {before["Low"]} | {before["Medium"]} | {before["High"]} | '
                         f'{added["Low"]} | {added["Medium"]} | {added["High"]} | {after["windows"]} | {ratio_text} |')
        lines += ['', ('Rasio tersebut menjelaskan proporsi, bukan ambang kecukupan statistik. '
                  + ('Delta adalah perubahan bersih; penggantian satu-banding-satu tidak menambah jumlah kelas. '
                     'Lihat bagian penggantian selektif untuk kontribusi setiap sumber tambahan dan row lama yang diganti.'
                     if summary.get('replacement_policy') else
                     'Builder mempertahankan semua window yang lolos, tanpa kuota kelas atau pemotongan manual.')), '']
    if summary['introduced_recordings']:
        lines += ['### Kontribusi rekaman yang belum ada pada build sebelumnya', '',
                  '| Window | Rekaman | Low | Medium | High | Durasi unik Low (menit) | Durasi unik Medium (menit) |',
                  '|---|---|---:|---:|---:|---:|---:|']
        for record in summary['introduced_recordings']:
            for seconds, count in record['windows'].items():
                minutes = count['unique_minutes']
                lines.append(f'| {seconds}s | {record["session_folder"]} | {count["Low"]} | {count["Medium"]} | '
                             f'{count["High"]} | {minutes.get("Low", 0):.2f} | {minutes.get("Medium", 0):.2f} |')
        lines += ['', ('Durasi unik merupakan gabungan interval yang diterima, bukan jumlah row dikali panjang window. '
                  'Satu sesi dapat menyumbang jumlah berbeda pada setiap panjang window. Rincian identitas dan delta tersimpan pada update_summary di manifest.'), '']
    return lines


def report_sections(results, audits, config, update_summary=None):
    replaced = bool(update_summary and update_summary.get('replacement_policy'))
    reports = results[25][2]
    recordings = len(reports)
    people = len({r['participant_id'].casefold() for r in reports})
    passed = sum(r['gaze_validation_passed'] is True for r in reports)
    failed = sum(r['gaze_validation_passed'] is False for r in reports)
    baseline = sum(r['baseline_usable'] for r in reports)
    sampled = sum(v['sampled_windows'] for a in audits for v in a['independent_sensor_check'].values())
    lines = [
        '## Proses pembuatan dan pembaruan', '',
        (f'Build ini membaca {recordings} rekaman dari {people} orang pada sessions/saved_session. '
        'Inventory memilih satu salinan acquisition dan menyatukan identitas orang yang mempunyai beberapa rekaman. '
        'Sumber yang diterima dan dikecualikan dicatat pada inventory serta manifest.'), '',
        '1. Builder memeriksa identitas sensor/ledger, readiness kedua kalibrasi, timestamp, dan grid HRV sumber.',
        '2. Builder mengolah RR mentah menurut urutan paket dan waktu ketersediaan, lalu menghitung ulang HRV pada tiga panjang window.',
        '3. Builder menggabungkan frame eye pada interval absolut yang sama dan menentukan target dari seluruh bagian fase di dalam window.',
        ('4. Builder menghitung tier sensor, kemudian menerapkan seleksi penggantian yang tercatat pada referensi dan pasangan window.'
         if replaced else '4. Builder menghitung tier sensor dan membentuk tiga view tanpa membatasi nama peserta atau kategori penilaian tim lama.'),
        '5. Builder memeriksa label, nilai sensor, subset, jumlah data, serta checksum sebelum mengganti bundle aktif.', '',
        (('Admission penggantian mempertahankan jumlah kelas referensi; tidak semua kandidat yang lolos tier masuk curated. '
          'Tier asli, data mentah, dan arsip v1 tetap utuh. Aturan dan pasangan ada pada replacement_selection.json dan replacement_pairs.csv.'
          if replaced else 'Pembaruan admission memperluas cakupan window yang sudah memenuhi ambang sensor. '
        'Kita tidak menambah baris sintetis, mengubah label untuk menyeimbangkan kelas, atau melonggarkan ambang kualitas. '
        'Data mentah dan arsip v1 tetap utuh. Manifest mencatat admission_policy=window_quality_only; '
        'team_review_group merupakan metadata historis.')), '',
        '## Ambang kualitas window', '',
        '| Kriteria | Good | Acceptable |', '|---|---:|---:|',
        '| HRV | Ready, RMSSD/SDNN finite dan nonnegatif | Sama |',
    ]
    labels = [('min_eye_temporal_coverage', 'Coverage temporal eye, bin 100 ms'),
              ('min_face_ear_ratio', 'Face dan EAR valid'), ('min_valid_gaze_ratio', 'Gaze tersedia/valid menurut recorder'),
              ('min_iris_available_ratio', 'Iris valid, nonblinking'), ('max_clipping_ratio', 'Clipping pada gaze valid')]
    for key, label in labels:
        sign = '≤' if key.startswith('max_') else '≥'
        good, acceptable = (config['quality'][tier][key]*100 for tier in ('good', 'acceptable'))
        lines.append(f'| {label} | {sign}{good:g}% | {sign}{acceptable:g}% |')
    lines += [
        '', ('Good/acceptable adalah tier kelengkapan/clipping pada window. '
        'Hasil validasi spasial gaze tetap merupakan pemeriksaan terpisah dan tidak berubah menjadi lolos karena sebuah window bertier good.'), '',
        '## Kondisi sensor dan nilai hilang', '',
        (f'Dari {recordings} rekaman sumber, validasi gaze asli mencatat {passed} lolos, {failed} gagal, '
        f'dan {recordings-passed-failed} tanpa keputusan. Baseline HRV layak pada {baseline} rekaman. '
        'Angka ini mencakup semua sumber, bukan hanya rekaman yang mempunyai window curated.'), '',
        ('Kegagalan validasi gaze tidak otomatis membuktikan EAR, blink, atau HRV pada rekaman yang sama tidak dapat digunakan. '
        'Kita mempertahankan hasil pengukuran serta status validasi aslinya; pemilihan kelompok fitur merupakan tahap pemodelan.'), '',
        '- Rasio RMSSD kosong ketika baseline tidak ready, tidak positif, atau coverage di bawah ambang; HRV absolut tetap tersedia bila window-nya layak.',
        '- Blink kosong ketika counter terputus/tidak diketahui atau tidak ada frame. Nol berarti tidak ada penambahan kedipan yang teramati.',
        '- Estimasi iris memerlukan frame valid dan tidak blinking; perubahan iris juga memerlukan baseline iris ready.',
        '- all_data dapat memuat HRV tidak layak atau fitur eye kosong. Kedua view curated tetap memerlukan tier sensor yang lolos.', '',
        '## Fitur dan kolom yang dipakai', '',
        ('Setiap CSV menyediakan 26 fitur sensor: tiga HRV, dua blink, empat EAR, empat iris, empat head pose, '
        'empat gaze, dan lima proporsi eye-state. feature_columns.json menetapkan allowlist dan cognitive_load_label sebagai target. '
        'data_dictionary.csv menjelaskan satuan dan definisi tiap kolom.'), '',
        ('RMSSD/SDNN memakai milidetik; blink rate memakai kejadian per menit. EAR serta ukuran iris memakai rasio; '
        'ukuran iris adalah estimasi landmark relatif terhadap lebar mata, bukan diameter pupil. '
        'Head pose memakai derajat; koordinat gaze dinormalisasi dengan ukuran layar dan dapat berada di luar rentang 0–1. '
        'Eye-state adalah kategori keluaran algoritma, bukan anotasi manual gerakan mata.'), '',
        ('Identitas, difficulty, fase, set/soal, waktu, dan kualitas tetap metadata. '
        'Label v2 harus dibaca dari cognitive_load_label, bukan dibuat kembali dari difficulty. '
        'Builder tidak melakukan imputasi, scaling, seleksi fitur, history, SMOTE, atau training.'), '',
        '## Pemeriksaan build', '',
        (f'Pemeriksaan pembanding memeriksa {sampled} window sensor pada ketiga panjang window, '
        'termasuk sampel awal/tengah/akhir dan perubahan status ready/gap/recovery bila tersedia. '
        'Perhitungan pembanding memakai RR hasil preprocessing yang tersedia sebelum endpoint dan agregasi NumPy atas frame eye yang telah diperiksa validitasnya. '
        'Pemeriksaan ini menguji penggabungan dan rumus, bukan memvalidasi algoritma pengukuran gaze terhadap ground truth baru.'), '',
        ('Validasi seluruh row memeriksa identitas/duplikasi, zona waktu dan durasi, grid 5 detik, target/fase, '
        'nilai infinity/negatif yang tidak sah, HRV readiness, baseline ratio, blink rate, dan jumlah proporsi eye-state. '
        'Setiap view harus memenuhi good_only ⊆ good_acceptable ⊆ all_data dengan nilai identik pada window yang sama. '
        'Checksum dan jumlah data diverifikasi sebelum dan setelah publikasi. Rincian mesin tersimpan pada manifest.json dan replay_validation.json.'), '',
        '## Menambahkan hasil pengambilan berikutnya', '',
        ('1. Tambahkan folder sesi lengkap ke sessions/saved_session: eye_tracking.csv, hrv.csv, rr_raw.csv, task_events.csv, '
        'serta file readiness/ringkasan kalibrasi yang digunakan builder. Jangan menimpa rekaman lama.'),
        ('2. Pastikan ID sensor dan ledger konsisten. Konfirmasikan identitas lintas sesi pada inventory; '
        'rekaman ulang orang yang sama harus memakai grup orang yang sama. Nama folder yang berubah tidak boleh membuat acquisition lama dihitung dua kali.'),
        ('3. Cohort penggantian saat ini dibekukan. --replace mengulang seleksi terhadap referensi awal yang sama; '
          'sumber tambahan di luar cohort menghentikan build sampai rencana seleksi diperbarui secara eksplisit.'
          if replaced else '3. Sumber baru tidak perlu masuk daftar proposed_good/questionable. Builder menilai setiap window memakai aturan sensor yang sama. '
        'Jika operator memakai alias baru untuk orang lama, perbarui pemetaan identitas yang dapat dikonfirmasi sebelum build.'),
        ('Sesi tanpa ledger harus dipulihkan penandanya atau dikecualikan secara eksplisit melalui session_exclusions; '
        'jangan membuat event atau label pengganti. Sumber bermasalah yang tidak dikecualikan menghentikan publikasi.'),
        '4. Jalankan tes, lalu rebuild dari root repository dengan perintah berikut:', '',
        '```powershell',
        'python -m unittest discover -s analysis_sessions/dataset_fusion -p "test_*.py"',
        'python analysis_sessions/dataset_fusion/build_dataset_v2.py --replace',
        '```', '',
        ('5. Periksa status berhasil, distribusi kelas, kualitas, seleksi dan alasan exclusion. Salin bundle beserta manifest dan allowlist bila dibutuhkan. '
        'Snapshot input pemodelan lama tidak berubah otomatis; evaluasi pada bundle baru memerlukan release/run baru.'), '',
        ('Builder membuat staging sementara dan menahan bundle lama sebagai rollback selama pemasangan. '
        'Setelah verifikasi berhasil, builder menghapus rollback; tidak ada arsip dataset permanen tambahan. '
        'Jika pemulihan juga gagal, builder menampilkan lokasi rollback dan mempertahankannya untuk pemulihan operator.'), '',
        '## Hubungan dengan hasil pemodelan', '',
        ('Hasil batch-00/batch-01 tetap merujuk hash dataset saat masing-masing training. '
        'Pembaruan CSV ini tidak melatih ulang model atau mengganti skor lama. '
        'Admission berbasis kualitas tetap berlaku setelah sumber tambahan digabung; kenaikan skor model belum diuji pada bundle ini. '
        'Tiga window good_acceptable adalah kandidat utama untuk pemodelan berikutnya, sedangkan v1 menjadi referensi historis. '
        'Ringkasannya tersedia pada [README output](../README.md).'), '',
    ]
    return (update_sections(update_summary) if update_summary else []) + lines


def render_output_index(archive_csv, counts, update_summary=None):
    replaced = bool(update_summary and update_summary.get('replacement_policy'))
    old = pd.read_csv(archive_csv)
    target = old.difficulty_level.map({'easy': 'Low', 'medium': 'Medium', 'hard': 'High'})
    if target.isna().any():
        raise ValueError('Invalid historical target while generating output index.')
    lines = [
        '# Pilihan Dataset Cognitive Load', '',
        ('Empat CSV berikut menjadi rujukan utama tim: tiga kandidat v2 untuk pekerjaan berikutnya dan satu referensi historis v1. '
        'Ketiga window v2 berasal dari sumber yang sama dan banyak intervalnya overlap; jangan menumpuk ketiganya sebagai dataset gabungan.'), '',
        '| Dataset | Window / step | Low | Medium | High | Total row | Orang | Peran |',
        '|---|---|---:|---:|---:|---:|---:|---|',
        (f'| [v1 good_acceptable](v1_active_task/good_acceptable.csv) | 25 / 5 detik | '
        f'{int(target.eq("Low").sum())} | {int(target.eq("Medium").sum())} | {int(target.eq("High").sum())} | '
        f'{len(old)} | {old.participant_id.nunique()} | Referensi dataset lama |'),
    ]
    for seconds in (15, 25, 45):
        variant = f'window_{seconds}s_step_5s'
        c = counts[variant]['good_acceptable']
        lines.append(f'| [v2 good_acceptable {seconds}s](v2_session_phase/{variant}/good_acceptable.csv) | '
                     f'{seconds} / 5 detik | {c["Low"]} | {c["Medium"]} | {c["High"]} | {c["windows"]} | '
                     f'{c["participants"]} | Kandidat dataset aktif |')
    lines += [
        '', '## Perbedaan v1 dan v2', '',
        '| Bagian | v1_active_task | v2_session_phase |', '|---|---|---|',
        '| Cakupan target | Pengerjaan aktif easy/medium/hard | Pengerjaan aktif dan fase sesi sesuai protokol v2 |',
        '| Low | Active easy | Active easy/practice/demo, pre-task setelah kalibrasi, feedback, rating, rest, post-task/interrupted yang terlacak |',
        '| Medium / High | Active medium / hard | Active medium / hard |',
        '| Target di CSV | difficulty_level, dipetakan easy→Low, medium→Medium, hard→High | cognitive_load_label; difficulty tetap metadata asal |',
        '| Window | 25 detik, fitur HRV dari cache sumber | HRV dihitung ulang dari RR mentah untuk 15/25/45 detik; eye mengikuti batas yang sama |',
        ('| Admission curated | Tier sensor dan daftar review tim lama | Tier sensor dan penggantian kualitas satu-banding-satu yang tercatat |'
         if replaced else '| Admission curated | Tier sensor dan daftar review tim lama | Tier sensor saja, tanpa pembatasan nama |'),
        '| Status | Arsip tetap, tidak ditimpa | Bundle aktif, dapat diperbarui dengan --replace |', '',
        ('V2 hanya menerima window dengan satu target sepanjang interval. Easy + rating dapat tetap Low; '
        'medium + rating atau hard + rest dikeluarkan. V2 juga menerapkan aturan endpoint RR yang konsisten, '
        'sehingga perubahan terhadap v1 tidak hanya berupa tambahan baris Low.'), '',
        ('Window panjang mencakup rentang waktu lebih besar, tetapi lebih mudah melintasi perubahan target. '
        'Karena itu jumlah row dan bagian rekaman yang lolos berbeda antarwindow. '
        'Step 5 detik menambah overlap; jumlah row bukan jumlah pengamatan independen.'), '',
        '## View lain yang tetap tersedia', '',
        (('Pada bundle penggantian, good_only memuat good terpilih dan good_acceptable memuat good/acceptable terpilih. '
          'Status curation_selected dan curation_reason menjelaskan keputusan terpisah dari tier sensor, '
          if replaced else 'Pada setiap panjang window v2, good_only memuat tier good, good_acceptable memuat tier good/acceptable, ')
        + 'dan all_data memuat seluruh window berlabel termasuk tier poor. '
        'Gunakan good_only untuk pembanding kualitas dan all_data untuk audit cakupan; '
        'jangan menganggap seluruh all_data layak menjadi input sensor tanpa memeriksa kualitasnya.'), '',
        ('Good/acceptable menilai kelengkapan/clipping, bukan jaminan akurasi gaze. '
        'Baca [laporan v2](v2_session_phase/QUALITY_REPORT.md) untuk proses, fitur, kondisi sensor, dan panduan menambah rekaman.'), '',
        '## Pembaruan dan pemodelan', '',
        ('Builder memperbarui tabel ini dari jumlah hasil build setelah mengganti bundle v2. '
        'Simpan metadata identitas untuk deduplikasi/grouping, bukan sebagai predictor. '
        'Gunakan allowlist fitur dan target sesuai versi dataset. '
        'Model lama tetap merujuk snapshot/hash saat training; menimpa v2 tidak memperbarui hasil eksperimen yang sudah selesai.'), '',
    ]
    if update_summary:
        lines += ['', '## Sumber pada pembaruan terakhir', '',
                  (f'Builder menerima {update_summary["accepted_recordings"]} rekaman dari '
                  f'{update_summary["accepted_people"]} orang. '
                  f'{len(update_summary["additional_session_ids"])} rekaman belum ada pada build sebelumnya. '
                  + ('Penggantian selektif mempertahankan jumlah kelas awal; tidak seluruh kandidat tambahan dimasukkan ke curated. '
                     'Tahap baru tidak mengubah hasil tahap sebelumnya. Window bertimestamp ambigu dikeluarkan sebelum pembentukan view.'
                     if replaced else 'Semua window yang lolos dipertahankan; tidak ada kuota tambahan per kelas atau window.')), '']
        for record in update_summary['excluded_sessions']:
            lines.append(f'- {record["folder"]} tidak digunakan: {record["reason"]}')
        lines += ['', ('Kontribusi per rekaman, perubahan jumlah kelas, durasi unik, dan batas kualitas dijelaskan pada '
                  '[QUALITY_REPORT v2](v2_session_phase/QUALITY_REPORT.md). '
                  'Pembaruan dataset tidak mengesahkan release pemodelan atau menjalankan training.'), '']
    return '\n'.join(lines)
