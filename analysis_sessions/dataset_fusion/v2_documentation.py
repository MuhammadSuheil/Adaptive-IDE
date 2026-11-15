"""Generate team-facing dataset documentation from the published counts."""

import pandas as pd


def report_sections(results, audits, config):
    reports = results[25][2]
    recordings = len(reports)
    people = len({r['participant_id'].casefold() for r in reports})
    passed = sum(r['gaze_validation_passed'] is True for r in reports)
    failed = sum(r['gaze_validation_passed'] is False for r in reports)
    baseline = sum(r['baseline_usable'] for r in reports)
    sampled = sum(v['sampled_windows'] for a in audits for v in a['independent_sensor_check'].values())
    lines = [
        '## Proses pembuatan dan pembaruan', '',
        f'Build ini membaca {recordings} rekaman dari {people} orang pada sessions/saved_session. '
        'Inventory memilih satu salinan acquisition dan menyatukan identitas orang yang mempunyai beberapa rekaman. '
        'Build ini belum memasukkan pengambilan Low tambahan yang direncanakan.', '',
        '1. Builder memeriksa identitas sensor/ledger, readiness kedua kalibrasi, timestamp, dan grid HRV sumber.',
        '2. Builder mengolah RR mentah menurut urutan paket dan waktu ketersediaan, lalu menghitung ulang HRV pada tiga panjang window.',
        '3. Builder menggabungkan frame eye pada interval absolut yang sama dan menentukan target dari seluruh bagian fase di dalam window.',
        '4. Builder menghitung tier sensor dan membentuk tiga view tanpa membatasi nama peserta atau kategori penilaian tim lama.',
        '5. Builder memeriksa label, nilai sensor, subset, jumlah data, serta checksum sebelum mengganti bundle aktif.', '',
        'Pembaruan admission memperluas cakupan window yang sudah memenuhi ambang sensor. '
        'Kita tidak menambah baris sintetis, mengubah label untuk menyeimbangkan kelas, atau melonggarkan ambang kualitas. '
        'Data mentah dan arsip v1 tetap utuh. Manifest mencatat admission_policy=window_quality_only; '
        'team_review_group merupakan metadata historis.', '',
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
        '', 'Good/acceptable adalah tier kelengkapan/clipping pada window. '
        'Hasil validasi spasial gaze tetap merupakan pemeriksaan terpisah dan tidak berubah menjadi lolos karena sebuah window bertier good.', '',
        '## Kondisi sensor dan nilai hilang', '',
        f'Dari {recordings} rekaman sumber, validasi gaze asli mencatat {passed} lolos, {failed} gagal, '
        f'dan {recordings-passed-failed} tanpa keputusan. Baseline HRV layak pada {baseline} rekaman. '
        'Angka ini mencakup semua sumber, bukan hanya rekaman yang mempunyai window curated.', '',
        'Kegagalan validasi gaze tidak otomatis membuktikan EAR, blink, atau HRV pada rekaman yang sama tidak dapat digunakan. '
        'Kita mempertahankan hasil pengukuran serta status validasi aslinya; pemilihan kelompok fitur merupakan tahap pemodelan.', '',
        '- Rasio RMSSD kosong ketika baseline tidak ready, tidak positif, atau coverage di bawah ambang; HRV absolut tetap tersedia bila window-nya layak.',
        '- Blink kosong ketika counter terputus/tidak diketahui atau tidak ada frame. Nol berarti tidak ada penambahan kedipan yang teramati.',
        '- Estimasi iris memerlukan frame valid dan tidak blinking; perubahan iris juga memerlukan baseline iris ready.',
        '- all_data dapat memuat HRV tidak layak atau fitur eye kosong. Kedua view curated tetap memerlukan tier sensor yang lolos.', '',
        '## Fitur dan kolom yang dipakai', '',
        'Setiap CSV menyediakan 26 fitur sensor: tiga HRV, dua blink, empat EAR, empat iris, empat head pose, '
        'empat gaze, dan lima proporsi eye-state. feature_columns.json menetapkan allowlist dan cognitive_load_label sebagai target. '
        'data_dictionary.csv menjelaskan satuan dan definisi tiap kolom.', '',
        'RMSSD/SDNN memakai milidetik; blink rate memakai kejadian per menit. EAR serta ukuran iris memakai rasio; '
        'ukuran iris adalah estimasi landmark relatif terhadap lebar mata, bukan diameter pupil. '
        'Head pose memakai derajat; koordinat gaze dinormalisasi dengan ukuran layar dan dapat berada di luar rentang 0–1. '
        'Eye-state adalah kategori keluaran algoritma, bukan anotasi manual gerakan mata.', '',
        'Identitas, difficulty, fase, set/soal, waktu, dan kualitas tetap metadata. '
        'Label v2 harus dibaca dari cognitive_load_label, bukan dibuat kembali dari difficulty. '
        'Builder tidak melakukan imputasi, scaling, seleksi fitur, history, SMOTE, atau training.', '',
        '## Pemeriksaan build', '',
        f'Pemeriksaan pembanding memeriksa {sampled} window sensor pada ketiga panjang window, '
        'termasuk sampel awal/tengah/akhir dan perubahan status ready/gap/recovery bila tersedia. '
        'Perhitungan pembanding memakai RR hasil preprocessing yang tersedia sebelum endpoint dan agregasi NumPy atas frame eye yang telah diperiksa validitasnya. '
        'Pemeriksaan ini menguji penggabungan dan rumus, bukan memvalidasi algoritma pengukuran gaze terhadap ground truth baru.', '',
        'Validasi seluruh row memeriksa identitas/duplikasi, zona waktu dan durasi, grid 5 detik, target/fase, '
        'nilai infinity/negatif yang tidak sah, HRV readiness, baseline ratio, blink rate, dan jumlah proporsi eye-state. '
        'Setiap view harus memenuhi good_only ⊆ good_acceptable ⊆ all_data dengan nilai identik pada window yang sama. '
        'Checksum dan jumlah data diverifikasi sebelum dan setelah publikasi. Rincian mesin tersimpan pada manifest.json dan replay_validation.json.', '',
        '## Menambahkan hasil pengambilan berikutnya', '',
        '1. Tambahkan folder sesi lengkap ke sessions/saved_session: eye_tracking.csv, hrv.csv, rr_raw.csv, task_events.csv, '
        'serta file readiness/ringkasan kalibrasi yang digunakan builder. Jangan menimpa rekaman lama.',
        '2. Pastikan ID sensor dan ledger konsisten. Konfirmasikan identitas lintas sesi pada inventory; '
        'rekaman ulang orang yang sama harus memakai grup orang yang sama. Nama folder yang berubah tidak boleh membuat acquisition lama dihitung dua kali.',
        '3. Sumber baru tidak perlu masuk daftar proposed_good/questionable. Builder menilai setiap window memakai aturan sensor yang sama. '
        'Jika operator memakai alias baru untuk orang lama, perbarui pemetaan identitas yang dapat dikonfirmasi sebelum build.',
        '4. Jalankan tes, lalu rebuild dari root repository dengan perintah berikut:', '',
        '```powershell',
        'python -m unittest discover -s analysis_sessions/dataset_fusion -p "test_*.py"',
        'python analysis_sessions/dataset_fusion/build_dataset_v2.py --replace',
        '```', '',
        '5. Periksa status berhasil, distribusi kelas, kualitas, dan alasan exclusion. Salin bundle beserta manifest dan allowlist ke Drive bila dibutuhkan. '
        'Setelah data tambahan lengkap, tetapkan input batch-02 dari bundle terbaru.', '',
        'Builder membuat staging sementara dan menahan bundle lama sebagai rollback selama pemasangan. '
        'Setelah verifikasi berhasil, builder menghapus rollback; tidak ada arsip dataset permanen tambahan. '
        'Jika pemulihan juga gagal, builder menampilkan lokasi rollback dan mempertahankannya untuk pemulihan operator.', '',
        '## Hubungan dengan hasil pemodelan', '',
        'Hasil batch-00/batch-01 tetap merujuk hash dataset saat masing-masing training. '
        'Pembaruan CSV ini tidak melatih ulang model atau mengganti skor lama. '
        'Angka data bertambah karena admission berbasis kualitas; kenaikan skor model belum diuji pada bundle hasil refresh ini. '
        'Tiga window good_acceptable adalah kandidat utama untuk pemodelan berikutnya, sedangkan v1 menjadi referensi historis. '
        'Ringkasannya tersedia pada [README output](../README.md).', '',
    ]
    return lines


def render_output_index(archive_csv, counts):
    old = pd.read_csv(archive_csv)
    target = old.difficulty_level.map({'easy': 'Low', 'medium': 'Medium', 'hard': 'High'})
    if target.isna().any():
        raise ValueError('Invalid historical target while generating output index.')
    lines = [
        '# Pilihan Dataset Cognitive Load', '',
        'Empat CSV berikut menjadi rujukan utama tim: tiga kandidat v2 untuk pekerjaan berikutnya dan satu referensi historis v1. '
        'Ketiga window v2 berasal dari sumber yang sama dan banyak intervalnya overlap; jangan menumpuk ketiganya sebagai dataset gabungan.', '',
        '| Dataset | Window / step | Low | Medium | High | Total row | Orang | Peran |',
        '|---|---|---:|---:|---:|---:|---:|---|',
        f'| [v1 good_acceptable](v1_active_task/good_acceptable.csv) | 25 / 5 detik | '
        f'{int(target.eq("Low").sum())} | {int(target.eq("Medium").sum())} | {int(target.eq("High").sum())} | '
        f'{len(old)} | {old.participant_id.nunique()} | Referensi dataset lama |',
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
        '| Admission curated | Tier sensor dan daftar review tim lama | Tier sensor saja, tanpa pembatasan nama |',
        '| Status | Arsip tetap, tidak ditimpa | Bundle aktif, dapat diperbarui dengan --replace |', '',
        'V2 hanya menerima window dengan satu target sepanjang interval. Easy + rating dapat tetap Low; '
        'medium + rating atau hard + rest dikeluarkan. V2 juga menerapkan aturan endpoint RR yang konsisten, '
        'sehingga perubahan terhadap v1 tidak hanya berupa tambahan baris Low.', '',
        'Window panjang mencakup rentang waktu lebih besar, tetapi lebih mudah melintasi perubahan target. '
        'Karena itu jumlah row dan bagian rekaman yang lolos berbeda antarwindow. '
        'Step 5 detik menambah overlap; jumlah row bukan jumlah pengamatan independen.', '',
        '## View lain yang tetap tersedia', '',
        'Pada setiap panjang window v2, good_only memuat tier good, good_acceptable memuat tier good/acceptable, '
        'dan all_data memuat seluruh window berlabel termasuk tier poor. '
        'Gunakan good_only untuk pembanding kualitas dan all_data untuk audit cakupan; '
        'jangan menganggap seluruh all_data layak menjadi input sensor tanpa memeriksa kualitasnya.', '',
        'Good/acceptable menilai kelengkapan/clipping, bukan jaminan akurasi gaze. '
        'Baca [laporan v2](v2_session_phase/QUALITY_REPORT.md) untuk proses, fitur, kondisi sensor, dan panduan menambah rekaman.', '',
        '## Pembaruan dan pemodelan', '',
        'Builder memperbarui tabel ini dari jumlah hasil build setelah mengganti bundle v2. '
        'Simpan metadata identitas untuk deduplikasi/grouping, bukan sebagai predictor. '
        'Gunakan allowlist fitur dan target sesuai versi dataset. '
        'Model lama tetap merujuk snapshot/hash saat training; menimpa v2 tidak memperbarui hasil eksperimen yang sudah selesai.', '',
    ]
    return '\n'.join(lines)
