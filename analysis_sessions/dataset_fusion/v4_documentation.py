"""Reports for observed difficulty ratings and the portable CSV package."""

import csv
from collections import Counter

import pandas as pd

from . import build_dataset as shared
from . import v2_features as features
from . import v4_ratings as ratings

WINDOW = 'window_15s_step_5s'


def write_reports(out, counts, coverage, summary, episodes, matches, source_exclusions):
    window = out/WINDOW
    primary = coverage.loc[coverage.window_seconds.eq(15)]
    assignments = pd.read_csv(window/'window_assignments.csv')
    exclusions = pd.read_csv(window/'excluded_windows.csv')
    rated = [e for e in episodes if e[ratings.TARGET] is not None]
    without_window = [e for e in rated if e['episode_id'] not in set(assignments.episode_id)]
    too_short = sum(max((end-start for start, end in e['intervals']), default=0) < 15000 for e in without_window)
    nonrated = Counter(e['closed_by'] or 'trace_tidak_selesai' for e in episodes if e[ratings.TARGET] is None)
    statuses = summary['rated_submission_types']
    lines = [
        '# Kualitas dataset v4', '',
        ('Target `difficulty_rating` adalah penilaian kesulitan soal yang dipilih peserta pada skala 1–10. '
        'Satu rating berlaku untuk satu episode pengerjaan. Skala ini tidak menggantikan label cognitive load v2 atau v3.'), '',
        '## Sumber rating dan cakupan episode', '',
        (f'Dari {summary["rating_log_rows"]} baris di empat log CodeArrange, {len(rated)} rating cocok dengan '
        f'{len(rated)} episode dari {summary["rated_participants"]} peserta. Sebanyak {statuses.get("manual", 0)} '
        f'rating mengikuti jawaban manual benar dan {statuses.get("timeout", 0)} mengikuti timeout. '
        f'Dari {summary["episodes"]} episode yang direkonstruksi, {summary["episodes"]-len(rated)} tidak memperoleh label.'), '',
        ('Pencocokan memakai `attempt_id`, `session_id`, set, identitas soal, jenis submission, hasil, dan timestamp. '
        'Lihat `rating_matches.csv` untuk sumber file dan baris, termasuk rating yang tidak menemukan pasangan. '
        'Nama peserta atau kemiripan judul saja tidak cukup untuk memasangkan log.'), '',
        '| Keputusan pada baris log | Baris |', '|---|---:|',
    ]
    lines.extend(f'| `{key}` | {value} |' for key, value in sorted(matches.status.value_counts().items()))
    lines += ['', 'Sesi sumber tanpa rating yang cocok: '+(', '.join(summary['source_sessions_without_ratings']) or 'tidak ada')+'.', '',
              '| Penutup episode tanpa label | Episode |', '|---|---:|']
    lines.extend(f'| `{key}` | {value} |' for key, value in sorted(nonrated.items()))
    lines += ['', ('Aplikasi meminta rating setelah jawaban benar atau timeout. Soal yang dilewati, ditinggalkan, '
        'atau berakhir tanpa angka rating tidak masuk CSV berlabel. Rating yang tercatat setelah timeout '
        'tetap memakai angka pilihan peserta. `episode_ratings.csv` menyimpan episode tanpa rating untuk audit. '
        'Cakupan ini membatasi kesimpulan: dataset tidak mewakili semua soal yang sempat dibuka.'), '',
        '## Pemilihan window dan kualitas sensor', '',
        ('Satu window harus utuh dalam satu interval aktif dari episode berating. Retry dan kembali ke soal '
        'sebelum episode ditutup tetap bagian dari episode itu. Feedback, rating, istirahat, dan perpindahan soal '
        'memutus interval. Dataset tidak menyambung potongan waktu untuk memenuhi durasi window.'), '',
        ('V4 memakai pool `all_data.csv` v2 sebelum kuota kelas v2. V4 mempertahankan agregat sensor, filter RR '
        'dan eksklusi timestamp v2. Pada CSV curated, fitur gaze yang tidak lolos syarat v2 tetap kosong. '
        'Tidak ada imputasi, normalisasi global, resampling, atau pembuatan rating sintetis.'), '',
        '| Dataset 15 detik | Window | Episode | Peserta | Durasi unik (menit) |', '|---|---:|---:|---:|---:|']
    for name, data in counts.items():
        lines.append(f'| `{name}` | {data["windows"]} | {data["episodes"]} | {data["participants"]} | {data["unique_seconds"]/60:.2f} |')
    lines += ['', (f'{len(without_window)} episode berating tidak memperoleh window 15 detik dalam pool yang cocok. '
        f'Dari jumlah itu, {too_short} tidak memiliki interval aktif tunggal sepanjang 15 detik. '
        f'Sisanya tidak memperoleh window utuh pada grid 5 detik atau terkendala cakupan dan eksklusi sumber. '
        f'Filter kualitas mengeluarkan {counts["all_data"]["windows"]-counts["good_acceptable"]["windows"]} '
        f'window dari dataset utama. Sebanyak {counts["all_data"]["episodes"]-counts["good_acceptable"]["episodes"]} '
        'episode yang semula punya window tidak lagi terwakili setelah filter kualitas.'), '',
        '| Eksklusi saat mencocokkan pool 15 detik | Window |', '|---|---:|']
    lines.extend(f'| `{key}` | {value} |' for key, value in sorted(exclusions.reason.value_counts().items()))
    lines += ['', ('Eksklusi sebelum pool v2 tetap berlaku. Jumlah berikut mencakup seluruh cohort sumber, '
              'bukan hanya episode berating.'), '', '| Window | Alasan eksklusi v2 | Jumlah |', '|---|---|---:|']
    for seconds, values in source_exclusions.items():
        lines.extend(f'| {seconds}s | `{key}` | {value} |' for key, value in sorted(values.items()))
    lines += ['', '## Distribusi dataset utama', '',
        '| Rating | Episode berating | Episode layak | Peserta layak | Window layak | Durasi unik (menit) |',
        '|---:|---:|---:|---:|---:|---:|']
    for row in primary.itertuples(index=False):
        rating = row.difficulty_rating
        lines.append(f'| {rating} | {summary["episode_rating_counts"][str(rating)]} | {row.episodes} | '
                     f'{row.participants} | {row.windows} | {row.unique_seconds/60:.2f} |')
    lines += ['', ('Durasi unik menghitung gabungan rentang window per rekaman. Window dengan step 5 detik '
        'bertumpang tindih, sehingga jumlah window tidak sama dengan jumlah rating independen.'), '',
        '## Perbandingan panjang window', '',
        '| Panjang | Window layak | Episode layak | Peserta pada rating 8 |', '|---:|---:|---:|---:|']
    for seconds, rows in coverage.groupby('window_seconds'):
        people_8 = int(rows.loc[rows.difficulty_rating.eq(8), 'participants'].iloc[0])
        lines.append(f'| {seconds} detik | {rows.windows.sum()} | {rows.episodes.sum()} | {people_8} |')
    rare = primary.loc[primary.difficulty_rating.eq(8)].iloc[0]
    lines += ['', (f'Pada window 15 detik, rating 8 hanya berasal dari {rare.episodes} episode dan '
        f'{rare.participants} peserta. Nilai tersebut lebih relevan untuk merancang evaluasi daripada '
        f'jumlah {rare.windows} windownya. Varian 25/45 detik hanya dihitung sebagai perbandingan cakupan. '
        'Bundle ini tidak menerbitkan CSV sensor untuk kedua varian tersebut.'), '',
        '## Penggunaan untuk pemodelan', '',
        ('Gunakan `good_acceptable.csv` dan daftar sepuluh fitur core pada `feature_columns.json`. '
        'Identitas, waktu, soal, difficulty protokol, durasi, kualitas, dan outcome berfungsi sebagai '
        'metadata. Rancang split dan bobot dari v4; jangan menyalinnya dari v3. Pertahankan kelompok '
        'peserta dan episode saat memeriksa kebocoran antarbagian. Laporan ini tidak menetapkan split, '
        'tidak menjalankan training, dan tidak menyatakan bahwa sepuluh kelas telah memadai secara statistik.'), '']
    (out/'QUALITY_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    (out/'README.md').write_text(
        '# Dataset v4: rating kesulitan peserta\n\n'
        'Dataset utama: `window_15s_step_5s/good_acceptable.csv`. Target: `difficulty_rating` (1–10). '
        'Lihat [QUALITY_REPORT.md](QUALITY_REPORT.md) untuk cakupan, asal label, eksklusi, dan batas data.\n\n'
        'Gunakan `episode_ratings.csv` untuk bukti label, `rating_matches.csv` untuk audit log, dan '
        '`window_assignments.csv` untuk pemetaan window–episode. `manifest.json` menyimpan hash sumber '
        'dan keluaran. Distribusi 25/45 detik hanya tersedia sebagai ringkasan cakupan.\n\n'
        'Build: `python -m analysis_sessions.dataset_fusion.build_dataset_v4`.\n\n'
        'Untuk memverifikasi bundle, gunakan modul yang sama dengan `--verify <folder-v4>`. '
        'Untuk menyalin paket Batch 06, tambahkan `--package models/batch-06/dataset/v4_self_reported_rating`. '
        'Tambahkan `--replace` hanya bila memperbarui bundle atau paket v4 yang sudah terverifikasi.\n', encoding='utf-8')


def package_readme(package, manifest):
    counts = manifest['datasets']['good_acceptable']
    files = ('episode_ratings.csv', f'{WINDOW}/good_acceptable.csv', f'{WINDOW}/window_assignments.csv')
    lines = ['# Dataset v4 untuk Batch 06', '',
        (f'Gunakan `{WINDOW}/good_acceptable.csv` sebagai dataset utama. CSV ini berisi '
        f'{counts["windows"]} window layak dari {counts["episodes"]} episode dan {counts["participants"]} peserta. '
        'Target `difficulty_rating` adalah angka 1–10 yang dipilih peserta untuk kesulitan soal.'), '',
        '| File | Fungsi |', '|---|---|',
        f'| `{WINDOW}/good_acceptable.csv` | Fitur eye tracking dan HRV serta target |',
        f'| `{WINDOW}/window_assignments.csv` | Hubungan window berlabel ke episode; mencakup window poor untuk audit |',
        '| `episode_ratings.csv` | Bukti rating per episode, termasuk episode tanpa label |', '',
        ('Mulai penggabungan dari `good_acceptable.csv`. Gunakan `window_id` untuk mencari episode pada '
        '`window_assignments.csv`, lalu `episode_id` untuk mencari bukti pada `episode_ratings.csv`. '
        'Gunakan `participant_id` sebagai kunci pengelompokan. Jangan gabungkan file menurut posisi baris.'), '',
        'Fitur awal classifier:', '', '```text', *features.CORE, '```', '',
        ('Skala iris menyatakan geometri landmark, bukan diameter pupil. Identitas, waktu, soal, difficulty '
        'protokol, outcome, dan data audit tidak masuk daftar fitur awal. Paket ini tidak menetapkan split '
        'atau bobot; tentukan keduanya dalam pipeline v4. Jangan memakai split atau bobot v3 untuk v4.'), '',
        '| Rating | Window layak |', '|---:|---:|']
    lines.extend(f'| {rating} | {counts[str(rating)]} |' for rating in ratings.RATINGS)
    lines += ['', ('Satu episode menyumbang satu rating, meski muncul pada banyak window. Periksa jumlah '
        'peserta dan episode per rating sebelum membagi data. Soal tanpa angka rating tidak masuk dataset '
        'utama; timeout dengan angka rating tetap memakai pilihan peserta.'), '',
        ('Laporan sumber di repository: [QUALITY_REPORT.md](../../../../analysis_sessions/dataset_fusion/output/v4_self_reported_rating/QUALITY_REPORT.md). '
        'Salin ketiga CSV bersama-sama ketika memperbarui paket dari bundle sumber yang sama.'), '',
        '| File | Baris data | SHA-256 |', '|---|---:|---|']
    for name in files:
        path = package/name
        with path.open(encoding='utf-8', newline='') as stream:
            reader = csv.reader(stream)
            next(reader)
            size = sum(1 for _ in reader)
        lines.append(f'| `{name}` | {size} | `{shared.digest(path)}` |')
    return '\n'.join([*lines, ''])
