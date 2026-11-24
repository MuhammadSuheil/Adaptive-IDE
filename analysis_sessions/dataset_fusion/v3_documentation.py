"""Reports for fixed-rule behavioral proxies; no claims of measured mental load."""


def write_reports(destination, counts, readiness, summary, per_window, splits, *, extras,
                  reference_audit, episodes):
    lines = ['# Dataset v3: label eksploratif berbasis perilaku', '',
        '## Tujuan dan arti label', '',
        ('Low, Medium, dan High pada v3 adalah kategori perkiraan dari kesesuaian waktu pengerjaan aktif dan pola '
        'kesalahan manual. Ini bukan pengukuran cognitive load yang sudah tervalidasi. V3 menguji sumber label '
        'alternatif tanpa menetapkan easy=Low, medium=Medium, atau hard=High. Skor model terhadap label ini '
        'mengukur kemampuan memprediksi proxy perilaku, bukan membuktikan akurasi membaca keadaan mental.'), '',
        ('Tidak ditemukan angka rating asli yang cocok dengan sesi aktif pada arsip proyek yang diperiksa. '
        'Penanda rating_started/rating_ended tidak menyimpan angka rating. Nilai yang hilang tidak diisi; timeout '
        'tidak diubah menjadi rating 10. Sensor, cognitive_load_score heuristik, difficulty, dan hasil classifier '
        'tidak dipakai untuk menentukan label.'), '',
        '## Rekonstruksi episode', '',
        ('Satu episode mengikuti satu soal dalam satu sesi set hingga jawaban pertama yang benar atau penutupan '
        'tanpa selesai. Coba Lagi dan kembali ke soal yang belum selesai menambahkan interval aktif ke episode '
        'yang sama. Membuka kembali soal yang sudah benar membuat episode baru. Waktu feedback, rating, '
        'istirahat, dan pengerjaan soal lain tidak dihitung. time_seconds kumulatif pada log bukan durasi soal.'), '',
        ('Kesalahan dihitung dari submission manual incorrect yang unik menurut attempt_id. Timeout dicatat '
        'terpisah. Durasi episode tidak selesai adalah batas bawah observasi, bukan waktu penyelesaian. '
        'Trace tidak lengkap atau bertentangan tidak dilengkapi dengan event/timestamp buatan.'), '',
        '## Pembagian dan referensi waktu', '',
        ('Kandidat awal memakai pembagian hash lama; kandidat berikutnya memakai shuffle deterministik '
        'dengan seed 20261124, maksimal 1.000 kandidat termasuk pembagian awal. Validation dan Test masing-masing '
        'max(1, round(0.15 × jumlah peserta)); sisanya Train. Semua rekaman satu orang berada pada bagian yang sama. '
        'Referensi dan label dihitung ulang dari Train pada setiap kandidat. Pemilihan hanya memeriksa '
        'cakupan kelas dan peserta, tanpa training atau skor classifier. Kandidat pertama pada prioritas '
        'terbaik dipakai, lalu pembagian dibekukan. Prioritas A memenuhi minimal 3 orang per kelas Train '
        'dan 2 pada validation/Test untuk semua window; B memenuhi syarat itu pada 15 detik dan kehadiran '
        'semua kelas pada window lain; C memenuhi syarat 15 detik dan kehadiran kelas pada 25 detik; '
        'D memenuhi syarat 15 detik saja. Jika semuanya gagal, pembagian awal dipertahankan dengan status '
        'unavailable. Jumlah peserta minimum ini adalah pemeriksaan teknis, bukan jaminan kecukupan statistik.'), '',
        '| Bagian | Orang |', '|---|---:|']
    lines += [f'| {s} | {list(splits.values()).count(s)} |' for s in ('Train','validation','Test')]
    lines += ['', ('Referensi hanya memakai paparan pertama pada soal yang sama dari peserta Train, bila trace valid '
        'dan berhasil diselesaikan secara manual. Paparan pertama yang gagal tidak diganti dengan keberhasilan '
        'pada paparan berikutnya. Paparan sebelumnya yang sudah diketahui dalam metadata juga dikeluarkan '
        'dari kontributor referensi. Identitas dasar pembanding adalah set, ID soal, dan judul setelah normalisasi nomor '
        'urut, kapitalisasi, Unicode NFKC, serta spasi. Perbedaan isi soal yang tidak terekam tidak dapat diperiksa '
        'ulang hanya dari judul. Pengecualian adalah tujuh soal dengan bukti seluruh JSON identik saat '
        'berpindah easy-2 ke easy-1; keduanya memakai identitas kanonis dari hash isi. Bukti tersimpan '
        'pada item_aliases.json, termasuk snapshot, commit, lokasi, dan hash. Kesamaan isi bukan bukti '
        'kesamaan config runtime: batas waktu easy pada config historis berubah setelah perluasan bank soal. '
        'Config yang benar-benar dijalankan pada setiap rekaman tidak tersedia, sehingga tekanan waktu '
        'tidak dapat dipastikan hanya dari Git atau durasi pengerjaan.'), '',
        ('Setiap orang menyumbang maksimal satu waktu per soal. Peserta yang sedang dilabeli dikeluarkan dari '
        'referensinya sendiri. Minimal lima orang pembanding diperlukan. Batas waktu adalah persentil 33⅓ '
        'dan 66⅔ dengan interpolasi linear. Referensi kurang atau kedua batas sama menyebabkan abstention.'), '',
        '## Rubrik yang dibekukan', '',
        '| Label | Bukti hasil pengerjaan | Bukti waktu aktif |', '|---|---|---|',
        '| Low | Submission manual pertama benar, tanpa kesalahan manual | Waktu ≤ batas bawah |',
        '| Medium | Berhasil setelah 1–2 kesalahan manual | Batas bawah < waktu ≤ batas atas |',
        '| High | Minimal 3 kesalahan manual; boleh belum selesai | Waktu teramati > batas atas |',
        '| Tidak berlabel | Bukti lemah, bertentangan, trace invalid, atau referensi kurang | Tidak dipaksa menjadi Medium |', '',
        ('Rubrik adalah definisi operasional eksperimen, bukan ambang psikologis baku. Waktu dan kesalahan '
        'tidak otomatis menunjukkan usaha mental: hasil juga dapat dipengaruhi keterampilan, strategi, motivasi, '
        'atau penilaian jawaban. Arsip tidak menyimpan seluruh urutan jawaban beserta versi grader historis '
        'untuk mengaudit ulang setiap keputusan benar/salah. Tidak dibuat probabilitas keyakinan yang tidak '
        'diukur dan tidak dilakukan tuning rubrik berdasarkan akurasi model.'), '',
        '## Cakupan episode', '',
        (f'Total {summary["total"]} episode; {summary["valid"]} mempunyai trace valid. '
        f'{sum(v for k,v in summary["labels"].items() if k != "unlabeled")} episode diberi label; '
        f'{summary["labels"].get("unlabeled",0)} tetap tidak berlabel.'), '',
        '| Kategori | Episode |', '|---|---:|']
    lines += [f'| {label} | {summary["labels"].get(label,0)} |' for label in ('Low','Medium','High','unlabeled')]
    lines += ['', '| Alasan keputusan episode | Episode |', '|---|---:|']
    lines += [f'| {reason} | {n} |' for reason,n in sorted(summary['label_reasons'].items())]
    lines += ['', '| Bagian | Low | Medium | High | Tidak berlabel |', '|---|---:|---:|---:|---:|']
    for split, values in summary['by_split'].items():
        lines.append(f'| {split} | {values.get("Low",0)} | {values.get("Medium",0)} | {values.get("High",0)} | {values.get("unlabeled",0)} |')
    lines += ['', '## Window dan kualitas sensor', '',
        ('Kandidat berasal dari all_data v2 yang terverifikasi, bukan hanya hasil seleksi keseimbangan v2. '
        'Window 15/25/45 detik dengan step 5 detik harus seluruhnya berada dalam satu interval aktif yang '
        'kontinu dari satu episode berlabel. Interval retry yang terpisah feedback tidak disambung. '
        'Pre-task, feedback, rating, rest, post-task, kalibrasi, dan window lintas episode tidak dijadikan '
        'tambahan Low. Target episode sama pada ketiga panjang window.'), '',
        ('Rumus sensor, ambang kualitas core, pengecualian RR lokal, dan penanganan timestamp ambigu mengikuti '
        'bundle v2 sumber. Nilai sensor tidak dikoreksi atau diimputasi. Gaze yang gagal validasi/kualitas '
        'tetap missing di CSV curated. Input default tetap sepuluh fitur core, bukan seluruh kolom numerik.'), '',
        '| Window | Dataset | Low | Medium | High | Total | Orang | Durasi unik (menit) |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for seconds in (15,25,45):
        key=f'window_{seconds}s_step_5s'
        for name,c in counts[key].items():
            lines.append(f'| {seconds}s | {name} | {c["Low"]} | {c["Medium"]} | {c["High"]} | '
                         f'{c["windows"]} | {c["participants"]} | {c["unique_seconds"]/60:.2f} |')
    lines += ['', ('all_data menyimpan semua window berlabel termasuk kualitas poor untuk audit. good_acceptable '
        'hanya memakai good/acceptable; good_only adalah subset good. Ketiganya tidak dipaksa seimbang. '
        'Penyeimbangan disiapkan sebagai dua alternatif khusus Train, tanpa mengganti label. Jumlah baris '
        'overlap bukan jumlah pengamatan independen. Ketiga panjang window adalah alternatif dan tidak ditumpuk.'), '',
        ('Rubrik High mensyaratkan waktu yang relatif lama. Episode panjang juga menghasilkan lebih banyak '
        'sliding window; akibatnya proporsi High pada CSV dapat jauh lebih besar daripada proporsi episode High. '
        'Ini adalah pengaruh durasi dan definisi label, bukan bukti bahwa mayoritas peserta mengalami CL High. '
        'Periksa jumlah episode dan durasi unik, bukan hanya jumlah baris. Pada pemodelan nanti, kontribusi '
        'episode panjang dan kelas mayoritas perlu ditangani di Train serta dievaluasi per episode.'), '',
        '## Ketersediaan kelas untuk evaluasi', '',
        '| Window | Tiga kelas pada dataset utama | Kelas hilang Train | Kelas hilang validation | Kelas hilang Test |',
        '|---|---|---|---|---|']
    for seconds in (15,25,45):
        status=readiness[f'window_{seconds}s_step_5s']
        missing=status['missing_classes_by_split']
        lines.append(f'| {seconds}s | {"Ada" if status["primary_has_all_classes"] else "Belum lengkap"} | '
                     f'{", ".join(missing["Train"]) or "Tidak ada"} | '
                     f'{", ".join(missing["validation"]) or "Tidak ada"} | '
                     f'{", ".join(missing["Test"]) or "Tidak ada"} |')
    lines += ['', ('Jika dataset utama kehilangan suatu kelas, varian itu belum layak untuk eksperimen tiga kelas. '
        'Jika validation/Test kehilangan suatu kelas, evaluasi lengkap ketiga kelas pada bagian tersebut '
        'tidak tersedia meskipun seluruh CSV secara gabungan mempunyai tiga kelas. Pembagian yang terpilih '
        'dari pencarian cakupan dibekukan sebelum pemodelan. Rubrik dan minimum referensi tidak dilonggarkan. '
        '15 detik adalah varian utama; 25/45 detik menjadi analisis sekunder sesuai dukungan peserta. '
        'Status tersimpan di manifest.'), '',
        '## Alasan window tidak digunakan', '', '| Window | Alasan | Window |', '|---|---|---:|']
    for seconds, tables in per_window.items():
        if not tables['excluded'].empty:
            for reason,n in tables['excluded'].groupby('reason').size().items():
                lines.append(f'| {seconds}s | {reason} | {n} |')
    lines += ['', ('excluded_windows.csv merinci window yang tersedia di pool v2 tetapi tidak lolos label/interval '
        'v3. Eksklusi awal v2—misalnya timestamp ambigu—tetap berlaku dan diringkas di source_pool_exclusions '
        'pada manifest. Builder berhenti jika menemukan window yang berpotensi valid v3 tetapi tidak mempunyai '
        'agregat sensor di pool sumber; tidak menciptakan nilai pengganti.'), '',
        '## Penggunaan dan pembaruan', '',
        ('episode_labels.csv menyimpan durasi, interval sumber, attempt, kesalahan, censoring, batas referensi, '
        'split, dan alasan keputusan. label_reference.json menyimpan kontributor Train. participant_splits.csv '
        'merupakan pembagian yang wajib dipakai. window_assignments.csv menghubungkan baris sensor ke episode '
        'dan split; source_v2_label hanya untuk audit, bukan target/predictor baru.'), '',
        ('Jangan sekadar mengganti path pada notebook lama yang membuat pembagian sendiri. Mengubah peserta '
        'Train memerlukan penghitungan ulang referensi dan label. Jika melakukan cross-validation di dalam '
        'Train dan memakai hasilnya untuk memilih model, hitung ulang referensi pada inner-Train; validation/Test '
        'akhir tetap tidak boleh menjadi sumber referensi. Statistik preprocessing dan balancing juga fit di Train.'), '',
        ('V3 belum memiliki validasi beban mental independen. Konsistensi rubrik, kelengkapan fitur, dan skor '
        'classifier tidak membuktikan bahwa proxy ini mengukur CL sebenarnya. Perbandingan dengan rating '
        'atau pengukuran independen memerlukan sumber baru yang benar-benar tercatat.'), '',
        'Build: `python -m analysis_sessions.dataset_fusion.build_dataset_v3`', '',
        'Rebuild setelah sumber v2 diperbarui: `python -m analysis_sessions.dataset_fusion.build_dataset_v3 --replace`', '',
        ('Verifikasi: `python -m analysis_sessions.dataset_fusion.build_dataset_v3 --verify '
        'analysis_sessions/dataset_fusion/output/v3_behavioral_proxy`'), '',
        ('Sumber dan v2 tetap read-only. Manifest mencatat hash, versi kebijakan, pembagian, jumlah kelas, '
        'serta status evaluasi. Keluaran dipublikasikan sebagai satu bundle dengan rollback.'), '',
        '## Rujukan', '',
        '- [Validitas dan perbedaan rating kesulitan serta usaha mental](https://link.springer.com/article/10.1007/s11251-024-09692-6).',
        ('- [Weak supervision sebagai sumber label yang tidak sempurna](https://cs.brown.edu/people/sbach/files/ratner-vldb17.pdf). '
        'Rubrik v3 bukan hasil validasi dari paper tersebut dan tidak memakai model label Snorkel.'), '']
    lines += revision_sections(extras, reference_audit, episodes, per_window, readiness)
    (destination/'QUALITY_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    (destination/'README.md').write_text('# Dataset v3: behavioral proxy\n\n'
        'Dataset ini eksploratif, bukan label CL terukur. Baca [QUALITY_REPORT.md](QUALITY_REPORT.md) untuk '
        'rubrik, cakupan, kekurangan kelas, dan cara menggunakan pembagian peserta.\n\n'
        'Input utama adalah window_15s_step_5s/good_acceptable.csv; 25/45 detik adalah analisis sekunder. '
        'Alternatif Train adalah full pool dengan train_sample_weights.csv atau train_balanced.csv, bukan '
        'keduanya sekaligus. Validation/Test tidak diseimbangkan. participant_splits.csv dan '
        'label_reference.json tidak boleh diabaikan saat pemodelan. V1/v2 tidak berubah.\n',encoding='utf-8')
    for seconds in (15,25,45):
        key=f'window_{seconds}s_step_5s'
        c=counts[key]['good_acceptable'];s=readiness[key]
        text=[f'# V3 window {seconds} detik, step 5 detik','',
            'Label: proxy perilaku, bukan CL terukur. [Laporan lengkap](../QUALITY_REPORT.md).','',
            '| Dataset utama | Low | Medium | High | Total |','|---|---:|---:|---:|---:|',
            f'| good_acceptable | {c["Low"]} | {c["Medium"]} | {c["High"]} | {c["windows"]} |','',
            (f'Tiga kelas tersedia secara gabungan: {s["primary_has_all_classes"]}. '
            f'Tiga kelas tersedia pada setiap bagian evaluasi: {s["evaluation_has_all_classes_in_every_split"]}.'),'',
            ('Gunakan participant_splits.csv dan window_assignments.csv untuk pembagian tetap. '
            'Train dapat memakai bobot hierarkis atau train_balanced.csv sebagai alternatif. '
            f'Status dukungan peserta: {s["technical_status"]}. '
            'Tidak ada interpolasi atau imputasi sensor. Kelas yang tidak tersedia '
            'pada validation/Test tidak dapat dinilai lengkap pada bagian tersebut.'),'']
        (destination/key/'QUALITY_REPORT.md').write_text('\n'.join(text),encoding='utf-8')


def revision_sections(extras, audit, episodes, per_window, readiness):
    search, sensitivity = extras['partition_search'], extras['reference_sensitivity']
    changed = int(sensitivity.changed_label_removals.gt(0).sum())
    abstained = int(sensitivity.abstained_removals.gt(0).sum())
    stable = int((sensitivity.changed_label_removals.eq(0) & sensitivity.abstained_removals.eq(0)).sum())
    lines = ['', '## Audit perbaikan referensi', '',
        '| Soal identik yang berpindah set | Bukti |', '|---|---|']
    lines += [f'| {item["puzzle_id"]} | Snapshot JSON dan hash identik |' for item in extras['item_aliases']['items']]
    lines += ['', ('Pengulangan easy pada bangsur_easy_medium dan medium pada Ramadhan-2 tidak menjadi '
        'kontributor referensi. Pengecualian berlaku konservatif untuk tingkat yang tercatat, karena '
        'detail soal paparan sebelumnya tidak lengkap. Episode sensornya tetap dapat berlabel bila '
        'memenuhi syarat. Kelayakan referensi tidak dimasukkan sebagai fitur classifier.'), '',
        (f'{audit["known_prior_exposure_episodes"]} episode mempunyai paparan sebelumnya yang diketahui. '
        f'Referensi akhir memakai {audit["donor_episodes"]} kontributor episode Train, dengan '
        f'{audit["known_repeat_donors"]} kontributor pengulangan yang diketahui.'), '',
        '| Alasan tidak menjadi calon referensi | Episode |', '|---|---:|']
    lines += [f'| {reason} | {n} |' for reason, n in sorted(audit['excluded_reasons'].items())]
    invalid = [e for e in episodes if not e['valid']]
    if invalid:
        lines += ['', '| Rekaman | Soal | Trace yang dikeluarkan |', '|---|---|---|']
        lines += [f'| {e["session_folder"]} | {e["puzzle_id"]} | {", ".join(e["invalid_reasons"])} |' for e in invalid]
    lines += ['', '## Dukungan independen per bagian', '',
        (f'Kandidat terpilih berindeks {search["selected_candidate_index"]} dengan prioritas '
        f'{search["selected_tier"]}. partition_search.json merekam kandidat dan hash pembagian.'), '',
        '| Window | Bagian | Low: baris/orang/episode | Medium: baris/orang/episode | High: baris/orang/episode |',
        '|---|---|---|---|---|']
    for seconds in (15, 25, 45):
        for split, classes in search['coverage'][str(seconds)]['counts'].items():
            cells = [f'{classes[c]["windows"]}/{classes[c]["participants"]}/{classes[c]["episodes"]}'
                     for c in ('Low', 'Medium', 'High')]
            lines.append(f'| {seconds}s | {split} | ' + ' | '.join(cells) + ' |')
    lines += ['', '| Window | Minimum dukungan orang terpenuhi | Status |', '|---|---|---|']
    for seconds in (15, 25, 45):
        s = readiness[f'window_{seconds}s_step_5s']
        lines.append(f'| {seconds}s | {s["minimum_people_met"]} | {s["technical_status"]} |')
    lines += ['', '## Penanganan dominasi pada Train', '',
        ('CSV utama mempertahankan seluruh window layak. Validation/Test tidak dipangkas, diulang, '
        'atau dibuat sintetis. Dua perlakuan berikut adalah alternatif:'), '',
        ('1. Full Train dari good_acceptable.csv dengan train_sample_weights.csv. Untuk N baris Train, '
        'P peserta pada kelas, E episode peserta tersebut pada kelas, dan W window episode: '
        'bobot = N / (3 * P * E * W). Setiap kelas mendapat total bobot N/3, lalu kontribusi disetarakan '
        'antarorang dalam kelas dan antarepisode orang tersebut. Tidak ada data sintetis.'),
        ('2. train_balanced.csv. Jumlah tiap kelas mengikuti kelas terkecil pada Train. Kuota dibagi '
        'merata bergiliran menurut peserta dan episode, dibatasi ketersediaan. Dalam episode, good '
        'didahulukan lalu acceptable, dengan pemilihan tersebar menurut waktu dan tie-break ID.'), '',
        ('Bobot tidak termasuk input classifier. Jangan mengalikan bobot full-pool ke Train seimbang. '
        'Jika memakai inner cross-validation, hitung ulang referensi, label, bobot, dan seleksi dari '
        'inner-Train; berkas bobot ini hanya untuk Train akhir pada pembagian yang dibekukan.'), '',
        '| Window | Train lengkap Low/Medium/High | Train seimbang per kelas | Total seimbang |', '|---|---|---:|---:|']
    for seconds, tables in per_window.items():
        train = tables['weights'].cognitive_load_label.value_counts()
        balanced = tables['balanced'].cognitive_load_label.value_counts()
        lines.append(f'| {seconds}s | {train.get("Low", 0)}/{train.get("Medium", 0)}/{train.get("High", 0)} | '
                     f'{balanced.get("Low", 0)} | {len(tables["balanced"])} |')
    lines += ['', '## Sensitivitas referensi', '',
        ('Setiap kontributor referensi dikeluarkan satu per satu dan keputusan episode dihitung ulang. '
        'Hasil tidak dipakai untuk mengubah label utama atau membuat probabilitas keyakinan.'), '',
        f'- {stable} episode tetap berlabel sama pada seluruh pengeluaran satu kontributor.',
        f'- {abstained} episode kehilangan label pada setidaknya satu pengeluaran kontributor.',
        f'- {changed} episode berubah ke kelas lain pada setidaknya satu pengeluaran kontributor.', '',
        ('Kategori kehilangan label dan perubahan kelas dapat beririsan. Kehilangan label dapat berasal '
        'dari referensi yang turun di bawah lima orang atau bukti waktu yang tidak lagi selaras. '
        'Rincian tersedia pada reference_sensitivity.csv; minimum referensi tidak diturunkan.'), '',
        '## Penutupan audit dan kebutuhan sumber tambahan', '',
        ('revision_changes.csv menelusuri perubahan seluruh episode bila bundle lama diganti. history '
        'menyimpan manifest, pembagian, label, serta referensi sebelumnya tanpa menggandakan CSV sensor.'), '',
        '| Bagian | Penanganan |', '|---|---|',
        '| Soal identik yang pindah set | Identitas diperbaiki dengan snapshot dan hash |',
        '| Paparan ulang dalam referensi | Dikeluarkan sebagai kontributor; episode layak dipertahankan |',
        '| Kelas hilang pada evaluasi | Pencarian cakupan terbatas dan status dukungan tiap window |',
        '| Dominasi kelas dan episode panjang | Bobot hierarkis atau alternatif Train seimbang |',
        '| Trace tidak lengkap | Dikeluarkan tanpa membuat event akhir |',
        '| Referensi kurang atau perilaku bertentangan | Tetap tidak berlabel |',
        '| Rating beban mental independen | Tidak tersedia pada arsip yang diperiksa |',
        '| Config deployed dan jawaban historis lengkap | Tidak dipastikan dari Git/ledger saja |', '',
        ('Lolos pemeriksaan teknis berarti siap diuji sebagai proxy perilaku dengan dukungan yang '
        'dilaporkan, bukan jaminan validitas CL sebenarnya atau accuracy tertentu. Kegunaan fitur tetap '
        'memerlukan pengujian pemodelan. Perubahan rumus tanpa bukti sumber baru tidak mengisi informasi '
        'yang memang tidak direkam.'), '',
        ('Pengambilan tambahan, bila dilakukan, perlu memprioritaskan paparan pertama soal yang sebanding, '
        'kualitas sensor dan interval aktif yang cukup, serta rating usaha mental yang benar-benar '
        'dicatat dekat dengan pengerjaan termasuk ketika salah atau timeout. Simpan versi soal, config '
        'runtime, urutan jawaban, dan identitas lintas sesi. Kebutuhan ditentukan dari peserta/episode '
        'per kelas dan referensi per soal, bukan sekadar target tambahan baris Low.'), '']
    return lines
