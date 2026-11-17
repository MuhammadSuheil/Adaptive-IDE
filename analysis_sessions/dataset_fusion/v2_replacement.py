"""Deterministic, count-preserving curation; measurements are never rewritten."""
from collections import deque

import numpy as np
import pandas as pd

POLICY = 'conservative_quality_replacement_v1'
TOLERANCE = 1e-9
METRICS = {
    'eye_temporal_coverage': 1, 'face_ear_ratio': 1,
    'valid_gaze_ratio': 1, 'iris_available_ratio': 1,
    'clipping_ratio_valid': -1, 'hrv_coverage': 1,
    'hrv_data_age_seconds': -1,
}
CURATION_META = {
    'curation_selected': 'Selected for curated views by the recorded replacement policy; never a predictor.',
    'curation_reason': 'Selection decision, separate from the original sensor quality tier.',
}
PAIR_COLUMNS = ['window_seconds', 'cognitive_load_label', 'old_window_id', 'new_window_id',
                'old_participant_id', 'new_participant_id', 'old_session_id', 'new_session_id',
                *[f'{side}_{key}' for side in ('old', 'new')
                  for key in (*METRICS, 'baseline_usable', 'gaze_validation_passed')]]


def boolean(series):
    values = series.astype(str).str.lower()
    if not values.isin(('true', 'false', '1', '0')).all():
        raise ValueError(f'Unknown replacement boolean: {series.name}')
    return values.isin(('true', '1'))


def _match(new, old, capacities):
    """Integral maximum flow: new -> old -> participant -> sink.

    Sorted insertion order makes augmenting-path tie breaks reproducible.
    No external optimization dependency or weighted quality score is needed.
    """
    people = sorted(capacities)
    n, m = len(new), len(old)
    source, sink = 0, 1+n+m+len(people)
    graph: list[dict[int, int]] = [{} for _ in range(sink+1)]

    def edge(left, right, capacity):
        graph[left][right] = capacity
        graph[right][left] = 0

    for i in range(n):
        edge(source, 1+i, 1)
    signs = np.array(list(METRICS.values()))
    a = old[list(METRICS)].to_numpy(float)*signs
    b = new[list(METRICS)].to_numpy(float)*signs
    eligible = np.all(b[:, None, :] >= a[None, :, :]-TOLERANCE, axis=2)
    eligible &= np.isfinite(b).all(axis=1)[:, None] & np.isfinite(a).all(axis=1)[None, :]
    eligible &= (~boolean(old.baseline_usable).to_numpy()[None, :]
                 | boolean(new.baseline_usable).to_numpy()[:, None])
    for i, j in zip(*np.where(eligible)):
        edge(1+int(i), 1+n+int(j), 1)
    group_nodes = {person: 1+n+m+i for i, person in enumerate(people)}
    for j, person in enumerate(old.participant_id):
        edge(1+n+j, group_nodes[person], 1)
    for person in people:
        edge(group_nodes[person], sink, capacities[person])
    while True:
        parents = [-1]*len(graph)
        parents[source] = source
        queue = deque([source])
        while queue and parents[sink] < 0:
            current = queue.popleft()
            for neighbor, capacity in graph[current].items():
                if capacity and parents[neighbor] < 0:
                    parents[neighbor] = current
                    queue.append(neighbor)
                    if neighbor == sink:
                        break
        if parents[sink] < 0:
            break
        current = sink
        while current != source:
            parent = parents[current]
            graph[parent][current] -= 1
            graph[current][parent] += 1
            current = parent
    return [(i, j) for i, j in zip(*np.where(eligible))
            if graph[1+int(i)][1+n+int(j)] == 0]


def curate_window(frame, reference_ids, candidate_session_ids):
    """Select replacements for one window length, preserving reference counts."""
    frame = frame.copy()
    reference_ids = list(reference_ids)
    candidates = set(candidate_session_ids)
    if frame.window_id.duplicated().any() or len(set(reference_ids)) != len(reference_ids):
        raise ValueError('Duplicate replacement window identity.')
    if frame.window_seconds.nunique() != 1 or not reference_ids or not candidates:
        raise ValueError('Replacement requires one window length and a nonempty reference/cohort.')
    reference = frame[frame.window_id.isin(reference_ids)].copy()
    if len(reference) != len(reference_ids):
        raise ValueError('Missing reference window in recomputed source data.')
    if reference.session_id.isin(candidates).any():
        raise ValueError('Replacement candidates overlap the reference cohort.')
    if not candidates.issubset(set(frame.session_id)):
        raise ValueError('Missing replacement session.')
    if not reference.quality_tier.isin(('good', 'acceptable')).all():
        raise ValueError('Reference curated data contains a poor window.')
    boolean(frame.baseline_usable)
    frame['curation_selected'] = frame.window_id.isin(reference_ids)
    frame['curation_reason'] = np.where(frame.quality_tier.eq('poor'), 'sensor_poor', 'eligible_not_selected')
    frame.loc[frame.curation_selected, 'curation_reason'] = 'retained_reference'
    passed = frame.gaze_validation_passed.astype(str).str.lower().isin(('true', '1'))
    failed = reference.gaze_validation_passed.astype(str).str.lower().isin(('false', '0'))
    old_pool = reference[reference.quality_tier.eq('acceptable') & failed]
    new_pool = frame[frame.session_id.isin(candidates) & passed
                     & frame.quality_tier.isin(('good', 'acceptable'))]
    pairs = []
    for label in ('Low', 'Medium', 'High'):
        old = old_pool[old_pool.cognitive_load_label.eq(label)].sort_values('window_id').reset_index(drop=True)
        new = new_pool[new_pool.cognitive_load_label.eq(label)].sort_values('window_id').reset_index(drop=True)
        if old.empty or new.empty:
            continue
        support = reference[reference.cognitive_load_label.eq(label)].participant_id.value_counts()
        capacities = {person: int(support[person])-1 for person in sorted(set(old.participant_id))}
        for i, j in _match(new, old, capacities):
            left, right = old.iloc[j], new.iloc[i]
            item = {'window_seconds': int(left.window_seconds), 'cognitive_load_label': label}
            for side, record in (('old', left), ('new', right)):
                for key in ('window_id', 'participant_id', 'session_id', *METRICS,
                            'baseline_usable', 'gaze_validation_passed'):
                    value = record[key]
                    item[f'{side}_{key}'] = value.item() if isinstance(value, np.generic) else value
            pairs.append(item)
    pairs.sort(key=lambda p: (p['window_seconds'], p['cognitive_load_label'], p['old_window_id'], p['new_window_id']))
    removed = frame.window_id.isin([p['old_window_id'] for p in pairs])
    added = frame.window_id.isin([p['new_window_id'] for p in pairs])
    frame.loc[removed, ['curation_selected', 'curation_reason']] = [False, 'replaced_reference']
    frame.loc[added, ['curation_selected', 'curation_reason']] = [True, 'replacement_selected']
    selected = frame[frame.curation_selected]
    if reference.cognitive_load_label.value_counts().to_dict() != selected.cognitive_load_label.value_counts().to_dict():
        raise ValueError('Replacement changed class counts.')
    support = set(zip(reference.participant_id, reference.cognitive_load_label))
    if not support.issubset(set(zip(selected.participant_id, selected.cognitive_load_label))):
        raise ValueError('Replacement removed existing participant/class support.')
    return frame, pairs


def selection_context(directory, previous, replacement_session):
    """Persist the original reference, rather than replacing Mon again on rebuild."""
    import json

    from .build_dataset import digest

    path = directory/'replacement_selection.json'
    if previous and previous.get('admission_policy') == POLICY:
        context = json.loads(path.read_text(encoding='utf-8'))
        if replacement_session and context['candidate_session_ids'] != [replacement_session]:
            raise ValueError('Changing an existing replacement cohort requires a new explicit selection plan.')
        return context
    if replacement_session is None:
        return None
    if not previous or previous.get('admission_policy') != 'window_quality_only':
        raise ValueError('Conservative replacement needs an existing quality-only v2 reference bundle.')
    if replacement_session in previous['selected_session_ids']:
        raise ValueError('Replacement session is already part of the reference bundle.')
    return {
        'policy': POLICY, 'tolerance': TOLERANCE,
        'algorithm': 'maximum_cardinality_flow_sorted_window_ids_with_participant_class_cap',
        'reference_manifest_sha256': digest(directory/'manifest.json'),
        'reference_session_ids': sorted(previous['selected_session_ids']),
        'reference_window_ids': {
            str(seconds): sorted(pd.read_csv(directory/f'window_{seconds}s_step_5s/good_acceptable.csv').window_id)
            for seconds in (15, 25, 45)},
        'candidate_session_ids': [replacement_session],
    }


def verify_context(context):
    if (context.get('policy') != POLICY or context.get('tolerance') != TOLERANCE
            or context.get('algorithm') != 'maximum_cardinality_flow_sorted_window_ids_with_participant_class_cap'
            or set(context.get('reference_window_ids', {})) != {'15', '25', '45'}
            or len(context.get('candidate_session_ids', [])) != 1
            or not context.get('reference_session_ids')
            or len(context.get('reference_manifest_sha256', '')) != 64):
        raise ValueError('Invalid replacement selection context.')


def report_lines(frame_by_seconds, context, pairs):
    """Describe actual curation without conflating it with a sensor-tier change."""
    from . import v2_history as history
    from .build_dataset_v2 import unique_duration

    if context['policy'] == history.POLICY:
        return history.report_lines(frame_by_seconds, context, pairs)

    lines = ['## Penggantian selektif dengan Mon', '',
             ('Kebijakan: `conservative_quality_replacement_v1`. Penggantian satu-banding-satu mempertahankan jumlah tiap kelas. '
             'Window lama yang diganti tetap ada di all_data, bukan dihapus dari rekaman mentah.'), '',
             ('Kandidat lama hanya acceptable dengan validasi gaze gagal. Semua good lama dan rekaman yang lolos validasi gaze dipertahankan. '
             'Mon harus lolos validasi gaze dan good/acceptable. Coverage eye, face/EAR, gaze, iris, dan HRV tidak boleh menurun; '
             'clipping dan usia RR tidak boleh meningkat (toleransi 1e-9). Baseline layak tidak diganti baseline tidak layak.'), '',
             ('Pencocokan memaksimalkan jumlah pasangan yang memenuhi seluruh syarat dan menyisakan sedikitnya satu window '
             'untuk setiap kombinasi peserta–kelas lama. Tidak ada skor model atau perubahan label dalam seleksi. '
             'Bukti peningkatan terbatas pada indikator ini, bukan validasi fisiologis baru.'), '',
             '| Window | Kelas | Sebelum | Diganti | Mon terpilih | Sesudah |',
             '|---|---|---:|---:|---:|---:|']
    for seconds, frame in frame_by_seconds.items():
        reference = frame[frame.window_id.isin(context['reference_window_ids'][str(seconds)])]
        selected = frame[boolean(frame.curation_selected)]
        for label in ('Low', 'Medium', 'High'):
            n = sum(p['window_seconds'] == seconds and p['cognitive_load_label'] == label for p in pairs)
            lines.append(f'| {seconds}s | {label} | {reference.cognitive_load_label.eq(label).sum()} | {n} | {n} | {selected.cognitive_load_label.eq(label).sum()} |')
    lines += ['', '| Window | Orang sebelum / sesudah | Durasi unik sebelum / sesudah (menit) | Gaze gagal sebelum / sesudah (row) |',
              '|---|---|---|---|']
    for seconds, frame in frame_by_seconds.items():
        old = frame[frame.window_id.isin(context['reference_window_ids'][str(seconds)])]
        new = frame[boolean(frame.curation_selected)]
        failed = lambda f: int(f.gaze_validation_passed.astype(str).str.lower().isin(('false', '0')).sum())
        lines.append(f'| {seconds}s | {old.participant_id.nunique()} / {new.participant_id.nunique()} | '
                     f'{unique_duration(old)/60:.2f} / {unique_duration(new)/60:.2f} | {failed(old)} / {failed(new)} |')
    return lines + ['', ('replacement_selection.json menyimpan referensi seleksi awal; replacement_pairs.csv menyimpan pasangan '
                    'dan indikatornya. curation_selected/curation_reason adalah metadata, bukan predictor. '
                    'good_only memuat good yang terpilih; good_acceptable memuat good/acceptable yang terpilih. '
                    'all_data mencakup semua window berlabel, termasuk yang tidak terpilih. '
                    'Rebuild --replace mempertahankan referensi dan aturan seleksi yang sama. '
                    'Penambahan sumber lain memerlukan pembaruan rencana seleksi, bukan append otomatis.'), '']


def condition_report(frames, context, pairs, reports, audits, operator_guide=''):
    """Regenerate the condition snapshot so stale pre-replacement counts cannot survive."""
    from datetime import UTC, datetime

    lines = ['# Kondisi Dataset Setelah Penggantian Selektif', '',
             f'Waktu pemeriksaan UTC: {datetime.now(UTC).isoformat()}.', '',
             '## Kesimpulan', '',
             ('Proporsi ketiga kelas dipertahankan dari dataset sebelumnya. Sebagian window acceptable lama '
              'diganti window sesi tambahan yang lolos validasi gaze dan tidak lebih buruk pada tujuh indikator kualitas. '
              'Ini pembaruan dataset, bukan hasil evaluasi model. Data lama tidak dinyatakan seluruhnya tidak layak.'), '',
             '| Window | Low | Medium | High | Total | Orang | Rasio terbesar/terkecil |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for seconds, frame in frames.items():
        selected = frame[boolean(frame.curation_selected)]
        counts = selected.cognitive_load_label.value_counts().reindex(['Low', 'Medium', 'High'], fill_value=0)
        ratio = f'{counts.max()/counts.min():.3f}' if counts.min() else 'tidak terdefinisi'
        lines.append(f'| {seconds}s | {counts.Low} | {counts.Medium} | {counts.High} | {len(selected)} | '
                     f'{selected.participant_id.nunique()} | {ratio} |')
    lines += ['', *report_lines(frames, context, pairs), '## Kualitas yang masih perlu diperhatikan', '',
              '| Window | Gaze gagal (row / persen) | Rasio baseline kosong | Blink kosong | HRV tidak usable di curated |',
              '|---|---:|---:|---:|---:|']
    for seconds, frame in frames.items():
        selected = frame[boolean(frame.curation_selected)]
        failed = selected.gaze_validation_passed.astype(str).str.lower().isin(('false', '0')).sum()
        percent = 100*failed/len(selected) if len(selected) else 0
        lines.append(f'| {seconds}s | {failed} / {percent:.1f}% | {selected.rmssd_baseline_ratio.isna().sum()} | '
                     f'{selected.blink_count.isna().sum()} | {(~boolean(selected.hrv_usable)).sum()} |')
    passed = sum(r['gaze_validation_passed'] is True for r in reports)
    failed = sum(r['gaze_validation_passed'] is False for r in reports)
    lines += ['', (f'Pada {len(reports)} rekaman sumber, {passed} lolos dan {failed} gagal validasi gaze. '
                   'Tabel curated menghitung status rekaman asal, bukan pemeriksaan akurasi baru per window.'), '',
              '- Kegagalan gaze tetap relevan untuk koordinat gaze dan fitur turunannya. Penggantian ini mengurangi kontribusinya, tidak memperbaiki kalibrasi sumber lama.',
              '- HRV absolut yang lolos aturan window tidak dibatalkan hanya karena baseline rasio tidak layak. Nilai rasio kosong tidak diganti nol.',
              '- EAR, blink, estimasi iris, dan HRV tidak otomatis tidak layak ketika validasi spasial gaze gagal.',
              '- Validasi awal yang lolos belum membuktikan tidak ada drift selama pengerjaan. Ambang berasal dari konfigurasi prototype, bukan standar universal.',
              '- Jumlah window overlap bukan pengamatan independen. Dukungan peserta–kelas dipertahankan, tetapi durasi unik dapat berubah.',
              '- Label tetap mengikuti protokol v2: fase sesi Low yang terlacak dan pengerjaan easy/practice menjadi Low; pengerjaan medium/hard menjadi Medium/High.', '',
              '## Kontribusi sumber tambahan', '',
              '| Sumber | Window | Berlabel, timestamp layak | Good/acceptable sebelum seleksi | Terpilih | Tidak terpilih |',
              '|---|---|---:|---:|---:|---:|']
    for seconds, frame in frames.items():
        for session in context['candidate_session_ids']:
            source = frame[frame.session_id.eq(session)]
            eligible = source.quality_tier.isin(('good', 'acceptable')).sum()
            selected = boolean(source.curation_selected).sum()
            name = ', '.join(sorted(source.session_folder.unique()))
            lines.append(f'| {name} | {seconds}s | {len(source)} | {eligible} | {selected} | {len(source)-selected} |')
    lines += ['', ('Kandidat tidak dipilih jika sensornya poor, tidak mempunyai pasangan dominasi kualitas, '
              'atau penggantiannya menghilangkan dukungan peserta–kelas lama. Window target campuran tetap berada pada excluded_windows.csv, '
              'bukan dianggap sensor buruk. all_data tidak identik dengan dataset utama.'), '',
              '## Kontinuitas HRV sumber sebelum seleksi', '',
              '| Sumber | Window cache 25s | Ready | Tidak ready | Ready (%) |',
              '|---|---:|---:|---:|---:|']
    for audit in audits:
        if 'source_hrv_status_counts' not in audit:
            continue
        statuses = audit['source_hrv_status_counts']
        total, ready = sum(statuses.values()), statuses.get('ready', 0)
        lines.append(f'| {audit["session_folder"]} | {total} | {ready} | {total-ready} | {100*ready/total:.1f} |')
    for audit in audits:
        timing = audit.get('eye_timestamp_audit')
        if not timing:
            continue
        lines += ['', f'## Timestamp ambigu: {audit["session_folder"]}', '',
                  ('Semua frame dan timestamp mentah dipertahankan. Tiga kelompok pada sesi Alif merupakan frame berbeda '
                   'dengan timestamp milidetik sama, bukan baris duplikat. Tidak ada pergeseran waktu atau interpolasi. '
                   'Setiap window dengan start < waktu ambigu <= end dikeluarkan, termasuk dari all_data. '
                   'Increment kedipan tetap mengikuti urutan frame asli. Hash sumber serta detail kelompok ada di replacement_selection.json '
                   'dan replay_validation.json.'), '',
                  '| Waktu WIB | Baris CSV | Frame | Capture ID |', '|---|---|---|---|']
        for group in timing['groups']:
            stamp = pd.Timestamp(group['timestamp_ms'], unit='ms', tz='UTC').tz_convert('Asia/Jakarta').isoformat()
            lines.append(f'| {stamp} | {group["csv_lines"]} | {group["frame_index"]} | {group["capture_frame_id"]} |')
        lines += ['', '| Window | Semua window menyentuh waktu ambigu | Tambahan exclusion di luar target campuran |',
                  '|---|---:|---:|']
        for seconds, counts in timing['windows'].items():
            lines.append(f'| {seconds}s | {counts["affected"]} | {counts["additional_excluded"]} |')
    lines += ['',
              '## Pemeriksaan dan pemakaian berikutnya', '',
              (f'Replay memeriksa {sum(a["cached_windows"] for a in audits)} window cache HRV; '
               f'{sum(a["unexplained_windows"] for a in audits)} perbedaan tidak terjelaskan. '
               'Validator mengulang pencocokan, memeriksa ledger pasangan, tier, label, subset, serta checksum sebelum dan setelah publikasi.'), '',
              ('Gunakan tiga CSV good_acceptable secara terpisah. Ikuti 26 predictor pada feature_columns.json; '
              'curation_selected, curation_reason, identitas, waktu, dan konteks tugas adalah metadata. '
              'Snapshot model lama tetap menggunakan CSV/hash lama. Evaluasi bundle ini memerlukan release/run baru, bukan mencampur hasil lama dan baru.'), '',
              ('Periksa [QUALITY_REPORT.md](QUALITY_REPORT.md), [manifest](manifest.json), '
              '[pasangan penggantian](replacement_pairs.csv), dan [referensi seleksi](replacement_selection.json).'), '']
    if operator_guide:
        lines += [operator_guide.rstrip(), '']
    return '\n'.join(lines)
