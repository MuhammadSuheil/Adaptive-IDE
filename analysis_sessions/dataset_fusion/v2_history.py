"""Append explicit selection stages without replacing earlier accepted cohorts."""
import copy
import json
from typing import Any

import pandas as pd

from . import build_dataset as shared
from . import v2_replacement as replacement
from . import v2_timestamps as timing

POLICY = 'conservative_quality_replacement_v2'
POLICIES = (replacement.POLICY, POLICY)
PAIR_COLUMNS = ['selection_stage', *replacement.PAIR_COLUMNS]


def stages(context):
    return context['stages'] if context['policy'] == POLICY else [context]


def selection_context(directory, previous, replacement_session):
    if not previous or previous.get('admission_policy') not in POLICIES:
        return replacement.selection_context(directory, previous, replacement_session)
    context = json.loads((directory/'replacement_selection.json').read_text(encoding='utf-8'))
    if not replacement_session or replacement_session in context['candidate_session_ids']:
        return context
    if replacement_session in previous['selected_session_ids']:
        raise ValueError('Replacement session is already an existing source, not a new cohort.')
    if context['policy'] == replacement.POLICY:
        context = {**copy.deepcopy(context), 'policy': POLICY, 'stages': [context], 'timestamp_policies': {}}
    new_stage = {
        'policy': replacement.POLICY, 'tolerance': replacement.TOLERANCE,
        'algorithm': context['algorithm'], 'reference_manifest_sha256': shared.digest(directory/'manifest.json'),
        'reference_session_ids': sorted(previous['selected_session_ids']),
        'reference_window_ids': {
            str(seconds): sorted(pd.read_csv(directory/f'window_{seconds}s_step_5s/good_acceptable.csv').window_id)
            for seconds in (15, 25, 45)},
        'candidate_session_ids': [replacement_session],
    }
    context['stages'].append(new_stage)
    context['candidate_session_ids'].append(replacement_session)
    verify_context(context)
    return context


def verify_context(context):
    if context.get('policy') == replacement.POLICY:
        replacement.verify_context(context)
        return
    if context.get('policy') != POLICY or len(context.get('stages', [])) < 2:
        raise ValueError('Invalid staged replacement context.')
    initial = context['stages'][0]
    for key in ('reference_window_ids', 'reference_manifest_sha256', 'reference_session_ids', 'algorithm', 'tolerance'):
        if context.get(key) != initial[key]:
            raise ValueError('Initial replacement reference changed.')
    seen = set(initial['reference_session_ids'])
    candidates = []
    for stage in context['stages']:
        replacement.verify_context(stage)
        if set(stage['reference_session_ids']) != seen:
            raise ValueError('Stage source history changed.')
        cohort = stage['candidate_session_ids']
        if seen.intersection(cohort):
            raise ValueError('Repeated replacement cohort.')
        seen.update(cohort)
        candidates.extend(cohort)
    if context['candidate_session_ids'] != candidates:
        raise ValueError('Stage cohort registry mismatch.')
    for session, policy in context.get('timestamp_policies', {}).items():
        if session not in candidates:
            raise ValueError('Timestamp exception is not part of an explicitly approved cohort.')
        timing.validate_policy(policy)


def curate_history(frame, context, seconds):
    verify_context(context)
    pairs: list[dict[str, Any]] = []
    previous_ids = None
    locked: set[str] = set()
    for index, stage in enumerate(stages(context), 1):
        reference = stage['reference_window_ids'][str(seconds)]
        if previous_ids is not None and set(reference) != previous_ids:
            raise ValueError('A later stage changed the previous selected reference.')
        frame, matches = replacement.curate_window(frame, reference, stage['candidate_session_ids'])
        expected = stage.get('expected_counts', {}).get(str(seconds))
        if expected is not None:
            actual = {label: sum(p['cognitive_load_label'] == label for p in matches)
                      for label in ('Low', 'Medium', 'High')}
            if actual != expected:
                raise ValueError(f'Expected replacement counts differ at stage {index}, {seconds}s: {actual}')
        selected = set(frame.loc[frame.curation_selected, 'window_id'])
        if not locked.issubset(selected):
            raise ValueError('An earlier accepted replacement was removed.')
        locked.update(p['new_window_id'] for p in matches)
        previous_ids = selected
        pairs.extend(({**p, 'selection_stage': index} if context['policy'] == POLICY else p) for p in matches)
    # Keep decisions from all stages visible, not just the last reference.
    frame.loc[frame.window_id.isin([p['old_window_id'] for p in pairs]), 'curation_reason'] = 'replaced_reference'
    frame.loc[frame.window_id.isin([p['new_window_id'] for p in pairs]), 'curation_reason'] = 'replacement_selected'
    return frame, pairs


def pair_columns(policy):
    return PAIR_COLUMNS if policy == POLICY else replacement.PAIR_COLUMNS


def report_lines(frames, context, pairs):
    from .build_dataset_v2 import unique_duration

    lines = ['## Penggantian bertahap: Mon kemudian Alif', '',
             (f'Kebijakan: `{context["policy"]}`. Referensi awal dan setiap tahap tersimpan pada replacement_selection.json. '
              'Tahap berikutnya memakai hasil curated tahap sebelumnya, bukan memilih ulang dari nol. '
              'Semua pasangan tahap lama dan semua window tambahan yang sudah dipilih tetap dipertahankan.'), '',
             ('Penggantian hanya terhadap acceptable lama yang gagal validasi gaze. Kandidat baru harus lolos validasi gaze dan '
              'good/acceptable. Coverage temporal eye, face/EAR, gaze, iris, serta HRV tidak menurun. Clipping dan usia RR '
              'tidak meningkat; baseline layak tidak diganti baseline tidak layak. Toleransi numerik 1e-9.'), '',
             ('Pencocokan memaksimalkan pasangan yang memenuhi seluruh syarat, secara deterministik, satu-banding-satu '
              'pada kelas dan panjang window sama. Dukungan setiap peserta–kelas lama tetap ada. '
              'Tidak ada pemilihan berdasarkan skor model, data sintetis, atau perubahan label.'), '',
             '| Tahap | Sumber | Window | Ganti Low | Ganti Medium | Ganti High | Total |',
             '|---|---|---|---:|---:|---:|---:|']
    for index, stage in enumerate(stages(context), 1):
        for seconds, frame in frames.items():
            source = frame[frame.session_id.isin(stage['candidate_session_ids'])]
            name = ', '.join(sorted(source.session_folder.unique()))
            matched = [p for p in pairs if p.get('selection_stage', 1) == index and p['window_seconds'] == seconds]
            counts = [sum(p['cognitive_load_label'] == label for p in matched) for label in ('Low', 'Medium', 'High')]
            lines.append(f'| {index} | {name} | {seconds}s | {counts[0]} | {counts[1]} | {counts[2]} | {sum(counts)} |')
    lines += ['', '| Window | Low / Medium / High akhir | Orang awal / akhir | Durasi unik awal / akhir (menit) |',
              '|---|---|---|---|']
    for seconds, frame in frames.items():
        original = frame[frame.window_id.isin(context['reference_window_ids'][str(seconds)])]
        selected = frame[replacement.boolean(frame.curation_selected)]
        counts = [int(selected.cognitive_load_label.eq(label).sum()) for label in ('Low', 'Medium', 'High')]
        lines.append(f'| {seconds}s | {counts[0]} / {counts[1]} / {counts[2]} | '
                     f'{original.participant_id.nunique()} / {selected.participant_id.nunique()} | '
                     f'{unique_duration(original)/60:.2f} / {unique_duration(selected)/60:.2f} |')
    return lines + ['', ('Awal berarti referensi sebelum tahap Mon. Jumlah tiap kelas tetap pada semua tahap. '
                       'replacement_pairs.csv mencatat selection_stage beserta indikator tiap pasangan. '
                       'curation_selected dan curation_reason adalah metadata, bukan predictor. '
                       'Row lama yang diganti tetap di all_data; window timestamp ambigu hanya di audit/excluded_windows. '
                       'good_only berisi good terpilih dan good_acceptable berisi good/acceptable terpilih.'), '']
