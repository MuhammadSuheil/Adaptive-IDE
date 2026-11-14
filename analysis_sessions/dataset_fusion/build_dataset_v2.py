"""Recompute aligned sensor windows with session-phase cognitive-load labels.

Raw recordings are read-only. A complete, verified bundle is published by one
directory rename; an existing bundle is never overwritten.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from itertools import groupby
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis_sessions.dataset_fusion import build_dataset as shared
from analysis_sessions.session_inventory import participant_group, select_session_directories
from fuse_and_analyze import calibration_ready_ms, context_at, summarize_segments
from hrv_monitor_prototype.calculate_hrv import LiveHRV
from hrv_monitor_prototype.preprocess_rr import RRPreprocessor

WINDOWS = (15, 25, 45)
LABELS = ('Low', 'Medium', 'High')
LOW_PHASES = ('pre_task', 'feedback', 'rating', 'rest', 'post_task', 'interrupted')
ACTIVE_LABELS = {'easy': 'Low', 'practice': 'Low', 'demo': 'Low', 'medium': 'Medium', 'hard': 'High'}
EXTRA_META = {
    'cognitive_load_label': 'Target Low/Medium/High according to phase/difficulty protocol; every segment must have the same target.',
    'window_label_mix': 'JSON target-duration map in seconds, including unresolved duration when present.',
    'baseline_recorder_ready': 'Baseline recorder reported ready, independently of builder coverage eligibility.',
    'baseline_rejection_reason': 'Why the baseline ratio is unavailable; empty when eligible.',
}
META_INFO = {
    **shared.META_INFO,
    'window_start': 'Timezone-aware recomputed window start, excluded sensor endpoint.',
    'window_end': 'Timezone-aware recomputed window end, included sensor endpoint.',
    'window_seconds': 'Actual window length in seconds: 15, 25, or 45.',
    'study_phase': 'Source-ledger phase; mixed phases are retained when every segment has the same target.',
    'difficulty_level': 'Original task difficulty metadata; never the v2 target or a predictor.',
    'hrv_status': 'Readiness recomputed from raw RR with the original sensor thresholds.',
    'hrv_reason': 'Recomputed HRV readiness explanation.',
    'hrv_coverage': 'Recomputed accepted-RR duration / window duration.',
    'hrv_rr_count': 'Recomputed usable RR count.',
    'hrv_rr_pairs': 'Recomputed usable adjacent-RR pair count.',
    'hrv_data_age_seconds': 'Recomputed age of latest usable RR.',
    **EXTRA_META,
}
DATA_COLUMNS = list(META_INFO) + shared.FEATURE_COLUMNS


def segment_label(segment):
    phase = segment.get('study_phase')
    if phase in LOW_PHASES:
        return 'Low'
    if phase == 'active_task':
        return ACTIVE_LABELS.get(segment.get('difficulty_level', '').lower())
    return None


def label_window(segments):
    totals = {}
    for segment in segments:
        label = segment_label(segment) or 'unresolved'
        totals[label] = totals.get(label, 0.) + segment['duration_seconds']
    totals = {label: round(value, 6) for label, value in totals.items()}
    if not totals or 'unresolved' in totals:
        label, reason = '', 'unresolved_or_calibration'
    elif len(totals) != 1:
        label, reason = '', 'mixed_cognitive_labels'
    else:
        label, reason = next(iter(totals)), ''
    return {'cognitive_load_label': label, 'window_label_mix': json.dumps(totals, separators=(',', ':')),
            'exclusion_reason': reason}


def segments_for_window(start, end, events, ready_ms):
    """Do not round away short phase transitions before the unanimous-label test."""
    if not events:
        return []
    boundaries = {start, end}
    if start < ready_ms < end:
        boundaries.add(ready_ms)
    boundaries.update(e['event_time_ms'] for e in events if start < e['event_time_ms'] < end)
    ordered = sorted(boundaries)
    return [{**context_at((left+right)/2, events, ready_ms),
             'duration_seconds': (right-left)/1000, 'proportion': (right-left)/(end-start)}
            for left, right in zip(ordered, ordered[1:])]


def replay_rr(raw, task_origin, stop_seconds, windows=WINDOWS, before_boundary=False):
    """Replay original packet order; packets at the endpoint are included by default.

    Datetime subtraction preserves microseconds. Subtracting two epoch floats
    changes packet/window membership near exact deadlines in these recordings.
    The alternative publication ordering is used only to audit the live cache.
    """
    if not raw or task_origin.utcoffset() is None or not math.isfinite(stop_seconds) or stop_seconds <= 0:
        raise ValueError('Invalid raw RR, task origin, or recording end.')
    recording_origin = datetime.fromisoformat(raw[0]['received_at'])-timedelta(seconds=float(raw[0]['elapsed_seconds']))
    offset = (task_origin-recording_origin).total_seconds()
    processor = RRPreprocessor()
    streams = {seconds: LiveHRV(task_origin, window_seconds=seconds, step_seconds=5) for seconds in windows}
    results = {seconds: [] for seconds in windows}
    task_rows = [row for row in raw if row.get('phase') == 'task']
    if not task_rows:
        raise ValueError('No task-phase RR packets for replay.')
    for _, packet in groupby(task_rows, key=lambda row: row['packet_id']):
        packet = [dict(row, elapsed_seconds=float(row['elapsed_seconds'])-offset) for row in packet]
        arrived = packet[0]['elapsed_seconds']
        if arrived < -1e-6:
            raise ValueError('Task RR precedes the reconstructed task origin.')
        if arrived > stop_seconds+1e-9:
            break
        for seconds, stream in streams.items():
            results[seconds].extend(stream.advance(arrived, include_boundary=before_boundary))
        processed = processor.process_packet(packet)
        for stream in streams.values():
            stream.add(processed, arrived, processor.recovering)
    # Unreleased recovery is not accepted retroactively. LiveHRV retains the gap
    # and recovering status through the final recording boundary.
    for seconds, stream in streams.items():
        results[seconds].extend(stream.advance(stop_seconds))
    return results


def same_hrv(source, replay):
    for key in ('rr_count', 'rr_pairs', 'rmssd_ms', 'sdnn_ms', 'coverage', 'accepted_rr_seconds', 'data_age_seconds'):
        left, right = shared.finite(source.get(key)), shared.finite(replay.get(key))
        if math.isnan(left) and math.isnan(right):
            continue
        if not math.isclose(left, right, rel_tol=1e-7, abs_tol=1e-6):
            return False
    return source['status'] == replay['status'] and source.get('reason', '') == replay.get('reason', '')


def audit_replay(raw, original, origin, replay):
    """All cached windows must match a causal live ordering; no unexplained drift."""
    end = (datetime.fromisoformat(original[-1]['window_end'])-origin).total_seconds()
    alternative = replay_rr(raw, origin, end, (25,), before_boundary=True)[25]
    primary = {r['window_end']: r for r in replay}
    alternative = {r['window_end']: r for r in alternative}
    # Compare absolute instants: source timezone text can differ from replay.
    primary = {shared.epoch_ms(k): value for k, value in primary.items()}
    alternative = {shared.epoch_ms(k): value for k, value in alternative.items()}
    recording_origin = datetime.fromisoformat(raw[0]['received_at'])-timedelta(seconds=float(raw[0]['elapsed_seconds']))
    offset = (origin-recording_origin).total_seconds()
    arrivals = [float(r['elapsed_seconds'])-offset for r in raw if r.get('phase') == 'task']
    details = []
    for source in original:
        end_ms = shared.epoch_ms(source['window_end'])
        actual = primary.get(end_ms)
        if actual is not None and same_hrv(source, actual):
            continue
        boundary_seconds = (datetime.fromisoformat(source['window_end'])-origin).total_seconds()
        near_boundary = any(abs(t-boundary_seconds) <= 1e-7 for t in arrivals)
        other = alternative.get(end_ms)
        if not near_boundary or other is None or not same_hrv(source, other):
            raise ValueError(f"Unexplained cached HRV mismatch at {source['window_end']}")
        details.append({'window_end': source['window_end'], 'reason': 'live_timer_before_endpoint_packet',
                        'cached_rr_count': int(source['rr_count']), 'recomputed_rr_count': actual['rr_count']})
    return {'cached_windows': len(original), 'exact_replay_windows': len(original)-len(details),
            'endpoint_publication_differences': len(details), 'unexplained_windows': 0, 'differences': details}


def load_session(directory, config):
    eye = pd.read_csv(directory/'eye_tracking.csv', keep_default_na=False, low_memory=False)
    hrv = pd.read_csv(directory/'hrv.csv', keep_default_na=False)
    with (directory/'rr_raw.csv').open(encoding='utf-8-sig', newline='') as stream:
        raw = list(csv.DictReader(stream))
    if eye.empty or hrv.empty or not raw:
        raise ValueError(f'{directory.name}: empty sensor source.')
    ids = set(eye.session_id.astype(str)) | set(hrv.session_id.astype(str)) | {r.get('session_id', '') for r in raw}
    if len(ids) != 1 or not next(iter(ids)):
        raise ValueError(f'{directory.name}: inconsistent sensor session IDs.')
    session_id = next(iter(ids))
    ready = calibration_ready_ms(directory)
    if ready is None or not math.isfinite(ready):
        raise ValueError(f'{directory.name}: missing both-sensor calibration readiness.')
    original = hrv.loc[hrv.phase.eq('task')].to_dict('records')
    if not original:
        raise ValueError(f'{directory.name}: no source task window to anchor replay.')
    origin = datetime.fromisoformat(original[0]['window_start'])
    previous_endpoint = None
    for row in original:
        duration = (datetime.fromisoformat(row['window_end'])-datetime.fromisoformat(row['window_start'])).total_seconds()
        endpoint = (datetime.fromisoformat(row['window_end'])-origin).total_seconds()
        if (not math.isclose(duration, 25) or not math.isclose(shared.finite(row.get('window_seconds')), 25)
                or endpoint < 25 or (previous_endpoint is not None and endpoint <= previous_endpoint)
                or not math.isclose(endpoint/5, round(endpoint/5), abs_tol=1e-7)):
            raise ValueError(f'{directory.name}: invalid cached HRV grid.')
        previous_endpoint = endpoint
    events = pd.read_csv(directory/'task_events.csv', keep_default_na=False)
    if not events.empty and not events.session_id.astype(str).eq(session_id).all():
        raise ValueError(f'{directory.name}: ledger session ID mismatch.')
    events['event_time_ms'] = shared.number(events, 'event_time_ms')
    if events.event_time_ms.isna().any():
        raise ValueError(f'{directory.name}: invalid event timestamp.')
    events = events.sort_values('event_time_ms', kind='stable').to_dict('records')
    calibration = shared.read_json(directory/'eye_summary_calibration.json')
    summary = shared.read_json(directory/'eye_summary.json')
    screen = calibration.get('config', {}).get('screen') or summary.get('config', {}).get('screen', {})
    if not all(shared.finite(screen.get(axis)) > 0 for axis in ('width', 'height')):
        raise ValueError(f'{directory.name}: invalid screen dimensions.')
    eye = shared.prepare_eye(eye, screen, config)
    baseline_rows = hrv[hrv.phase.eq('baseline')]
    baseline = baseline_rows.iloc[-1] if len(baseline_rows) else {}
    baseline_rmssd, coverage = shared.finite(baseline.get('rmssd_ms')), shared.finite(baseline.get('coverage'))
    baseline_ready = baseline.get('status') == 'ready'
    reasons = []
    if not baseline_ready:
        reasons.append('baseline_not_ready')
    if not baseline_rmssd > 0:
        reasons.append('nonpositive_or_missing_rmssd')
    if not coverage >= config['baseline_min_coverage']:
        reasons.append('insufficient_baseline_coverage')
    baseline_usable = not reasons
    participant = participant_group(directory.name)
    group = next((name for name, members in config['participant_review_groups'].items()
                  if participant.casefold() in {v.casefold() for v in members}), 'unreviewed')
    validations = [a for a in calibration.get('attempts', []) if a.get('phase') == 'held_out_validation']
    validation = validations[-1] if validations else {}
    camera = summary.get('summary', {}).get('camera', {})
    metadata = {
        'participant_id': participant, 'session_id': session_id, 'session_folder': directory.name,
        'team_review_group': group, 'participant_note': shared.lookup(config['participant_notes'], participant, ''),
        'baseline_recorder_ready': baseline_ready, 'baseline_rejection_reason': ';'.join(reasons),
        'baseline_rmssd_ms': baseline_rmssd if baseline_usable else np.nan,
        'baseline_coverage': coverage, 'baseline_usable': baseline_usable,
        'screen_width_px': screen['width'], 'screen_height_px': screen['height'],
        'camera_width_px': shared.finite(camera.get('width')), 'camera_height_px': shared.finite(camera.get('height')),
        'camera_reported_fps': shared.finite(camera.get('fps_reported')),
        'camera_observed_capture_fps': shared.finite(camera.get('capture_fps_observed')),
        'gaze_validation_passed': validation.get('passed_thresholds', ''),
        'gaze_validation_median_px': shared.finite(validation.get('mean_target_error_median_px')),
        'gaze_validation_p95_px': shared.finite(validation.get('mean_target_error_p95_px')),
        'gaze_validation_max_corner_px': shared.finite(validation.get('max_corner_error_px')),
    }
    last_raw = datetime.fromisoformat(raw[-1]['received_at'])
    stop = max((datetime.fromisoformat(original[-1]['window_end'])-origin).total_seconds(),
               (last_raw-origin).total_seconds(), (float(eye.timestamp_ms.iloc[-1])-origin.timestamp()*1000)/1000)
    replay = replay_rr(raw, origin, stop)
    audit = {'session_folder': directory.name, **audit_replay(raw, original, origin, replay[25])}
    return eye, replay, events, ready, metadata, audit


def build_session(directory, config):
    eye, replay, events, ready, metadata, audit = load_session(directory, config)
    result = {}
    for seconds, windows in replay.items():
        rows, excluded = [], []
        for source in windows:
            start, end = (shared.epoch_ms(source[key]) for key in ('window_start', 'window_end'))
            segments = segments_for_window(start, end, events, ready)
            context = summarize_segments(segments)
            target = label_window(segments)
            reason = target.pop('exclusion_reason')
            identity = {'window_id': f"{metadata['session_id']}|{source['window_start']}|{source['window_end']}",
                        'window_start': source['window_start'], 'window_end': source['window_end'], 'window_seconds': seconds}
            if reason:
                excluded.append({**metadata, **identity, **context, **target, 'exclusion_reason': reason})
                continue
            rmssd, sdnn = shared.finite(source['rmssd_ms']), shared.finite(source['sdnn_ms'])
            usable = source['status'] == 'ready' and rmssd >= 0 and sdnn >= 0
            row = {
                **metadata, **identity, **context, **target,
                'repeated_task': context['difficulty_level'] in shared.lookup(config['repeated_difficulties_by_session'], directory.name, []),
                'hrv_status': source['status'], 'hrv_reason': source['reason'],
                'hrv_coverage': source['coverage'], 'hrv_rr_count': source['rr_count'], 'hrv_rr_pairs': source['rr_pairs'],
                'hrv_data_age_seconds': source['data_age_seconds'], 'hrv_usable': usable,
                'rmssd_ms': rmssd if usable else np.nan, 'sdnn_ms': sdnn if usable else np.nan,
                'rmssd_baseline_ratio': rmssd/metadata['baseline_rmssd_ms'] if usable and metadata['baseline_usable'] else np.nan,
                **shared.aggregate_eye(eye, start, end, config),
            }
            row.update(shared.quality_tier(row, config))
            rows.append(row)
        tiers = {tier: sum(r['quality_tier'] == tier for r in rows) for tier in shared.TIERS}
        report = {**metadata, 'window_seconds': seconds, 'candidate_windows': len(windows),
                  'retained_windows': len(rows), 'excluded_windows': len(excluded),
                  **{f'{tier}_windows': count for tier, count in tiers.items()}}
        result[seconds] = rows, excluded, report
    return result, audit


def unique_duration(frame):
    """Union sensor time, not overlapping row duration, per recording."""
    total = 0.
    for _, rows in frame.groupby('session_id'):
        covered_until = -math.inf
        for start, end in sorted((shared.epoch_ms(r.window_start), shared.epoch_ms(r.window_end)) for r in rows.itertuples()):
            total += max(0., end-max(start, covered_until))/1000
            covered_until = max(covered_until, end)
    return total


def export_window(destination, rows, excluded, reports, inventory, seconds):
    destination.mkdir()
    frame = pd.DataFrame(rows, columns=DATA_COLUMNS).replace([np.inf, -np.inf], np.nan)
    views = shared.dataset_views(frame)
    counts, distribution = {}, []
    for name, view in views.items():
        view.to_csv(destination/f'{name}.csv', index=False)
        counts[name] = {label: int(view.cognitive_load_label.eq(label).sum()) for label in LABELS}
        counts[name].update(windows=len(view), participants=int(view.participant_id.nunique()), unique_seconds=unique_duration(view))
        for label in LABELS:
            subset = view[view.cognitive_load_label.eq(label)]
            distribution.append({'dataset': name, 'label': label, 'windows': len(subset),
                                 'participants': subset.participant_id.nunique(), 'unique_seconds': unique_duration(subset)})
    pd.DataFrame(distribution).to_csv(destination/'class_distribution.csv', index=False)
    pd.DataFrame(inventory).to_csv(destination/'session_inventory.csv', index=False)
    pd.DataFrame(reports).to_csv(destination/'session_quality_summary.csv', index=False)
    excluded_columns = [*META_INFO, 'exclusion_reason']
    pd.DataFrame(excluded, columns=excluded_columns).to_csv(destination/'excluded_windows.csv', index=False)
    frame.groupby(['session_folder', 'cognitive_load_label', 'study_phase', 'quality_tier'], dropna=False).size().rename('windows').reset_index().to_csv(destination/'counts_by_session_label_phase_tier.csv', index=False)
    frame.groupby(['quality_tier', 'hrv_status', 'hrv_reason'], dropna=False).size().rename('windows').reset_index().to_csv(destination/'sensor_quality_counts.csv', index=False)
    dictionary = [{'column': key, 'role': 'target' if key == 'cognitive_load_label' else 'metadata', 'unit': '', 'description': value}
                  for key, value in META_INFO.items()]
    dictionary += [{'column': key, 'role': 'feature', 'unit': unit,
                    'description': description.replace('Recorded time-domain HRV', 'Recomputed time-domain HRV')}
                   for key, (unit, description) in shared.FEATURE_INFO.items()]
    pd.DataFrame(dictionary).to_csv(destination/'data_dictionary.csv', index=False)
    (destination/'feature_columns.json').write_text(json.dumps({'target': 'cognitive_load_label', 'group': 'participant_id',
                                                              'features': shared.FEATURE_COLUMNS}, indent=2)+'\n', encoding='utf-8')
    lines = [f'# Window {seconds} detik, step 5 detik', '',
             'Target: `cognitive_load_label`. Difficulty dan fase tetap metadata. Hanya window dengan satu target sepanjang interval yang masuk CSV.', '',
             '| Dataset | Low | Medium | High | Total | Peserta |', '|---|---:|---:|---:|---:|---:|']
    lines += [f"| {name} | {c['Low']} | {c['Medium']} | {c['High']} | {c['windows']} | {c['participants']} |" for name, c in counts.items()]
    lines += ['', 'Baca laporan induk untuk aturan fase, replay RR, admission peserta, dan penggunaan fitur. `class_distribution.csv` melaporkan durasi unik; penjumlahan durasi semua row akan menghitung overlap berulang.', '']
    (destination/'QUALITY_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return counts


def build_dataset(input_dir, output_dir, config_path=shared.DEFAULT_CONFIG, subjects=None):
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f'Preserve existing bundle; choose a new version directory: {output_dir}')
    if output_dir.is_relative_to(input_dir) or input_dir.is_relative_to(output_dir) or Path(__file__).resolve().is_relative_to(output_dir):
        raise ValueError('Output must be separate from raw sources and builder.')
    config = shared.load_config(config_path)
    sources_before = shared.snapshot(input_dir)
    directories, inventory = select_session_directories(input_dir, subjects)
    if not directories:
        raise ValueError('No selected sensor sessions.')
    results = {seconds: ([], [], []) for seconds in WINDOWS}
    audits = []
    for directory in directories:
        session, audit = build_session(directory, config)
        audits.append(audit)
        for seconds, (rows, excluded, report) in session.items():
            results[seconds][0].extend(rows)
            results[seconds][1].extend(excluded)
            results[seconds][2].append(report)
        print(f"{directory.name}: retained " + ', '.join(f'{s}s={len(session[s][0])}' for s in WINDOWS), flush=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.fusion-v2-', dir=output_dir.parent) as temporary:
        destination = Path(temporary)/'bundle'
        destination.mkdir()
        counts = {f'window_{s}s_step_5s': export_window(destination/f'window_{s}s_step_5s', *results[s], inventory, s) for s in WINDOWS}
        (destination/'replay_validation.json').write_text(json.dumps(audits, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        report_lines = ['# Dataset fusion v2: session phase', '',
            '## Label dan window', '',
            '`cognitive_load_label` adalah satu-satunya target. Active easy/practice/demo, pre-task setelah kedua kalibrasi selesai, feedback, rating, jeda antarset, dan post-task diberi Low. Rating/feedback setelah medium/hard juga Low. Active medium diberi Medium; active hard diberi High. Interrupted hanya menunjuk interval setelah event set_ended yang tercatat, tanpa event akhir penelitian berikutnya. Baseline/kalibrasi dan interval tanpa ledger/readiness yang dapat ditelusuri tidak diberi Low.', '',
            'Setiap bagian interval harus menghasilkan target yang sama. Campuran easy + rating tetap Low; medium + rating tidak masuk dataset. Aturan ini bukan majority vote. Window mixed-phase dapat dipertahankan; `window_phase_mix` dan `window_label_mix` menyimpan rinciannya. Difficulty/set/soal sumber tidak ditimpa agar sama dengan target.', '',
            '## Rekonstruksi sensor', '',
            'Window 15/25/45 detik dihitung ulang dari rr_raw.csv dan eye_tracking.csv dengan step 5 detik, memakai origin task dan grid yang sama per sesi. Tidak ada interpolasi, averaging RMSSD tersimpan, atau perubahan timestamp mentah. RR sebelum task sensor tidak masuk replay. Paket yang tiba setelah endpoint tidak dapat memperbaiki window yang telah lewat. Paket yang tiba tepat di endpoint diproses sebelum window diterbitkan. Recovery menunggu waktu rilisnya; pasangan RMSSD tidak melintasi RR terbuang atau segmen terputus.', '',
            'Syarat HRV tetap coverage 80%, minimal 10 pasangan, umur data maksimal 3 detik. Aturan eye dan admission peserta identik dengan config v1. good_only memerlukan proposed_good dan tier good; good_acceptable memerlukan proposed_good/questionable dan tier good/acceptable; all_data memuat semua window berlabel termasuk sensor poor. Peserta baru memerlukan review dan pembaruan daftar admission, bukan otomatis diterima.', '',
            'Baseline recorder ready dipisahkan dari baseline usable builder. Rasio hanya tersedia untuk baseline ready dengan RMSSD positif dan coverage minimal 80%. Penolakan tercatat pada baseline_rejection_reason; nilai hilang tetap kosong, bukan nol.', '',
            '## Validasi cache HRV 25 detik', '',
            f"Diperiksa {sum(a['cached_windows'] for a in audits)} window: {sum(a['exact_replay_windows'] for a in audits)} sesuai replay utama; {sum(a['endpoint_publication_differences'] for a in audits)} sesuai urutan timer live sebelum paket tepat di endpoint; tidak ada selisih yang tidak terjelaskan. Semua rincian ada di replay_validation.json. Dataset baru memakai aturan endpoint yang konsisten, tidak menyalin angka cache agar tampak identik.", '',
            '## Distribusi', '', '| Window | Dataset | Low | Medium | High | Total | Peserta |', '|---|---|---:|---:|---:|---:|---:|']
        for variant, views in counts.items():
            for name, c in views.items():
                report_lines.append(f"| {variant} | {name} | {c['Low']} | {c['Medium']} | {c['High']} | {c['windows']} | {c['participants']} |")
        report_lines += ['', '## Penyaringan window', '',
            '| Window | Kandidat | Target campuran | Kalibrasi/tidak terlacak | Berlabel, HRV tidak layak | Gagal admission peserta |',
            '|---|---:|---:|---:|---:|---:|']
        for seconds, (rows, excluded, reports) in results.items():
            mixed = sum(r['exclusion_reason'] == 'mixed_cognitive_labels' for r in excluded)
            unresolved = len(excluded)-mixed
            unusable = sum(not r['hrv_usable'] for r in rows)
            admission = sum(r['team_review_group'] == 'unreviewed' for r in rows)
            report_lines.append(f'| {seconds}s | {sum(r["candidate_windows"] for r in reports)} | {mixed} | {unresolved} | {unusable} | {admission} |')
        report_lines += ['', 'Kolom HRV/admission di atas merupakan diagnostik pada window berlabel dan dapat saling overlap. Tier poor juga dapat berasal dari kualitas eye; rincian kriteria tersimpan di CSV. all_data tidak membuang tier poor. CSV curated menerapkan quality dan admission bersama-sama.', '',
            '## Komposisi fase pada Low good_acceptable', '',
            '| Window | Fase sumber | Window Low |', '|---|---|---:|']
        for seconds, (rows, _, _) in results.items():
            frame = pd.DataFrame(rows, columns=DATA_COLUMNS)
            view = shared.dataset_views(frame)['good_acceptable']
            phases = view.loc[view.cognitive_load_label.eq('Low')].groupby('study_phase').size()
            report_lines += [f'| {seconds}s | {phase} | {count} |' for phase, count in phases.items()]
        report_lines += ['', 'mixed berarti beberapa fase dengan target Low yang sama. Detail per segmen ada pada window_phase_mix; jumlah window bukan durasi unik dan tidak menyatakan semua Low berasal dari pengerjaan easy.', '',
            '## Pemodelan berikutnya', '',
            'Gunakan 26 predictor sensor pada feature_columns.json, bukan semua kolom numerik. Target/difficulty/fase, nama peserta, ID soal, waktu, dan kualitas adalah metadata atau label, bukan X. Identitas tetap dipakai untuk grouping/deduplikasi. Imputasi, scaling, seleksi fitur, class weight dan oversampling dilakukan pada training fold, tidak pada CSV sumber. Belum ada training model atau rekaman Low tambahan pada build ini. Batch-01 tetap arsip; batch-02 memakai evaluasinya sendiri.', '',
            'Window overlap bukan observasi independen. Durasi unik dilaporkan per kelas/rekaman pada class_distribution.csv. Mutu sensor dan komposisi kelas dapat berubah antarwindow; jumlah row terbesar bukan bukti window terbaik. Endpoint pembanding dan manifest evaluasi batch-02 dibuat setelah data tambahan diterima.', '']
        (destination/'QUALITY_REPORT.md').write_text('\n'.join(report_lines), encoding='utf-8')
        if shared.snapshot(input_dir) != sources_before:
            raise RuntimeError('Raw sources changed during build; no bundle published.')
        code_paths = [Path(__file__).resolve(), Path(shared.__file__).resolve(), ROOT/'analysis_sessions/session_inventory.py',
                      ROOT/'fuse_and_analyze.py', ROOT/'hrv_monitor_prototype/calculate_hrv.py',
                      ROOT/'hrv_monitor_prototype/preprocess_rr.py', ROOT/'hrv_monitor_prototype/config.py']
        manifest = {
            'schema_version': 2, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
            'input_directory': str(input_dir), 'selected_folders': [p.name for p in directories],
            'window_seconds': list(WINDOWS), 'step_seconds': 5, 'target': 'cognitive_load_label',
            'label_policy': {'active_task': ACTIVE_LABELS, 'Low_phases': list(LOW_PHASES), 'window_rule': 'unanimous_target'},
            'quality_configuration': config, 'quality_configuration_sha256': shared.digest(config_path),
            'endpoint_packet_policy': 'include_available_packets_at_endpoint',
            'dataset_eye_review_groups': shared.DATASET_REVIEW_GROUPS,
            'source_integrity_verified': True, 'source_sha256': sources_before,
            'code_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): shared.digest(p) for p in code_paths},
            'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
            'datasets': counts, 'feature_columns': shared.FEATURE_COLUMNS,
            'output_sha256': shared.snapshot(destination),
        }
        (destination/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        os.rename(destination, output_dir)
        try:
            # A Windows rename retains the private temporary parent's ACL.
            # Reset AFTER moving, so inheritance comes from the real output
            # parent, and propagate to every generated child as well.
            if os.name == 'nt':
                import subprocess
                subprocess.run(['icacls', str(output_dir), '/reset', '/T', '/q'], check=True, capture_output=True)
        except Exception:
            os.rename(output_dir, destination)
            raise
    print(json.dumps(counts, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'sessions/saved_session')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('output')/'v2_session_phase')
    parser.add_argument('--config', type=Path, default=shared.DEFAULT_CONFIG)
    parser.add_argument('--subjects', nargs='+')
    args = parser.parse_args()
    build_dataset(args.input, args.output, args.config, args.subjects)


if __name__ == '__main__':
    main()
