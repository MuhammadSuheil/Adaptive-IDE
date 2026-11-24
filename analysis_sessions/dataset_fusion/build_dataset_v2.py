"""Recompute aligned sensor windows with session-phase cognitive-load labels.

Raw recordings are read-only. Complete verified bundles can replace an existing
v2 output explicitly, with rollback if publication fails.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from io import StringIO
from itertools import groupby, pairwise
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis_sessions.dataset_fusion import build_dataset as shared
from analysis_sessions.dataset_fusion import v2_features as sensor_features
from analysis_sessions.dataset_fusion import v2_phase_policy as phases
from analysis_sessions.dataset_fusion import v2_history as history
from analysis_sessions.dataset_fusion import v2_replacement as replacement
from analysis_sessions.dataset_fusion import v2_timestamps as timing
from analysis_sessions.dataset_fusion.v2_documentation import (
    build_update_summary,
    render_output_index,
    report_sections,
)
from analysis_sessions.dataset_fusion.v2_integrity import (
    ADMISSION_POLICY,
    guard_output,
    publish_bundle,
    quality_views,
    validate_frame,
    verify_bundle,
)
from analysis_sessions.dataset_fusion.v2_sensor_checks import audit_sensor_samples
from analysis_sessions.session_inventory import (
    participant_group,
    select_session_directories,
)
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
    'team_review_group': 'Historical team review category; metadata only, never a v2 admission gate or predictor.',
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


def segment_label(segment, policy=None):
    if policy == phases.LABEL_POLICY:
        return phases.segment_label(segment)
    phase = segment.get('study_phase')
    if phase in LOW_PHASES:
        return 'Low'
    if phase == 'active_task':
        return ACTIVE_LABELS.get(segment.get('difficulty_level', '').lower())
    return None


def label_window(segments, policy=None):
    if policy == phases.LABEL_POLICY:
        return phases.label_window(segments)
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
            for left, right in pairwise(ordered)]


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


def load_session(directory, config, timestamp_policy=None):
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
    if timestamp_policy:
        timing.check_policy(directory, eye, timestamp_policy)
    eye = shared.prepare_eye(eye, screen, config, allow_equal_timestamps=bool(timestamp_policy))
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
    audit['source_hrv_status_counts'] = {status: sum(r['status'] == status for r in original)
                                       for status in sorted({r['status'] for r in original})}
    if timestamp_policy:
        audit['eye_timestamp_audit'] = {**timestamp_policy, 'session_id': session_id}
    audit['independent_sensor_check'] = audit_sensor_samples(raw, eye, replay, origin, config)
    return eye, replay, events, ready, metadata, audit


def build_session(directory, config, timestamp_policy=None, *, label_policy=None):
    eye, replay, events, ready, metadata, audit = load_session(directory, config, timestamp_policy)
    if label_policy not in (None, phases.LABEL_POLICY):
        raise ValueError('Unsupported label policy.')
    timeline = phases.PhaseTimeline(events, ready) if label_policy else None
    result = {}
    for seconds, windows in replay.items():
        rows, excluded = [], []
        for source in windows:
            start, end = (shared.epoch_ms(source[key]) for key in ('window_start', 'window_end'))
            segments = timeline.segments(start, end) if timeline else segments_for_window(start, end, events, ready)
            context = summarize_segments(segments)
            target = label_window(segments, label_policy)
            reason = target.pop('exclusion_reason')
            identity = {'window_id': f"{metadata['session_id']}|{source['window_start']}|{source['window_end']}",
                        'window_start': source['window_start'], 'window_end': source['window_end'], 'window_seconds': seconds}
            touched = timing.touches(start, end, timestamp_policy)
            if reason or touched:
                flags = ([reason] if reason else []) + (['ambiguous_eye_timestamp'] if touched else [])
                excluded.append({**metadata, **identity, **context, **target,
                                 'exclusion_reason': reason or 'ambiguous_eye_timestamp',
                                 'exclusion_flags': json.dumps(flags)})
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
        if timestamp_policy:
            report['timestamp_affected_windows'] = sum('ambiguous_eye_timestamp' in r['exclusion_flags'] for r in excluded)
            report['timestamp_only_excluded_windows'] = sum(r['exclusion_reason'] == 'ambiguous_eye_timestamp' for r in excluded)
        result[seconds] = rows, excluded, report
    if timestamp_policy:
        audit['eye_timestamp_audit']['windows'] = {
            str(s): {'affected': result[s][2]['timestamp_affected_windows'],
                     'additional_excluded': result[s][2]['timestamp_only_excluded_windows']} for s in WINDOWS}
    if timeline:
        audit['phase_context_source'] = {'events': events, 'ready_ms': ready,
            'task_events_sha256': shared.digest(directory/'task_events.csv')}
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


def export_window(destination, rows, excluded, reports, inventory, seconds, *, config=None, policy=ADMISSION_POLICY,
                  validation_checks=None, label_policy=None, curation_policy=None):
    destination.mkdir()
    config = config if config is not None else shared.load_config()
    metadata = {**META_INFO, **(replacement.CURATION_META if policy in (*history.POLICIES, sensor_features.POLICY) else {})}
    if policy == sensor_features.POLICY:
        metadata.update(quality_tier='Core-feature quality tier, not spatial-gaze accuracy.',
                        good_failed_criteria='Failed core good rules; empty when all pass.',
                        acceptable_failed_criteria='Failed core acceptable rules; empty when all pass.',
                        curation_selected='Selected by core quality; never a predictor.',
                        curation_reason='Core selection or rejection reason; never a predictor.')
    frame = pd.DataFrame(rows, columns=[*metadata, *shared.FEATURE_COLUMNS]).replace([np.inf, -np.inf], np.nan)
    if policy == sensor_features.POLICY:
        frame = sensor_features.apply_policy(frame, config)
    if curation_policy == phases.CURATION_POLICY:
        metadata.update(curation_selected='Selected by core quality and balanced phase curation; never a predictor.',
                        curation_reason='Basis/optional selection, balance non-selection, or sensor-quality reason.')
        frame, selection_summary = phases.curate(frame)
        (destination/'curation_summary.json').write_text(json.dumps(selection_summary, indent=2)+'\n', encoding='utf-8')
    validate_frame(frame, config=config, seconds=seconds, policy=policy,
                   label_policy=label_policy, curation_policy=curation_policy)
    views = quality_views(frame, policy, config=config, validation_checks=validation_checks,
                          curation_policy=curation_policy)
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
    excluded_columns = [*META_INFO, 'exclusion_reason', 'exclusion_flags']
    pd.DataFrame(excluded, columns=excluded_columns).to_csv(destination/'excluded_windows.csv', index=False)
    frame.groupby(['session_folder', 'cognitive_load_label', 'study_phase', 'quality_tier'], dropna=False).size().rename('windows').reset_index().to_csv(destination/'counts_by_session_label_phase_tier.csv', index=False)
    frame.groupby(['quality_tier', 'hrv_status', 'hrv_reason'], dropna=False).size().rename('windows').reset_index().to_csv(destination/'sensor_quality_counts.csv', index=False)
    dictionary = [{'column': key, 'role': 'target' if key == 'cognitive_load_label' else 'metadata', 'unit': '', 'description': value}
                  for key, value in metadata.items()]
    dictionary += [{'column': key, 'role': ('feature' if key in sensor_features.CORE else 'optional_or_audit')
                    if policy == sensor_features.POLICY else 'feature', 'unit': unit,
                    'description': description.replace('Recorded time-domain HRV', 'Recomputed time-domain HRV')}
                   for key, (unit, description) in shared.FEATURE_INFO.items()]
    pd.DataFrame(dictionary).to_csv(destination/'data_dictionary.csv', index=False)
    feature_contract = (sensor_features.contract(config, validation_checks, curation_policy=curation_policy) if policy == sensor_features.POLICY else
                        {'target': 'cognitive_load_label', 'group': 'participant_id', 'features': shared.FEATURE_COLUMNS})
    (destination/'feature_columns.json').write_text(json.dumps(feature_contract, indent=2)+'\n', encoding='utf-8')
    lines = [f'# Window {seconds} detik, step 5 detik', '',
             'Target: `cognitive_load_label`. Difficulty dan fase tetap metadata. Hanya window dengan satu target sepanjang interval yang masuk CSV.', '',
             '| Dataset | Low | Medium | High | Total | Peserta |', '|---|---:|---:|---:|---:|---:|']
    lines += [f"| {name} | {c['Low']} | {c['Medium']} | {c['High']} | {c['windows']} | {c['participants']} |" for name, c in counts.items()]
    lines += ['', 'Baca laporan induk untuk aturan fase, replay RR, admission peserta, dan penggunaan fitur. `class_distribution.csv` melaporkan durasi unik; penjumlahan durasi semua row akan menghitung overlap berulang.', '']
    (destination/'QUALITY_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return counts


def build_dataset(input_dir, output_dir, config_path=shared.DEFAULT_CONFIG, subjects=None, *, replace=False,
                  replacement_session=None, exclude_tied_eye_windows=False, expected_replacement_counts=None,
                  admission_policy=None, label_policy=None):
    input_dir = Path(input_dir).resolve()
    output_dir = guard_output(input_dir, output_dir, replace=replace)
    previous_manifest = verify_bundle(output_dir) if output_dir.exists() else None
    if admission_policy not in (None, ADMISSION_POLICY, sensor_features.POLICY):
        raise ValueError('Unsupported explicit admission policy.')
    requested = admission_policy or (previous_manifest or {}).get('admission_policy', ADMISSION_POLICY)
    inherited_label = (previous_manifest or {}).get('label_policy', {}).get('version')
    label_policy = label_policy or inherited_label
    if label_policy not in (None, phases.LABEL_POLICY):
        raise ValueError('Unsupported label policy.')
    if label_policy and requested != sensor_features.POLICY:
        raise ValueError('Context phase labels require feature-aware admission.')
    if requested == sensor_features.POLICY:
        if replacement_session or exclude_tied_eye_windows or expected_replacement_counts is not None:
            raise ValueError('Feature-aware admission cannot use count-preserving replacement options.')
        from analysis_sessions.dataset_fusion.v2_feature_build import (
            build_feature_dataset,
        )
        return build_feature_dataset(input_dir, output_dir, config_path, subjects, previous_manifest,
                                     replace=replace, label_policy=label_policy)
    if previous_manifest and previous_manifest['admission_policy'] == sensor_features.POLICY:
        raise ValueError('Do not silently downgrade a feature-aware bundle to a legacy policy.')
    selection = history.selection_context(output_dir, previous_manifest, replacement_session)
    policy = selection['policy'] if selection else ADMISSION_POLICY
    if exclude_tied_eye_windows and (not replacement_session or policy != history.POLICY):
        raise ValueError('Timestamp exclusion requires an explicit new staged replacement session.')
    if expected_replacement_counts is not None:
        if not replacement_session or policy != history.POLICY:
            raise ValueError('Expected counts require an explicit staged replacement session.')
        stage = next(s for s in history.stages(selection) if s['candidate_session_ids'] == [replacement_session])
        if 'expected_counts' in stage and stage['expected_counts'] != expected_replacement_counts:
            raise ValueError('Do not change expected counts in an existing audited stage.')
        stage['expected_counts'] = expected_replacement_counts
    config = shared.load_config(config_path)
    sources_before = shared.snapshot(input_dir)
    directories, inventory = select_session_directories(input_dir, subjects,
        session_exclusions=config.get('session_exclusions', {}))
    if not directories:
        raise ValueError('No selected sensor sessions.')
    for directory in directories:
        if not (directory/'task_events.csv').is_file():
            raise ValueError(f'{directory.name}: missing task_events.csv; source is not explicitly excluded.')
    results = {seconds: ([], [], []) for seconds in WINDOWS}
    audits = []
    for directory in directories:
        session_id = str(pd.read_csv(directory/'eye_tracking.csv', nrows=1, usecols=['session_id']).session_id.iloc[0])
        timestamp_policy = selection.get('timestamp_policies', {}).get(session_id) if selection else None
        if exclude_tied_eye_windows and session_id == replacement_session and timestamp_policy is None:
            timestamp_policy = timing.make_policy(directory)
            selection['timestamp_policies'][session_id] = timestamp_policy
        session, audit = build_session(directory, config, timestamp_policy)
        audits.append(audit)
        for seconds, (rows, excluded, report) in session.items():
            results[seconds][0].extend(rows)
            results[seconds][1].extend(excluded)
            results[seconds][2].append(report)
        print(f"{directory.name}: retained " + ', '.join(f'{s}s={len(session[s][0])}' for s in WINDOWS), flush=True)
    selected_frames, pairs = {}, []
    if selection:
        history.verify_context(selection)
        allowed = set(selection['reference_session_ids']) | set(selection['candidate_session_ids'])
        if {r['session_id'] for r in results[25][2]} != allowed:
            raise ValueError('Replacement source cohort changed; update the selection plan explicitly.')
        if config != previous_manifest['quality_configuration']:
            raise ValueError('Do not change quality configuration during count-preserving replacement.')
        for seconds, (rows, excluded, reports) in results.items():
            frame = pd.DataFrame(rows, columns=DATA_COLUMNS).replace([np.inf, -np.inf], np.nan)
            before = pd.read_csv(output_dir/f'window_{seconds}s_step_5s/all_data.csv')
            original_columns = [c for c in before if c not in replacement.CURATION_META]
            normalized = pd.read_csv(StringIO(frame.to_csv(index=False))).set_index('window_id')
            pd.testing.assert_frame_equal(before[original_columns].set_index('window_id'),
                normalized.loc[before.window_id, [c for c in original_columns if c != 'window_id']],
                check_dtype=False, rtol=1e-10, atol=1e-10)
            selected, matches = history.curate_history(frame, selection, seconds)
            if policy == history.POLICY and previous_manifest['admission_policy'] in history.POLICIES:
                previous_pairs = pd.read_csv(output_dir/'replacement_pairs.csv')
                old_pairs = previous_pairs[previous_pairs.window_seconds.eq(seconds)].copy()
                if 'selection_stage' not in old_pairs:
                    old_pairs['selection_stage'] = 1
                kept_pairs = pd.DataFrame(matches, columns=history.PAIR_COLUMNS)
                kept_pairs = kept_pairs[kept_pairs.selection_stage.le(old_pairs.selection_stage.max())]
                pd.testing.assert_frame_equal(old_pairs[history.PAIR_COLUMNS].reset_index(drop=True),
                    kept_pairs.reset_index(drop=True), check_dtype=False, rtol=1e-10, atol=1e-10)
            selected_frames[seconds] = selected
            pairs.extend(matches)
            results[seconds] = (selected.to_dict('records'), excluded, reports)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.fusion-v2-', dir=output_dir.parent) as temporary:
        destination = Path(temporary)/'bundle'
        destination.mkdir()
        counts = {f'window_{s}s_step_5s': export_window(destination/f'window_{s}s_step_5s', *results[s], inventory, s, config=config, policy=policy) for s in WINDOWS}
        update_summary = build_update_summary(results, counts, inventory, previous_manifest, policy=policy)
        if selection:
            (destination/'replacement_selection.json').write_text(json.dumps(selection, indent=2)+'\n', encoding='utf-8')
            pd.DataFrame(pairs, columns=history.pair_columns(policy)).to_csv(destination/'replacement_pairs.csv', index=False)
            update_summary['replacement_policy'] = policy
            update_summary['replacement_counts'] = {
                str(s): {label: sum(p['window_seconds'] == s and p['cognitive_load_label'] == label for p in pairs)
                         for label in LABELS} for s in WINDOWS}
            update_summary['replacement_stages'] = [
                {'stage': i, 'candidate_session_ids': stage['candidate_session_ids'],
                 'counts': {str(s): {label: sum(p['window_seconds'] == s and p['cognitive_load_label'] == label
                                      and p.get('selection_stage', 1) == i for p in pairs) for label in LABELS}
                            for s in WINDOWS}}
                for i, stage in enumerate(history.stages(selection), 1)]
        (destination/'replay_validation.json').write_text(json.dumps(audits, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        report_lines = ['# Dataset fusion v2: session phase', '',
            '## Label dan window', '',
            '`cognitive_load_label` adalah satu-satunya target. Active easy/practice/demo, pre-task setelah kedua kalibrasi selesai, feedback, rating, jeda antarset, dan post-task diberi Low. Rating/feedback setelah medium/hard juga Low. Active medium diberi Medium; active hard diberi High. Interrupted hanya menunjuk interval setelah event set_ended yang tercatat, tanpa event akhir penelitian berikutnya. Baseline/kalibrasi dan interval tanpa ledger/readiness yang dapat ditelusuri tidak diberi Low.', '',
            'Setiap bagian interval harus menghasilkan target yang sama. Campuran easy + rating tetap Low; medium + rating tidak masuk dataset. Aturan ini bukan majority vote. Window mixed-phase dapat dipertahankan; `window_phase_mix` dan `window_label_mix` menyimpan rinciannya. Difficulty/set/soal sumber tidak ditimpa agar sama dengan target.', '',
            '## Rekonstruksi sensor', '',
            'Window 15/25/45 detik dihitung ulang dari rr_raw.csv dan eye_tracking.csv dengan step 5 detik, memakai origin task dan grid yang sama per sesi. Tidak ada interpolasi, averaging RMSSD tersimpan, atau perubahan timestamp mentah. RR sebelum task sensor tidak masuk replay. Paket yang tiba setelah endpoint tidak dapat memperbaiki window yang telah lewat. Paket yang tiba tepat di endpoint diproses sebelum window diterbitkan. Recovery menunggu waktu rilisnya; pasangan RMSSD tidak melintasi RR terbuang atau segmen terputus.', '',
            'Syarat HRV tetap coverage 80%, minimal 10 pasangan, umur data maksimal 3 detik. Ambang eye tetap mengikuti config. Admission v2 tidak menggunakan daftar nama: good_only memuat tier good; good_acceptable memuat tier good/acceptable; all_data memuat semua window berlabel termasuk sensor poor. team_review_group hanya metadata historis. Tier menilai kelengkapan/clipping, bukan akurasi spasial gaze; hasil validasi asli tetap dipertahankan.', '',
            'Baseline recorder ready dipisahkan dari baseline usable builder. Rasio hanya tersedia untuk baseline ready dengan RMSSD positif dan coverage minimal 80%. Penolakan tercatat pada baseline_rejection_reason; nilai hilang tetap kosong, bukan nol.', '',
            '## Validasi cache HRV 25 detik', '',
            f"Diperiksa {sum(a['cached_windows'] for a in audits)} window: {sum(a['exact_replay_windows'] for a in audits)} sesuai replay utama; {sum(a['endpoint_publication_differences'] for a in audits)} sesuai urutan timer live sebelum paket tepat di endpoint; tidak ada selisih yang tidak terjelaskan. Semua rincian ada di replay_validation.json. Dataset baru memakai aturan endpoint yang konsisten, tidak menyalin angka cache agar tampak identik.", '',
            '## Distribusi', '', '| Window | Dataset | Low | Medium | High | Total | Peserta |', '|---|---|---:|---:|---:|---:|---:|']
        for variant, views in counts.items():
            for name, c in views.items():
                report_lines.append(f"| {variant} | {name} | {c['Low']} | {c['Medium']} | {c['High']} | {c['windows']} | {c['participants']} |")
        report_lines += ['', '## Penyaringan window', '',
            '| Window | Kandidat | Target campuran | Kalibrasi/tidak terlacak | Timestamp ambigu saja | Berlabel, HRV tidak layak | Tier poor |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for seconds, (rows, excluded, reports) in results.items():
            mixed = sum(r['exclusion_reason'] == 'mixed_cognitive_labels' for r in excluded)
            ambiguous = sum(r['exclusion_reason'] == 'ambiguous_eye_timestamp' for r in excluded)
            unresolved = len(excluded)-mixed-ambiguous
            unusable = sum(not r['hrv_usable'] for r in rows)
            poor = sum(r['quality_tier'] == 'poor' for r in rows)
            report_lines.append(f'| {seconds}s | {sum(r["candidate_windows"] for r in reports)} | {mixed} | {unresolved} | {ambiguous} | {unusable} | {poor} |')
        report_lines += ['', 'Kolom HRV/tier poor merupakan diagnostik pada window berlabel dan dapat saling overlap. Tier poor juga dapat berasal dari kualitas eye; rincian kriteria tersimpan di CSV. all_data tidak membuang tier poor. CSV curated hanya menerapkan tier sensor, tanpa pembatasan nama peserta.', '',
            '## Komposisi fase pada Low good_acceptable', '',
            '| Window | Fase sumber | Window Low |', '|---|---|---:|']
        for seconds, (rows, _, _) in results.items():
            frame = pd.DataFrame(rows, columns=DATA_COLUMNS)
            if selection:
                frame = selected_frames[seconds]
            view = quality_views(frame, policy)['good_acceptable']
            phase_counts = view.loc[view.cognitive_load_label.eq('Low')].groupby('study_phase').size()
            report_lines += [f'| {seconds}s | {phase} | {count} |' for phase, count in phase_counts.items()]
        report_lines += ['', 'mixed berarti beberapa fase dengan target Low yang sama. Detail per segmen ada pada window_phase_mix; jumlah window bukan durasi unik dan tidak menyatakan semua Low berasal dari pengerjaan easy.', '',
            '## Pemodelan berikutnya', '',
            'Gunakan 26 predictor sensor pada feature_columns.json, bukan semua kolom numerik. Target/difficulty/fase, nama peserta, ID soal, waktu, dan kualitas adalah metadata atau label, bukan X. Identitas tetap dipakai untuk grouping/deduplikasi. Imputasi, scaling, seleksi fitur, class weight dan oversampling dilakukan pada training fold, tidak pada CSV sumber. Builder tidak melatih model; sumber tambahan yang diterima dijelaskan pada ringkasan pembaruan. Batch-01 tetap arsip; batch-02 memakai evaluasinya sendiri.', '',
            'Window overlap bukan observasi independen. Durasi unik dilaporkan per kelas/rekaman pada class_distribution.csv. Mutu sensor dan komposisi kelas dapat berubah antarwindow; jumlah row terbesar bukan bukti window terbaik. Endpoint pembanding dan manifest evaluasi batch-02 dibuat setelah data tambahan diterima.', '']
        report_lines += report_sections(results, audits, config, update_summary)
        if selection:
            report_lines += replacement.report_lines(selected_frames, selection, pairs)
            if selection.get('timestamp_policies'):
                report_lines += ['', '## Timestamp eye ambigu', '',
                    ('Penanganan hanya berlaku untuk sesi yang disetujui secara eksplisit pada timestamp_policies. '
                     'Seluruh frame, urutan, dan waktu mentah dipertahankan, termasuk urutan increment kedipan. '
                     'Tidak ada interpolasi atau timestamp buatan. Setiap window dengan start < timestamp ambigu <= end '
                     'dikeluarkan dari semua view dan dicatat pada excluded_windows. exclusion_flags juga mempertahankan '
                     'alasan target campuran jika keduanya terjadi. Sumber dipin dengan SHA-256; perubahan file memerlukan audit baru. '
                     'Rincian kelompok dan jumlah window terdampak ada di DATASET_CONDITION_REPORT.md dan replay_validation.json.'), '']
            report_lines = [line.replace(
                'Admission v2 tidak menggunakan daftar nama: good_only memuat tier good; good_acceptable memuat tier good/acceptable; all_data memuat semua window berlabel termasuk sensor poor.',
                'Admission curated memakai tier sensor serta seleksi penggantian yang tercatat. good_only berisi good terpilih; good_acceptable berisi good/acceptable terpilih; all_data memuat window berlabel dengan waktu tidak ambigu, termasuk poor dan kandidat tidak terpilih.'
            ).replace('CSV curated hanya menerapkan tier sensor, tanpa pembatasan nama peserta.',
                      'CSV curated menerapkan tier sensor dan kebijakan penggantian yang tercatat.') for line in report_lines]
            prior_report = output_dir/'DATASET_CONDITION_REPORT.md'
            operator_guide = ''
            if prior_report.exists():
                text = prior_report.read_text(encoding='utf-8')
                start = text.find('## 7. Panduan operator sebelum dan saat pengambilan')
                if start >= 0:
                    stop = text.find('\n## ', start+4)
                    operator_guide = text[start:stop if stop >= 0 else len(text)]
            (destination/'DATASET_CONDITION_REPORT.md').write_text(
                replacement.condition_report(selected_frames, selection, pairs, results[25][2], audits, operator_guide),
                encoding='utf-8')
        (destination/'QUALITY_REPORT.md').write_text('\n'.join(report_lines), encoding='utf-8')
        if shared.snapshot(input_dir) != sources_before:
            raise RuntimeError('Raw sources changed during build; no bundle published.')
        code_paths = [Path(__file__).resolve(), Path(shared.__file__).resolve(), Path(__file__).with_name('v2_integrity.py'),
                      Path(__file__).with_name('v2_sensor_checks.py'), Path(__file__).with_name('v2_documentation.py'),
                      Path(replacement.__file__).resolve(),
                      Path(history.__file__).resolve(), Path(timing.__file__).resolve(),
                      ROOT/'analysis_sessions/session_inventory.py',
                      ROOT/'fuse_and_analyze.py', ROOT/'hrv_monitor_prototype/calculate_hrv.py',
                      ROOT/'hrv_monitor_prototype/preprocess_rr.py', ROOT/'hrv_monitor_prototype/config.py']
        manifest = {
            'schema_version': 2, 'generated_at_utc': datetime.now(UTC).isoformat(),
            'input_directory': str(input_dir), 'selected_folders': [p.name for p in directories],
            'selected_session_ids': [r['session_id'] for r in results[25][2]],
            'excluded_sessions': [r for r in inventory if r['reason'].startswith('excluded_session:')],
            'update_summary': update_summary,
            'window_seconds': list(WINDOWS), 'step_seconds': 5, 'target': 'cognitive_load_label',
            'label_policy': {'active_task': ACTIVE_LABELS, 'Low_phases': list(LOW_PHASES), 'window_rule': 'unanimous_target'},
            'quality_configuration': config, 'quality_configuration_sha256': shared.digest(config_path),
            'endpoint_packet_policy': 'include_available_packets_at_endpoint',
            'admission_policy': policy,
            'dataset_eye_review_groups': {name: None for name in shared.DATASETS},
            'source_integrity_verified': True, 'source_sha256': sources_before,
            'code_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): shared.digest(p) for p in code_paths},
            'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
            'datasets': counts, 'feature_columns': shared.FEATURE_COLUMNS,
            'output_sha256': shared.snapshot(destination),
        }
        (destination/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        archive_csv = output_dir.parent/'v1_active_task/good_acceptable.csv'
        index = Path(temporary)/'README.md'
        if output_dir.name == 'v2_session_phase' and archive_csv.exists():
            index.write_text(render_output_index(archive_csv, counts, update_summary), encoding='utf-8')
        publish_bundle(destination, output_dir, replace=replace)
        if index.exists():
            shared.publish_artifact(index, output_dir.parent/'README.md')
    print(json.dumps(counts, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'sessions/saved_session')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('output')/'v2_session_phase')
    parser.add_argument('--config', type=Path, default=shared.DEFAULT_CONFIG)
    parser.add_argument('--subjects', nargs='+')
    parser.add_argument('--replace', action='store_true', help='Replace a verified v2 bundle; restore it if publication fails.')
    parser.add_argument('--admission-policy', choices=(ADMISSION_POLICY, sensor_features.POLICY),
                        help='Explicit feature-policy migration; omitted rebuilds inherit the active bundle policy.')
    parser.add_argument('--replacement-session', help='Replace weaker curated reference windows using this new session ID.')
    parser.add_argument('--phase-policy', choices=(phases.LABEL_POLICY,),
                        help='Explicit migration to task-context phase labels and balanced primary datasets.')
    parser.add_argument('--exclude-tied-eye-windows', action='store_true',
                        help='Audit ties in the explicit replacement session and exclude all touched windows.')
    args = parser.parse_args()
    build_dataset(args.input, args.output, args.config, args.subjects, replace=args.replace,
                  replacement_session=args.replacement_session, exclude_tied_eye_windows=args.exclude_tied_eye_windows,
                  admission_policy=args.admission_policy, label_policy=args.phase_policy)


if __name__ == '__main__':
    main()
