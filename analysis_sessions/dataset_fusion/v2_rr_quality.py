"""Label-blind RR uncertainty exclusion; no replacement or editing of measurements."""

import argparse
import csv
import math
from datetime import datetime, timedelta
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd

from hrv_monitor_prototype.preprocess_rr import (
    HISTORY_SIZE,
    MIN_HISTORY,
    RRPreprocessor,
)

from . import build_dataset as shared

POLICY = 'exclude_local_rr_jump_v1'
MARKER = 'rr_local_median_jump'
THRESHOLD_MS = 250.


def settings():
    return {'policy': POLICY, 'threshold_ms': THRESHOLD_MS, 'history_size': HISTORY_SIZE,
            'minimum_reference_rr': MIN_HISTORY, 'reference': 'packet_causal_preprocessor_reference_median',
            'action': 'exclude_affected_window_from_curated_without_replacing_rr',
            'window_membership': '(start,end] and available_at<=end'}


def audit_session(directory):
    directory = Path(directory)
    with (directory/'rr_raw.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = [row for row in csv.DictReader(stream) if row.get('phase') == 'task']
    ids = {row.get('session_id') for row in rows}
    if not rows or len(ids) != 1 or not next(iter(ids)):
        raise ValueError('RR quality audit requires one identified task stream.')
    # Mirror timing/contact/range/recovery checks, but never turn the live median filter on.
    processor = RRPreprocessor(filter_enabled=False, threshold_ms=THRESHOLD_MS)
    candidates, accepted = [], 0
    for _, packet in groupby(rows, key=lambda row: row['packet_id']):
        for row in processor.process_packet(list(packet)):
            if row['rr_processed_ms'] is None:
                continue
            accepted += 1
            reference = shared.finite(row['reference_median_ms'])
            value = float(row['rr_original_ms'])
            deviation = abs(value-reference)
            if math.isfinite(reference) and deviation > THRESHOLD_MS:
                assert processor.origin is not None
                available = processor.origin+timedelta(seconds=float(row['available_seconds']))
                candidates.append({'raw_id': row['raw_id'], 'packet_id': row['packet_id'],
                                   'beat_time': row['beat_time'], 'available_at': available.isoformat(),
                                   'rr_original_ms': value, 'reference_median_ms': reference,
                                   'deviation_ms': deviation})
    return {'session_id': next(iter(ids)), 'session_folder': directory.name,
            'rr_raw_sha256': shared.digest(directory/'rr_raw.csv'), 'task_rr_count': len(rows),
            'accepted_rr_count': accepted, 'candidates': candidates}


def affected_windows(frame, audit):
    affected: set[str] = set()
    selected = frame.loc[frame.session_id.eq(audit['session_id'])]
    if selected.empty:
        return affected
    start = np.array([pd.Timestamp(v).value for v in selected.window_start], dtype=np.int64)
    end = np.array([pd.Timestamp(v).value for v in selected.window_end], dtype=np.int64)
    for point in audit['candidates']:
        beat = pd.Timestamp(point['beat_time']).value
        available = pd.Timestamp(point['available_at']).value
        affected.update(selected.loc[(start < beat) & (beat <= end) & (available <= end), 'window_id'])
    return affected


def reason_without_marker(value):
    text = '' if pd.isna(value) else str(value)
    return ';'.join(token for token in text.split(';') if token and token != MARKER)


def apply_window_policy(frame, audit):
    result = frame.copy()
    affected = affected_windows(frame, audit)
    reason = result.hrv_reason.map(reason_without_marker)
    result['hrv_reason'] = [
        ';'.join(filter(None, (value, MARKER))) if identity in affected else value
        for identity, value in zip(result.window_id, reason)]
    return result


def window_summary(frame, config):
    from . import v2_features as features

    marked = frame.hrv_reason.fillna('').str.split(';').map(lambda tokens: MARKER in tokens)
    without_rr_rule = features.apply_policy(frame.assign(
        hrv_reason=frame.hrv_reason.map(reason_without_marker)), config)
    eligible = marked & without_rr_rule.curation_selected
    return {'all_data_affected': int(marked.sum()), 'core_eligible_excluded': int(eligible.sum()),
            'excluded_by_class': {label: int((eligible & frame.cognitive_load_label.eq(label)).sum())
                                  for label in ('Low', 'Medium', 'High')}}


def verify_frame(frame, registry, source_hashes):
    if any(registry.get(key) != value for key, value in settings().items()):
        raise ValueError('RR quality policy mismatch.')
    affected = set()
    for identity, audit in registry['sessions'].items():
        path = f"{audit['session_folder']}/rr_raw.csv"
        if audit['session_id'] != identity or source_hashes.get(path) != audit['rr_raw_sha256']:
            raise ValueError('RR quality source hash mismatch.')
        if not 0 <= len(audit['candidates']) <= audit['accepted_rr_count'] <= audit['task_rr_count']:
            raise ValueError('Invalid RR audit counts.')
        raw_ids = []
        for row in audit['candidates']:
            value, reference = row['rr_original_ms'], row['reference_median_ms']
            if not (math.isfinite(value) and math.isfinite(reference) and 300 <= value <= 2000
                    and abs(value-reference) > THRESHOLD_MS
                    and math.isclose(abs(value-reference), row['deviation_ms'], rel_tol=0, abs_tol=1e-9)):
                raise ValueError('Invalid RR jump evidence.')
            for timestamp in ('beat_time', 'available_at'):
                if datetime.fromisoformat(row[timestamp]).utcoffset() is None:
                    raise ValueError('RR quality timestamp must include a timezone.')
            raw_ids.append(row['raw_id'])
        if raw_ids != sorted(set(raw_ids)):
            raise ValueError('Duplicate or unordered RR audit identities.')
        affected.update(affected_windows(frame, audit))
    actual = set(frame.loc[frame.hrv_reason.fillna('').str.split(';').map(lambda v: MARKER in v), 'window_id'])
    if actual != affected:
        raise ValueError('RR window markers do not match source evidence.')
    if not set(frame.session_id).issubset(registry['sessions']):
        raise ValueError('Missing RR source audit.')


def refresh_bundle(output):
    """Reuse verified aggregates only when raw sources, settings and sensor code match."""
    from . import v2_integrity as integrity
    from .v2_feature_build import build_feature_dataset

    output = Path(output).resolve()
    previous = integrity.verify_bundle(output)
    return build_feature_dataset(Path(previous['input_directory']), output, shared.DEFAULT_CONFIG,
                                 None, previous, replace=True, reuse_verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', type=Path, required=True, help='Verified active feature-aware bundle.')
    args = parser.parse_args()
    refresh_bundle(args.refresh)


if __name__ == '__main__':
    main()
