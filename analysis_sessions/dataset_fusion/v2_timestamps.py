"""Explicit source-pinned exclusions; never synthesize capture timestamps."""
import numpy as np
import pandas as pd

from . import build_dataset as shared

STRATEGY = 'exclude_windows_touching_equal_eye_timestamps'


def _groups(frame):
    times = shared.number(frame, 'timestamp_ms')
    if not np.isfinite(times).all() or times.diff().dropna().lt(0).any():
        raise ValueError('Nonfinite or backward eye timestamps cannot be repaired by tie exclusion.')
    for column in ('frame_index', 'capture_frame_id'):
        values = shared.number(frame, column)
        if (not np.isfinite(values).all() or not values.mod(1).eq(0).all()
                or values.diff().dropna().le(0).any()):
            raise ValueError('Timestamp ties require unique increasing frame and capture identities.')
    if 'timestamp_wib' in frame:
        written = pd.to_datetime(frame.timestamp_wib, utc=True).dt.as_unit('ns').astype('int64')/1e6
        if not np.isclose(written, times, rtol=0, atol=.001).all():
            raise ValueError('Eye timestamp fields disagree.')
    groups = []
    for timestamp in sorted(times[times.duplicated(keep=False)].unique()):
        positions = np.flatnonzero(times.eq(timestamp).to_numpy())
        groups.append({'timestamp_ms': float(timestamp), 'csv_lines': (positions+2).tolist(),
                       'frame_index': frame.iloc[positions].frame_index.astype(int).tolist(),
                       'capture_frame_id': frame.iloc[positions].capture_frame_id.astype(int).tolist()})
    return groups


def make_policy(directory):
    frame = pd.read_csv(directory/'eye_tracking.csv')
    groups = _groups(frame)
    if not groups:
        raise ValueError('No ambiguous timestamp groups to approve.')
    return {'strategy': STRATEGY, 'source_eye_sha256': shared.digest(directory/'eye_tracking.csv'),
            'groups': groups}


def check_policy(directory, frame, policy):
    validate_policy(policy)
    if (shared.digest(directory/'eye_tracking.csv') != policy['source_eye_sha256']
            or _groups(frame) != policy['groups']):
        raise ValueError('Audited eye source changed; timestamp handling requires a new audit.')


def validate_policy(policy):
    if (policy.get('strategy') != STRATEGY or len(policy.get('source_eye_sha256', '')) != 64
            or not policy.get('groups')):
        raise ValueError('Invalid eye timestamp policy.')
    previous = -np.inf
    for group in policy['groups']:
        timestamp = group['timestamp_ms']
        if not np.isfinite(timestamp) or timestamp <= previous:
            raise ValueError('Invalid timestamp group order.')
        previous = timestamp
        size = len(group['csv_lines'])
        if size < 2 or any(len(group[k]) != size for k in ('frame_index', 'capture_frame_id')):
            raise ValueError('Invalid timestamp group identities.')
        for key in ('csv_lines', 'frame_index', 'capture_frame_id'):
            values = group[key]
            if any(not isinstance(v, int) or v < 1 for v in values) or values != sorted(set(values)):
                raise ValueError('Invalid timestamp group identities.')


def touches(start_ms, end_ms, policy):
    return bool(policy) and any(start_ms < g['timestamp_ms'] <= end_ms for g in policy['groups'])
