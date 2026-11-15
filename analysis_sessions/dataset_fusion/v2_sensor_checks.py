"""Small independent numeric checks on accepted RR and prepared eye frames.

These checks verify aggregation, not physiological ground truth or gaze accuracy.
"""

from datetime import datetime, timedelta
from itertools import groupby
import math

import numpy as np

from . import build_dataset as shared
from hrv_monitor_prototype.preprocess_rr import RRPreprocessor


def _same(actual, expected, name):
    if not np.isclose(shared.finite(actual), shared.finite(expected), rtol=1e-7, atol=1e-6, equal_nan=True):
        raise ValueError(f'Independent sensor check failed: {name}: {actual} != {expected}')


def audit_sensor_samples(raw, eye, replay, origin, config):
    recording_origin = datetime.fromisoformat(raw[0]['received_at'])-timedelta(seconds=float(raw[0]['elapsed_seconds']))
    offset = (origin-recording_origin).total_seconds()
    processor = RRPreprocessor()
    processed = []
    for _, packet in groupby((r for r in raw if r.get('phase') == 'task'), key=lambda r: r['packet_id']):
        packet = [dict(r, elapsed_seconds=float(r['elapsed_seconds'])-offset) for r in packet]
        processed.extend(processor.process_packet(packet))
    result = {}
    for seconds, windows in replay.items():
        indexes = {0, len(windows)//2, len(windows)-1} if windows else set()
        statuses = {}
        for index, window in enumerate(windows):
            statuses.setdefault(window['status'], []).append(index)
            if index and window['status'] != windows[index-1]['status']:
                indexes.update((index-1, index))
        for positions in statuses.values():
            indexes.update((positions[0], positions[-1]))
        # Bounded, deterministic samples across ready/gap/recovery boundaries.
        indexes = sorted(indexes)
        if len(indexes) > 16:
            indexes = [indexes[i] for i in np.linspace(0, len(indexes)-1, 16, dtype=int)]
        for index in indexes:
            window = windows[index]
            end = (datetime.fromisoformat(window['window_end'])-origin).total_seconds()
            start = end-seconds
            rows = [r for r in processed
                    if r['elapsed_seconds'] <= end+1e-9
                    and start+1e-9 < r['beat_elapsed_seconds'] <= end+1e-9
                    and r.get('available_seconds', r['elapsed_seconds']) <= end+1e-9
                    and r['rr_processed_ms'] is not None]
            values = np.array([r['rr_processed_ms'] for r in rows], dtype=float)
            pairs = [i for i in range(1, len(rows))
                     if rows[i]['raw_id'] == rows[i-1]['raw_id']+1
                     and rows[i]['segment_id'] == rows[i-1]['segment_id']]
            rmssd = math.sqrt(sum((values[i]-values[i-1])**2 for i in pairs)/len(pairs)) if pairs else np.nan
            sdnn = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            if window['status'] != 'ready':
                rmssd = sdnn = np.nan
            for key, expected in [('rr_count', len(rows)), ('rr_pairs', len(pairs)),
                                  ('rmssd_ms', rmssd), ('sdnn_ms', sdnn)]:
                _same(window[key], expected, f'{seconds}s/{index}/{key}')
            left, right = (shared.epoch_ms(window[k]) for k in ('window_start', 'window_end'))
            frame = eye[eye.timestamp_ms.gt(left) & eye.timestamp_ms.le(right)]
            actual = shared.aggregate_eye(eye, left, right, config)
            _same(actual['eye_frames_count'], len(frame), 'eye frame count')
            bins = set(np.ceil((frame.timestamp_ms.to_numpy()-left)/config['eye_coverage_bin_ms']).astype(int)-1)
            coverage = len(bins)/round(seconds*1000/config['eye_coverage_bin_ms'])
            _same(actual['eye_temporal_coverage'], coverage, 'eye coverage')
            count = (float(frame.blink_increment.sum())
                     if len(frame) and not frame.blink_counter_bad.any() else np.nan)
            _same(actual['blink_count'], count, 'blink count')
            _same(actual['blink_rate_per_minute'], count*60/seconds, 'blink rate')
            for name in shared.MEASUREMENTS:
                values = frame[name].dropna().to_numpy(float)
                _same(actual[f'{name}_mean'], np.mean(values) if len(values) else np.nan, name+' mean')
                _same(actual[f'{name}_std'], np.std(values, ddof=1) if len(values) > 1 else np.nan, name+' std')
            states = frame.loc[frame.valid_gaze, 'Eye_State'].astype(str).tolist()
            for state in shared.STATES:
                hits = sum(s == state if state != 'other' else s not in shared.STATES[:-1] for s in states)
                _same(actual[f'eye_state_{state.lower()}_ratio'], hits/len(states) if states else np.nan, 'eye state '+state)
        result[str(seconds)] = {'sampled_windows': len(indexes), 'mismatches': 0}
    return result
